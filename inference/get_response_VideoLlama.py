import argparse
import os
import json
import sys
import time
import logging
from datetime import datetime
import traceback
import torch
import av
import numpy as np
from PIL import Image
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm
from threading import Lock

# 固定的文件夹路径配置
MODEL_FOLDER = "/mnt/shared-storage-user/colab-share/liujiaheng/pjlab-oss/models/VideoLlama"
OUTPUT_FOLDER = "response"
LOG_FOLDER = "logs"


def load_video_frames(video_path: str, max_frames: int = 8, image_size: int = 224) -> List[Image.Image]:
    """
    从视频中提取固定数量的帧（32帧），并进行预处理
    """
    frames = []
    
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        
        # 获取视频总帧数
        total_frames = stream.frames
        if total_frames == 0:
            frame_count = 0
            for _ in container.decode(stream):
                frame_count += 1
            total_frames = frame_count
            container.seek(0)
        
        # 固定采样帧数
        target_frame_count = max_frames
        
        # 计算采样的帧索引
        if total_frames <= target_frame_count:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, target_frame_count, dtype=int).tolist()
        
        # 采样帧
        frame_idx = 0
        collected_indices = set(frame_indices)
        
        for frame in container.decode(stream):
            if frame_idx in collected_indices:
                img = frame.to_image()
                
                # 调整图像大小
                if img.size != (image_size, image_size):
                    img = img.resize((image_size, image_size), Image.LANCZOS)
                
                frames.append(img)
                
                if len(frames) >= len(frame_indices):
                    break
            
            frame_idx += 1
        
        container.close()
        
    except Exception as e:
        raise Exception(f"Error loading video {video_path}: {str(e)}")
    
    return frames


class VideoLLaMAProcessor:
    """VideoLLaMA 视频对比处理主类"""
    
    # 添加类级别的文件锁
    _output_file_lock = Lock()
    
    def __init__(self, config):
        """初始化处理器"""
        self.model_name = config.get('model_name')
        self.model_path = os.path.join(MODEL_FOLDER, self.model_name)
        self.input_json_file = config.get('input_json_file', 'input_videos.json')
        self.prompt_file = config.get('prompt_file', 'prompt_generate.txt')
        self.thinking = config.get('thinking', False)
        self.max_frames_per_video = config.get('max_frames_per_video', 8)
        self.fps = 1.0  # VideoLLaMA 默认设置
        
        # 设置输出文件路径
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        if self.thinking:
            self.output_file = os.path.join(OUTPUT_FOLDER, f"{self.model_name}_thinking_results.json")
        else:
            self.output_file = os.path.join(OUTPUT_FOLDER, f"{self.model_name}_nothinking_results.json")
        
        # 设置日志
        self._setup_logging()
        
        # 验证输入文件存在
        if not os.path.exists(self.input_json_file):
            raise FileNotFoundError(f"输入JSON文件不存在: {self.input_json_file}")
        
        # 验证模型路径存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
        
        # 检测 GPU 数量
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            self.logger.error("❌ 未检测到GPU，此脚本需要GPU支持")
            raise RuntimeError("需要GPU才能运行此脚本")
        
        self.logger.info(f"✅ 检测到 {gpu_count} 个GPU")
        
        # 打印GPU信息
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            self.logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        
        # 初始化 VideoLLaMA 模型
        self.logger.info(f"正在加载 VideoLLaMA 模型: {self.model_path}")
        self.logger.info(f"启用思考模式: {self.thinking}")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.logger.info("✅ VideoLLaMA 模型加载成功")
        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {e}")
            raise
        
        # 生成参数
        self.gen_kwargs = {
            "max_new_tokens": 1024 if self.thinking else 512,
            "temperature": 0.6 if self.thinking else 0.1,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
            "do_sample": True
        }
        
        self.logger.info(f"配置信息:")
        self.logger.info(f"  - 模型名称: {self.model_name}")
        self.logger.info(f"  - 模型路径: {self.model_path}")
        self.logger.info(f"  - 每个视频最大帧数: {self.max_frames_per_video}")
        self.logger.info(f"  - 输入文件: {self.input_json_file}")
        self.logger.info(f"  - 输出文件: {self.output_file}")
        self.logger.info(f"  - 提示词文件: {self.prompt_file}")
        self.logger.info(f"  - 思考模式: {'启用' if self.thinking else '禁用'}")
        
        # 统计信息
        self.successful = 0
        self.failed = 0
        self.skipped_processed = 0
        self.start_time = None
        
        # 从输出文件加载已处理的记录
        self.processed_indices = self._load_processed_indices()
        
        # 提示词
        self.system_prompt = self._load_system_prompt()
        
        # 如果启用思考模式，添加思考提示
        if self.thinking:
            self.thinking_prompt = "\n\nPlease think step by step before answering. Start your thinking process with 'Thinking:' and then provide the final answer after 'Answer:'"
        
        # 初始化或加载已有的结果文件
        self._initialize_output_file()
    
    def _setup_logging(self):
        """设置日志配置"""
        log_dir = os.path.join(LOG_FOLDER, self.model_name)
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"processing_{timestamp}.log")
        error_log_file = os.path.join(log_dir, f"errors_{timestamp}.log")
        
        self.logger = logging.getLogger(f"VideoLLaMAProcessor_{self.model_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.error_logger = logging.getLogger(f"error_logger_{self.model_name}")
        self.error_logger.setLevel(logging.ERROR)
        self.error_logger.handlers.clear()
        
        error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
        error_handler.setFormatter(logging.Formatter('%(asctime)s - [ERROR] - %(message)s'))
        self.error_logger.addHandler(error_handler)
    
    def _load_processed_indices(self):
        """从输出文件加载已处理的索引"""
        processed = set()
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if 'index' in item:
                                processed.add(item['index'])
                        self.logger.info(f"从输出文件加载了 {len(processed)} 条已处理记录")
            except Exception as e:
                self.logger.warning(f"加载已处理记录失败: {e}")
        return processed
    
    def _initialize_output_file(self):
        """初始化输出文件"""
        with self._output_file_lock:
            if os.path.exists(self.output_file):
                try:
                    with open(self.output_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        if isinstance(existing_data, list):
                            self.logger.info(f"输出文件已存在，包含 {len(existing_data)} 条历史记录")
                        else:
                            with open(self.output_file, 'w', encoding='utf-8') as f:
                                json.dump([], f, ensure_ascii=False)
                            self.logger.info("输出文件格式错误，已重新初始化")
                except (json.JSONDecodeError, Exception) as e:
                    backup_file = f"{self.output_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    os.rename(self.output_file, backup_file)
                    self.logger.warning(f"输出文件读取失败，已备份至: {backup_file}")
                    with open(self.output_file, 'w', encoding='utf-8') as f:
                        json.dump([], f, ensure_ascii=False)
            else:
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False)
                self.logger.info("创建新的输出文件")
    
    def _append_result_to_file(self, result):
        """增量写入单个结果到文件"""
        with self._output_file_lock:
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    data = []
                
                clean_result = {
                    "index": result["index"],
                    "video1_path": result["video1_path"],
                    "video2_path": result["video2_path"],
                    "response": result["response"]
                }
                data.append(clean_result)
                
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                self.logger.debug(f"成功增量写入结果，当前共 {len(data)} 条记录")
                
            except Exception as e:
                self.logger.error(f"增量写入失败: {e}")
                backup_file = f"{self.output_file}.incremental"
                try:
                    if os.path.exists(backup_file):
                        with open(backup_file, 'r', encoding='utf-8') as f:
                            backup_data = json.load(f)
                    else:
                        backup_data = []
                    
                    clean_result = {
                        "index": result["index"],
                        "video1_path": result["video1_path"],
                        "video2_path": result["video2_path"],
                        "response": result["response"]
                    }
                    backup_data.append(clean_result)
                    
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        json.dump(backup_data, f, ensure_ascii=False, indent=2)
                    
                    self.logger.warning(f"结果已保存到备用文件: {backup_file}")
                except Exception as e2:
                    self.logger.error(f"备用文件写入也失败: {e2}")
    
    def _load_system_prompt(self):
        """加载系统提示词"""
        prompt_path = self.prompt_file
        
        if not os.path.exists(prompt_path):
            error_msg = f"❌ 错误: 提示词文件不存在: {prompt_path}"
            self.logger.error(error_msg)
            print("\n" + "="*60)
            print(error_msg)
            print("请创建提示词文件后再运行程序！")
            print("="*60)
            sys.exit(1)
        
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            
            if not prompt:
                error_msg = f"❌ 错误: 提示词文件为空: {prompt_path}"
                self.logger.error(error_msg)
                print("\n" + "="*60)
                print(error_msg)
                print("请在提示词文件中添加内容！")
                print("="*60)
                sys.exit(1)
            
            self.logger.info(f"✅ 成功加载系统提示词文件: {prompt_path}")
            self.logger.info(f"提示词长度: {len(prompt)} 字符")
            
            return prompt
            
        except Exception as e:
            error_msg = f"❌ 错误: 读取提示词文件失败: {e}"
            self.logger.error(error_msg)
            sys.exit(1)
    
    def _log_error(self, error_info):
        """记录错误信息到日志"""
        self.error_logger.error(json.dumps(error_info, ensure_ascii=False, indent=2))
    
    def load_input_data(self):
        """从JSON文件加载输入数据"""
        self.logger.info(f"开始加载输入文件: {self.input_json_file}")
        
        if not os.path.exists(self.input_json_file):
            raise FileNotFoundError(f"输入文件不存在: {self.input_json_file}")
        
        data_list = []
        
        try:
            with open(self.input_json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            if isinstance(json_data, list):
                for idx, item in enumerate(json_data):
                    if 'video1_path' in item and 'video2_path' in item:
                        entry = {
                            'index': idx,
                            'video1_path': item['video1_path'],
                            'video2_path': item['video2_path']
                        }
                        data_list.append(entry)
                    else:
                        self.logger.warning(f"第{idx}项缺少必要的视频路径字段")
            elif isinstance(json_data, dict):
                video_pairs = json_data.get('video_pairs', json_data.get('data', [json_data]))
                if isinstance(video_pairs, list):
                    for idx, item in enumerate(video_pairs):
                        if 'video1_path' in item and 'video2_path' in item:
                            entry = {
                                'index': idx,
                                'video1_path': item['video1_path'],
                                'video2_path': item['video2_path']
                            }
                            data_list.append(entry)
                elif 'video1_path' in json_data and 'video2_path' in json_data:
                    entry = {
                        'index': 0,
                        'video1_path': json_data['video1_path'],
                        'video2_path': json_data['video2_path']
                    }
                    data_list.append(entry)
            
            self.logger.info(f"✅ 成功加载 {len(data_list)} 条数据")
            return data_list
            
        except Exception as e:
            self.logger.error(f"加载输入文件失败: {e}")
            raise
    
    def process_video_pair(self, entry):
        self.logger.info("开始=======")
        video1_path = entry['video1_path']
        video2_path = entry['video2_path']
        
        # 步骤 1: 手动加载两个视频的帧
        self.logger.info(f"手动加载帧: {video1_path}")
        video1_frames = load_video_frames(video1_path, max_frames=self.max_frames_per_video)
        
        self.logger.info(f"手动加载帧: {video2_path}")
        video2_frames = load_video_frames(video2_path, max_frames=self.max_frames_per_video)

        # 步骤 2: 将所有 PIL 图像合并到一个列表中
        all_pil_images = video1_frames + video2_frames
        
        self.logger.info("FUCK")
        if video1_frames == None:
            self.logger.info("videoframes none")
        # 步骤 3: 构建包含文本和图像占位符的 content 列表
        content = []
        content.append({"type": "text", "text": "Source video:"})
        for _ in range(len(video1_frames)):
            content.append({"type": "image"})
        
        content.append({"type": "text", "text": "\nDestination video:"})
        for _ in range(len(video2_frames)):
            content.append({"type": "image"})
            
        content.append({"type": "text", "text": "\nPlease analyze these two videos and provide a comparison."})

        # 准备系统提示
        system_content = self.system_prompt
        if self.thinking:
            system_content += self.thinking_prompt
            
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": content}
        ]
        self.logger.info("开始应用聊天模板")
        # 步骤 4: 应用聊天模板
        try:
            text_prompt_with_placeholders = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception as e:
            raise RuntimeError(f"应用聊天模板失败: {e}. 请检查您的 transformers 库版本。")
        
        self.logger.info("处理文本图像")
        # 步骤 5: 联合处理文本和图像
        inputs = self.processor(
            text=text_prompt_with_placeholders,
            images=all_pil_images,
            return_tensors="pt"
        )
        self.logger.info("转移设备，移动张量")
        # 步骤 6: 健壮性检查和设备转移 (关键修复)
        if inputs is None:
            raise RuntimeError(f"Processor returned None for video pair ({video1_path}, {video2_path}).")

        # 将所有 Tensor 移动到模型所在的设备
        inputs = {k: v.to(self.model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        # 确保 pixel_values 是正确的 bfloat16 类型
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        
        # 清理内
        self.logger.info("输出")
        # 生成输出
        output_ids = self.model.generate(**inputs, **self.gen_kwargs)
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # 提取最终答案
        if self.thinking and 'Answer:' in response:
            response = response.split('Answer:', 1)[-1].strip()
        
        return response

    
    def process_all(self):
        """处理所有数据"""
        self.start_time = time.time()
        
        data_list = self.load_input_data()
        
        if not data_list:
            self.logger.info("没有需要处理的数据")
            return
        
        pending_data = [entry for entry in data_list if entry['index'] not in self.processed_indices]
        
        if not pending_data:
            self.logger.info("✅ 所有数据已处理完成")
            return
        
        total = len(data_list)
        pending = len(pending_data)
        
        self.logger.info(f"总数据: {total} 条")
        self.logger.info(f"已处理: {len(self.processed_indices)} 条")
        self.logger.info(f"待处理: {pending} 条")
        
        self.logger.info("="*60)
        self.logger.info("开始处理 (使用 VideoLLaMA 模型)")
        self.logger.info(f"增量写入模式: 启用")
        self.logger.info("="*60)
        
        with tqdm(total=pending, desc="处理进度") as pbar:
            for entry in pending_data:
                self.logger.info(f"\n处理视频对 {entry['index']}")
                
                max_retries = 3
                retry_count = 0
                success = False
                
                while retry_count < max_retries and not success:
                    try:
                        response = self.process_video_pair(entry)
                        
                        result = {
                            "index": entry['index'],
                            "video1_path": entry['video1_path'],
                            "video2_path": entry['video2_path'],
                            "response": response
                        }
                        
                        self._append_result_to_file(result)
                        self.processed_indices.add(entry['index'])
                        
                        self.successful += 1
                        self.logger.info(f"[Entry {entry['index']}] ✅ 处理成功并已保存")
                        
                        success = True
                        pbar.update(1)
                        
                    except Exception as e:
                        self.logger.error(f"处理错误: {str(e)}")
                        self.error_logger.error(f"处理错误详情: {traceback.format_exc()}")
                        retry_count += 1
                        
                        if retry_count < max_retries:
                            self.logger.warning(f"正在重试 {retry_count}/{max_retries}...")
                            time.sleep(2)
                        else:
                            self.failed += 1
                            self.logger.error(f"[Entry {entry['index']}] ❌ 处理失败")
                            
                            error_info = {
                                "index": entry['index'],
                                "video1_path": entry['video1_path'],
                                "video2_path": entry['video2_path'],
                                "error": str(e),
                                "timestamp": datetime.now().isoformat()
                            }
                            self._log_error(error_info)
                            pbar.update(1)
        
        self.logger.info(f"✅ 所有结果已保存到: {self.output_file}")
        self.print_summary()
    
    def print_summary(self):
        """打印处理摘要"""
        elapsed = time.time() - self.start_time
        total_processed = self.successful + self.failed
        
        self.logger.info("\n" + "="*60)
        self.logger.info("处理完成 - 统计摘要")
        self.logger.info("="*60)
        self.logger.info(f"总耗时: {elapsed/60:.2f} 分钟")
        self.logger.info(f"处理总数: {total_processed}")
        self.logger.info(f"成功: {self.successful}")
        self.logger.info(f"失败: {self.failed}")
        self.logger.info(f"跳过: {self.skipped_processed}")
        
        if total_processed > 0:
            self.logger.info(f"成功率: {self.successful/total_processed*100:.2f}%")
            self.logger.info(f"平均处理时间: {elapsed/total_processed:.2f} 秒/条")
        
        self.logger.info(f"\n输出文件: {self.output_file}")
        self.logger.info(f"日志目录: {os.path.join(LOG_FOLDER, self.model_name)}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='VideoLLaMA 视频对比分析处理程序',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model_name', type=str, required=True, 
                       help='模型名称（模型文件夹下的子文件夹名）')
    parser.add_argument('--input_json', type=str, default='videos.json', 
                       help='输入JSON文件路径')
    parser.add_argument('--prompt_file', type=str, default='prompt_generate.txt', 
                       help='系统提示词文件路径')
    parser.add_argument('--max_frames', type=int, default=8, 
                       help='每个视频最大帧数')
    parser.add_argument('-t', '--thinking', action='store_true',
                       help='启用思考模式')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("="*60)
    print("VideoLLaMA 视频对比分析处理程序")
    print("增量写入模式：已启用")
    print("文件锁保护：已启用")
    print("="*60)
    
    config = {
        "model_name": args.model_name,
        "input_json_file": args.input_json,
        "prompt_file": args.prompt_file,
        "max_frames_per_video": args.max_frames,
        "thinking": args.thinking,
    }
    
    print(f"配置信息:")
    print(f"  - 模型名称: {config['model_name']}")
    print(f"  - 模型路径: {os.path.join(MODEL_FOLDER, config['model_name'])}")
    print(f"  - 输入文件: {config['input_json_file']}")
    
    # 根据思考模式选择输出文件名
    suffix = "_thinking_results.json" if config['thinking'] else "_nothinking_results.json"
    print(f"  - 输出文件: {os.path.join(OUTPUT_FOLDER, config['model_name'] + suffix)}")
    print(f"  - 日志目录: {os.path.join(LOG_FOLDER, config['model_name'])}")
    print(f"  - 每个视频最大帧数: {config['max_frames_per_video']}")
    print(f"  - 思考模式: {'启用' if config['thinking'] else '禁用'}")
    print("="*60)
    
    try:
        processor = VideoLLaMAProcessor(config)
        processor.process_all()
        print("\n✅ 处理完成!")
    except KeyboardInterrupt:
        print("\n⚠️ 处理被用户中断")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
