import argparse
import os
import json
import sys
import time
import logging
from datetime import datetime
import traceback
import torch
from tqdm import tqdm
from threading import Lock

from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

# 固定的文件夹路径配置
MODEL_FOLDER = "/mnt/shared-storage-user/colab-share/liujiaheng/pjlab-oss/models/Llava"
OUTPUT_FOLDER = "response"
LOG_FOLDER = "logs"


class VideoLLaVAProcessor:
    """Video-LLaVA 视频对比处理主类"""
    
    # 添加类级别的文件锁
    _output_file_lock = Lock()
    
    def __init__(self, config):
        """初始化处理器"""
        disable_torch_init()
        
        self.model_name = config.get('model_name', 'Video-LLaVA')
        self.model_path = config.get('model_path')
        
        self.model_path = os.path.join(MODEL_FOLDER, "Video-LLaVA")
        
        self.input_json_file = config.get('input_json_file', 'input_videos.json')
        self.prompt_file = config.get('prompt_file', 'prompt_generate.txt')
        self.thinking = config.get('thinking', False)
        self.device = config.get('device', 'cuda:0')
        
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
        
        # 设置CUDA设备
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.cuda.set_device(0)
        
        # 初始化 Video-LLaVA 模型
        self.logger.info(f"正在加载 Video-LLaVA 模型: {self.model_path}")
        self.logger.info(f"启用思考模式: {self.thinking}")
        
        try:
            cache_dir = ''
            load_4bit, load_8bit = False, False
            model_name = get_model_name_from_path(self.model_path)
            
            self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
                self.model_path, None, model_name, load_8bit, load_4bit, 
                device=self.device, cache_dir=cache_dir, local_files_only=True
            )
            self.model = self.model.to(self.device)
            self.video_processor = self.processor['video']
            self.conv_mode = "llava_v1"
            
            if next(self.model.parameters()).device.type == 'cuda':
                self.logger.info("✅ Video-LLaVA 模型加载成功 (GPU)")
            else:
                self.logger.warning("⚠️ 模型在CPU上运行，性能可能受影响")
                
        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {e}")
            raise
        
        # 生成配置
        if self.thinking:
            self.temperature = 0.6
            self.max_new_tokens = 8192
        else:
            self.temperature = 0.1
            self.max_new_tokens = 2048
        
        self.logger.info(f"配置信息:")
        self.logger.info(f"  - 模型名称: {self.model_name}")
        self.logger.info(f"  - 模型路径: {self.model_path}")
        self.logger.info(f"  - 输入文件: {self.input_json_file}")
        self.logger.info(f"  - 输出文件: {self.output_file}")
        self.logger.info(f"  - 提示词文件: {self.prompt_file}")
        self.logger.info(f"  - 思考模式: {'启用' if self.thinking else '禁用'}")
        self.logger.info(f"  - 温度: {self.temperature}")
        self.logger.info(f"  - 最大生成tokens: {self.max_new_tokens}")
        
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
            R1_SYSTEM_PROMPT = """You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within 
 tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.

"""
            self.system_prompt = R1_SYSTEM_PROMPT + self.system_prompt
        
        # 初始化或加载已有的结果文件
        self._initialize_output_file()
    
    def _setup_logging(self):
        """设置日志配置"""
        log_dir = os.path.join(LOG_FOLDER, self.model_name)
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"processing_{timestamp}.log")
        error_log_file = os.path.join(log_dir, f"errors_{timestamp}.log")
        
        self.logger = logging.getLogger(f"VideoLLaVAProcessor_{self.model_name}")
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
    
    def process_video(self, video_path: str, prompt_text: str):
        """处理单个视频"""
        conv = conv_templates[self.conv_mode].copy()
        
        # 处理视频
        video_tensor = self.video_processor(video_path, return_tensors='pt')['pixel_values']
        if type(video_tensor) is list:
            tensor = [v.to(self.model.device, dtype=torch.float16) for v in video_tensor]
        else:
            tensor = video_tensor.to(self.model.device, dtype=torch.float16)
        
        # 准备输入
        inp = ' '.join([DEFAULT_IMAGE_TOKEN] * self.model.get_video_tower().config.num_frames) + '\n' + prompt_text
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        real_prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(real_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        input_ids = input_ids.to(self.model.device)
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=tensor,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )
        
        response = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        return response
    
    def process_video_pair(self, entry):
        """处理视频对"""
        video1_path = entry['video1_path']
        video2_path = entry['video2_path']
        
        if not os.path.exists(video1_path):
            raise FileNotFoundError(f"视频文件未找到: {video1_path}")
        if not os.path.exists(video2_path):
            raise FileNotFoundError(f"视频文件未找到: {video2_path}")
        
        # 构建完整的提示
        full_prompt = f"{self.system_prompt}\n\n"
        full_prompt += f"Please analyze these two videos and provide a comparison.\n"
        full_prompt += f"Source video: {video1_path}\n"
        full_prompt += f"Destination video: {video2_path}\n"
        
        # 注意：Video-LLaVA 可能一次只能处理一个视频
        # 这里我们分别处理两个视频，然后合并结果
        try:
            # 处理第一个视频
            prompt1 = f"{self.system_prompt}\n\nAnalyze this source video and describe its content in detail:"
            response1 = self.process_video(video1_path, prompt1)
            
            # 处理第二个视频
            prompt2 = f"Now analyze this destination video and describe its content in detail:"
            response2 = self.process_video(video2_path, prompt2)
            
            # 合并响应并生成对比分析
            combined_prompt = f"Based on the following analysis of two videos, provide a comprehensive comparison:\n\n"
            combined_prompt += f"Source video analysis:\n{response1}\n\n"
            combined_prompt += f"Destination video analysis:\n{response2}\n\n"
            combined_prompt += "Please compare these two videos in terms of content, style, and differences."
            
            # 由于Video-LLaVA可能不支持纯文本输入，我们直接使用第一个视频作为视觉上下文
            final_response = self.process_video(video1_path, combined_prompt)
            
            # 如果启用了思考模式，提取最终答案
            if self.thinking and '</think>' in final_response:
                final_response = final_response.split('</think>', 1)[-1].strip()
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"视频对处理过程中发生错误: {str(e)}")
            raise e
    
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
        self.logger.info("开始处理 (使用 Video-LLaVA 模型)")
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
        description='Video-LLaVA 视频对比分析处理程序',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model_name', type=str, default='videollava',
                       help='模型名称')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型路径（如不指定则使用默认路径）')
    parser.add_argument('--input_json', type=str, default='videos.json', 
                       help='输入JSON文件路径')
    parser.add_argument('--prompt_file', type=str, default='prompt_generate.txt', 
                       help='系统提示词文件路径')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='使用的设备')
    parser.add_argument('-t', '--thinking', action='store_true',
                       help='启用思考模式')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("="*60)
    print("Video-LLaVA 视频对比分析处理程序")
    print("增量写入模式：已启用")
    print("文件锁保护：已启用")
    print("="*60)
    
    config = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "input_json_file": args.input_json,
        "prompt_file": args.prompt_file,
        "device": args.device,
        "thinking": args.thinking,
    }
    
    print(f"配置信息:")
    print(f"  - 模型名称: {config['model_name']}")
    print(f"  - 输入文件: {config['input_json_file']}")
    
    # 根据思考模式选择输出文件名
    suffix = "_thinking_results.json" if config['thinking'] else "_nothinking_results.json"
    print(f"  - 输出文件: {os.path.join(OUTPUT_FOLDER, config['model_name'] + suffix)}")
    print(f"  - 日志目录: {os.path.join(LOG_FOLDER, config['model_name'])}")
    print(f"  - 设备: {config['device']}")
    print(f"  - 思考模式: {'启用' if config['thinking'] else '禁用'}")
    print("="*60)
    
    try:
        processor = VideoLLaVAProcessor(config)
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
