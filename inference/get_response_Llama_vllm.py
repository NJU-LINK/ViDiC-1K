import os
import json
import sys
from pathlib import Path
import time
import logging
from datetime import datetime
import traceback
import torch
from threading import Lock
import numpy as np
from decord import VideoReader, cpu
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from modelscope import AutoProcessor
from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse

# 固定的文件夹路径配置
MODEL_FOLDER = "/mnt/shared-storage-user/colab-share/liujiaheng/pjlab-oss/models/Llama/LLM-Research"  # 模型文件夹路径
OUTPUT_FOLDER = "response"  # 输出文件夹
LOG_FOLDER = "log"  # 日志文件夹

# 图像处理常量
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VideoProcessor:
    """视频处理主类 - 使用本地 Llama 模型"""
    
    # 添加类级别的文件锁
    _output_file_lock = Lock()
  
    def __init__(self, config):
        """初始化处理器"""
        self.model_name = config.get('model_name')
        self.model_path = os.path.join(MODEL_FOLDER, self.model_name)
        self.input_json_file = config.get('input_json_file', 'input_videos.json')
        self.batch_size = config.get('batch_size', 1)
        self.prompt_file = config.get('prompt_file', 'prompt_generate.txt')
        self.gpu_memory_utilization = config.get('gpu_memory_utilization', 0.85)
        
        # 视频处理配置
        self.input_size = 448
        self.fps = 1.0
        self.max_num = 1
        self.max_frames = 32
        
        # 设置输出文件路径
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        self.output_file = os.path.join(OUTPUT_FOLDER, f"{self.model_name}_results.json")
        
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
        
        self.tensor_parallel_size = gpu_count
        self.logger.info(f"✅ 检测到 {gpu_count} 个GPU")
        
        # 打印GPU信息
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            self.logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
      
        # 初始化 vLLM 模型
        self.logger.info(f"正在加载 Llama 模型: {self.model_path}")
        self.logger.info(f"GPU 内存利用率设置: {self.gpu_memory_utilization}")
        
        try:
            self.model = LLM(
                model=self.model_path,
                trust_remote_code=True,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization= 0.85,
                enforce_eager=True,
                max_model_len=60960,
                max_num_seqs= 1,
                limit_mm_per_prompt={"image": 32, "video": 2}
            )
            self.logger.info("✅ Llama 模型加载成功")
        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {e}")
            raise
      
        # 设置采样参数
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            max_tokens=4096,
            stop_token_ids=[],
        )
      
        # 加载处理器
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            self.logger.info("✅ Processor 加载成功")
        except Exception as e:
            self.logger.error(f"❌ Processor 加载失败: {e}")
            raise
      
        self.logger.info(f"配置信息:")
        self.logger.info(f"  - 模型名称: {self.model_name}")
        self.logger.info(f"  - 模型路径: {self.model_path}")
        self.logger.info(f"  - 批处理大小: {self.batch_size}")
        self.logger.info(f"  - 输入文件: {self.input_json_file}")
        self.logger.info(f"  - 输出文件: {self.output_file}")
        self.logger.info(f"  - 提示词文件: {self.prompt_file}")
        self.logger.info(f"  - GPU显存利用率: {self.gpu_memory_utilization}")
      
        # 统计信息
        self.successful = 0
        self.failed = 0
        self.skipped_processed = 0
        self.start_time = None
      
        # 从输出文件加载已处理的记录
        self.processed_indices = self._load_processed_indices()
      
        # 提示词
        self.system_prompt = self._load_system_prompt()
      
        # 初始化或加载已有的结果文件
        self._initialize_output_file()
    
    def _setup_logging(self):
        """设置日志配置"""
        # 创建模型专属的日志目录
        log_dir = os.path.join(LOG_FOLDER, self.model_name)
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"processing_{timestamp}.log")
        error_log_file = os.path.join(log_dir, f"errors_{timestamp}.log")
        
        # 创建专属的logger
        self.logger = logging.getLogger(f"VideoProcessor_{self.model_name}")
        self.logger.setLevel(logging.INFO)
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 错误日志记录器
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
        """初始化输出文件（支持增量写入）"""
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
        """增量写入单个结果到文件（带文件锁）"""
        with self._output_file_lock:
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
                if not isinstance(data, list):
                    data = []
            
                # 只保留需要的字段
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
                # 备份处理
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
  
    def _validate_video_files(self, video1_path, video2_path):
        """验证视频文件并记录大小信息"""
        if not os.path.exists(video1_path):
            raise FileNotFoundError(f"视频文件不存在: {video1_path}")
        if not os.path.exists(video2_path):
            raise FileNotFoundError(f"视频文件不存在: {video2_path}")
        
        try:
            size1_mb = os.path.getsize(video1_path) / (1024 * 1024)
            size2_mb = os.path.getsize(video2_path) / (1024 * 1024)
            self.logger.info(f"视频1大小: {size1_mb:.2f}MB, 视频2大小: {size2_mb:.2f}MB")
            
            max_size_mb = 500
            if size1_mb > max_size_mb:
                self.logger.warning(f"⚠️ 视频1文件较大 ({size1_mb:.2f}MB)，可能影响处理速度")
            if size2_mb > max_size_mb:
                self.logger.warning(f"⚠️ 视频2文件较大 ({size2_mb:.2f}MB)，可能影响处理速度")
                
        except Exception as e:
            self.logger.warning(f"无法获取文件大小信息: {e}")
    
    # 视频处理相关函数
    def build_transform(self, input_size):
        """构建图像变换"""
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """找到最接近的宽高比"""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """动态预处理图像"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def get_index(self, bound, fps, max_frame, first_idx=0, num_segments=16):
        """获取帧索引"""
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices

    def load_images_from_video(self, video_path, bound=None, input_size=448, max_num=1, num_segments=32, max_frames=32):
        """从视频加载图像帧"""
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        # 获取帧索引
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        if len(frame_indices) > max_frames:
            frame_indices = frame_indices[:max_frames]  # 限制最大帧数
        
        pil_images = []
        for frame_index in frame_indices:
            # 读取帧并转换为 PIL Image
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            
            # 应用动态分块预处理，这会返回一个 PIL Image 列表
            processed_imgs = self.dynamic_preprocess(img, image_size=input_size, use_thumbnail=False, max_num=max_num)
            
            # 将处理后的图像块添加到主列表中
            pil_images.extend(processed_imgs)
            
        return pil_images

    def process_video_pairs_batch(self, entries):
        """批量处理视频对"""
        llm_inputs = []
        
        for entry in entries:
            video1_path = entry['video1_path']
            video2_path = entry['video2_path']
            
            self._validate_video_files(video1_path, video2_path)
            
            # 加载两个视频的帧
            pil_images1 = self.load_images_from_video(
                video1_path,
                input_size=self.input_size,
                max_num=self.max_num,
                num_segments=16, 
                max_frames=16
            )
            
            pil_images2 = self.load_images_from_video(
                video2_path,
                input_size=self.input_size,
                max_num=self.max_num,
                num_segments=16,  
                max_frames=16
            )
            
            # 合并所有图像
            all_images = pil_images1 + pil_images2
            
            # 构建消息内容
            content = []
            
            # 添加视频1的图像占位符
            content.append({"type": "text", "text": "Source video:"})
            for _ in range(len(pil_images1)):
                content.append({"type": "image"})
            
            # 添加视频2的图像占位符
            content.append({"type": "text", "text": "\nDestination video:"})
            for _ in range(len(pil_images2)):
                content.append({"type": "image"})
            
            # 添加系统提示词
            content.append({"type": "text", "text": f"\n{self.system_prompt}"})
            
            # 构建完整的消息结构
            messages = [{"role": "user", "content": content}]
            
            # 应用聊天模板
            text_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            # 准备多模态数据
            multi_modal_data = {
                "image": all_images
            }
            
            # 构建vLLM输入
            llm_inputs.append({
                "prompt": text_prompt,
                "multi_modal_data": multi_modal_data,
            })
        
        # 批量生成
        outputs = self.model.generate(llm_inputs, self.sampling_params)
        generated_texts = [output.outputs[0].text.strip() for output in outputs]
        
        return generated_texts
  
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
        self.logger.info("开始批量处理 (使用本地 Llama 模型)")
        self.logger.info(f"批处理大小: {self.batch_size}")
        self.logger.info(f"增量写入模式: 启用")
        self.logger.info("="*60)
      
        with tqdm(total=pending, desc="处理进度") as pbar:
            for batch_start in range(0, pending, self.batch_size):
                batch_end = min(batch_start + self.batch_size, pending)
                batch_entries = pending_data[batch_start:batch_end]
              
                self.logger.info(f"\n处理批次 {batch_start//self.batch_size + 1}: {len(batch_entries)} 个视频对")
              
                max_retries = 3
                retry_count = 0
                success = False
              
                while retry_count < max_retries and not success:
                    try:
                        responses = self.process_video_pairs_batch(batch_entries)
                      
                        for entry, response in zip(batch_entries, responses):
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
                        pbar.update(len(batch_entries))
                      
                    except Exception as e:
                        self.logger.error(f"批处理错误: {str(e)}")
                        self.error_logger.error(f"批处理错误详情: {traceback.format_exc()}")
                        retry_count += 1
                      
                        if retry_count < max_retries:
                            self.logger.warning(f"正在重试 {retry_count}/{max_retries}...")
                            time.sleep(2)
                            
                            # 清理内存
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            self.logger.warning("批处理失败，尝试逐个处理...")
                            for entry in batch_entries:
                                try:
                                    responses = self.process_video_pairs_batch([entry])
                                  
                                    result = {
                                        "index": entry['index'],
                                        "video1_path": entry['video1_path'],
                                        "video2_path": entry['video2_path'],
                                        "response": responses[0]
                                    }
                                  
                                    self._append_result_to_file(result)
                                    self.processed_indices.add(entry['index'])
                                    self.successful += 1
                                    self.logger.info(f"[Entry {entry['index']}] ✅ 单独处理成功")
                                    pbar.update(1)
                                  
                                except Exception as e2:
                                    self.failed += 1
                                    self.logger.error(f"[Entry {entry['index']}] ❌ 处理失败: {str(e2)}")
                                  
                                    error_info = {
                                        "index": entry['index'],
                                        "video1_path": entry['video1_path'],
                                        "video2_path": entry['video2_path'],
                                        "error": str(e2),
                                        "timestamp": datetime.now().isoformat()
                                    }
                                    self._log_error(error_info)
                                    pbar.update(1)
                          
                            success = True
                
                # 批次处理后清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
      
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
        description='视频对比分析处理程序 - 使用本地 Llama 模型',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model_name', type=str, required=True, help='模型名称（模型文件夹下的子文件夹名）')
    parser.add_argument('--input_json', type=str, default='videos.json', help='输入JSON文件路径')
    parser.add_argument('--prompt_file', type=str, default='prompt_generate.txt', help='系统提示词文件路径')
    parser.add_argument('--batch_size', type=int, default=2, help='批处理大小')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95, help='GPU显存利用率 (0.0-1.0)')
    
    return parser.parse_args()


def main():
    """主函数"""
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    args = parse_args()
    
    print("="*60)
    print("视频对比分析处理程序")
    print("使用本地 Llama 模型")
    print("增量写入模式：已启用")
    print("文件锁保护：已启用")
    print("="*60)
    
    config = {
        "model_name": args.model_name,
        "input_json_file": args.input_json,
        "prompt_file": args.prompt_file,
        "batch_size": args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    
    print(f"配置信息:")
    print(f"  - 模型名称: {config['model_name']}")
    print(f"  - 模型路径: {os.path.join(MODEL_FOLDER, config['model_name'])}")
    print(f"  - 输入文件: {config['input_json_file']}")
    print(f"  - 输出文件: {os.path.join(OUTPUT_FOLDER, config['model_name'] + '_results.json')}")
    print(f"  - 日志目录: {os.path.join(LOG_FOLDER, config['model_name'])}")
    print(f"  - 批处理大小: {config['batch_size']}")
    print(f"  - GPU显存利用率: {config['gpu_memory_utilization']}")
    print("="*60)
    
    try:
        processor = VideoProcessor(config)
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
