#!/usr/bin/env python3
"""
视频帧提取对比分析工具 - LLaVA-Video本地版本（完全离线）
修复版 - 正确调用LLaVA-Video模型
"""

import os
import json
import sys

# 设置离线模式 - 必须在导入transformers之前设置
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
# 清除代理设置
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import logging
from datetime import datetime
import traceback
import torch
import copy
import warnings
import numpy as np
from decord import VideoReader, cpu

# 从 LLaVA-NeXT 库中导入必要的模块
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

# 忽略不必要的警告
warnings.filterwarnings("ignore")

sys.stdout.reconfigure(encoding='utf-8')

# 创建日志目录
log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"video_processing_{timestamp}.log")
error_log_file = os.path.join(log_dir, f"video_processing_errors_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

error_logger = logging.getLogger('error_logger')
error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
error_handler.setFormatter(logging.Formatter('%(asctime)s - [ERROR] - %(message)s'))
error_logger.addHandler(error_handler)
error_logger.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

file_lock = Lock()
model_lock = Lock()  # 用于模型推理的锁


def load_video(video_path, max_frames_num):
    """
    从视频文件中均匀采样指定数量的帧。
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        # 如果视频帧数少于或等于目标帧数，则全部读取
        if total_frame_num <= max_frames_num:
            frame_idx = np.arange(0, total_frame_num).tolist()
        else:
            # 均匀采样 max_frames_num 帧
            frame_idx = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int).tolist()
        
        frames = vr.get_batch(frame_idx).asnumpy()
        
        # 计算视频时长和采样帧的时间戳
        video_time = total_frame_num / vr.get_avg_fps()
        frame_time_list = [idx / vr.get_avg_fps() for idx in frame_idx]
        frame_time_str = ",".join([f"{t:.2f}s" for t in frame_time_list])
        
        return frames, frame_time_str, video_time
        
    except Exception as e:
        logger.error(f"Error loading video {video_path}: {e}")
        return None, None, None


class VideoProcessor:
    def __init__(self, config):
        # 文件路径配置
        self.input_json_file = config.get('input_json_file', 'input_videos.json')
        self.output_file = config.get('output_file', 'video_analysis_results.json')
        self.error_file = config.get('error_file', 'video_analysis_errors.json')
        self.checkpoint_file = config.get('checkpoint_file', 'processing_checkpoint.json')
        
        # LLaVA-Video模型配置
        self.model_path = config.get('model_path')
        self.model_name = config.get('model_name', 'llava_qwen')
        self.device = config.get('device', 'cuda')
        self.device_map = config.get('device_map', 'auto')
        self.torch_dtype = config.get('torch_dtype', 'bfloat16')
        self.num_frames = config.get('num_frames', 32)
        self.conv_template = config.get('conv_template', 'qwen_2')  # 使用qwen_2模板
        
        # 处理配置
        self.max_workers = config.get('max_workers', 1)
        self.max_pairs = config.get('max_pairs', None)
        self.model_delay = config.get('model_delay', 0.5)
        self.timeout = config.get('timeout', 300)
        self.resume_from_checkpoint = config.get('resume_from_checkpoint', True)
        self.max_retries = config.get('max_retries', 3)
        
        # 创建必要的目录
        for file_path in [self.output_file, self.error_file, self.checkpoint_file]:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
        
        # 加载模型
        logger.info(f"加载本地模型: {self.model_path}")
        logger.info(f"模型名称: {self.model_name}")
        
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            self.model_path,
            None,
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="flash_attention_2"
        )
        self.model.eval()
        
        logger.info(f"="*80)
        logger.info(f"配置: Device={self.device}, Dtype={self.torch_dtype}, Frames={self.num_frames}")
        logger.info(f"模型加载成功！")
        logger.info(f"="*80)
        
        # 统计信息
        self.successful = 0
        self.failed = 0
        self.skipped_processed = 0
        self.start_time = None
        
        # 加载检查点和系统提示词
        self.checkpoint_data = self._load_checkpoint()
        self.processed_indices = set(self.checkpoint_data.get('successful_indices', []))
        self.system_prompt = self._load_system_prompt()
        self._initialize_output_file()
    
    def analyze_single_video(self, video_path, video_label, custom_prompt=None):
        """
        分析单个视频
        """
        # 加载视频
        frames, frame_time, video_time = load_video(video_path, self.num_frames)
        if frames is None:
            raise ValueError(f"无法加载视频: {video_path}")
        
        # 预处理视频帧 - 关键：正确的预处理方式
        video_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        video_tensor = video_tensor.to(device=self.model.device, dtype=getattr(torch, self.torch_dtype))
        video_tensor_list = [video_tensor]  # 关键：必须用列表包装！
        
        # 构建时间戳信息
        time_instruction = (
            f"The {video_label} video lasts for {video_time:.2f} seconds, and {len(frames)} frames are uniformly "
            f"sampled from it. These frames are located at timestamps: {frame_time}."
        )
        
        # 构建问题
        if custom_prompt:
            question = f"{DEFAULT_IMAGE_TOKEN}\n{time_instruction}\n{custom_prompt}"
        else:
            question = f"{DEFAULT_IMAGE_TOKEN}\n{time_instruction}\nPlease describe this video in detail, including the main content, actions, objects, and any notable features."
        
        # 使用对话模板 - 关键：使用qwen_2模板
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_for_tokenizer = conv.get_prompt()
        
        # Tokenize 输入
        input_ids = tokenizer_image_token(
            prompt_for_tokenizer,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)
        
        # 模型推理 - 关键：正确的推理参数
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=video_tensor_list,  # 列表格式的视频张量
                modalities=["video"],       # 指定视频模态
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
                use_cache=True,
            )
        
        # 解码输出
        response_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # 清理输出 - 移除prompt部分
        assistant_role_marker = "assistant\n"
        if assistant_role_marker in response_text:
            response_text = response_text.split(assistant_role_marker)[-1].strip()
        
        return response_text, len(frames), video_time
    
    def process_single_entry(self, entry):
        index = entry['index']
        video1_path = entry['video1_path']
        video2_path = entry['video2_path']
        
        logger.info(f"\n[Entry {index}] 开始处理")
        
        if index in self.processed_indices:
            logger.info(f"[Entry {index}] 已处理，跳过")
            self.skipped_processed += 1
            return None
        
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retries:
            try:
                # 检查视频文件是否存在
                for video_path in [video1_path, video2_path]:
                    if not os.path.exists(video_path):
                        raise FileNotFoundError(f"视频不存在: {video_path}")
                
                with model_lock:
                    # 步骤1: 分析源视频
                    logger.info(f"[Entry {index}] 分析源视频: {os.path.basename(video1_path)}")
                    video1_description, frames1_count, video1_duration = self.analyze_single_video(
                        video1_path, 
                        "source"
                    )
                    time.sleep(self.model_delay)
                    
                    # 步骤2: 分析目标视频
                    logger.info(f"[Entry {index}] 分析目标视频: {os.path.basename(video2_path)}")
                    video2_description, frames2_count, video2_duration = self.analyze_single_video(
                        video2_path,
                        "destination"
                    )
                    time.sleep(self.model_delay)
                    
                    # 步骤3: 基于两个描述进行对比分析
                    logger.info(f"[Entry {index}] 生成对比分析...")
                    comparison_prompt = f"""Based on the following two video descriptions, please provide a detailed comparison analysis:

Source Video Description:
{video1_description}

Destination Video Description:
{video2_description}

{self.system_prompt}"""
                    
                    # 由于对比是纯文本任务，我们可以直接用文本生成
                    # 或者选择重新分析其中一个视频并包含对比指令
                    # 这里我们使用源视频作为视觉输入，添加对比指令
                    frames1, frame_time1, video_time1 = load_video(video1_path, self.num_frames)
                    video_tensor1 = self.image_processor.preprocess(frames1, return_tensors="pt")["pixel_values"]
                    video_tensor1 = video_tensor1.to(device=self.model.device, dtype=getattr(torch, self.torch_dtype))
                    video_tensor_list1 = [video_tensor1]
                    
                    time_instruction1 = (
                        f"The source video lasts for {video_time1:.2f} seconds, and {len(frames1)} frames are uniformly "
                        f"sampled from it. These frames are located at timestamps: {frame_time1}."
                    )
                    
                    final_question = f"{DEFAULT_IMAGE_TOKEN}\n{time_instruction1}\n{comparison_prompt}"
                    
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                    conv.append_message(conv.roles[0], final_question)
                    conv.append_message(conv.roles[1], None)
                    prompt_for_tokenizer = conv.get_prompt()
                    
                    input_ids = tokenizer_image_token(
                        prompt_for_tokenizer,
                        self.tokenizer,
                        IMAGE_TOKEN_INDEX,
                        return_tensors="pt"
                    ).unsqueeze(0).to(self.model.device)
                    
                    with torch.inference_mode():
                        output_ids = self.model.generate(
                            input_ids,
                            images=video_tensor_list1,
                            modalities=["video"],
                            do_sample=False,
                            temperature=0,
                            max_new_tokens=4096,
                            use_cache=True,
                        )
                    
                    final_response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                    
                    # 清理输出
                    assistant_role_marker = "assistant\n"
                    if assistant_role_marker in final_response:
                        final_response = final_response.split(assistant_role_marker)[-1].strip()
                
                # 保存结果
                result = {
                    "index": index,
                    "video1_path": video1_path,
                    "video2_path": video2_path,
                    "frames_extracted": {
                        "video1": frames1_count,
                        "video2": frames2_count
                    },
                    "video_info": {
                        "video1_duration": f"{video1_duration:.2f}s",
                        "video2_duration": f"{video2_duration:.2f}s"
                    },
                    "video1_description": video1_description,
                    "video2_description": video2_description,
                    "comparison_analysis": final_response,
                    "response": final_response,  # 保持兼容性
                    "timestamp": datetime.now().isoformat()
                }
                
                self._append_result_to_file(result)
                self._save_checkpoint(index, success=True)
                self.successful += 1
                logger.info(f"[Entry {index}] ✅ 成功")
                return result
                
            except Exception as e:
                last_error = e
                retry_count += 1
                logger.error(f"[Entry {index}] ❌ 尝试{retry_count}失败: {e}")
                error_logger.error(f"[Entry {index}] Error: {traceback.format_exc()}")
                if retry_count < self.max_retries:
                    time.sleep(retry_count * 3)
        
        # 所有重试都失败
        self.failed += 1
        error_info = {
            "index": index,
            "video1_path": video1_path,
            "video2_path": video2_path,
            "error": str(last_error),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }
        self._append_error_to_file(error_info)
        return None
    
    def _initialize_output_file(self):
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        logger.info(f"输出文件已存在: {len(data)}条记录")
                        return
            except:
                pass
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False)
    
    def _append_result_to_file(self, result):
        with file_lock:
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = []
                data.append(result)
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"写入结果失败: {e}")
    
    def _load_system_prompt(self):
        prompt_path = "prompt_generate.txt"
        if not os.path.exists(prompt_path):
            default_prompt = "Please provide a detailed comparison of these two videos, highlighting the key differences and similarities."
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(default_prompt)
            return default_prompt
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    
    def _load_checkpoint(self):
        if not self.resume_from_checkpoint:
            return {}
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_checkpoint(self, index, success=True):
        try:
            if success:
                self.processed_indices.add(index)
                if 'successful_indices' not in self.checkpoint_data:
                    self.checkpoint_data['successful_indices'] = []
                if index not in self.checkpoint_data['successful_indices']:
                    self.checkpoint_data['successful_indices'].append(index)
            
            with file_lock:
                with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(self.checkpoint_data, f, ensure_ascii=False, indent=2)
        except:
            pass
    
    def _append_error_to_file(self, error_info):
        with file_lock:
            try:
                if os.path.exists(self.error_file):
                    with open(self.error_file, 'r', encoding='utf-8') as f:
                        errors = json.load(f)
                else:
                    errors = []
                errors.append(error_info)
                with open(self.error_file, 'w', encoding='utf-8') as f:
                    json.dump(errors, f, ensure_ascii=False, indent=2)
            except:
                pass
    
    def load_input_data(self):
        if not os.path.exists(self.input_json_file):
            raise FileNotFoundError(f"输入文件不存在: {self.input_json_file}")
        
        data_list = []
        with open(self.input_json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        if isinstance(json_data, list):
            for idx, item in enumerate(json_data):
                if 'video1_path' in item and 'video2_path' in item:
                    data_list.append({
                        'index': idx,
                        'video1_path': item['video1_path'],
                        'video2_path': item['video2_path']
                    })
        elif isinstance(json_data, dict):
            video_pairs = json_data.get('video_pairs', json_data.get('data', []))
            for idx, item in enumerate(video_pairs):
                if 'video1_path' in item and 'video2_path' in item:
                    data_list.append({
                        'index': idx,
                        'video1_path': item['video1_path'],
                        'video2_path': item['video2_path']
                    })
        
        if self.max_pairs and len(data_list) > self.max_pairs:
            data_list = data_list[:self.max_pairs]
        
        return data_list
    
    def run(self):
        logger.info(f"\n{'='*80}")
        logger.info(f"开始批量处理视频对比分析 - 离线模式")
        logger.info(f"{'='*80}\n")
        
        self.start_time = time.time()
        
        try:
            # 加载输入数据
            data_list = self.load_input_data()
            if not data_list:
                logger.error("没有找到有效的视频对数据")
                return
            
            total_pairs = len(data_list)
            logger.info(f"共找到 {total_pairs} 对视频待处理\n")
            
            # 使用线程池并发处理
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.process_single_entry, entry): entry 
                    for entry in data_list
                }
                
                for future in as_completed(futures):
                    try:
                        future.result(timeout=self.timeout)
                    except Exception as e:
                        logger.error(f"处理任务失败: {e}")
            
            # 统计结果
            elapsed = time.time() - self.start_time
            logger.info(f"\n{'='*80}")
            logger.info(f"✅ 处理完成！")
            logger.info(f"总计: {total_pairs} 对")
            logger.info(f"成功: {self.successful} 对")
            logger.info(f"失败: {self.failed} 对")
            logger.info(f"跳过: {self.skipped_processed} 对")
            logger.info(f"总用时: {elapsed:.2f}秒")
            logger.info(f"{'='*80}\n")
            
        except KeyboardInterrupt:
            logger.warning(f"\n用户中断处理")
        except Exception as e:
            logger.error(f"\n处理过程中出现错误: {e}")
            traceback.print_exc()


def main():
    # 配置参数
    config = {
        'input_json_file': '/mnt/shared-storage-user/colab-share/liujiaheng/workspace/shihao/VDC/checklist.json',
        'output_file': '/mnt/shared-storage-user/colab-share/liujiaheng/workspace/shihao/VDC/response/response_llava_video.json',
        'error_file': '/mnt/shared-storage-user/colab-share/liujiaheng/workspace/shihao/VDC/response/video_analysis_errors_llava_video.json',
        'checkpoint_file': '/mnt/shared-storage-user/colab-share/liujiaheng/workspace/shihao/VDC/response/processing_checkpoint_llava_video.json',
        
        # LLaVA-Video模型配置
        'model_path': '/mnt/shared-storage-user/colab-share/liujiaheng/workspace/shihao/IFEval-Caption/rebuttal/models/llava-video',
        'model_name': 'llava_qwen',  # 与模型路径对应
        'device': 'cuda',
        'device_map': 'auto',
        'torch_dtype': 'bfloat16',
        'num_frames': 32,
        'conv_template': 'qwen_2',  # 使用qwen_2模板
        
        # 处理配置
        'max_workers': 1,  # 建议使用1，避免GPU内存问题
        'max_pairs': None,  # None表示处理所有
        'model_delay': 0.5,
        'timeout': 300,
        'resume_from_checkpoint': True,
        'max_retries': 3,
    }
    
    # 确认模型路径存在
    if not os.path.exists(config['model_path']):
        logger.error(f"模型路径不存在: {config['model_path']}")
        return
    
    logger.info(f"使用本地模型: {config['model_path']}")
    logger.info("运行在完全离线模式")
    
    processor = VideoProcessor(config)
    processor.run()


if __name__ == "__main__":
    main()
