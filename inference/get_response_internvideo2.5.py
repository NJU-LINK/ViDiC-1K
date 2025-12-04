"""
视频帧提取对比分析工具 - InternVideo本地版本
"""

import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import logging
from datetime import datetime
import traceback
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

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

# InternVideo相关的常量
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# InternVideo的辅助函数
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
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


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
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


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


class VideoProcessor:
    def __init__(self, config):
        # 文件路径配置
        self.input_json_file = config.get('input_json_file', 'input_videos.json')
        self.output_file = config.get('output_file', 'video_analysis_results.json')
        self.error_file = config.get('error_file', 'video_analysis_errors.json')
        self.checkpoint_file = config.get('checkpoint_file', 'processing_checkpoint.json')
        
        # InternVideo模型配置
        self.model_path = config.get('model_path', '/path/to/Internvideo')
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = config.get('dtype', torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)
        self.num_segments = config.get('num_segments', 32)
        self.input_size = config.get('input_size', 448)
        self.max_num = config.get('max_num', 1)
        
        # 处理配置
        self.max_workers = config.get('max_workers', 2)
        self.max_pairs = config.get('max_pairs', None)
        self.model_delay = config.get('model_delay', 0.5)  # 模型推理间隔
        self.timeout = config.get('timeout', 300)
        self.resume_from_checkpoint = config.get('resume_from_checkpoint', True)
        self.max_retries = config.get('max_retries', 3)
        
        # 创建必要的目录
        for file_path in [self.output_file, self.error_file, self.checkpoint_file]:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
        
        # 加载模型
        logger.info(f"加载模型: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="flash_attention_2"
        ).to(self.device).to(self.dtype)
        
        # 生成配置
        self.generation_config = dict(
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            top_p=None,
            num_beams=1,
        )
        
        logger.info(f"="*80)
        logger.info(f"配置: Device={self.device}, Dtype={self.dtype}, Segments={self.num_segments}")
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
                
                # 加载视频1
                logger.info(f"[Entry {index}] 处理视频1: {os.path.basename(video1_path)}")
                pixel_values1, num_patches_list1 = load_video(
                    video1_path, 
                    num_segments=self.num_segments,
                    input_size=self.input_size,
                    max_num=self.max_num
                )
                pixel_values1 = pixel_values1.to(self.dtype).to(self.device)
                
                # 加载视频2
                logger.info(f"[Entry {index}] 处理视频2: {os.path.basename(video2_path)}")
                pixel_values2, num_patches_list2 = load_video(
                    video2_path,
                    num_segments=self.num_segments,
                    input_size=self.input_size,
                    max_num=self.max_num
                )
                pixel_values2 = pixel_values2.to(self.dtype).to(self.device)
                
                # 构建视频前缀
                video1_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list1))])
                video2_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list2))])
                
                # 构建完整的prompt
                full_prompt = f"""Source video:
{video1_prefix}

Destination video:
{video2_prefix}

{self.system_prompt}"""
                
                # 合并两个视频的像素值
                combined_pixel_values = torch.cat([pixel_values1, pixel_values2], dim=0)
                combined_num_patches = num_patches_list1 + num_patches_list2
                
                # 模型推理
                with torch.no_grad():
                    with model_lock:
                        logger.info(f"[Entry {index}] 开始模型推理...")
                        response, _ = self.model.chat(
                            self.tokenizer,
                            combined_pixel_values,
                            full_prompt,
                            self.generation_config,
                            num_patches_list=combined_num_patches,
                            history=None,
                            return_history=True
                        )
                        time.sleep(self.model_delay)
                
                # 保存结果
                result = {
                    "index": index,
                    "video1_path": video1_path,
                    "video2_path": video2_path,
                    "frames_extracted": {
                        "video1": len(num_patches_list1),
                        "video2": len(num_patches_list2)
                    },
                    "response": response,
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
                if retry_count < self.max_retries:
                    time.sleep(retry_count * 3)
        
        # 所有重试都失败
        self.failed += 1
        error_info = {
            "index": index,
            "video1_path": video1_path,
            "video2_path": video2_path,
            "error": str(last_error),
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
            default_prompt = "Please analyze and compare these two videos, describing their differences in detail."
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
        logger.info(f"开始批量处理视频对比分析")
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
        # 文件路径配置
        'input_json_file': 'checklist.json',
        'output_file': '/mnt/shared-storage-user/colab-share/liujiaheng/workspace/shihao/VDC/response/response_internvideo.json',
        'error_file': '/mnt/shared-storage-user/colab-share/liujiaheng/workspace/shihao/VDC/response/video_analysis_errors_internvideo.json',
        'checkpoint_file': '/mnt/shared-storage-user/colab-share/liujiaheng/workspace/shihao/VDC/response/processing_checkpoint_internvideo.json',
        
        # InternVideo模型配置
        'model_path': '/mnt/shared-storage-user/colab-share/liujiaheng/workspace/shihao/IFEval-Caption/rebuttal/models/Internvideo',
        'device': 'cuda',
        'dtype': torch.bfloat16,
        'num_segments': 32,
        'input_size': 448,
        'max_num': 1,
        
        # 处理配置
        'max_workers': 1,  # 建议使用1，避免GPU内存问题
        'max_pairs': None,  # None表示处理所有
        'model_delay': 0.5,  # 模型推理间隔
        'timeout': 300,
        'resume_from_checkpoint': True,
        'max_retries': 3,
    }
    
    processor = VideoProcessor(config)
    processor.run()


if __name__ == "__main__":
    main()
