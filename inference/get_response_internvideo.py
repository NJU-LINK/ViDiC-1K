"""
视频对比分析工具 - InternVideo2.5版本（增强日志和视频区分）
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
import torch
import numpy as np
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import gc
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

# 设置环境变量优化显存
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 日志设置
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"video_processing_{timestamp}.log"
error_log_file = f"video_processing_errors_{timestamp}.log"
detail_log_file = f"video_processing_details_{timestamp}.log"  # 新增详细日志

# 创建多个logger用于不同目的
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# 创建详细日志记录器
detail_logger = logging.getLogger("detail_logger")
detail_handler = logging.FileHandler(detail_log_file, encoding='utf-8')
detail_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
detail_handler.setFormatter(detail_formatter)
detail_logger.addHandler(detail_handler)
detail_logger.setLevel(logging.DEBUG)

file_lock = Lock()
model_lock = Lock()

# --- 模型和路径设置 ---
# 本地模型路径（与成功示例相同）
MODEL_PATH = '/mnt/shared-storage-user/colab-share/liujiaheng/workspace/shihao/IFEval-Caption/rebuttal/models/Internvideo'

# 推理参数
NUM_SEGMENTS = 32  # 视频采样帧数
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# 图像预处理常量
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- 全局模型加载（与成功示例完全相同） ---
print(f"Loading tokenizer from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,    
    trust_remote_code=True,
    local_files_only=True,
)

print(f"Loading model from {MODEL_PATH}...")
model = AutoModel.from_pretrained(
    MODEL_PATH, 
    trust_remote_code=True, 
    local_files_only=True,
    attn_implementation="flash_attention_2"
).to(DEVICE).to(DTYPE)

# 生成配置
generation_config = dict(
    do_sample=False,
    temperature=0.0,
    max_new_tokens=4096,
    top_p=None,
    num_beams=1,
)

print(f"Model loaded successfully! Device: {DEVICE}, Dtype: {DTYPE}")


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


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, video_name="Video"):
    """加载视频并记录详细日志"""
    try:
        detail_logger.info(f"[{video_name}] 开始加载: {video_path}")
        
        # 获取视频信息
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        detail_logger.info(f"[{video_name}] 文件大小: {file_size:.2f} MB")
        
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        detail_logger.info(f"[{video_name}] 视频信息 - 总帧数: {max_frame + 1}, FPS: {fps:.2f}, 时长: {(max_frame + 1) / fps:.2f}秒")
        
        pixel_values_list, num_patches_list = [], []
        transform = build_transform(input_size=input_size)
        frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        
        detail_logger.info(f"[{video_name}] 选取的帧索引: {frame_indices.tolist()}")
        
        for i, frame_index in enumerate(frame_indices):
            img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
            
            if i % 8 == 0:  # 每8帧记录一次进度
                detail_logger.debug(f"[{video_name}] 已处理 {i+1}/{len(frame_indices)} 帧")
        
        pixel_values = torch.cat(pixel_values_list)
        detail_logger.info(f"[{video_name}] 加载完成 - 张量形状: {pixel_values.shape}")
        
        return pixel_values, num_patches_list
        
    except Exception as e:
        detail_logger.error(f"[{video_name}] 加载失败: {e}")
        raise


class VideoProcessor:
    def __init__(self, config):
        self.input_json_file = config.get('input_json_file', 'input_videos.json')
        self.output_file = config.get('output_file', 'video_analysis_results.json')
        self.error_file = config.get('error_file', 'video_analysis_errors.json')
        self.checkpoint_file = config.get('checkpoint_file', 'processing_checkpoint.json')
        self.max_workers = config.get('max_workers', 1)
        self.max_pairs = config.get('max_pairs', None)
        self.model_delay = config.get('model_delay', 2)
        self.timeout = config.get('timeout', 300)
        self.resume_from_checkpoint = config.get('resume_from_checkpoint', True)
        self.max_retries = config.get('max_retries', 3)
        
        # InternVideo2.5 特定配置
        self.num_segments = config.get('num_segments', NUM_SEGMENTS)
        self.input_size = config.get('input_size', 448)
        self.max_num = config.get('max_num', 1)
        
        # 创建输出目录
        for file_path in [self.output_file, self.error_file, self.checkpoint_file]:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                except:
                    logger.warning(f"Cannot create directory {directory}, using current directory")
        
        logger.info(f"="*80)
        logger.info(f"视频处理器初始化完成")
        logger.info(f"Model: InternVideo2.5 (Local)")
        logger.info(f"Device: {DEVICE}, Dtype: {DTYPE}")
        logger.info(f"Frames per video: {self.num_segments}")
        logger.info(f"Input size: {self.input_size}")
        logger.info(f"日志文件:")
        logger.info(f"  - 主日志: {log_file}")
        logger.info(f"  - 错误日志: {error_log_file}")
        logger.info(f"  - 详细日志: {detail_log_file}")
        logger.info(f"="*80)
        
        self.successful = 0
        self.failed = 0
        self.skipped_processed = 0
        self.start_time = None
        
        self.checkpoint_data = self._load_checkpoint()
        self.processed_indices = set(self.checkpoint_data.get('successful_indices', []))
        self.system_prompt = self._load_system_prompt()
        self._initialize_output_file()
    
    def process_video_pair(self, video1_path, video2_path):
        """处理两个视频并生成对比分析"""
        try:
            # 记录开始时间
            process_start_time = time.time()
            
            # 加载源视频
            logger.info(f"[Source Video] Loading: {os.path.basename(video1_path)}")
            video1_start_time = time.time()
            pixel_values1, num_patches_list1 = load_video(
                video1_path, 
                num_segments=self.num_segments, 
                max_num=self.max_num,
                input_size=self.input_size,
                video_name="Source Video"
            )
            video1_load_time = time.time() - video1_start_time
            logger.info(f"[Source Video] 加载耗时: {video1_load_time:.2f}秒")
            
            # 加载目标视频
            logger.info(f"[Destination Video] Loading: {os.path.basename(video2_path)}")
            video2_start_time = time.time()
            pixel_values2, num_patches_list2 = load_video(
                video2_path, 
                num_segments=self.num_segments, 
                max_num=self.max_num,
                input_size=self.input_size,
                video_name="Destination Video"
            )
            video2_load_time = time.time() - video2_start_time
            logger.info(f"[Destination Video] 加载耗时: {video2_load_time:.2f}秒")
            
            # 记录视频对比信息
            detail_logger.info(f"===== 视频对比信息 =====")
            detail_logger.info(f"Source Video: {video1_path}")
            detail_logger.info(f"  - 帧数: {len(num_patches_list1)}")
            detail_logger.info(f"  - 补丁数: {sum(num_patches_list1)}")
            detail_logger.info(f"Destination Video: {video2_path}")
            detail_logger.info(f"  - 帧数: {len(num_patches_list2)}")
            detail_logger.info(f"  - 补丁数: {sum(num_patches_list2)}")
            detail_logger.info(f"========================")
            
            # 合并两个视频的帧
            pixel_values = torch.cat([pixel_values1, pixel_values2], dim=0)
            num_patches_list = num_patches_list1 + num_patches_list2
            
            # 转换到正确的设备和数据类型
            pixel_values = pixel_values.to(DTYPE).to(DEVICE)
            
            # 构建更清晰的视频前缀，明确区分源视频和目标视频
            video1_prefix = f"[SOURCE VIDEO: {os.path.basename(video1_path)}]\n"
            video1_prefix += "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list1))])
            
            video2_prefix = f"\n[DESTINATION VIDEO: {os.path.basename(video2_path)}]\n"
            video2_prefix += "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list2))])
            
            # 组合完整的提示词
            full_prompt = video1_prefix + video2_prefix + "\n" + self.system_prompt
            
            detail_logger.debug(f"Prompt length: {len(full_prompt)} characters")
            
            # 模型推理
            inference_start_time = time.time()
            with torch.no_grad():
                response, _ = model.chat(
                    tokenizer,
                    pixel_values,
                    full_prompt,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=None,  # 确保每次对话都是独立的
                    return_history=True
                )
            inference_time = time.time() - inference_start_time
            
            # 记录推理信息
            detail_logger.info(f"模型推理完成 - 耗时: {inference_time:.2f}秒")
            detail_logger.info(f"响应长度: {len(response)} 字符")
            
            # 清理GPU内存
            del pixel_values, pixel_values1, pixel_values2
            torch.cuda.empty_cache()
            
            # 记录总处理时间
            total_process_time = time.time() - process_start_time
            detail_logger.info(f"视频对处理完成 - 总耗时: {total_process_time:.2f}秒")
            
            # 返回结果，包含更多时间信息
            return {
                "response": response,
                "source_video_load_time": video1_load_time,
                "destination_video_load_time": video2_load_time,
                "inference_time": inference_time,
                "total_time": total_process_time
            }
            
        except Exception as e:
            logger.error(f"Error processing video pair: {e}")
            detail_logger.error(f"详细错误信息: {traceback.format_exc()}")
            torch.cuda.empty_cache()
            raise e
    
    def process_single_entry(self, entry):
        """处理单个视频对"""
        index = entry['index']
        video1_path = entry['video1_path']
        video2_path = entry['video2_path']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"[Entry {index}] 开始处理")
        logger.info(f"[Source Video]: {video1_path}")
        logger.info(f"[Destination Video]: {video2_path}")
        logger.info(f"{'='*60}")
        
        # 检查是否已处理
        if index in self.processed_indices:
            logger.info(f"[Entry {index}] 已处理，跳过")
            self.skipped_processed += 1
            return None
        
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retries:
            try:
                # 检查视频文件是否存在
                for i, (video_path, video_type) in enumerate([(video1_path, 'Source'), (video2_path, 'Destination')], 1):
                    if not os.path.exists(video_path):
                        raise FileNotFoundError(f"{video_type} Video 文件不存在: {video_path}")
                    detail_logger.info(f"[Entry {index}] {video_type} Video 文件检查通过: {video_path}")
                
                # 处理视频对
                with model_lock:
                    start_time = time.time()
                    result_data = self.process_video_pair(video1_path, video2_path)
                    
                    logger.info(f"[Entry {index}] 推理完成")
                    logger.info(f"  - Source Video 加载: {result_data['source_video_load_time']:.2f}秒")
                    logger.info(f"  - Destination Video 加载: {result_data['destination_video_load_time']:.2f}秒")
                    logger.info(f"  - 模型推理: {result_data['inference_time']:.2f}秒")
                    logger.info(f"  - 总耗时: {result_data['total_time']:.2f}秒")
                    
                    time.sleep(self.model_delay)
                
                # 保存结果
                result = {
                    "index": index,
                    "source_video_path": video1_path,
                    "destination_video_path": video2_path,
                    "source_video_filename": os.path.basename(video1_path),
                    "destination_video_filename": os.path.basename(video2_path),
                    "response": result_data["response"],
                    "processing_times": {
                        "source_video_load_time": result_data['source_video_load_time'],
                        "destination_video_load_time": result_data['destination_video_load_time'],
                        "inference_time": result_data['inference_time'],
                        "total_time": result_data['total_time']
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                self._append_result_to_file(result)
                self._save_checkpoint(index, success=True)
                self.successful += 1
                
                logger.info(f"[Entry {index}] ✅ 处理成功")
                return result
                
            except torch.cuda.OutOfMemoryError as e:
                last_error = e
                retry_count += 1
                logger.error(f"[Entry {index}] GPU内存不足，尝试 {retry_count}/{self.max_retries}")
                detail_logger.error(f"[Entry {index}] GPU OOM详情: {traceback.format_exc()}")
                torch.cuda.empty_cache()
                gc.collect()
                if retry_count < self.max_retries:
                    time.sleep(retry_count * 5)
                    
            except Exception as e:
                last_error = e
                retry_count += 1
                logger.error(f"[Entry {index}] 处理失败，尝试 {retry_count}/{self.max_retries}: {e}")
                detail_logger.error(f"[Entry {index}] 详细错误: {traceback.format_exc()}")
                if retry_count < self.max_retries:
                    time.sleep(retry_count * 3)
        
        # 所有重试都失败
        self.failed += 1
        error_info = {
            "index": index,
            "source_video_path": video1_path,
            "destination_video_path": video2_path,
            "source_video_filename": os.path.basename(video1_path),
            "destination_video_filename": os.path.basename(video2_path),
            "error": str(last_error),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }
        self._append_error_to_file(error_info)
        logger.error(f"[Entry {index}] ❌ 处理失败")
        return None
    
    def load_input_data(self):
        """加载输入数据"""
        if not os.path.exists(self.input_json_file):
            raise FileNotFoundError(f"输入文件不存在: {self.input_json_file}")
        
        logger.info(f"Loading input data from {self.input_json_file}")
        
        with open(self.input_json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        data_list = []
        
        # 支持多种输入格式
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
        
        # 限制处理数量
        if self.max_pairs and len(data_list) > self.max_pairs:
            data_list = data_list[:self.max_pairs]
            logger.info(f"限制处理数量为 {self.max_pairs} 对")
        
        logger.info(f"找到 {len(data_list)} 对视频待处理")
        
        # 记录详细的输入信息
        detail_logger.info("===== 输入数据详情 =====")
        for i, item in enumerate(data_list[:5]):  # 只记录前5个
            detail_logger.info(f"Entry {i}:")
            detail_logger.info(f"  Source Video: {item['video1_path']}")
            detail_logger.info(f"  Destination Video: {item['video2_path']}")
        if len(data_list) > 5:
            detail_logger.info(f"... 还有 {len(data_list) - 5} 对视频 ...")
        
        return data_list
    
    def run(self):
        """运行批处理"""
        logger.info(f"\n{'='*80}")
        logger.info(f"开始视频批量处理")
        logger.info(f"模型: InternVideo2.5 (本地模型)")
        logger.info(f"日志文件已创建:")
        logger.info(f"  - 主日志: {log_file}")
        logger.info(f"  - 错误日志: {error_log_file}")
        logger.info(f"  - 详细日志: {detail_log_file}")
        logger.info(f"{'='*80}\n")
        
        self.start_time = time.time()
        
        try:
            # 加载输入数据
            data_list = self.load_input_data()
            
            if not data_list:
                logger.error("没有找到可处理的视频对")
                return
            
            total_pairs = len(data_list)
            
            # 使用进度条
            progress_bar = tqdm(data_list, desc="Processing video pairs")
            
            # 定期记录进度
            def log_progress():
                elapsed = time.time() - self.start_time
                progress = self.successful + self.failed + self.skipped_processed
                if progress > 0:
                    avg_time = elapsed / progress
                    remaining = total_pairs - progress
                    eta = remaining * avg_time
                    logger.info(f"\n--- 进度报告 ---")
                    logger.info(f"已处理: {progress}/{total_pairs}")
                    logger.info(f"成功: {self.successful}, 失败: {self.failed}, 跳过: {self.skipped_processed}")
                    logger.info(f"平均处理时间: {avg_time:.2f}秒/对")
                    logger.info(f"预计剩余时间: {eta/60:.1f}分钟")
                    logger.info(f"---------------\n")
            
            # 处理每一对视频
            if self.max_workers == 1:
                # 单线程处理
                for i, entry in enumerate(progress_bar):
                    progress_bar.set_description(f"Processing entry {entry['index']}")
                    self.process_single_entry(entry)
                    
                    # 每处理10个记录一次进度
                    if (i + 1) % 10 == 0:
                        log_progress()
            else:
                # 多线程处理
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {
                        executor.submit(self.process_single_entry, entry): entry 
                        for entry in data_list
                    }
                    
                    for i, future in enumerate(as_completed(futures)):
                        try:
                            future.result(timeout=self.timeout)
                        except Exception as e:
                            logger.error(f"Task execution error: {e}")
                        progress_bar.update(1)
                        
                        # 每处理10个记录一次进度
                        if (i + 1) % 10 == 0:
                            log_progress()
            
            progress_bar.close()
            
            # 统计结果
            elapsed = time.time() - self.start_time
            logger.info(f"\n{'='*80}")
            logger.info(f"✅ 处理完成!")
            logger.info(f"总计: {total_pairs}, 成功: {self.successful}, 失败: {self.failed}, 跳过: {self.skipped_processed}")
            logger.info(f"总耗时: {elapsed:.2f}秒 ({elapsed/60:.2f}分钟)")
            if self.successful > 0:
                logger.info(f"平均耗时: {elapsed/self.successful:.2f}秒/对")
            logger.info(f"输出文件:")
            logger.info(f"  - 结果: {self.output_file}")
            logger.info(f"  - 错误: {self.error_file}")
            logger.info(f"  - 检查点: {self.checkpoint_file}")
            logger.info(f"{'='*80}\n")
            
            # 写入最终统计到详细日志
            detail_logger.info(f"\n{'='*80}")
            detail_logger.info(f"===== 最终统计 =====")
            detail_logger.info(f"总视频对数: {total_pairs}")
            detail_logger.info(f"成功处理: {self.successful}")
            detail_logger.info(f"处理失败: {self.failed}")
            detail_logger.info(f"跳过已处理: {self.skipped_processed}")
            detail_logger.info(f"总耗时: {elapsed/60:.2f}分钟")
            detail_logger.info(f"{'='*80}\n")
            
        except KeyboardInterrupt:
            logger.warning("\n⚠️ 用户中断处理")
            detail_logger.warning("处理被用户中断")
        except Exception as e:
            logger.error(f"\n❌ 处理出错: {e}")
            detail_logger.error(f"主程序错误: {traceback.format_exc()}")
            traceback.print_exc()
        finally:
            # 清理资源
            torch.cuda.empty_cache()
            gc.collect()
    
    def _initialize_output_file(self):
        """初始化输出文件"""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        logger.info(f"输出文件已存在，包含 {len(data)} 条记录")
                        return
            except:
                pass
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False)
    
    def _append_result_to_file(self, result):
        """追加结果到文件"""
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
                logger.error(f"写入结果文件失败: {e}")
    
    def _append_error_to_file(self, error_info):
        """追加错误信息到文件"""
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
            except Exception as e:
                logger.error(f"写入错误文件失败: {e}")
    
    def _load_checkpoint(self):
        """加载检查点"""
        if not self.resume_from_checkpoint:
            return {}
        
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                    logger.info(f"从检查点恢复，已处理 {len(checkpoint.get('successful_indices', []))} 个条目")
                    return checkpoint
            except:
                return {}
        return {}
    
    def _save_checkpoint(self, index, success=True):
        """保存检查点"""
        try:
            if success:
                self.processed_indices.add(index)
                
                if 'successful_indices' not in self.checkpoint_data:
                    self.checkpoint_data['successful_indices'] = []
                
                if index not in self.checkpoint_data['successful_indices']:
                    self.checkpoint_data['successful_indices'].append(index)
            
            # 更新统计信息
            self.checkpoint_data['stats'] = {
                'successful': self.successful,
                'failed': self.failed,
                'skipped': self.skipped_processed,
                'last_update': datetime.now().isoformat()
            }
            
            with file_lock:
                with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(self.checkpoint_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
    
    def _load_system_prompt(self):
        """加载系统提示词"""
        prompt_path = "prompt_generate.txt"
        
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"提示词文件不存在: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
            logger.info(f"加载提示词文件: {prompt_path}")
            return prompt


def main():
    config = {
        'input_json_file': '/mnt/shared-storage-user/colab-share/liujiaheng/workspace/shihao/VDC/checklist.json',  # 输入文件路径
        'output_file': '/mnt/shared-storage-user/colab-share/liujiaheng/workspace/shihao/VDC/response/video_analysis_results_internvideo.json',       # 输出文件
        'error_file': '/mnt/shared-storage-user/colab-share/liujiaheng/workspace/shihao/VDC/response/video_analysis_errors_internvideo.json',         # 错误文件
        'checkpoint_file': '/mnt/shared-storage-user/colab-share/liujiaheng/workspace/shihao/VDC/response/processing_checkpoint_internvideo.json',    # 检查点文件
        'max_workers': 1,          # 工作线程数（建议使用1以避免GPU内存问题）
        'max_pairs': None,         # 最大处理数量（None表示处理全部）
        'model_delay': 1,          # 模型调用间隔（秒）
        'timeout': 600,            # 单个任务超时（秒）
        'resume_from_checkpoint': True,  # 是否从断点恢复
        'max_retries': 3,          # 最大重试次数
        'num_segments': 32,        # 每个视频提取的帧数
        'input_size': 448,         # 输入图像大小
        'max_num': 1,              # 动态预处理的最大块数
    }
    
    processor = VideoProcessor(config)
    processor.run()


if __name__ == "__main__":
    main()
