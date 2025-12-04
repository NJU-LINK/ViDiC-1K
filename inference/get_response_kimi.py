"""
视频帧提取对比分析工具 - Kimi本地版本（显存优化）
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
import cv2
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import gc

sys.stdout.reconfigure(encoding='utf-8')

# 设置环境变量优化显存
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

log_dir = "logs"
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
model_lock = Lock()


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
        self.model_path = config.get('model_path', '/root/bayes-tmp/workplace/model/kimi')
        self.resume_from_checkpoint = config.get('resume_from_checkpoint', True)
        self.max_retries = config.get('max_retries', 3)
        self.frame_interval_seconds = config.get('frame_interval_seconds', 1.0)
        self.max_frames_per_video = config.get('max_frames_per_video', 8)
        self.max_frame_width = config.get('max_frame_width', 512)
        self.temp_frame_dir = config.get('temp_frame_dir', 'temp_frames')
        
        if not os.path.exists(self.temp_frame_dir):
            os.makedirs(self.temp_frame_dir)
        
        for file_path in [self.output_file, self.error_file, self.checkpoint_file]:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
        
        # 清理显存
        torch.cuda.empty_cache()
        gc.collect()
        
        # 初始化Kimi模型（优化显存）
        logger.info("正在加载Kimi模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,  # 使用float16节省显存
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,  # 减少CPU内存使用
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        # 设置模型为评估模式
        self.model.eval()
        
        logger.info("✅ Kimi模型加载完成")
        
        # 显示显存使用情况
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"GPU {i}: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
        
        logger.info(f"="*80)
        logger.info(f"配置: Model=Kimi-VL, FPS={1/self.frame_interval_seconds:.1f}, MaxFrames={self.max_frames_per_video}")
        logger.info(f"="*80)
        
        self.successful = 0
        self.failed = 0
        self.skipped_processed = 0
        self.start_time = None
        
        self.checkpoint_data = self._load_checkpoint()
        self.processed_indices = set(self.checkpoint_data.get('successful_indices', []))
        self.system_prompt = self._load_system_prompt()
        self._initialize_output_file()
    
    def extract_frames_from_video(self, video_path, video_label=""):
        """提取视频帧并保存到磁盘，返回帧路径列表"""
        frame_paths = []
        cap = None
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"视频: {os.path.basename(video_path)} - {total_frames}帧, {fps:.1f}fps, {duration:.1f}s")
            
            # 计算需要提取的帧索引
            frame_indices = []
            current_time = 0
            while current_time < duration:
                frame_idx = int(current_time * fps)
                if frame_idx < total_frames:
                    frame_indices.append(frame_idx)
                    current_time += self.frame_interval_seconds
                else:
                    break
            
            if len(frame_indices) > self.max_frames_per_video:
                frame_indices = frame_indices[:self.max_frames_per_video]
            
            logger.info(f"提取 {len(frame_indices)} 帧")
            
            # 创建视频专用文件夹
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_frame_dir = os.path.join(self.temp_frame_dir, f"{video_name}_{video_label}")
            os.makedirs(video_frame_dir, exist_ok=True)
            
            # 提取并保存帧
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # 调整尺寸以节省显存
                if width > self.max_frame_width:
                    scale = self.max_frame_width / width
                    new_w = int(width * scale)
                    new_h = int(height * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # 转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # 保存为JPEG以节省空间
                frame_filename = f"frame_{i:04d}.jpg"
                frame_filepath = os.path.join(video_frame_dir, frame_filename)
                img.save(frame_filepath, quality=85, optimize=True)
                frame_paths.append(frame_filepath)
            
            cap.release()
            logger.info(f"✅ 提取完成: {len(frame_paths)}帧保存至 {video_frame_dir}")
            return frame_paths
            
        except Exception as e:
            logger.error(f"❌ 提取失败: {e}")
            if cap:
                cap.release()
            raise
    
    def cleanup_temp_frames(self, frame_paths):
        """清理临时帧文件"""
        for path in frame_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass
        
        # 清理空文件夹
        if frame_paths:
            folder = os.path.dirname(frame_paths[0])
            try:
                if os.path.exists(folder) and not os.listdir(folder):
                    os.rmdir(folder)
            except:
                pass
    
    def process_with_kimi(self, frame_paths_video1, frame_paths_video2):
        """使用Kimi模型处理视频帧（显存优化版本 - 指令模式优化）"""
        
        # 清理显存
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            # 加载图像
            loaded_images = []
            for frame_path in frame_paths_video1 + frame_paths_video2:
                if os.path.exists(frame_path):
                    img = Image.open(frame_path)
                    # 确保图片不会太大
                    if max(img.size) > self.max_frame_width:
                        img.thumbnail((self.max_frame_width, self.max_frame_width), Image.Resampling.LANCZOS)
                    loaded_images.append(img)
                else:
                    logger.warning(f"帧文件不存在: {frame_path}")
            
            logger.info(f'加载了 {len(loaded_images)} 帧用于处理')
            
            # 构建消息内容
            content = []
            content.append({"type": "text", "text": self.system_prompt})
            content.append({"type": "text", "text": f"\nSource video ({len(frame_paths_video1)} frames):"})
            for frame_path in frame_paths_video1:
                content.append({"type": "image", "image": frame_path})
            
            content.append({"type": "text", "text": f"\nDestination video ({len(frame_paths_video2)} frames):"})
            for frame_path in frame_paths_video2:
                content.append({"type": "image", "image": frame_path})
            
            messages = [{"role": "user", "content": content}]
            
            # 处理文本和图像
            text = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
            )
            
            # 模型推理（显存优化 + 指令模式优化）
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    inputs = self.processor(
                        images=loaded_images[0] if len(loaded_images) == 1 else loaded_images, 
                        text=text, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True
                    ).to(self.model.device)
                    
                    # 指令模型推荐配置：启用采样 + temperature=0.2
                    generated_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=1024,
                        do_sample=True,           # 启用采样
                        temperature=0.2,          # 低温度，更确定性但仍有轻微随机性
                        top_p=0.9,               # nucleus sampling
                        top_k=50,                # top-k sampling
                        num_beams=1,             # 不使用beam search以节省显存
                        use_cache=True
                    )
                    
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    response = self.processor.batch_decode(
                        generated_ids_trimmed, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=False
                    )[0]
            
            # 释放显存
            del inputs, generated_ids, generated_ids_trimmed, loaded_images
            torch.cuda.empty_cache()
            gc.collect()
            
            return response
            
        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            raise e

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
        frame_paths_video1 = []
        frame_paths_video2 = []
        
        while retry_count < self.max_retries:
            try:
                # 清理显存
                torch.cuda.empty_cache()
                gc.collect()
                
                for video_path in [video1_path, video2_path]:
                    if not os.path.exists(video_path):
                        raise FileNotFoundError(f"不存在: {video_path}")
                
                logger.info(f"[Entry {index}] 提取视频1帧...")
                frame_paths_video1 = self.extract_frames_from_video(video1_path, "source")
                
                logger.info(f"[Entry {index}] 提取视频2帧...")
                frame_paths_video2 = self.extract_frames_from_video(video2_path, "destination")
                
                # 显示当前显存使用
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    logger.info(f"推理前显存使用: {allocated:.2f}GB")
                
                # 使用Kimi模型处理
                with model_lock:
                    logger.info(f"[Entry {index}] 调用Kimi模型...")
                    start_time = time.time()
                    response_content = self.process_with_kimi(frame_paths_video1, frame_paths_video2)
                    inference_time = time.time() - start_time
                    logger.info(f"[Entry {index}] 推理耗时: {inference_time:.2f}秒")
                    time.sleep(self.model_delay)
                
                # 清理临时文件
                self.cleanup_temp_frames(frame_paths_video1)
                self.cleanup_temp_frames(frame_paths_video2)
                
                result = {
                    "index": index,
                    "video1_path": video1_path,
                    "video2_path": video2_path,
                    "frames_extracted": {
                        "video1": len(frame_paths_video1), 
                        "video2": len(frame_paths_video2)
                    },
                    "response": response_content,
                    "inference_time": inference_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                self._append_result_to_file(result)
                self._save_checkpoint(index, success=True)
                self.successful += 1
                logger.info(f"[Entry {index}] ✅ 成功")
                return result
                
            except torch.cuda.OutOfMemoryError as e:
                last_error = e
                retry_count += 1
                logger.error(f"[Entry {index}] ❌ 显存不足，尝试{retry_count}次")
                
                # 清理
                self.cleanup_temp_frames(frame_paths_video1)
                self.cleanup_temp_frames(frame_paths_video2)
                torch.cuda.empty_cache()
                gc.collect()
                
                if retry_count < self.max_retries:
                    logger.info(f"等待{retry_count * 5}秒后重试...")
                    time.sleep(retry_count * 5)
                    
            except Exception as e:
                last_error = e
                retry_count += 1
                self.cleanup_temp_frames(frame_paths_video1)
                self.cleanup_temp_frames(frame_paths_video2)
                logger.error(f"[Entry {index}] ❌ 尝试{retry_count}失败: {e}")
                traceback.print_exc()
                
                # 清理显存
                torch.cuda.empty_cache()
                gc.collect()
                
                if retry_count < self.max_retries:
                    time.sleep(retry_count * 3)
        
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
                        logger.info(f"输出文件已存在: {len(data)}条")
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
                logger.error(f"写入失败: {e}")
    
    def _load_system_prompt(self):
        prompt_path = "prompt_generate.txt"
        if not os.path.exists(prompt_path):
            default_prompt = "请对比分析这两个视频的差异，详细描述它们在内容、风格、质量等方面的不同。"
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
        logger.info(f"开始批量处理 (Kimi本地模型 - 显存优化版)")
        logger.info(f"{'='*80}\n")
        
        self.start_time = time.time()
        
        try:
            data_list = self.load_input_data()
            if not data_list:
                logger.error("没有数据")
                return
            
            total_pairs = len(data_list)
            logger.info(f"共 {total_pairs} 对视频\n")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.process_single_entry, entry): entry 
                    for entry in data_list
                }
                for future in as_completed(futures):
                    try:
                        future.result(timeout=self.timeout)
                    except Exception as e:
                        logger.error(f"任务执行错误: {e}")
            
            elapsed = time.time() - self.start_time
            logger.info(f"\n{'='*80}")
            logger.info(f"✅ 完成！")
            logger.info(f"总数: {total_pairs}, 成功: {self.successful}, 失败: {self.failed}, 跳过: {self.skipped_processed}")
            logger.info(f"用时: {elapsed:.2f}秒")
            logger.info(f"{'='*80}\n")
            
        except KeyboardInterrupt:
            logger.warning(f"\n用户中断")
        except Exception as e:
            logger.error(f"\n错误: {e}")
            traceback.print_exc()
        finally:
            # 清理临时目录和显存
            if os.path.exists(self.temp_frame_dir):
                try:
                    import shutil
                    shutil.rmtree(self.temp_frame_dir)
                    os.makedirs(self.temp_frame_dir)
                except:
                    pass
            
            torch.cuda.empty_cache()
            gc.collect()


def main():
    config = {
        'input_json_file': '/root/bayes-tmp/workplace/checklist.json',  
        'output_file': 'video_analysis_results_kimi.json',
        'error_file': 'video_analysis_errors_kimi.json',
        'checkpoint_file': 'processing_checkpoint_kimi.json',
        'max_workers': 1,  
        'max_pairs': None,
        'model_delay': 1,
        'timeout': 600,
        'model_path': '/root/bayes-tmp/workplace/model/kimi',
        'resume_from_checkpoint': True,
        'max_retries': 3,
        
        # 显存优化配置
        'frame_interval_seconds': 2.0,    # 1fps，减少帧数
        'max_frames_per_video': 4,        # 每个视频最多4帧（总共8帧）
        'max_frame_width': 512,           # 降低分辨率到512
        'temp_frame_dir': 'temp_kimi_frames'
    }
    
    processor = VideoProcessor(config)
    processor.run()


if __name__ == "__main__":
    main()
