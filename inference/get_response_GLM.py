"""
视频帧提取对比分析工具 - GLM-4.1V-Thinking 本地版本（显存优化）
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
from transformers import AutoProcessor, Glm4vForConditionalGeneration
import gc

sys.stdout.reconfigure(encoding='utf-8')

# 设置环境变量优化显存
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"video_processing_glm_{timestamp}.log")
error_log_file = os.path.join(log_dir, f"video_processing_errors_glm_{timestamp}.log")

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
        self.model_delay = config.get('model_delay', 1)
        self.timeout = config.get('timeout', 600)
        self.model_path = config.get('model_path', 'THUDM/GLM-4.1V-9B-Thinking')
        self.resume_from_checkpoint = config.get('resume_from_checkpoint', True)
        self.max_retries = config.get('max_retries', 3)
        self.frame_interval_seconds = config.get('frame_interval_seconds', 1.0)
        self.max_frames_per_video = config.get('max_frames_per_video', 8)
        self.max_frame_width = config.get('max_frame_width', 512)
        self.temp_frame_dir = config.get('temp_frame_dir', 'temp_frames')
        
        # GLM-specific generation parameters
        self.first_max_tokens = config.get('first_max_tokens', 4096)
        self.force_max_tokens = config.get('force_max_tokens', 4096)
        self.temperature = config.get('temperature', 0.2)
        self.do_sample = config.get('do_sample', True)
  
        if not os.path.exists(self.temp_frame_dir):
            os.makedirs(self.temp_frame_dir)
  
        for file_path in [self.output_file, self.error_file, self.checkpoint_file]:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
  
        # 清理显存
        torch.cuda.empty_cache()
        gc.collect()
  
        # 初始化GLM-4V模型
        logger.info("正在加载 GLM-4.1V-Thinking 模型...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        # 定义GLM的特殊tokens（假设Thinking模型使用<thinking>等标签，根据实际模型调整）
        self.special_tokens = {
            "think_start": self.processor.tokenizer.convert_tokens_to_ids("<thinking>"),
            "think_end": self.processor.tokenizer.convert_tokens_to_ids("</thinking>"),
            "answer_start": self.processor.tokenizer.convert_tokens_to_ids("<answer>"),
            "answer_end": self.processor.tokenizer.convert_tokens_to_ids("</answer>"),
        }
        
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            torch_dtype=torch.bfloat16, # 使用bfloat16以获得更好性能
            device_map="auto",
            trust_remote_code=True
            # 根据用户要求，已移除 attn_implementation="flash_attention_2"
        )
        self.model.eval()
  
        logger.info("✅ GLM-4.1V-Thinking 模型加载完成")
  
        # 显示显存使用情况
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"GPU {i}: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
  
        logger.info(f"="*80)
        logger.info(f"配置: Model=GLM-4.1V-Thinking, FPS={1/self.frame_interval_seconds:.1f}, MaxFrames={self.max_frames_per_video}")
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
      
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_frame_dir = os.path.join(self.temp_frame_dir, f"{video_name}_{video_label}")
            os.makedirs(video_frame_dir, exist_ok=True)
      
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
          
                if width > self.max_frame_width:
                    scale = self.max_frame_width / width
                    new_w = int(width * scale)
                    new_h = int(height * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
          
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
  
        if frame_paths:
            folder = os.path.dirname(frame_paths[0])
            try:
                if os.path.exists(folder) and not os.listdir(folder):
                    os.rmdir(folder)
            except:
                pass

    def process_with_glm(self, frame_paths_video1, frame_paths_video2):
        """使用 GLM-4V 模型处理视频帧（带强制完成逻辑）"""
        torch.cuda.empty_cache()
        gc.collect()

        try:
            # 加载图像
            loaded_images_v1 = [Image.open(p).convert("RGB") for p in frame_paths_video1 if os.path.exists(p)]
            loaded_images_v2 = [Image.open(p).convert("RGB") for p in frame_paths_video2 if os.path.exists(p)]
            all_loaded_images = loaded_images_v1 + loaded_images_v2
        
            logger.info(f'加载了 {len(all_loaded_images)} 帧用于处理')

            # 构建符合 transformers 库多模态格式的 content 列表
            content_list = []

            # 1. 添加系统提示和视频1的描述文本
            content_list.append({
                "type": "text",
                "text": (
                    f"{self.system_prompt}\n\n"
                    f"Source video ({len(loaded_images_v1)} frames are provided):"
                )
            })

            # 2. 为视频1的每一帧添加一个图片占位符
            for _ in loaded_images_v1:
                content_list.append({"type": "image"})

            # 3. 添加视频2的描述文本
            content_list.append({
                "type": "text",
                "text": f"\n\nDestination video ({len(loaded_images_v2)} frames are provided):"
            })

            # 4. 为视频2的每一帧添加一个图片占位符
            for _ in loaded_images_v2:
                content_list.append({"type": "image"})

            # 构建最终的 messages 结构
            messages = [
                {
                    "role": "user",
                    "content": content_list
                }
            ]
        
            # 调用带强制补全逻辑的生成函数
            with torch.no_grad():
                result_dict = self.generate_with_force_completion(
                    messages=messages,
                    images=all_loaded_images
                )
        
            response = result_dict.get("output_text", "")

            # 释放显存
            del all_loaded_images, loaded_images_v1, loaded_images_v2
            torch.cuda.empty_cache()
            gc.collect()
        
            return response

        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            raise e

    def generate_with_force_completion(self, messages, images):
        """
        GLM-4V的 robust inference 实现，确保输出完整性。
        """
        # 1. 使用 apply_chat_template 生成带占位符的提示文本
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 2. 使用 processor 处理提示文本和图像
        inputs = self.processor(
            text=prompt_text,
            images=images,
            return_tensors="pt",
            padding=False  # 单样本，无需填充
        ).to(self.model.device)
        
        inputs.pop("token_type_ids", None)
        input_length = inputs["input_ids"].shape[1]

        # 第一轮生成
        with torch.no_grad():
            first_generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.first_max_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
            )

        first_output_ids = first_generated_ids[0][input_length:]
    
        needs_completion = self._check_needs_completion_by_tokens(
            first_output_ids, self.first_max_tokens
        )

        if not needs_completion:
            final_output = self.processor.decode(
                first_output_ids, skip_special_tokens=False
            )
            return {
                "output_text": final_output,
                "complete": True,
                "reason": "first_generation_complete",
            }

        force_input_ids = self._prepare_force_input(
            inputs["input_ids"], first_output_ids
        )

        force_inputs = {
            "input_ids": force_input_ids.to(self.model.device),
            "attention_mask": torch.ones_like(force_input_ids).to(self.model.device),
        }

        if "pixel_values" in inputs:
            force_inputs["pixel_values"] = inputs["pixel_values"]

        second_generated_ids = self.model.generate(
            **force_inputs,
            max_new_tokens=self.force_max_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
        )

        second_output_ids = second_generated_ids[0][force_input_ids.shape[1] :]
    
        added_tokens = force_input_ids[0][input_length + len(first_output_ids) :]
        complete_output_ids = torch.cat(
            [first_output_ids, added_tokens, second_output_ids], dim=0
        )
        complete_output_text = self.processor.decode(
            complete_output_ids, skip_special_tokens=False
        )

        return {
            "output_text": complete_output_text,
            "complete": (self.special_tokens["answer_end"] in complete_output_ids.tolist()),
            "reason": "force_completion_success",
        }

    def _check_needs_completion_by_tokens(self, output_token_ids, max_tokens):
        token_list = output_token_ids.tolist()
        reached_max = len(token_list) >= max_tokens
        has_answer_end = self.special_tokens["answer_end"] in token_list
        has_think_start = self.special_tokens["think_start"] in token_list
        has_think_end = self.special_tokens["think_end"] in token_list

        if has_answer_end:
            return False
        if reached_max:
            return True
        if has_think_start and not has_think_end:
            return True
        return False

    def _prepare_force_input(self, original_input_ids, first_output_ids):
        first_output_list = first_output_ids.tolist()
        has_think_end = self.special_tokens["think_end"] in first_output_list
        has_answer_start = self.special_tokens["answer_start"] in first_output_list
        tokens_to_add = []

        if not has_think_end:
            tokens_to_add.extend(
                [self.special_tokens["think_end"], self.special_tokens["answer_start"]]
            )
        elif not has_answer_start:
            tokens_to_add.append(self.special_tokens["answer_start"])

        if tokens_to_add:
            additional_tokens = torch.tensor(tokens_to_add).unsqueeze(0).to(self.model.device)
            force_input_ids = torch.cat(
                [original_input_ids, first_output_ids.unsqueeze(0), additional_tokens], dim=1
            )
        else:
            force_input_ids = torch.cat(
                [original_input_ids, first_output_ids.unsqueeze(0)], dim=1
            )
        return force_input_ids

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
                torch.cuda.empty_cache()
                gc.collect()
          
                for video_path in [video1_path, video2_path]:
                    if not os.path.exists(video_path):
                        raise FileNotFoundError(f"不存在: {video_path}")
          
                logger.info(f"[Entry {index}] 提取视频1帧...")
                frame_paths_video1 = self.extract_frames_from_video(video1_path, "source")
          
                logger.info(f"[Entry {index}] 提取视频2帧...")
                frame_paths_video2 = self.extract_frames_from_video(video2_path, "destination")
          
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    logger.info(f"推理前显存使用: {allocated:.2f}GB")
          
                with model_lock:
                    logger.info(f"[Entry {index}] 调用 GLM-4V 模型...")
                    start_time = time.time()
                    response_content = self.process_with_glm(frame_paths_video1, frame_paths_video2)
                    inference_time = time.time() - start_time
                    logger.info(f"[Entry {index}] 推理耗时: {inference_time:.2f}秒")
                    time.sleep(self.model_delay)
          
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

    def _append_error_to_file(self, error_info):
        with file_lock:
            try:
                if os.path.exists(self.error_file):
                    with open(self.error_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    data = []
                data.append(error_info)
                with open(self.error_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"写入错误文件失败: {e}")

    def _load_checkpoint(self):
        if self.resume_from_checkpoint and os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                logger.warning("加载检查点失败，使用空数据")
        return {"successful_indices": []}

    def _save_checkpoint(self, index, success):
        with file_lock:
            if success:
                self.checkpoint_data['successful_indices'].append(index)
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(self.checkpoint_data, f, ensure_ascii=False, indent=2)

    def _load_system_prompt(self):
        prompt_path = "prompt_generate.txt"
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            default_prompt = "请对比分析这两个视频的差异，详细描述它们在内容、风格、质量等方面的不同。"
            logger.warning(f"提示文件不存在，使用默认: {default_prompt}")
            return default_prompt

    def process_batch(self):
        logger.info("\n" + "="*80)
        logger.info("开始批量处理 (GLM-4.1V-Thinking - 显存优化版)")
        logger.info("="*80)

        # 加载输入数据
        if not os.path.exists(self.input_json_file):
            logger.error(f"输入文件不存在: {self.input_json_file}")
            return

        with open(self.input_json_file, 'r', encoding='utf-8') as f:
            entries = json.load(f)

        if self.max_pairs:
            entries = entries[:self.max_pairs]

        logger.info(f"共 {len(entries)} 对视频")

        self.start_time = time.time()
        self.total_pairs = len(entries)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_single_entry, entry) for entry in entries]

            for future in as_completed(futures):
                try:
                    future.result(timeout=self.timeout)
                except TimeoutError:
                    logger.error("处理超时")
                except Exception as e:
                    logger.error(f"异常: {e}")

        elapsed = time.time() - self.start_time
        logger.info("="*80)
        logger.info(f"处理完成: 成功 {self.successful}, 失败 {self.failed}, 跳过 {self.skipped_processed}")
        logger.info(f"总耗时: {elapsed:.2f}秒")


if __name__ == "__main__":
    # 示例配置（根据需要调整）
    config = {
        "input_json_file": "/root/bayes-tmp/workplace/checklist.json",  # 包含视频对的JSON文件，格式: [{"index": 0, "video1_path": "...", "video2_path": "..."}, ...]
        "output_file": "video_analysis_results_glm.json",
        "error_file": "video_analysis_errors_glm.json",
        "checkpoint_file": "processing_checkpoint.json",
        "max_workers": 1,  # 并发处理数量（受显存限制，通常为1）
        "max_pairs": None,  # 处理的最大视频对数，None为全部
        "model_path": "/root/bayes-tmp/workplace/model/GLM",  # 模型路径
        "frame_interval_seconds": 1.0,  # 帧提取间隔
        "max_frames_per_video": 8,  # 每视频最大帧数
        "first_max_tokens": 4096,
        "force_max_tokens": 4096,
        "temperature": 0.2,
        "do_sample": True,
    }

    processor = VideoProcessor(config)
    processor.process_batch()
