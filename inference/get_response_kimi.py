"""
Video Frame Extraction and Comparison Analysis Tool - Kimi Local Version (VRAM Optimized)
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

# Set environment variables to optimize VRAM
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
        self.model_path = config.get('model_path', 'moonshot-ai/Kimi-VL')
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
        
        # Clear VRAM
        torch.cuda.empty_cache()
        gc.collect()
        
        # Initialize Kimi model (VRAM optimized)
        logger.info("Loading Kimi model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,  # Use float16 to save VRAM
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,  # Reduce CPU memory usage
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
        logger.info("✅ Kimi model loaded successfully")
        
        # Display VRAM usage
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"GPU {i}: Allocated {allocated:.2f}GB, Reserved {reserved:.2f}GB")
        
        logger.info(f"="*80)
        logger.info(f"Config: Model=Kimi-VL, FPS={1/self.frame_interval_seconds:.1f}, MaxFrames={self.max_frames_per_video}")
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
        """Extract video frames and save to disk, return list of frame paths"""
        frame_paths = []
        cap = None
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video: {os.path.basename(video_path)} - {total_frames} frames, {fps:.1f}fps, {duration:.1f}s")
            
            # Calculate frame indices to extract
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
            
            logger.info(f"Extracting {len(frame_indices)} frames")
            
            # Create video-specific folder
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_frame_dir = os.path.join(self.temp_frame_dir, f"{video_name}_{video_label}")
            os.makedirs(video_frame_dir, exist_ok=True)
            
            # Extract and save frames
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Resize to save VRAM
                if width > self.max_frame_width:
                    scale = self.max_frame_width / width
                    new_w = int(width * scale)
                    new_h = int(height * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Save as JPEG to save space
                frame_filename = f"frame_{i:04d}.jpg"
                frame_filepath = os.path.join(video_frame_dir, frame_filename)
                img.save(frame_filepath, quality=85, optimize=True)
                frame_paths.append(frame_filepath)
            
            cap.release()
            logger.info(f"✅ Extraction complete: {len(frame_paths)} frames saved to {video_frame_dir}")
            return frame_paths
            
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            if cap:
                cap.release()
            raise
    
    def cleanup_temp_frames(self, frame_paths):
        """Clean up temporary frame files"""
        for path in frame_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass
        
        # Clean up empty folders
        if frame_paths:
            folder = os.path.dirname(frame_paths[0])
            try:
                if os.path.exists(folder) and not os.listdir(folder):
                    os.rmdir(folder)
            except:
                pass
    
    def process_with_kimi(self, frame_paths_video1, frame_paths_video2):
        """Process video frames using Kimi model (VRAM optimized version - instruction mode optimized)"""
        
        # Clear VRAM
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            # Load images
            loaded_images = []
            for frame_path in frame_paths_video1 + frame_paths_video2:
                if os.path.exists(frame_path):
                    img = Image.open(frame_path)
                    # Ensure image is not too large
                    if max(img.size) > self.max_frame_width:
                        img.thumbnail((self.max_frame_width, self.max_frame_width), Image.Resampling.LANCZOS)
                    loaded_images.append(img)
                else:
                    logger.warning(f"Frame file does not exist: {frame_path}")
            
            logger.info(f'Loaded {len(loaded_images)} frames for processing')
            
            # Build message content
            content = []
            content.append({"type": "text", "text": self.system_prompt})
            content.append({"type": "text", "text": f"\nVideo A ({len(frame_paths_video1)} frames):"})
            for frame_path in frame_paths_video1:
                content.append({"type": "image", "image": frame_path})

            content.append({"type": "text", "text": f"\nVideo B ({len(frame_paths_video2)} frames):"})
            for frame_path in frame_paths_video2:
                content.append({"type": "image", "image": frame_path})
            
            messages = [{"role": "user", "content": content}]
            
            # Process text and images
            text = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
            )
            
            # Model inference (VRAM optimized + instruction mode optimized)
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    inputs = self.processor(
                        images=loaded_images[0] if len(loaded_images) == 1 else loaded_images, 
                        text=text, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True
                    ).to(self.model.device)
                    
                    # Instruction model recommended configuration: enable sampling + temperature=0.2
                    generated_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=1024,
                        do_sample=True,           # Enable sampling
                        temperature=0.2,          # Low temperature, more deterministic but still slightly random
                        top_p=0.9,               # nucleus sampling
                        top_k=50,                # top-k sampling
                        num_beams=1,             # Don't use beam search to save VRAM
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
            
            # Release VRAM
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
        
        logger.info(f"\n[Entry {index}] Starting processing")
        
        if index in self.processed_indices:
            logger.info(f"[Entry {index}] Already processed, skipping")
            self.skipped_processed += 1
            return None
        
        retry_count = 0
        last_error = None
        frame_paths_video1 = []
        frame_paths_video2 = []
        
        while retry_count < self.max_retries:
            try:
                # Clear VRAM
                torch.cuda.empty_cache()
                gc.collect()
                
                for video_path in [video1_path, video2_path]:
                    if not os.path.exists(video_path):
                        raise FileNotFoundError(f"Does not exist: {video_path}")
                
                logger.info(f"[Entry {index}] Extracting video A frames...")
                frame_paths_video1 = self.extract_frames_from_video(video1_path, "video_a")

                logger.info(f"[Entry {index}] Extracting video B frames...")
                frame_paths_video2 = self.extract_frames_from_video(video2_path, "video_b")

                # Display current VRAM usage
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    logger.info(f"VRAM usage before inference: {allocated:.2f}GB")
                
                # Process using Kimi model
                with model_lock:
                    logger.info(f"[Entry {index}] Calling Kimi model...")
                    start_time = time.time()
                    response_content = self.process_with_kimi(frame_paths_video1, frame_paths_video2)
                    inference_time = time.time() - start_time
                    logger.info(f"[Entry {index}] Inference time: {inference_time:.2f}s")
                    time.sleep(self.model_delay)
                
                # Clean up temporary files
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
                logger.info(f"[Entry {index}] ✅ Success")
                return result
                
            except torch.cuda.OutOfMemoryError as e:
                last_error = e
                retry_count += 1
                logger.error(f"[Entry {index}] ❌ Out of VRAM, attempt {retry_count}")
                
                # Clean up
                self.cleanup_temp_frames(frame_paths_video1)
                self.cleanup_temp_frames(frame_paths_video2)
                torch.cuda.empty_cache()
                gc.collect()
                
                if retry_count < self.max_retries:
                    logger.info(f"Waiting {retry_count * 5}s before retry...")
                    time.sleep(retry_count * 5)
                    
            except Exception as e:
                last_error = e
                retry_count += 1
                self.cleanup_temp_frames(frame_paths_video1)
                self.cleanup_temp_frames(frame_paths_video2)
                logger.error(f"[Entry {index}] ❌ Attempt {retry_count} failed: {e}")
                traceback.print_exc()
                
                # Clear VRAM
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
                        logger.info(f"Output file exists: {len(data)} entries")
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
                logger.error(f"Write failed: {e}")
    
    def _load_system_prompt(self):
        prompt_path = "prompt_generate.txt"
        if not os.path.exists(prompt_path):
            default_prompt = "Please compare and analyze the differences between these two videos, describing in detail their differences in content, style, quality, etc."
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
            raise FileNotFoundError(f"Input file does not exist: {self.input_json_file}")
        
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
        logger.info(f"Starting batch processing (Kimi local model - VRAM optimized version)")
        logger.info(f"{'='*80}\n")
        
        self.start_time = time.time()
        
        try:
            data_list = self.load_input_data()
            if not data_list:
                logger.error("No data to process")
                return
            
            total_pairs = len(data_list)
            logger.info(f"Total {total_pairs} video pairs\n")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.process_single_entry, entry): entry 
                    for entry in data_list
                }
                for future in as_completed(futures):
                    try:
                        future.result(timeout=self.timeout)
                    except Exception as e:
                        logger.error(f"Task execution error: {e}")
            
            elapsed = time.time() - self.start_time
            logger.info(f"\n{'='*80}")
            logger.info(f"✅ Complete!")
            logger.info(f"Total: {total_pairs}, Success: {self.successful}, Failed: {self.failed}, Skipped: {self.skipped_processed}")
            logger.info(f"Time elapsed: {elapsed:.2f}s")
            logger.info(f"{'='*80}\n")
            
        except KeyboardInterrupt:
            logger.warning(f"\nInterrupted by user")
        except Exception as e:
            logger.error(f"\nError: {e}")
            traceback.print_exc()
        finally:
            # Clean up temporary directory and VRAM
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
        'input_json_file': 'input_videos.json',  
        'output_file': 'video_analysis_results_kimi.json',
        'error_file': 'video_analysis_errors_kimi.json',
        'checkpoint_file': 'processing_checkpoint_kimi.json',
        'max_workers': 1,  
        'max_pairs': None,
        'model_delay': 1,
        'timeout': 600,
        'model_path': 'moonshot-ai/Kimi-VL',
        'resume_from_checkpoint': True,
        'max_retries': 3,
        
        # VRAM optimization configuration
        'frame_interval_seconds': 2.0,    # 1fps, reduce frame count
        'max_frames_per_video': 4,        # Max 4 frames per video (8 frames total)
        'max_frame_width': 512,           # Reduce resolution to 512
        'temp_frame_dir': 'temp_kimi_frames'
    }
    
    processor = VideoProcessor(config)
    processor.run()


if __name__ == "__main__":
    main()
