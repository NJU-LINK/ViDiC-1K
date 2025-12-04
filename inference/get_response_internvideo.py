"""
Video Comparison and Analysis Tool - InternVideo2.5 Version
(Enhanced with detailed logging and clear video differentiation)
"""

import os
import json
import sys
import gc
import time
import logging
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import torch
import numpy as np
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import argparse

# Ensure stdout can handle UTF-8 characters, useful in some environments
sys.stdout.reconfigure(encoding='utf-8')

# Set environment variable to optimize CUDA memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# --- Global Locks ---
file_lock = Lock()  # For thread-safe file writing
model_lock = Lock() # For thread-safe model inference if max_workers > 1

# --- Constants ---
# Inference parameters
DEFAULT_NUM_SEGMENTS = 32  # Default number of frames to sample from a video
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# Image preprocessing constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --- Utility Functions for Image/Video Processing ---

def build_transform(input_size):
    """Builds the image transformation pipeline."""
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Finds the best grid-like aspect ratio for image patching."""
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
    """Dynamically preprocesses an image by splitting it into patches based on aspect ratio."""
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
            (i % target_aspect_ratio[0]) * image_size,
            (i // target_aspect_ratio[0]) * image_size,
            ((i % target_aspect_ratio[0]) + 1) * image_size,
            ((i // target_aspect_ratio[0]) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
        
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
        
    return processed_images


def get_frame_indices(total_frames, fps, num_segments=32, bound=None):
    """Calculates indices for frame sampling."""
    if bound:
        start, end = bound
        start_idx = max(0, round(start * fps))
        end_idx = min(total_frames, round(end * fps))
    else:
        start_idx, end_idx = 0, total_frames

    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    # Ensure indices are within bounds
    frame_indices = np.clip(frame_indices, start_idx, end_idx - 1).astype(int)
    return frame_indices


def load_video(video_path, detail_logger, input_size=448, max_num=1, num_segments=32, video_name="Video"):
    """Loads and processes frames from a video file with detailed logging."""
    try:
        detail_logger.info(f"[{video_name}] Starting to load: {video_path}")
        
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        detail_logger.info(f"[{video_name}] File size: {file_size_mb:.2f} MB")
        
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        fps = float(vr.get_avg_fps())
        duration = total_frames / fps
        
        detail_logger.info(f"[{video_name}] Video Info - Frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
        
        pixel_values_list, num_patches_list = [], []
        transform = build_transform(input_size=input_size)
        frame_indices = get_frame_indices(total_frames, fps, num_segments=num_segments)
        
        detail_logger.info(f"[{video_name}] Selected frame indices: {frame_indices.tolist()}")
        
        for i, frame_index in enumerate(frame_indices):
            img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
            patches = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = torch.stack([transform(tile) for tile in patches])
            
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
            
            if (i + 1) % 8 == 0:
                detail_logger.debug(f"[{video_name}] Processed frame {i+1}/{len(frame_indices)}")
        
        pixel_values = torch.cat(pixel_values_list)
        detail_logger.info(f"[{video_name}] Loading complete - Tensor shape: {pixel_values.shape}")
        
        return pixel_values, num_patches_list
        
    except Exception as e:
        detail_logger.error(f"[{video_name}] Failed to load video: {e}")
        raise


class VideoProcessor:
    def __init__(self, config, logger, detail_logger):
        self.logger = logger
        self.detail_logger = detail_logger
        
        self.input_json_file = config['input_json_file']
        self.output_file = config['output_file']
        self.error_file = config['error_file']
        self.checkpoint_file = config['checkpoint_file']
        self.max_workers = config['max_workers']
        self.max_pairs = config['max_pairs']
        self.model_delay = config['model_delay']
        self.timeout = config['timeout']
        self.resume_from_checkpoint = config['resume_from_checkpoint']
        self.max_retries = config['max_retries']
        self.prompt_file = config['prompt_file']
        
        # InternVideo specific settings
        self.num_segments = config['num_segments']
        self.input_size = config['input_size']
        self.max_num_patches = config['max_num_patches']
        
        # Create output directories if they don't exist
        for file_path in [self.output_file, self.error_file, self.checkpoint_file]:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
        
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Video Processor Initialized")
        self.logger.info(f"Model: InternVideo2.5 (Local)")
        self.logger.info(f"Device: {DEVICE}, Dtype: {DTYPE}")
        self.logger.info(f"Frames per video: {self.num_segments}")
        self.logger.info(f"Input image size: {self.input_size}")
        self.logger.info(f"{'='*80}")
        
        self.successful_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.start_time = None
        
        self.checkpoint_data = self._load_checkpoint()
        self.processed_indices = set(self.checkpoint_data.get('successful_indices', []))
        self.system_prompt = self._load_system_prompt()
        self._initialize_output_file()
    
    def process_video_pair(self, video1_path, video2_path, model, tokenizer, generation_config):
        """Processes a pair of videos and generates a comparative analysis."""
        try:
            process_start_time = time.time()
            
            self.logger.info(f"[Source Video] Loading: {os.path.basename(video1_path)}")
            video1_start_time = time.time()
            pixel_values1, num_patches_list1 = load_video(
                video1_path, self.detail_logger, num_segments=self.num_segments,
                max_num=self.max_num_patches, input_size=self.input_size, video_name="Source Video"
            )
            video1_load_time = time.time() - video1_start_time
            self.logger.info(f"[Source Video] Load time: {video1_load_time:.2f}s")
            
            self.logger.info(f"[Destination Video] Loading: {os.path.basename(video2_path)}")
            video2_start_time = time.time()
            pixel_values2, num_patches_list2 = load_video(
                video2_path, self.detail_logger, num_segments=self.num_segments,
                max_num=self.max_num_patches, input_size=self.input_size, video_name="Destination Video"
            )
            video2_load_time = time.time() - video2_start_time
            self.logger.info(f"[Destination Video] Load time: {video2_load_time:.2f}s")
            
            self.detail_logger.info(f"===== Video Comparison Details =====")
            self.detail_logger.info(f"Source Video: {video1_path} ({len(num_patches_list1)} frames, {sum(num_patches_list1)} patches)")
            self.detail_logger.info(f"Destination Video: {video2_path} ({len(num_patches_list2)} frames, {sum(num_patches_list2)} patches)")
            self.detail_logger.info(f"====================================")
            
            pixel_values = torch.cat([pixel_values1, pixel_values2], dim=0).to(DTYPE).to(DEVICE)
            num_patches_list = num_patches_list1 + num_patches_list2
            
            video1_prefix = f"[SOURCE VIDEO: {os.path.basename(video1_path)}]\n" + "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list1))])
            video2_prefix = f"\n[DESTINATION VIDEO: {os.path.basename(video2_path)}]\n" + "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list2))])
            
            full_prompt = video1_prefix + video2_prefix + "\n" + self.system_prompt
            self.detail_logger.debug(f"Prompt length: {len(full_prompt)} characters")
            
            inference_start_time = time.time()
            with torch.no_grad():
                response, _ = model.chat(
                    tokenizer, pixel_values=pixel_values, question=full_prompt,
                    generation_config=generation_config, num_patches_list=num_patches_list,
                    history=None, return_history=True
                )
            inference_time = time.time() - inference_start_time
            
            self.detail_logger.info(f"Model inference complete - Time: {inference_time:.2f}s, Response length: {len(response)} chars")
            
            del pixel_values, pixel_values1, pixel_values2
            torch.cuda.empty_cache()
            
            total_process_time = time.time() - process_start_time
            self.detail_logger.info(f"Video pair processing complete - Total time: {total_process_time:.2f}s")
            
            return {
                "response": response,
                "source_video_load_time": video1_load_time,
                "destination_video_load_time": video2_load_time,
                "inference_time": inference_time,
                "total_time": total_process_time
            }
            
        except Exception as e:
            self.logger.error(f"Error processing video pair: {e}")
            self.detail_logger.error(f"Detailed error: {traceback.format_exc()}")
            torch.cuda.empty_cache()
            raise

    def process_single_entry(self, entry, model, tokenizer, generation_config):
        """Wrapper to process a single entry with retry logic."""
        index = entry['index']
        video1_path = entry['video1_path']
        video2_path = entry['video2_path']
        
        self.logger.info(f"\n{'='*60}\n[Entry {index}] Starting processing...\n  Source: {video1_path}\n  Destination: {video2_path}\n{'='*60}")
        
        if index in self.processed_indices:
            self.logger.info(f"[Entry {index}] Already processed, skipping.")
            self.skipped_count += 1
            return None
        
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                for path, v_type in [(video1_path, 'Source'), (video2_path, 'Destination')]:
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"{v_type} video file not found: {path}")
                
                with model_lock:
                    result_data = self.process_video_pair(video1_path, video2_path, model, tokenizer, generation_config)
                    time.sleep(self.model_delay)
                
                self.logger.info(f"[Entry {index}] Inference complete. Times: "
                                 f"SrcLoad={result_data['source_video_load_time']:.2f}s, "
                                 f"DstLoad={result_data['destination_video_load_time']:.2f}s, "
                                 f"Infer={result_data['inference_time']:.2f}s, "
                                 f"Total={result_data['total_time']:.2f}s")

                result = {
                    "index": index,
                    "source_video_path": video1_path,
                    "destination_video_path": video2_path,
                    "response": result_data["response"],
                    "processing_times": {k: v for k, v in result_data.items() if k != "response"},
                    "timestamp": datetime.now().isoformat()
                }
                
                self._append_to_file(self.output_file, result)
                self._save_checkpoint(index, success=True)
                self.successful_count += 1
                self.logger.info(f"[Entry {index}] ✅ Successfully processed and saved.")
                return result
                
            except torch.cuda.OutOfMemoryError as e:
                last_error = e
                self.logger.error(f"[Entry {index}] Out of Memory on attempt {attempt}/{self.max_retries}. Freeing memory and retrying...")
                self.detail_logger.error(f"[Entry {index}] OOM details: {traceback.format_exc()}")
                torch.cuda.empty_cache()
                gc.collect()
                if attempt < self.max_retries:
                    time.sleep(attempt * 5)
                    
            except Exception as e:
                last_error = e
                self.logger.error(f"[Entry {index}] Failed on attempt {attempt}/{self.max_retries}: {e}")
                self.detail_logger.error(f"[Entry {index}] Error details: {traceback.format_exc()}")
                if attempt < self.max_retries:
                    time.sleep(attempt * 3)
        
        self.failed_count += 1
        error_info = {
            "index": index, "source_video_path": video1_path, "destination_video_path": video2_path,
            "error": str(last_error), "traceback": traceback.format_exc(), "timestamp": datetime.now().isoformat()
        }
        self._append_to_file(self.error_file, error_info)
        self.logger.error(f"[Entry {index}] ❌ Failed after all retries.")
        return None

    def load_input_data(self):
        """Loads and validates input data from the JSON file."""
        if not os.path.exists(self.input_json_file):
            raise FileNotFoundError(f"Input file not found: {self.input_json_file}")
        
        self.logger.info(f"Loading input data from {self.input_json_file}")
        with open(self.input_json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        data_list = []
        source_list = []
        if isinstance(json_data, list):
            source_list = json_data
        elif isinstance(json_data, dict):
            source_list = json_data.get('video_pairs', json_data.get('data', []))
        
        for idx, item in enumerate(source_list):
            if 'video1_path' in item and 'video2_path' in item:
                data_list.append({'index': idx, 'video1_path': item['video1_path'], 'video2_path': item['video2_path']})
        
        if self.max_pairs and len(data_list) > self.max_pairs:
            data_list = data_list[:self.max_pairs]
            self.logger.info(f"Limited processing to the first {self.max_pairs} pairs.")
        
        self.logger.info(f"Found {len(data_list)} video pairs to process.")
        self.detail_logger.info(f"===== Input Data Preview (first 5) =====")
        for item in data_list[:5]:
            self.detail_logger.info(f"  Index {item['index']}: {item['video1_path']} vs {item['video2_path']}")
        
        return data_list

    def run(self, model, tokenizer, generation_config):
        """Runs the main batch processing loop."""
        self.logger.info(f"\n{'='*80}\nStarting batch processing...\n{'='*80}")
        self.start_time = time.time()
        
        try:
            data_list = self.load_input_data()
            if not data_list:
                self.logger.error("No video pairs found to process.")
                return

            total_pairs = len(data_list)
            
            def log_progress():
                elapsed = time.time() - self.start_time
                processed = self.successful_count + self.failed_count
                total_attempted = processed + self.skipped_count
                if processed > 0:
                    avg_time = elapsed / processed
                    remaining = total_pairs - total_attempted
                    eta_seconds = remaining * avg_time
                    self.logger.info(f"\n--- Progress Report ---\n"
                                     f"Processed: {total_attempted}/{total_pairs} | "
                                     f"Successful: {self.successful_count}, Failed: {self.failed_count}, Skipped: {self.skipped_count}\n"
                                     f"Avg. Time/Pair: {avg_time:.2f}s | "
                                     f"ETA: {eta_seconds/60:.1f} min\n"
                                     f"---------------------\n")
            
            with tqdm(total=total_pairs, desc="Processing video pairs") as pbar:
                if self.max_workers <= 1:
                    for i, entry in enumerate(data_list):
                        pbar.set_description(f"Processing entry {entry['index']}")
                        self.process_single_entry(entry, model, tokenizer, generation_config)
                        pbar.update(1)
                        if (i + 1) % 10 == 0: log_progress()
                else:
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        futures = {executor.submit(self.process_single_entry, entry, model, tokenizer, generation_config): entry for entry in data_list}
                        for i, future in enumerate(as_completed(futures)):
                            try:
                                future.result(timeout=self.timeout)
                            except Exception as e:
                                self.logger.error(f"A task in the thread pool failed: {e}")
                            pbar.update(1)
                            if (i + 1) % 10 == 0: log_progress()
        
        except KeyboardInterrupt:
            self.logger.warning("\n⚠️ User interrupted processing.")
            self.detail_logger.warning("Processing was interrupted by the user.")
        except Exception as e:
            self.logger.critical(f"\n❌ A critical error occurred: {e}")
            self.detail_logger.critical(f"Main program error: {traceback.format_exc()}")
            traceback.print_exc()
        finally:
            self.print_summary()
            torch.cuda.empty_cache()
            gc.collect()

    def print_summary(self):
        """Prints a final summary of the processing run."""
        elapsed = time.time() - self.start_time
        total_attempted = self.successful_count + self.failed_count
        
        self.logger.info(f"\n{'='*80}\n✅ Processing Finished!\n{'='*80}")
        self.logger.info(f"Total pairs in input: {total_attempted + self.skipped_count}")
        self.logger.info(f"Attempted in this run: {total_attempted}")
        self.logger.info(f"  - Successful: {self.successful_count}")
        self.logger.info(f"  - Failed: {self.failed_count}")
        self.logger.info(f"Skipped (already processed): {self.skipped_count}")
        
        self.logger.info(f"Total run time: {elapsed/60:.2f} minutes")
        if self.successful_count > 0:
            self.logger.info(f"Average time per successful pair: {elapsed/self.successful_count:.2f} seconds")
            
        self.logger.info(f"Output files:")
        self.logger.info(f"  - Results: {self.output_file}")
        self.logger.info(f"  - Errors: {self.error_file}")
        self.logger.info(f"  - Checkpoint: {self.checkpoint_file}")
        self.logger.info(f"{'='*80}\n")
    
    def _initialize_output_file(self):
        """Initializes the output file as an empty JSON list if it doesn't exist or is invalid."""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    if isinstance(json.load(f), list):
                        return
            except (json.JSONDecodeError, IOError):
                pass
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
    
    def _append_to_file(self, file_path, data_to_append):
        """Appends a JSON object to a file containing a JSON list in a thread-safe manner."""
        with file_lock:
            try:
                with open(file_path, 'r+', encoding='utf-8') as f:
                    data = json.load(f)
                    if not isinstance(data, list): data = []
                    data.append(data_to_append)
                    f.seek(0)
                    f.truncate()
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except (FileNotFoundError, json.JSONDecodeError):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump([data_to_append], f, ensure_ascii=False, indent=2)
            except Exception as e:
                self.logger.error(f"Failed to write to file {file_path}: {e}")

    def _load_checkpoint(self):
        """Loads checkpoint data if resuming is enabled."""
        if not self.resume_from_checkpoint or not os.path.exists(self.checkpoint_file):
            return {}
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                count = len(checkpoint.get('successful_indices', []))
                self.logger.info(f"Resuming from checkpoint, found {count} successfully processed entries.")
                return checkpoint
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_checkpoint(self, index, success=True):
        """Saves the current processing state to a checkpoint file."""
        if success:
            self.processed_indices.add(index)
            if 'successful_indices' not in self.checkpoint_data:
                self.checkpoint_data['successful_indices'] = []
            if index not in self.checkpoint_data['successful_indices']:
                self.checkpoint_data['successful_indices'].append(index)
        
        self.checkpoint_data['stats'] = {
            'successful': self.successful_count, 'failed': self.failed_count, 'skipped': self.skipped_count,
            'last_update': datetime.now().isoformat()
        }
        self._append_to_file(self.checkpoint_file, self.checkpoint_data)
        # For simplicity, we just overwrite the checkpoint file.
        with file_lock:
             with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(self.checkpoint_data, f, ensure_ascii=False, indent=2)

    def _load_system_prompt(self):
        """Loads the system prompt from the specified file."""
        if not os.path.exists(self.prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_file}")
        with open(self.prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
            self.logger.info(f"Loaded prompt from: {self.prompt_file}")
            return prompt

def setup_logging(log_dir):
    """Configures and returns the main and detail loggers."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"processing_{timestamp}.log")
    detail_log_file = os.path.join(log_dir, f"details_{timestamp}.log")

    # Main logger
    logger = logging.getLogger("main_logger")
    logger.setLevel(logging.INFO)
    logger.handlers = [
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    # Detail logger
    detail_logger = logging.getLogger("detail_logger")
    detail_logger.setLevel(logging.DEBUG)
    detail_handler = logging.FileHandler(detail_log_file, encoding='utf-8')
    detail_handler.setFormatter(formatter)
    detail_logger.addHandler(detail_handler)
    
    logger.info(f"Logging initialized. Main log: {log_file}, Detail log: {detail_log_file}")
    return logger, detail_logger

def main(args):
    """Main execution function."""
    logger, detail_logger = setup_logging(args.log_dir)

    try:
        logger.info(f"Loading tokenizer from {args.model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

        logger.info(f"Loading model from {args.model_path}...")
        model = AutoModel.from_pretrained(
            args.model_path, trust_remote_code=True,
            attn_implementation="flash_attention_2"  # Use flash attention if available
        ).to(DEVICE).to(DTYPE)

        generation_config = dict(
            do_sample=False, temperature=0.0, max_new_tokens=4096,
            top_p=None, num_beams=1,
        )
        logger.info(f"Model loaded successfully! Device: {DEVICE}, Dtype: {DTYPE}")

        config = {
            'input_json_file': args.input_json,
            'output_file': args.output_file,
            'error_file': args.error_file,
            'checkpoint_file': args.checkpoint_file,
            'prompt_file': args.prompt_file,
            'max_workers': args.max_workers,
            'max_pairs': args.max_pairs,
            'model_delay': 1,
            'timeout': 600,
            'resume_from_checkpoint': not args.no_resume,
            'max_retries': 3,
            'num_segments': DEFAULT_NUM_SEGMENTS,
            'input_size': 448,
            'max_num_patches': 1,
        }
    
        processor = VideoProcessor(config, logger, detail_logger)
        processor.run(model, tokenizer, generation_config)
    
    except Exception as e:
        logger.critical(f"An error occurred during setup or execution: {e}")
        logger.critical(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Comparison and Analysis Tool using InternVideo2.5")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the local InternVideo model directory.")
    parser.add_argument('--input_json', type=str, required=True, help="Path to the input JSON file with video pairs.")
    parser.add_argument('--prompt_file', type=str, default='prompt.txt', help="Path to the file containing the system prompt.")
    parser.add_argument('--output_file', type=str, default='output/results.json', help="Path to save successful results.")
    parser.add_argument('--error_file', type=str, default='output/errors.json', help="Path to save error logs.")
    parser.add_argument('--checkpoint_file', type=str, default='output/checkpoint.json', help="Path to save processing checkpoint.")
    parser.add_argument('--log_dir', type=str, default='logs', help="Directory to store log files.")
    parser.add_argument('--max_workers', type=int, default=1, help="Number of worker threads (recommend 1 to avoid GPU OOM issues).")
    parser.add_argument('--max_pairs', type=int, default=None, help="Maximum number of pairs to process (None for all).")
    parser.add_argument('--no_resume', action='store_true', help="Disable resuming from a checkpoint.")
    
    cli_args = parser.parse_args()
    main(cli_args)
