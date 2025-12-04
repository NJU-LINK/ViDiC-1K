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

# --- Image Processing Constants ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VideoProcessor:
    """Main class for video processing using a local LLM."""

    # Class-level lock for safe concurrent file writing
    _output_file_lock = Lock()

    def __init__(self, config):
        """Initialize the processor with the given configuration."""
        self.model_path = config.get('model_path')
        self.model_name = Path(self.model_path).name  # Derive model name from path for organization
        self.input_json_file = config.get('input_json_file', 'input_videos.json')
        self.output_folder = config.get('output_folder', 'response')
        self.log_folder = config.get('log_folder', 'log')
        self.batch_size = config.get('batch_size', 1)
        self.prompt_file = config.get('prompt_file', 'prompt.txt')
        self.gpu_memory_utilization = config.get('gpu_memory_utilization', 0.85)

        # Video processing configuration
        self.input_size = 448
        self.max_num_patches = 1
        self.max_frames = 32

        # Set up output file path
        os.makedirs(self.output_folder, exist_ok=True)
        self.output_file = os.path.join(self.output_folder, f"{self.model_name}_results.json")

        # Set up logging
        self._setup_logging()

        # Validate existence of input files and model path
        if not os.path.exists(self.input_json_file):
            raise FileNotFoundError(f"Input JSON file not found: {self.input_json_file}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        # Detect GPUs
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            self.logger.error("❌ No GPU detected. This script requires GPU support.")
            raise RuntimeError("A GPU is required to run this script.")
        
        self.tensor_parallel_size = gpu_count
        self.logger.info(f"✅ Detected {gpu_count} GPU(s).")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            self.logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")

        # Initialize vLLM model
        self.logger.info(f"Loading LLM from: {self.model_path}")
        self.logger.info(f"GPU memory utilization set to: {self.gpu_memory_utilization}")
        try:
            self.model = LLM(
                model=self.model_path,
                trust_remote_code=True,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                enforce_eager=True,
                max_model_len=60960,
                max_num_seqs=1,
                limit_mm_per_prompt={"image": 32, "video": 2}
            )
            self.logger.info("✅ LLM loaded successfully.")
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise

        # Set sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            max_tokens=4096,
            stop_token_ids=[],
        )

        # Load processor
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.logger.info("✅ Processor loaded successfully.")
        except Exception as e:
            self.logger.error(f"❌ Failed to load processor: {e}")
            raise

        self.logger.info("Configuration details:")
        self.logger.info(f"  - Model Name: {self.model_name}")
        self.logger.info(f"  - Model Path: {self.model_path}")
        self.logger.info(f"  - Batch Size: {self.batch_size}")
        self.logger.info(f"  - Input File: {self.input_json_file}")
        self.logger.info(f"  - Output File: {self.output_file}")
        self.logger.info(f"  - Prompt File: {self.prompt_file}")
        self.logger.info(f"  - GPU Memory Utilization: {self.gpu_memory_utilization}")

        # Statistics
        self.successful_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.start_time = None

        # Load already processed records from the output file
        self.processed_indices = self._load_processed_indices()
        self.prompt_template = self._load_prompt_from_file()
        self._initialize_output_file()

    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = os.path.join(self.log_folder, self.model_name)
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"processing_{timestamp}.log")
        error_log_file = os.path.join(log_dir, f"errors_{timestamp}.log")

        self.logger = logging.getLogger(f"VideoProcessor_{self.model_name}")
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.error_logger = logging.getLogger(f"error_logger_{self.model_name}")
        self.error_logger.setLevel(logging.ERROR)
        if self.error_logger.hasHandlers():
            self.error_logger.handlers.clear()

        error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
        error_handler.setFormatter(logging.Formatter('%(asctime)s - [ERROR] - %(message)s'))
        self.error_logger.addHandler(error_handler)

    def _load_processed_indices(self):
        """Load indices of already processed items from the output file."""
        processed = set()
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if 'index' in item:
                                processed.add(item['index'])
                        self.logger.info(f"Loaded {len(processed)} processed records from output file.")
            except Exception as e:
                self.logger.warning(f"Could not load processed records: {e}")
        return processed

    def _initialize_output_file(self):
        """Initialize the output file, preserving existing data if valid."""
        with self._output_file_lock:
            if os.path.exists(self.output_file):
                try:
                    with open(self.output_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        self.logger.warning("Output file has incorrect format, re-initializing.")
                        with open(self.output_file, 'w', encoding='utf-8') as f:
                            json.dump([], f)
                    else:
                        self.logger.info(f"Output file exists with {len(existing_data)} records. Appending new results.")
                except (json.JSONDecodeError, IOError) as e:
                    backup_file = f"{self.output_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    os.rename(self.output_file, backup_file)
                    self.logger.warning(f"Failed to read output file, backed up to: {backup_file}")
                    with open(self.output_file, 'w', encoding='utf-8') as f:
                        json.dump([], f)
            else:
                self.logger.info("Creating new output file.")
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)

    def _append_result_to_file(self, result):
        """Append a single result to the JSON output file with a file lock."""
        with self._output_file_lock:
            try:
                with open(self.output_file, 'r+', encoding='utf-8') as f:
                    # Load data, append new result, and write back
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
                    
                    f.seek(0)
                    f.truncate()
                    json.dump(data, f, ensure_ascii=False, indent=2)
                self.logger.debug(f"Successfully appended result. Total records: {len(data)}")
            except Exception as e:
                self.logger.error(f"Failed to append to main output file: {e}")
                # Fallback to a backup incremental file
                backup_file = f"{self.output_file}.incremental_error_log"
                try:
                    with open(backup_file, 'a', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False)
                        f.write('\n')
                    self.logger.warning(f"Result saved to fallback file: {backup_file}")
                except Exception as e2:
                    self.logger.error(f"CRITICAL: Failed to write to fallback file: {e2}")

    def _load_prompt_from_file(self):
        """Load the system prompt from a text file."""
        if not os.path.exists(self.prompt_file):
            error_msg = f"❌ ERROR: Prompt file not found: {self.prompt_file}"
            self.logger.error(error_msg)
            sys.exit(error_msg)
        
        try:
            with open(self.prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            if not prompt:
                error_msg = f"❌ ERROR: Prompt file is empty: {self.prompt_file}"
                self.logger.error(error_msg)
                sys.exit(error_msg)

            self.logger.info(f"✅ Successfully loaded prompt file: {self.prompt_file}")
            self.logger.info(f"Prompt length: {len(prompt)} characters.")
            return prompt
        except Exception as e:
            error_msg = f"❌ ERROR: Failed to read prompt file: {e}"
            self.logger.error(error_msg)
            sys.exit(error_msg)

    def _log_error_details(self, error_info):
        """Log detailed error information to a separate error log file."""
        self.error_logger.error(json.dumps(error_info, ensure_ascii=False, indent=2))

    def _validate_video_files(self, video1_path, video2_path):
        """Validate video file existence and log their sizes."""
        if not os.path.exists(video1_path):
            raise FileNotFoundError(f"Video file not found: {video1_path}")
        if not os.path.exists(video2_path):
            raise FileNotFoundError(f"Video file not found: {video2_path}")
        
        try:
            size1_mb = os.path.getsize(video1_path) / (1024 * 1024)
            size2_mb = os.path.getsize(video2_path) / (1024 * 1024)
            self.logger.info(f"Video1 size: {size1_mb:.2f}MB, Video2 size: {size2_mb:.2f}MB")
            
            max_size_mb = 500
            if size1_mb > max_size_mb:
                self.logger.warning(f"⚠️ Video1 is large ({size1_mb:.2f}MB), which may slow down processing.")
            if size2_mb > max_size_mb:
                self.logger.warning(f"⚠️ Video2 is large ({size2_mb:.2f}MB), which may slow down processing.")
        except Exception as e:
            self.logger.warning(f"Could not get file size information: {e}")

    # --- Video and Image Processing Functions ---
    def build_transform(self, input_size):
        """Builds the image transformation pipeline."""
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """Finds the best grid-like aspect ratio for image patching."""
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

    def dynamic_preprocess(self, image, min_patches=1, max_patches=12, image_size=448, use_thumbnail=False):
        """Dynamically preprocesses an image by splitting it into patches."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        target_ratios = set(
            (i, j) for n in range(min_patches, max_patches + 1)
            for i in range(1, n + 1) for j in range(1, n + 1)
            if i * j <= max_patches and i * j >= min_patches
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        num_blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(num_blocks):
            box = (
                (i % target_aspect_ratio[0]) * image_size,
                (i // target_aspect_ratio[0]) * image_size,
                ((i % target_aspect_ratio[0]) + 1) * image_size,
                ((i // target_aspect_ratio[0]) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        assert len(processed_images) == num_blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
            
        return processed_images

    def get_frame_indices(self, total_frames, fps, num_segments=16, bound=None):
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

    def load_images_from_video(self, video_path, num_segments=16, max_frames=32):
        """Loads and processes frames from a video file."""
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        
        frame_indices = self.get_frame_indices(total_frames, fps, num_segments)
        if len(frame_indices) > max_frames:
            frame_indices = frame_indices[:max_frames]

        pil_images = []
        for frame_index in frame_indices:
            img_array = vr[frame_index].asnumpy()
            img = Image.fromarray(img_array).convert('RGB')
            
            processed_patches = self.dynamic_preprocess(
                img, image_size=self.input_size, max_patches=self.max_num_patches, use_thumbnail=False
            )
            pil_images.extend(processed_patches)
            
        return pil_images

    def process_video_pairs_batch(self, entries):
        """Processes a batch of video pairs."""
        llm_inputs = []

        for entry in entries:
            video1_path = entry['video1_path']
            video2_path = entry['video2_path']
            
            self._validate_video_files(video1_path, video2_path)
            
            # Load frames for both videos
            pil_images1 = self.load_images_from_video(video1_path, num_segments=16, max_frames=16)
            pil_images2 = self.load_images_from_video(video2_path, num_segments=16, max_frames=16)
            
            all_images = pil_images1 + pil_images2
            
            # Construct multimodal message content
            content = []
            content.append({"type": "text", "text": "Source video:"})
            for _ in pil_images1:
                content.append({"type": "image"})
            
            content.append({"type": "text", "text": "\nDestination video:"})
            for _ in pil_images2:
                content.append({"type": "image"})
            
            content.append({"type": "text", "text": f"\n{self.prompt_template}"})
            
            messages = [{"role": "user", "content": content}]
            
            text_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            multi_modal_data = {"image": all_images}
            
            llm_inputs.append({
                "prompt": text_prompt,
                "multi_modal_data": multi_modal_data,
            })

        # Generate responses in a batch
        outputs = self.model.generate(llm_inputs, self.sampling_params)
        generated_texts = [output.outputs[0].text.strip() for output in outputs]
        
        return generated_texts

    def load_input_data(self):
        """Loads input data from the specified JSON file."""
        self.logger.info(f"Loading input file: {self.input_json_file}")
        
        try:
            with open(self.input_json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            data_list = []
            
            # Handle list format
            if isinstance(json_data, list):
                for idx, item in enumerate(json_data):
                    if 'video1_path' in item and 'video2_path' in item:
                        data_list.append({
                            'index': idx,
                            'video1_path': item['video1_path'],
                            'video2_path': item['video2_path']
                        })
                    else:
                        self.logger.warning(f"Item at index {idx} is missing required video path keys.")
            # Handle dictionary format (e.g., with a 'data' or 'video_pairs' key)
            elif isinstance(json_data, dict):
                video_pairs = json_data.get('video_pairs', json_data.get('data'))
                if isinstance(video_pairs, list):
                    for idx, item in enumerate(video_pairs):
                        if 'video1_path' in item and 'video2_path' in item:
                            data_list.append({
                                'index': idx,
                                'video1_path': item['video1_path'],
                                'video2_path': item['video2_path']
                            })
            
            self.logger.info(f"✅ Loaded {len(data_list)} data entries.")
            return data_list
            
        except Exception as e:
            self.logger.error(f"Failed to load or parse input file: {e}")
            raise

    def process_all(self):
        """Main processing loop for all data entries."""
        self.start_time = time.time()
        
        data_list = self.load_input_data()
        if not data_list:
            self.logger.info("No data to process.")
            return
            
        pending_data = [entry for entry in data_list if entry['index'] not in self.processed_indices]
        
        if not pending_data:
            self.logger.info("✅ All data has already been processed.")
            return
            
        total_count = len(data_list)
        pending_count = len(pending_data)
        self.skipped_count = total_count - pending_count

        self.logger.info(f"Total entries: {total_count}")
        self.logger.info(f"Already processed (skipped): {self.skipped_count}")
        self.logger.info(f"Pending: {pending_count}")
        self.logger.info("="*60)
        self.logger.info(f"Starting batch processing with batch size: {self.batch_size}")
        self.logger.info("Incremental writing mode: Enabled")
        self.logger.info("="*60)

        with tqdm(total=pending_count, desc="Processing Progress") as pbar:
            for i in range(0, pending_count, self.batch_size):
                batch_entries = pending_data[i:i + self.batch_size]
                self.logger.info(f"\nProcessing batch {i//self.batch_size + 1}: {len(batch_entries)} video pairs")
                
                max_retries = 3
                attempt = 0
                batch_processed = False
                
                while attempt < max_retries and not batch_processed:
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
                            self.successful_count += 1
                            self.logger.info(f"[Entry {entry['index']}] ✅ Processed and saved successfully.")
                        
                        batch_processed = True
                        pbar.update(len(batch_entries))

                    except Exception as e:
                        self.logger.error(f"Error during batch processing: {e}")
                        self.error_logger.error(f"Batch processing error details:\n{traceback.format_exc()}")
                        attempt += 1
                        
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache() # Clear cache before retry
                        
                        if attempt < max_retries:
                            self.logger.warning(f"Retrying batch... (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(2)
                        else:
                            self.logger.warning("Batch failed after retries. Attempting to process items individually...")
                            for entry in batch_entries:
                                try:
                                    response = self.process_video_pairs_batch([entry])
                                    result = {
                                        "index": entry['index'],
                                        "video1_path": entry['video1_path'],
                                        "video2_path": entry['video2_path'],
                                        "response": response[0]
                                    }
                                    self._append_result_to_file(result)
                                    self.processed_indices.add(entry['index'])
                                    self.successful_count += 1
                                    self.logger.info(f"[Entry {entry['index']}] ✅ Processed individually after batch failure.")
                                except Exception as e2:
                                    self.failed_count += 1
                                    self.logger.error(f"[Entry {entry['index']}] ❌ FAILED to process: {e2}")
                                    self._log_error_details({
                                        "index": entry['index'],
                                        "video1_path": entry['video1_path'],
                                        "video2_path": entry['video2_path'],
                                        "error": str(e2),
                                        "timestamp": datetime.now().isoformat()
                                    })
                                finally:
                                    pbar.update(1)
                            # Mark batch as handled to exit the retry loop
                            batch_processed = True 

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.logger.info(f"✅ All results have been saved to: {self.output_file}")
        self.print_summary()

    def print_summary(self):
        """Prints a summary of the processing results."""
        elapsed_time = time.time() - self.start_time
        total_processed = self.successful_count + self.failed_count

        self.logger.info("\n" + "="*60)
        self.logger.info("Processing Complete - Summary")
        self.logger.info("="*60)
        self.logger.info(f"Total time taken: {elapsed_time / 60:.2f} minutes")
        self.logger.info(f"Total items attempted in this run: {total_processed}")
        self.logger.info(f"  - Successful: {self.successful_count}")
        self.logger.info(f"  - Failed: {self.failed_count}")
        self.logger.info(f"Items skipped (already processed): {self.skipped_count}")

        if total_processed > 0:
            success_rate = (self.successful_count / total_processed * 100) if total_processed > 0 else 0
            avg_time_per_item = elapsed_time / total_processed if total_processed > 0 else 0
            self.logger.info(f"Success rate: {success_rate:.2f}%")
            self.logger.info(f"Average processing time: {avg_time_per_item:.2f} seconds/item")
        
        self.logger.info(f"\nOutput file: {self.output_file}")
        self.logger.info(f"Log directory: {os.path.join(self.log_folder, self.model_name)}")


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Video Comparison and Analysis Script using a Local LLM.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model_path', type=str, required=True, help='Full path to the local model directory.')
    parser.add_argument('--input_json', type=str, default='videos.json', help='Path to the input JSON file containing video pairs.')
    parser.add_argument('--output_folder', type=str, default='responses', help='Directory to save the output JSON results.')
    parser.add_argument('--log_folder', type=str, default='logs', help='Directory to save log files.')
    parser.add_argument('--prompt_file', type=str, default='prompt.txt', help='Path to the file containing the system prompt.')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of video pairs to process in a single batch.')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='Proportion of GPU memory to be used by the model (0.0-1.0).')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    args = parse_args()
    model_name = Path(args.model_path).name

    print("="*60)
    print("Video Comparison and Analysis Script")
    print(f"Model: {model_name}")
    print("Incremental write mode: Enabled")
    print("File lock protection: Enabled")
    print("="*60)

    config = {
        "model_path": args.model_path,
        "input_json_file": args.input_json,
        "output_folder": args.output_folder,
        "log_folder": args.log_folder,
        "prompt_file": args.prompt_file,
        "batch_size": args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }

    print("Configuration:")
    print(f"  - Model Path: {config['model_path']}")
    print(f"  - Input File: {config['input_json_file']}")
    print(f"  - Output Folder: {config['output_folder']}")
    print(f"  - Log Folder: {config['log_folder']}")
    print(f"  - Batch Size: {config['batch_size']}")
    print(f"  - GPU Memory Usage: {config['gpu_memory_utilization']:.0%}")
    print("="*60)

    try:
        processor = VideoProcessor(config)
        processor.process_all()
        print("\n✅ Processing finished successfully!")
    except KeyboardInterrupt:
        print("\n⚠️ Processing was interrupted by the user.")
    except Exception as e:
        print(f"\n❌ A critical error occurred: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
