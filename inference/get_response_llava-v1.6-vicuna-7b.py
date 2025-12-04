import argparse
import os
import json
import sys
import time
import logging
from datetime import datetime
import traceback
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from threading import Lock
import numpy as np
import re

# LLaVA-specific imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images
from llava.utils import disable_torch_init

# --- Constants ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Utility Functions for Image/Video Processing ---

def build_transform(input_size):
    """Builds the image transformation pipeline."""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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

def dynamic_preprocess(image, min_patches=1, max_patches=12, image_size=448, use_thumbnail=False):
    """Dynamically preprocesses an image by splitting it into patches."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_patches, max_patches + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_patches and i * j >= min_patches
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

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
    """Calculates indices for frame sampling from a video."""
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
    return np.clip(frame_indices, start_idx, end_idx - 1).astype(int)

def process_video_to_image_list(video_path, input_size=448, max_patches=1, num_segments=32):
    """Processes a video file and returns a list of PIL Image objects."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    frame_indices = get_frame_indices(total_frames, fps, num_segments=num_segments)
    
    image_list = []
    for frame_index in frame_indices:
        frame = vr[frame_index].asnumpy()
        pil_image = Image.fromarray(frame).convert('RGB')
        processed_images = dynamic_preprocess(
            pil_image, image_size=input_size, use_thumbnail=True, max_patches=max_patches
        )
        image_list.extend(processed_images)
    return image_list


class LLaVAProcessor:
    """Main class for video comparison using LLaVA."""
    
    _output_file_lock = Lock()
    
    def __init__(self, config):
        """Initializes the processor."""
        disable_torch_init()
        
        self.model_path = config['model_path']
        self.model_name = get_model_name_from_path(self.model_path)
        self.input_json_file = config['input_json_file']
        self.prompt_file = config['prompt_file']
        self.output_folder = config['output_folder']
        self.log_folder = config['log_folder']
        self.use_thinking_prompt = config['use_thinking_prompt']
        self.max_frames_per_video = config['max_frames_per_video']
        
        # Set output file path based on thinking mode
        os.makedirs(self.output_folder, exist_ok=True)
        mode_suffix = "thinking" if self.use_thinking_prompt else "no_thinking"
        self.output_file = os.path.join(self.output_folder, f"{self.model_name}_{mode_suffix}_results.json")
        
        # Setup logging
        self._setup_logging()
        
        # Validate input files and paths
        if not os.path.exists(self.input_json_file):
            raise FileNotFoundError(f"Input JSON file not found: {self.input_json_file}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        
        # Check for GPU
        if not torch.cuda.is_available():
            self.logger.error("❌ No GPU detected. This script requires GPU support.")
            raise RuntimeError("A GPU is required to run this script.")
        
        gpu_count = torch.cuda.device_count()
        self.logger.info(f"✅ Detected {gpu_count} GPU(s).")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            self.logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        
        # Initialize LLaVA model
        self.logger.info(f"Loading LLaVA model from: {self.model_path}")
        self.logger.info(f"Thinking mode enabled: {self.use_thinking_prompt}")
        
        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=self.model_path, model_base=None, model_name=self.model_name
            )
            if next(self.model.parameters()).device.type == 'cuda':
                self.logger.info("✅ LLaVA model loaded successfully onto GPU.")
            else:
                self.logger.warning("⚠️ Model is running on CPU, which may be slow.")
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise
        
        # Video and generation settings
        self.input_size = 448
        self.max_patches = 1
        
        self.temperature = 0.6 if self.use_thinking_prompt else 0.1
        self.max_new_tokens = 8192 if self.use_thinking_prompt else 2048
        
        self.logger.info("Configuration:")
        self.logger.info(f"  - Model Name: {self.model_name}")
        self.logger.info(f"  - Model Path: {self.model_path}")
        self.logger.info(f"  - Max Frames per Video: {self.max_frames_per_video}")
        self.logger.info(f"  - Input File: {self.input_json_file}")
        self.logger.info(f"  - Output File: {self.output_file}")
        self.logger.info(f"  - Prompt File: {self.prompt_file}")
        self.logger.info(f"  - Thinking Mode: {'Enabled' if self.use_thinking_prompt else 'Disabled'}")
        self.logger.info(f"  - Temperature: {self.temperature}")
        self.logger.info(f"  - Max New Tokens: {self.max_new_tokens}")
        
        # Statistics
        self.successful_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.start_time = None
        
        self.processed_indices = self._load_processed_indices()
        self.system_prompt = self._load_prompt_from_file()
        
        if self.use_thinking_prompt:
            THINKING_PROMPT_WRAPPER = """You are an AI assistant that rigorously follows this response protocol:
1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.
2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.
Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.\n\n"""
            self.system_prompt = THINKING_PROMPT_WRAPPER + self.system_prompt
        
        self._initialize_output_file()
    
    def _setup_logging(self):
        """Configures logging for the application."""
        log_dir = os.path.join(self.log_folder, self.model_name)
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"processing_{timestamp}.log")
        error_log_file = os.path.join(log_dir, f"errors_{timestamp}.log")
        
        self.logger = logging.getLogger(f"LLaVAProcessor_{self.model_name}")
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers(): self.logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.error_logger = logging.getLogger(f"error_logger_{self.model_name}")
        self.error_logger.setLevel(logging.ERROR)
        if self.error_logger.hasHandlers(): self.error_logger.handlers.clear()
        
        error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
        error_handler.setFormatter(logging.Formatter('%(asctime)s - [ERROR] - %(message)s'))
        self.error_logger.addHandler(error_handler)
    
    def _load_processed_indices(self):
        """Loads indices of already processed items from the output file."""
        processed = set()
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    for item in json.load(f):
                        if 'index' in item:
                            processed.add(item['index'])
                self.logger.info(f"Loaded {len(processed)} processed record indices from output file.")
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"Could not load processed records: {e}")
        return processed
    
    def _initialize_output_file(self):
        """Initializes the output file, preserving existing valid data."""
        with self._output_file_lock:
            if not os.path.exists(self.output_file):
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
                self.logger.info("Created new output file.")
            else:
                try:
                    with open(self.output_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if not isinstance(data, list):
                        raise ValueError("Output file is not a JSON list.")
                    self.logger.info(f"Output file exists with {len(data)} records. Appending new results.")
                except (json.JSONDecodeError, ValueError, IOError) as e:
                    backup_file = f"{self.output_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    os.rename(self.output_file, backup_file)
                    self.logger.warning(f"Invalid output file found, backed up to: {backup_file}. Re-initializing.")
                    with open(self.output_file, 'w', encoding='utf-8') as f:
                        json.dump([], f)
    
    def _append_result_to_file(self, result):
        """Appends a single result to the JSON output file in a thread-safe manner."""
        with self._output_file_lock:
            try:
                with open(self.output_file, 'r+', encoding='utf-8') as f:
                    data = json.load(f)
                    
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
    
    def _load_prompt_from_file(self):
        """Loads the system prompt from the specified text file."""
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
    
    def _log_error(self, error_info):
        """Logs detailed error information to the separate error log."""
        self.error_logger.error(json.dumps(error_info, ensure_ascii=False, indent=2))
    
    def load_input_data(self):
        """Loads and parses the input JSON file."""
        self.logger.info(f"Loading input file: {self.input_json_file}")
        
        try:
            with open(self.input_json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            data_list = []
            source_list = []
            if isinstance(json_data, list):
                source_list = json_data
            elif isinstance(json_data, dict):
                source_list = json_data.get('video_pairs', json_data.get('data', [json_data]))

            for idx, item in enumerate(source_list):
                if 'video1_path' in item and 'video2_path' in item:
                    data_list.append({
                        'index': idx,
                        'video1_path': item['video1_path'],
                        'video2_path': item['video2_path']
                    })
                else:
                    self.logger.warning(f"Item at index {idx} is missing required 'video1_path' or 'video2_path' keys.")
            
            self.logger.info(f"✅ Loaded {len(data_list)} data entries.")
            return data_list
        except Exception as e:
            self.logger.error(f"Failed to load or parse input file: {e}")
            raise
    
    def generate_response(self, image_files, question: str):
        """Generates a response from the LLaVA model for the given images and question."""
        image_token_seq = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        
        # Replace placeholder or prepend image token sequence
        if IMAGE_PLACEHOLDER in question:
            question = re.sub(IMAGE_PLACEHOLDER, image_token_seq if self.model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN, question)
        else:
            prefix = image_token_seq if self.model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN
            question = prefix + "\n" + question

        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images_tensor = process_images(
            image_files, self.image_processor, self.model.config
        ).to(self.model.device, dtype=torch.float16)

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=[img.size for img in image_files],
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                num_beams=1,
            )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    def process_video_pair(self, entry):
        """Processes a single pair of videos."""
        video1_path = entry['video1_path']
        video2_path = entry['video2_path']
        
        if not os.path.exists(video1_path):
            raise FileNotFoundError(f"Video file not found: {video1_path}")
        if not os.path.exists(video2_path):
            raise FileNotFoundError(f"Video file not found: {video2_path}")
        
        frame_images1 = process_video_to_image_list(
            video1_path, self.input_size, self.max_patches, self.max_frames_per_video
        )
        frame_images2 = process_video_to_image_list(
            video2_path, self.input_size, self.max_patches, self.max_frames_per_video
        )
        
        all_frames = frame_images1 + frame_images2
        
        # The prompt guides the model on how to reference the frames.
        full_prompt = f"Source video is represented by the first {len(frame_images1)} frames. Destination video is represented by the next {len(frame_images2)} frames.\n\n{self.system_prompt}"
        
        try:
            response = self.generate_response(all_frames, full_prompt)
            if self.use_thinking_prompt and '</think>' in response:
                response = response.split('</think>', 1)[-1].strip()
            return response
        except Exception as e:
            self.logger.error(f"Error during model inference: {e}")
            raise

    def process_all(self):
        """Main loop to process all video pairs."""
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
        
        self.logger.info(f"Total entries: {total_count}, Processed: {self.skipped_count}, Pending: {pending_count}")
        self.logger.info("="*60 + "\nStarting processing with LLaVA model...\n" + "="*60)
        
        with tqdm(total=pending_count, desc="Processing Progress") as pbar:
            for entry in pending_data:
                self.logger.info(f"\nProcessing video pair index {entry['index']}")
                
                max_retries = 3
                for attempt in range(1, max_retries + 1):
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
                        self.successful_count += 1
                        self.logger.info(f"[Index {entry['index']}] ✅ Successfully processed and saved.")
                        break # Success, exit retry loop
                    except Exception as e:
                        self.logger.error(f"Error on attempt {attempt}/{max_retries}: {e}")
                        self.error_logger.error(f"Error details for index {entry['index']}:\n{traceback.format_exc()}")
                        if attempt == max_retries:
                            self.failed_count += 1
                            self.logger.error(f"[Index {entry['index']}] ❌ FAILED after all retries.")
                            self._log_error({
                                "index": entry['index'], "video1_path": entry['video1_path'], "video2_path": entry['video2_path'],
                                "error": str(e), "timestamp": datetime.now().isoformat()
                            })
                        else:
                            self.logger.warning("Retrying...")
                            time.sleep(2)
                pbar.update(1)
        
        self.logger.info(f"✅ All results saved to: {self.output_file}")
        self.print_summary()
    
    def print_summary(self):
        """Prints a final summary of the processing run."""
        elapsed_time = time.time() - self.start_time
        total_processed = self.successful_count + self.failed_count
        
        self.logger.info("\n" + "="*60 + "\nProcessing Complete - Summary\n" + "="*60)
        self.logger.info(f"Total run time: {elapsed_time/60:.2f} minutes")
        self.logger.info(f"Total items attempted in this run: {total_processed}")
        self.logger.info(f"  - Successful: {self.successful_count}")
        self.logger.info(f"  - Failed: {self.failed_count}")
        self.logger.info(f"Items skipped (already processed): {self.skipped_count}")
        
        if total_processed > 0:
            success_rate = (self.successful_count / total_processed) * 100
            avg_time = elapsed_time / total_processed
            self.logger.info(f"Success rate: {success_rate:.2f}%")
            self.logger.info(f"Average processing time: {avg_time:.2f} seconds/item")
        
        self.logger.info(f"\nOutput file: {self.output_file}")
        self.logger.info(f"Log directory: {os.path.join(self.log_folder, self.model_name)}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='LLaVA-based Video Comparison and Analysis Tool.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model_path', type=str, required=True, help="Path to the local LLaVA model directory.")
    parser.add_argument('--input_json', type=str, default='videos.json', help='Path to the input JSON file with video pairs.')
    parser.add_argument('--prompt_file', type=str, default='prompt.txt', help='Path to the file containing the system prompt.')
    parser.add_argument('--output_folder', type=str, default='response', help='Directory to save output JSON results.')
    parser.add_argument('--log_folder', type=str, default='logs', help='Directory to save log files.')
    parser.add_argument('--max_frames', type=int, default=16, help='Maximum number of frames to sample from each video.')
    parser.add_argument('--thinking', action='store_true', help='Enable the "thinking" prompt for step-by-step reasoning.')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LLaVA Video Comparison and Analysis Tool")
    print("Incremental write mode: Enabled")
    print("File lock protection: Enabled")
    print("="*60)
    
    config = {
        "model_path": args.model_path,
        "input_json_file": args.input_json,
        "prompt_file": args.prompt_file,
        "output_folder": args.output_folder,
        "log_folder": args.log_folder,
        "max_frames_per_video": args.max_frames,
        "use_thinking_prompt": args.thinking,
    }
    
    model_name_for_display = get_model_name_from_path(config['model_path'])
    mode_suffix = "thinking" if config['use_thinking_prompt'] else "no_thinking"
    
    print("Configuration:")
    print(f"  - Model Name: {model_name_for_display}")
    print(f"  - Input File: {config['input_json_file']}")
    print(f"  - Output File: {os.path.join(config['output_folder'], f'{model_name_for_display}_{mode_suffix}_results.json')}")
    print(f"  - Log Folder: {config['log_folder']}")
    print(f"  - Max Frames per Video: {config['max_frames_per_video']}")
    print(f"  - Thinking Mode: {'Enabled' if config['use_thinking_prompt'] else 'Disabled'}")
    print("="*60)
    
    try:
        processor = LLaVAProcessor(config)
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
