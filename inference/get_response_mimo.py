import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
from datetime import datetime
import traceback
import gc

# Import image and PyTorch related libraries
import cv2
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

sys.stdout.reconfigure(encoding='utf-8')

# Set environment variables to optimize GPU memory (using PyTorch recommended new name)
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# Global locks
file_lock = Lock()
model_lock = Lock()


class VideoProcessor:
    def __init__(self, config):
        self.config = config
        
        # File path configuration
        self.input_json_file = config.get('input_json_file', 'input_videos.json')
        output_dir = config.get('output_dir', '.')
        os.makedirs(output_dir, exist_ok=True)
        self.output_file = os.path.join(output_dir, config.get('output_file', 'video_analysis_results.json'))
        self.checkpoint_file = os.path.join(output_dir, config.get('checkpoint_file', 'processing_checkpoint.json'))
        
        # Performance and retry configuration
        self.max_workers = config.get('max_workers', 1)
        self.max_pairs = config.get('max_pairs', None)
        self.model_delay = config.get('model_delay', 2)
        self.timeout = config.get('timeout', 300)
        self.max_retries = config.get('max_retries', 3)
        self.resume_from_checkpoint = config.get('resume_from_checkpoint', True)
        
        # Model and inference configuration
        self.model_path = config.get('model_path')
        self.temperature = config.get('temperature', 0.3)
        self.top_p = config.get('top_p', 0.95)
        self.max_new_tokens = config.get('max_new_tokens', 16384)
        self.do_sample = config.get('do_sample', True)
      
        # Clean GPU memory
        torch.cuda.empty_cache()
        gc.collect()
      
        # Initialize MiMo model
        print(f"Loading MiMo model from path: {self.model_path}...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
      
        self.model.eval()
        print("✅ MiMo model loaded successfully")
      
        self.log_gpu_memory("After model loading")
      
        print(f"="*80)
        print(f"Configuration: Model=MiMo-VL-7B, MaxNewTokens={self.max_new_tokens}")
        print(f"="*80)
      
        self.successful = 0
        self.failed = 0
        self.skipped_processed = 0
        self.start_time = None
      
        self.checkpoint_data = self._load_checkpoint()
        self.processed_indices = set(self.checkpoint_data.get('successful_indices', []))
        self.system_prompt = self._load_system_prompt()
        self._initialize_output_file()

    def log_gpu_memory(self, stage=""):
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"[{stage}] GPU {i}: Allocated {allocated:.2f}GB, Reserved {reserved:.2f}GB")

    def inference(self, messages):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
      
        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
      
        return output_text[0]
  
    def process_with_mimo(self, video1_path, video2_path):
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.system_prompt + "/no_think"},
                        {"type": "text", "text": "/no_think"},
                        {"type": "text", "text": "\nSource video:"},
                        {"type": "video", "video": video1_path},
                        {"type": "text", "text": "\nDestination video:"},
                        {"type": "video", "video": video2_path},
                    ],
                }
            ]
            start_time = time.time()
            response = self.inference(messages)
            inference_time = time.time() - start_time
            return response, inference_time
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    def process_single_entry(self, entry):
        index = entry['index']
        video1_path = entry['video1_path']
        video2_path = entry['video2_path']
      
        print(f"\n[Entry {index}] Starting processing")
      
        if index in self.processed_indices:
            print(f"[Entry {index}] Already processed, skipping")
            self.skipped_processed += 1
            return None
      
        retry_count = 0
        last_error = None
      
        while retry_count < self.max_retries:
            try:
                torch.cuda.empty_cache()
                gc.collect()
              
                for video_path in [video1_path, video2_path]:
                    if not os.path.exists(video_path):
                        raise FileNotFoundError(f"File does not exist: {video_path}")
              
                self.log_gpu_memory(f"Entry {index} before inference")
              
                with model_lock:
                    print(f"[Entry {index}] Calling MiMo model...")
                    response_content, inference_time = self.process_with_mimo(video1_path, video2_path)
                    print(f"[Entry {index}] Inference time: {inference_time:.2f} seconds")
                    time.sleep(self.model_delay)
              
                result = {
                    "index": index,
                    "video1_path": video1_path,
                    "video2_path": video2_path,
                    "response": response_content,
                    "inference_time": inference_time,
                    "timestamp": datetime.now().isoformat()
                }
              
                self._append_result_to_file(result)
                self._save_checkpoint(index, success=True)
                self.successful += 1
                print(f"[Entry {index}] ✅ Success")
                return result
              
            except torch.cuda.OutOfMemoryError as e:
                last_error = e
                retry_count += 1
                print(f"[ERROR][Entry {index}] Out of memory (OOM), retry attempt {retry_count}/{self.max_retries}...")
                torch.cuda.empty_cache()
                gc.collect()
                if retry_count < self.max_retries:
                    wait_time = retry_count * 5
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                  
            except Exception as e:
                last_error = e
                retry_count += 1
                print(f"[ERROR][Entry {index}] Unknown error occurred, retry attempt {retry_count}/{self.max_retries}. Error: {e}")
                traceback.print_exc()
                torch.cuda.empty_cache()
                gc.collect()
                if retry_count < self.max_retries:
                    time.sleep(retry_count * 3)
      
        self.failed += 1
        print(f"[FAILURE][Entry {index}] Maximum retry attempts reached, processing failed. Last error: {last_error}")
        # No longer writing errors to file since logging is abandoned
        return None
  
    def _initialize_output_file(self):
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        print(f"Output file '{self.output_file}' already exists, contains {len(data)} records.")
                        return
            except (json.JSONDecodeError, IOError):
                pass
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False)
  
    def _append_result_to_file(self, result):
        with file_lock:
            try:
                with open(self.output_file, 'r+', encoding='utf-8') as f:
                    # Read existing data
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = []  # Create new list if file is empty or invalid
                    
                    if not isinstance(data, list):
                        data = []
                    
                    data.append(result)
                    
                    # Move to beginning of file and truncate
                    f.seek(0)
                    f.truncate()
                    
                    # Write back updated data
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[ERROR] Failed to write result to '{self.output_file}': {e}")
  
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
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
  
    def _save_checkpoint(self, index, success=True):
        if not success:
            return
        try:
            self.processed_indices.add(index)
            if 'successful_indices' not in self.checkpoint_data:
                self.checkpoint_data['successful_indices'] = []
            if index not in self.checkpoint_data['successful_indices']:
                self.checkpoint_data['successful_indices'].append(index)
          
            with file_lock:
                with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(self.checkpoint_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint file '{self.checkpoint_file}': {e}")
  
    def load_input_data(self):
        if not os.path.exists(self.input_json_file):
            raise FileNotFoundError(f"Input file does not exist: {self.input_json_file}")
      
        with open(self.input_json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        data_list = []
        video_pairs = []
        if isinstance(json_data, list):
            video_pairs = json_data
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
        print(f"\n{'='*80}")
        print(f"Starting batch processing (MiMo local model - no logging version)")
        print(f"{'='*80}\n")
      
        self.start_time = time.time()
      
        try:
            data_list = self.load_input_data()
            if not data_list:
                print("[ERROR] No valid video pair data found in input file.")
                return
          
            total_pairs = len(data_list)
            print(f"Found {total_pairs} video pairs to process.\n")
          
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.process_single_entry, entry): entry 
                    for entry in data_list
                }
                for future in as_completed(futures):
                    try:
                        future.result(timeout=self.timeout)
                    except Exception as e:
                        print(f"[ERROR] Unexpected error occurred during task execution: {e}")
          
            elapsed = time.time() - self.start_time
            print(f"\n{'='*80}")
            print(f"✅ All tasks completed!")
            print(f"Total tasks: {total_pairs}, Success: {self.successful}, Failed: {self.failed}, Skipped (already processed): {self.skipped_processed}")
            print(f"Total time: {elapsed:.2f} seconds")
            print(f"{'='*80}\n")
          
        except KeyboardInterrupt:
            print(f"\n[WARNING] Program execution interrupted by user.")
        except Exception as e:
            print(f"\n[FATAL ERROR] Program terminated due to critical error: {e}")
            traceback.print_exc()
        finally:
            print("Cleaning up resources...")
            torch.cuda.empty_cache()
            gc.collect()


def main():
    config = {
        'input_json_file': 'checklist.json',
        'output_dir': '.',  # Output file and checkpoint file will be saved in current directory
        'output_file': 'video_analysis_results_mimo_nothink.json',
        'checkpoint_file': 'processing_checkpoint_mimo_nothink.json',
        'max_workers': 1,
        'max_pairs': None,  # Set to None to process all video pairs
        'model_delay': 1,
        'timeout': 600,
        'model_path': '',  # Model path
        'resume_from_checkpoint': True,
        'max_retries': 3,
        
        # MiMo specific configuration
        'temperature': 0.3,
        'top_p': 0.95,
        'max_new_tokens': 16384,
        'do_sample': True,
    }
  
    processor = VideoProcessor(config)
    processor.run()


if __name__ == "__main__":
    main()
