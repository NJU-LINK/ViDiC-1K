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
from transformers import AutoModelForImageTextToText, AutoProcessor
from tqdm import tqdm
import argparse
import gc


MODEL_FOLDER = ""  
OUTPUT_FOLDER = ""  
LOG_FOLDER = ""  


class VideoProcessor:
    """Video processing main class - using local Qwen3-VL model"""
    
    # Add class-level file lock
    _output_file_lock = Lock()
  
    def __init__(self, config):
        """Initialize processor"""
        self.model_name = config.get('model_name')
        self.model_path = os.path.join(MODEL_FOLDER, self.model_name)
        self.input_json_file = config.get('input_json_file', 'input_videos.json')
        self.batch_size = config.get('batch_size', 2)
        self.prompt_file = config.get('prompt_file', 'prompt_generate.txt')
        self.fps = config.get('fps', 2.0)
        self.gpu_memory_utilization = config.get('gpu_memory_utilization', 0.9)
        
        # Set output file path
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        self.output_file = os.path.join(OUTPUT_FOLDER, f"{self.model_name}_results.json")
        
        # Setup logging
        self._setup_logging()
        
        # Verify input file exists
        if not os.path.exists(self.input_json_file):
            raise FileNotFoundError(f"Input JSON file does not exist: {self.input_json_file}")
        
        # Verify model path exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        # Detect GPU count
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            self.logger.error("❌ No GPU detected, this script requires GPU support")
            raise RuntimeError("GPU is required to run this script")
        
        self.logger.info(f"✅ Detected {gpu_count} GPU(s)")
        
        # Print GPU information
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            self.logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
      
        # Initialize Qwen3-VL model
        self.logger.info(f"Loading Qwen3-VL model: {self.model_path}")
        
        try:
            # Load Qwen3-VL model using transformers
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                trust_remote_code=True,
            )
            self.logger.info("✅ Qwen3-VL model loaded successfully")
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {e}")
            raise
      
        # Load processor
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                use_fast=True,
                trust_remote_code=True
            )
            
            # Configure video processing parameters
            self.processor.video_processor.size = {
                "longest_edge": 23520*32*32, 
                "shortest_edge": 256*32*32
            }
            
            # Set padding_side to left to support batch generation
            self.processor.tokenizer.padding_side = 'left'
            
            self.logger.info("✅ Processor loaded successfully")
        except Exception as e:
            self.logger.error(f"❌ Processor loading failed: {e}")
            raise
      
        # Generation configuration
        self.generation_config = {
            "max_new_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty": 1.05,
        }
      
        self.logger.info(f"Configuration:")
        self.logger.info(f"  - Model name: {self.model_name}")
        self.logger.info(f"  - Model path: {self.model_path}")
        self.logger.info(f"  - Batch size: {self.batch_size}")
        self.logger.info(f"  - Video sampling FPS: {self.fps} fps")
        self.logger.info(f"  - Input file: {self.input_json_file}")
        self.logger.info(f"  - Output file: {self.output_file}")
        self.logger.info(f"  - Prompt file: {self.prompt_file}")
      
        # Statistics
        self.successful = 0
        self.failed = 0
        self.skipped_processed = 0
        self.start_time = None
      
        # Load processed records from output file
        self.processed_indices = self._load_processed_indices()
      
        # Prompt
        self.system_prompt = self._load_system_prompt()
      
        # Initialize or load existing results file
        self._initialize_output_file()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Create model-specific log directory
        log_dir = os.path.join(LOG_FOLDER, self.model_name)
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"processing_{timestamp}.log")
        error_log_file = os.path.join(log_dir, f"errors_{timestamp}.log")
        
        # Create dedicated logger
        self.logger = logging.getLogger(f"VideoProcessor_{self.model_name}")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Error logger
        self.error_logger = logging.getLogger(f"error_logger_{self.model_name}")
        self.error_logger.setLevel(logging.ERROR)
        self.error_logger.handlers.clear()
        
        error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
        error_handler.setFormatter(logging.Formatter('%(asctime)s - [ERROR] - %(message)s'))
        self.error_logger.addHandler(error_handler)
    
    def _load_processed_indices(self):
        """Load processed indices from output file"""
        processed = set()
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if 'index' in item:
                                processed.add(item['index'])
                        self.logger.info(f"Loaded {len(processed)} processed records from output file")
            except Exception as e:
                self.logger.warning(f"Failed to load processed records: {e}")
        return processed
  
    def _initialize_output_file(self):
        """Initialize output file (supports incremental writing)"""
        with self._output_file_lock:
            if os.path.exists(self.output_file):
                try:
                    with open(self.output_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        if isinstance(existing_data, list):
                            self.logger.info(f"Output file exists, contains {len(existing_data)} historical records")
                        else:
                            with open(self.output_file, 'w', encoding='utf-8') as f:
                                json.dump([], f, ensure_ascii=False)
                            self.logger.info("Output file format error, reinitialized")
                except (json.JSONDecodeError, Exception) as e:
                    backup_file = f"{self.output_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    os.rename(self.output_file, backup_file)
                    self.logger.warning(f"Output file read failed, backed up to: {backup_file}")
                    with open(self.output_file, 'w', encoding='utf-8') as f:
                        json.dump([], f, ensure_ascii=False)
            else:
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False)
                self.logger.info("Created new output file")
  
    def _append_result_to_file(self, result):
        """Incrementally write single result to file (with file lock)"""
        with self._output_file_lock:
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
                if not isinstance(data, list):
                    data = []
            
                # Only keep required fields
                clean_result = {
                    "index": result["index"],
                    "video1_path": result["video1_path"],
                    "video2_path": result["video2_path"],
                    "response": result["response"]
                }
                data.append(clean_result)
            
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            
                self.logger.debug(f"Successfully wrote result incrementally, total {len(data)} records")
            
            except Exception as e:
                self.logger.error(f"Incremental write failed: {e}")
                # Backup handling
                backup_file = f"{self.output_file}.incremental"
                try:
                    if os.path.exists(backup_file):
                        with open(backup_file, 'r', encoding='utf-8') as f:
                            backup_data = json.load(f)
                    else:
                        backup_data = []
                
                    clean_result = {
                        "index": result["index"],
                        "video1_path": result["video1_path"],
                        "video2_path": result["video2_path"],
                        "response": result["response"]
                    }
                    backup_data.append(clean_result)
                
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        json.dump(backup_data, f, ensure_ascii=False, indent=2)
                
                    self.logger.warning(f"Result saved to backup file: {backup_file}")
                except Exception as e2:
                    self.logger.error(f"Backup file write also failed: {e2}")
  
    def _load_system_prompt(self):
        """Load system prompt"""
        prompt_path = self.prompt_file
      
        if not os.path.exists(prompt_path):
            error_msg = f"❌ Error: Prompt file does not exist: {prompt_path}"
            self.logger.error(error_msg)
            print("\n" + "="*60)
            print(error_msg)
            print("Please create prompt file before running the program!")
            print("="*60)
            sys.exit(1)
      
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
          
            if not prompt:
                error_msg = f"❌ Error: Prompt file is empty: {prompt_path}"
                self.logger.error(error_msg)
                print("\n" + "="*60)
                print(error_msg)
                print("Please add content to prompt file!")
                print("="*60)
                sys.exit(1)
          
            self.logger.info(f"✅ Successfully loaded system prompt file: {prompt_path}")
            self.logger.info(f"Prompt length: {len(prompt)} characters")
          
            return prompt
          
        except Exception as e:
            error_msg = f"❌ Error: Failed to read prompt file: {e}"
            self.logger.error(error_msg)
            sys.exit(1)
  
    def _log_error(self, error_info):
        """Log error information to log file"""
        self.error_logger.error(json.dumps(error_info, ensure_ascii=False, indent=2))
  
    def _validate_video_files(self, video1_path, video2_path):
        """Validate video files and log size information"""
        if not os.path.exists(video1_path):
            raise FileNotFoundError(f"Video file does not exist: {video1_path}")
        if not os.path.exists(video2_path):
            raise FileNotFoundError(f"Video file does not exist: {video2_path}")
        
        try:
            size1_mb = os.path.getsize(video1_path) / (1024 * 1024)
            size2_mb = os.path.getsize(video2_path) / (1024 * 1024)
            self.logger.info(f"Video1 size: {size1_mb:.2f}MB, Video2 size: {size2_mb:.2f}MB")
            
            max_size_mb = 500
            if size1_mb > max_size_mb:
                self.logger.warning(f"⚠️ Video1 file is large ({size1_mb:.2f}MB), may affect processing speed")
            if size2_mb > max_size_mb:
                self.logger.warning(f"⚠️ Video2 file is large ({size2_mb:.2f}MB), may affect processing speed")
                
        except Exception as e:
            self.logger.warning(f"Unable to get file size information: {e}")
  
    def load_input_data(self):
        """Load input data from JSON file"""
        self.logger.info(f"Starting to load input file: {self.input_json_file}")
      
        if not os.path.exists(self.input_json_file):
            raise FileNotFoundError(f"Input file does not exist: {self.input_json_file}")
      
        data_list = []
      
        try:
            with open(self.input_json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
          
            if isinstance(json_data, list):
                for idx, item in enumerate(json_data):
                    if 'video1_path' in item and 'video2_path' in item:
                        entry = {
                            'index': idx,
                            'video1_path': item['video1_path'],
                            'video2_path': item['video2_path']
                        }
                        data_list.append(entry)
                    else:
                        self.logger.warning(f"Item {idx} missing required video path fields")
            elif isinstance(json_data, dict):
                video_pairs = json_data.get('video_pairs', json_data.get('data', [json_data]))
                if isinstance(video_pairs, list):
                    for idx, item in enumerate(video_pairs):
                        if 'video1_path' in item and 'video2_path' in item:
                            entry = {
                                'index': idx,
                                'video1_path': item['video1_path'],
                                'video2_path': item['video2_path']
                            }
                            data_list.append(entry)
                elif 'video1_path' in json_data and 'video2_path' in json_data:
                    entry = {
                        'index': 0,
                        'video1_path': json_data['video1_path'],
                        'video2_path': json_data['video2_path']
                    }
                    data_list.append(entry)
          
            self.logger.info(f"✅ Successfully loaded {len(data_list)} data entries")
            return data_list
          
        except Exception as e:
            self.logger.error(f"Failed to load input file: {e}")
            raise
  
    def process_video_pairs_batch(self, entries):
        """Process video pairs in batch - using Qwen3-VL format"""
        batch_messages = []
      
        for entry in entries:
            video1_path = entry['video1_path']
            video2_path = entry['video2_path']
          
            self._validate_video_files(video1_path, video2_path)
          
            # Qwen3-VL message format
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Video A:"},
                        {
                            "type": "video",
                            "video": video1_path,
                            "fps": self.fps,
                            "min_pixels": 4 * 32 * 32,
                            "max_pixels": 256 * 32 * 32,
                            "total_pixels": 16384 * 32 * 32,
                        },
                        {"type": "text", "text": "Video B:"},
                        {
                            "type": "video",
                            "video": video2_path,
                            "fps": self.fps,
                            "min_pixels": 4 * 32 * 32,
                            "max_pixels": 256 * 32 * 32,
                            "total_pixels": 16384 * 32 * 32,
                        },
                    ],
                },
            ]
            batch_messages.append(messages)
      
        # Prepare batch inputs
        with torch.no_grad():
            try:
                # Use Qwen3-VL's processor
                inputs = self.processor.apply_chat_template(
                    batch_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    fps=self.fps,
                    padding=True  # Enable padding to support batch processing
                )
                
                # Move to GPU
                inputs = inputs.to(self.model.device)
                
                # Batch generation
                generated_ids = self.model.generate(
                    **inputs,
                    **self.generation_config
                )
                
                # Decode output
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                output_texts = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                
                # Clean GPU memory
                del inputs
                del generated_ids
                torch.cuda.empty_cache()
                
                return output_texts
                
            except Exception as e:
                self.logger.error(f"Batch processing failed: {e}")
                raise
  
    def process_all(self):
        """Process all data"""
        self.start_time = time.time()
      
        data_list = self.load_input_data()
      
        if not data_list:
            self.logger.info("No data to process")
            return
      
        pending_data = [entry for entry in data_list if entry['index'] not in self.processed_indices]
      
        if not pending_data:
            self.logger.info("✅ All data has been processed")
            return
      
        total = len(data_list)
        pending = len(pending_data)
      
        self.logger.info(f"Total data: {total} entries")
        self.logger.info(f"Processed: {len(self.processed_indices)} entries")
        self.logger.info(f"Pending: {pending} entries")
      
        self.logger.info("="*60)
        self.logger.info("Starting batch processing (using local Qwen3-VL model)")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Incremental write mode: Enabled")
        self.logger.info("="*60)
      
        with tqdm(total=pending, desc="Processing progress") as pbar:
            for batch_start in range(0, pending, self.batch_size):
                batch_end = min(batch_start + self.batch_size, pending)
                batch_entries = pending_data[batch_start:batch_end]
              
                self.logger.info(f"\nProcessing batch {batch_start//self.batch_size + 1}: {len(batch_entries)} video pairs")
              
                max_retries = 3
                retry_count = 0
                success = False
              
                while retry_count < max_retries and not success:
                    try:
                        responses = self.process_video_pairs_batch(batch_entries)
                      
                        for entry, response in zip(batch_entries, responses):
                            # Handle special output for Thinking models
                            if 'Thinking' in self.model_name and '</think>' in response:
                                response = response.split("</think>\n\n")[-1]
                            
                            result = {
                                "index": entry['index'],
                                "video1_path": entry['video1_path'],
                                "video2_path": entry['video2_path'],
                                "response": response
                            }
                        
                            self._append_result_to_file(result)
                            self.processed_indices.add(entry['index'])
                        
                            self.successful += 1
                            self.logger.info(f"[Entry {entry['index']}] ✅ Processed successfully and saved")
                      
                        success = True
                        pbar.update(len(batch_entries))
                        
                        # Periodically clean GPU memory
                        if batch_start % (self.batch_size * 10) == 0:
                            gc.collect()
                            torch.cuda.empty_cache()
                      
                    except Exception as e:
                        self.logger.error(f"Batch processing error: {str(e)}")
                        self.error_logger.error(f"Batch processing error details: {traceback.format_exc()}")
                        retry_count += 1
                      
                        if retry_count < max_retries:
                            self.logger.warning(f"Retrying {retry_count}/{max_retries}...")
                            time.sleep(2)
                            
                            # Clean GPU memory before retry
                            gc.collect()
                            torch.cuda.empty_cache()
                        else:
                            self.logger.warning("Batch processing failed, trying individual processing...")
                            for entry in batch_entries:
                                try:
                                    # Individual processing
                                    responses = self.process_video_pairs_batch([entry])
                                    
                                    if 'Thinking' in self.model_name and '</think>' in responses[0]:
                                        response = responses[0].split("</think>\n\n")[-1]
                                    else:
                                        response = responses[0]
                                  
                                    result = {
                                        "index": entry['index'],
                                        "video1_path": entry['video1_path'],
                                        "video2_path": entry['video2_path'],
                                        "response": response
                                    }
                                  
                                    self._append_result_to_file(result)
                                    self.processed_indices.add(entry['index'])
                                    self.successful += 1
                                    self.logger.info(f"[Entry {entry['index']}] ✅ Individually processed successfully")
                                    pbar.update(1)
                                    
                                    # Clean GPU memory
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                  
                                except Exception as e2:
                                    self.failed += 1
                                    self.logger.error(f"[Entry {entry['index']}] ❌ Processing failed: {str(e2)}")
                                  
                                    error_info = {
                                        "index": entry['index'],
                                        "video1_path": entry['video1_path'],
                                        "video2_path": entry['video2_path'],
                                        "error": str(e2),
                                        "timestamp": datetime.now().isoformat()
                                    }
                                    self._log_error(error_info)
                                    pbar.update(1)
                          
                            success = True
      
        self.logger.info(f"✅ All results saved to: {self.output_file}")
        self.print_summary()
  
    def print_summary(self):
        """Print processing summary"""
        elapsed = time.time() - self.start_time
        total_processed = self.successful + self.failed
      
        self.logger.info("\n" + "="*60)
        self.logger.info("Processing Complete - Statistics Summary")
        self.logger.info("="*60)
        self.logger.info(f"Total time: {elapsed/60:.2f} minutes")
        self.logger.info(f"Total processed: {total_processed}")
        self.logger.info(f"Successful: {self.successful}")
        self.logger.info(f"Failed: {self.failed}")
        self.logger.info(f"Skipped: {self.skipped_processed}")
      
        if total_processed > 0:
            self.logger.info(f"Success rate: {self.successful/total_processed*100:.2f}%")
            self.logger.info(f"Average processing time: {elapsed/total_processed:.2f} seconds/entry")
      
        self.logger.info(f"\nOutput file: {self.output_file}")
        self.logger.info(f"Log directory: {os.path.join(LOG_FOLDER, self.model_name)}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Video comparison analysis processing program - using local Qwen3-VL model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model_name', type=str, required=True, help='Model name (subfolder name under model folder)')
    parser.add_argument('--input_json', type=str, default='videos.json', help='Input JSON file path')
    parser.add_argument('--prompt_file', type=str, default='prompt_generate.txt', help='System prompt file path')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--fps', type=float, default=2.0, help='Video sampling frame rate')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95, help='GPU memory utilization (0.0-1.0)')
    
    return parser.parse_args()


def main():
    """Main function"""
    # Set environment variables to optimize performance
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(torch.cuda.device_count()))
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    args = parse_args()
    
    config = {
        "model_name": args.model_name,
        "input_json_file": args.input_json,
        "prompt_file": args.prompt_file,
        "batch_size": args.batch_size,
        "fps": args.fps,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    
    print("="*60)
    print("Video Comparison Analysis Processing Program")
    print("Using local Qwen3-VL model")
    print("Incremental write mode: Enabled")
    print("File lock protection: Enabled")
    print("="*60)
    
    print(f"Configuration:")
    print(f"  - Model name: {config['model_name']}")
    print(f"  - Model path: {os.path.join(MODEL_FOLDER, config['model_name'])}")
    print(f"  - Input file: {config['input_json_file']}")
    print(f"  - Output file: {os.path.join(OUTPUT_FOLDER, config['model_name'] + '_results.json')}")
    print(f"  - Log directory: {os.path.join(LOG_FOLDER, config['model_name'])}")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Video sampling FPS: {config['fps']} fps")
    print(f"  - GPU memory utilization: {config['gpu_memory_utilization']}")
    print("="*60)
    
    try:
        processor = VideoProcessor(config)
        processor.process_all()
        print("\n✅ Processing complete!")
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Program error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
