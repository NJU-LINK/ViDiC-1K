import argparse
import os
import json
import sys
import time
import logging
from datetime import datetime
import traceback
import torch
from transformers import AutoModel, AutoProcessor
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
from tqdm import tqdm
from threading import Lock
from keye_vl_utils import process_vision_info


# Configuration: these can be overridden via command line arguments or environment variables
MODEL_FOLDER = os.getenv("MODEL_FOLDER", "models")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "response")
LOG_FOLDER = os.getenv("LOG_FOLDER", "logs")


class KwaiKeyeProcessor:
    """
    Main video processing class using Kwai-Keye model.
    Processes pairs of videos for comparative analysis with optional thinking mode.
    """
    
    # Class-level file lock for thread-safe output file operations
    _output_file_lock = Lock()
    
    def __init__(self, config):
        """
        Initialize the Kwai-Keye processor.
        
        Args:
            config (dict): Configuration dictionary containing:
                - model_name: Name of the model folder
                - input_json_file: Path to input JSON file with video pairs
                - prompt_file: Path to system prompt file
                - thinking: Enable thinking mode (default: False)
                - max_frames_per_video: Maximum frames per video (default: 32)
        """
        self.model_name = config.get('model_name')
        self.model_path = os.path.join(MODEL_FOLDER, self.model_name)
        self.input_json_file = config.get('input_json_file', 'input_videos.json')
        self.prompt_file = config.get('prompt_file', 'prompt_generate.txt')
        self.thinking = config.get('thinking', False)
        self.max_frames_per_video = config.get('max_frames_per_video', 32)
        self.fps = 2.0  # Kwai-Keye default uses 2fps
        
        # Set up output file path (different names for thinking/non-thinking modes)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        if self.thinking:
            self.output_file = os.path.join(OUTPUT_FOLDER, f"{self.model_name}_thinking_results.json")
        else:
            self.output_file = os.path.join(OUTPUT_FOLDER, f"{self.model_name}_nothinking_results.json")
        
        # Set up logging
        self._setup_logging()
        
        # Validate input file exists
        if not os.path.exists(self.input_json_file):
            raise FileNotFoundError(f"Input JSON file not found: {self.input_json_file}")
        
        # Validate model path exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
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
        
        # Initialize Kwai-Keye model
        self.logger.info(f"Loading Kwai-Keye model: {self.model_path}")
        self.logger.info(f"Thinking mode enabled: {self.thinking}")
        
        try:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            ).eval()
            self.model.to("cuda")
            
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.logger.info("✅ Kwai-Keye model loaded successfully")
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {e}")
            raise
        
        # Generation configuration (different settings for thinking vs non-thinking modes)
        if self.thinking:
            self.temperature = 0.6
            self.max_new_tokens = 8192
        else:
            self.temperature = 0.1
            self.max_new_tokens = 2048
        
        self.logger.info(f"Configuration:")
        self.logger.info(f"  - Model name: {self.model_name}")
        self.logger.info(f"  - Model path: {self.model_path}")
        self.logger.info(f"  - Max frames per video: {self.max_frames_per_video}")
        self.logger.info(f"  - Input file: {self.input_json_file}")
        self.logger.info(f"  - Output file: {self.output_file}")
        self.logger.info(f"  - Prompt file: {self.prompt_file}")
        self.logger.info(f"  - Thinking mode: {'Enabled' if self.thinking else 'Disabled'}")
        self.logger.info(f"  - Temperature: {self.temperature}")
        self.logger.info(f"  - Max new tokens: {self.max_new_tokens}")
        
        # Statistics tracking
        self.successful = 0
        self.failed = 0
        self.skipped_processed = 0
        self.start_time = None
        
        # Load already processed records from output file
        self.processed_indices = self._load_processed_indices()
        
        # Load system prompt
        self.system_prompt = self._load_system_prompt()
        
        # Initialize or load existing results file
        self._initialize_output_file()
    
    def _setup_logging(self):
        """
        Set up logging configuration with both file and console handlers.
        Creates model-specific log directories and separate error logs.
        """
        log_dir = os.path.join(LOG_FOLDER, self.model_name)
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"processing_{timestamp}.log")
        error_log_file = os.path.join(log_dir, f"errors_{timestamp}.log")
        
        # Create dedicated logger
        self.logger = logging.getLogger(f"KwaiKeyeProcessor_{self.model_name}")
        self.logger.setLevel(logging.INFO)
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
        """
        Load indices of already processed video pairs from output file.
        This enables resumable processing.
        
        Returns:
            set: Set of processed indices
        """
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
        """
        Initialize output file (supports incremental writing).
        If file exists, validate its format. Otherwise, create new empty file.
        """
        with self._output_file_lock:
            if os.path.exists(self.output_file):
                try:
                    with open(self.output_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        if isinstance(existing_data, list):
                            self.logger.info(f"Output file exists with {len(existing_data)} historical records")
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
        """
        Incrementally write a single result to file (with file lock protection).
        
        Args:
            result (dict): Processing result to append
        """
        with self._output_file_lock:
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    data = []
                
                # Only keep necessary fields
                clean_result = {
                    "index": result["index"],
                    "video1_path": result["video1_path"],
                    "video2_path": result["video2_path"],
                    "response": result["response"]
                }
                data.append(clean_result)
                
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                self.logger.debug(f"Successfully appended result, total {len(data)} records")
                
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
        """
        Load system prompt from file.
        
        Returns:
            str: System prompt text
        """
        prompt_path = self.prompt_file
        
        if not os.path.exists(prompt_path):
            error_msg = f"❌ Error: Prompt file not found: {prompt_path}"
            self.logger.error(error_msg)
            print("\n" + "="*60)
            print(error_msg)
            print("Please create the prompt file before running!")
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
                print("Please add content to the prompt file!")
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
        """
        Log error information to error log file.
        
        Args:
            error_info (dict): Error information dictionary
        """
        self.error_logger.error(json.dumps(error_info, ensure_ascii=False, indent=2))
    
    def load_input_data(self):
        """
        Load input data from JSON file.
        Supports multiple JSON formats: list, dict with 'video_pairs' or 'data', single dict.
        
        Returns:
            list: List of video pair entries
        """
        self.logger.info(f"Loading input file: {self.input_json_file}")
        
        if not os.path.exists(self.input_json_file):
            raise FileNotFoundError(f"Input file not found: {self.input_json_file}")
        
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
                        self.logger.warning(f"Entry {idx} missing required video path fields")
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
            
            self.logger.info(f"✅ Successfully loaded {len(data_list)} entries")
            return data_list
            
        except Exception as e:
            self.logger.error(f"Failed to load input file: {e}")
            raise
    
    def process_video_pair(self, entry):
        """
        Process a single video pair using Kwai-Keye model.
        
        Args:
            entry (dict): Dictionary containing video1_path and video2_path
            
        Returns:
            str: Model's analysis response
        """
        video1_path = entry['video1_path']
        video2_path = entry['video2_path']
        
        if not os.path.exists(video1_path):
            raise FileNotFoundError(f"Video file not found: {video1_path}")
        if not os.path.exists(video2_path):
            raise FileNotFoundError(f"Video file not found: {video2_path}")
        
        # Build message with two videos
        prompt_text = f"{self.system_prompt}\n\nPlease analyze these two videos and provide a comparison."
        
        # Add thinking indicator according to Kwai-Keye format requirements
        if self.thinking:
            prompt_text = prompt_text + "/think"
        else:
            prompt_text = prompt_text + "/no_think"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "\nVideo A:"},
                    {
                        "type": "video",
                        "video": video1_path,
                        "fps": self.fps,
                        "max_frames": self.max_frames_per_video
                    },
                    {"type": "text", "text": "\nVideo B:"},
                    {
                        "type": "video",
                        "video": video2_path,
                        "fps": self.fps,
                        "max_frames": self.max_frames_per_video
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, mm_processor_kwargs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **mm_processor_kwargs
        )
        inputs = inputs.to("cuda")
        
        # Inference: generate output
        try:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                top_p=0.001,
                repetition_penalty=1.05
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            response = output_text[0] if output_text else ""
            
            if self.thinking:
                if '<think>' in response and '</think>' in response:
                    think_start = response.find('<think>') + len('<think>')
                    think_end = response.find('</think>')
                    think_content = response[think_start:think_end].strip()
                    final_answer = response[think_end + len('</think>'):].strip()
                    
                    if '<answer>' in final_answer and '</answer>' in final_answer:
                        answer_start = final_answer.find('<answer>') + len('<answer>')
                        answer_end = final_answer.find('</answer>')
                        final_answer = final_answer[answer_start:answer_end].strip()
                    
                    return final_answer
                else:
                    if '<answer>' in response and '</answer>' in response:
                        answer_start = response.find('<answer>') + len('<answer>')
                        answer_end = response.find('</answer>')
                        return response[answer_start:answer_end].strip()
                    else:
                        return response.strip()
            else:
                if '<answer>' in response and '</answer>' in response:
                    answer_start = response.find('<answer>') + len('<answer>')
                    answer_end = response.find('</answer>')
                    return response[answer_start:answer_end].strip()
                else:
                    return response.strip()
                
        except Exception as e:
            self.logger.error(f"Error during model inference: {str(e)}")
            raise e
    
    def process_all(self):
        """
        Process all video pairs from input file.
        Supports resumable processing by skipping already processed entries.
        """
        self.start_time = time.time()
        
        data_list = self.load_input_data()
        
        if not data_list:
            self.logger.info("No data to process")
            return
        
        pending_data = [entry for entry in data_list if entry['index'] not in self.processed_indices]
        
        if not pending_data:
            self.logger.info("✅ All data already processed")
            return
        
        total = len(data_list)
        pending = len(pending_data)
        
        self.logger.info(f"Total data: {total} entries")
        self.logger.info(f"Already processed: {len(self.processed_indices)} entries")
        self.logger.info(f"Pending: {pending} entries")
        
        self.logger.info("="*60)
        self.logger.info("Starting processing (using Kwai-Keye model)")
        self.logger.info(f"Incremental write mode: Enabled")
        self.logger.info("="*60)
        
        with tqdm(total=pending, desc="Processing progress") as pbar:
            for entry in pending_data:
                self.logger.info(f"\nProcessing video pair {entry['index']}")
                
                max_retries = 3
                retry_count = 0
                success = False
                
                while retry_count < max_retries and not success:
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
                        
                        self.successful += 1
                        self.logger.info(f"[Entry {entry['index']}] ✅ Processing successful and saved")
                        
                        success = True
                        pbar.update(1)
                        
                    except Exception as e:
                        self.logger.error(f"Processing error: {str(e)}")
                        self.error_logger.error(f"Processing error details: {traceback.format_exc()}")
                        retry_count += 1
                        
                        if retry_count < max_retries:
                            self.logger.warning(f"Retrying {retry_count}/{max_retries}...")
                            time.sleep(2)
                        else:
                            self.failed += 1
                            self.logger.error(f"[Entry {entry['index']}] ❌ Processing failed")
                            
                            error_info = {
                                "index": entry['index'],
                                "video1_path": entry['video1_path'],
                                "video2_path": entry['video2_path'],
                                "error": str(e),
                                "timestamp": datetime.now().isoformat()
                            }
                            self._log_error(error_info)
                            pbar.update(1)
        
        self.logger.info(f"✅ All results saved to: {self.output_file}")
        self.print_summary()
    
    def print_summary(self):
        """Print processing summary with statistics."""
        elapsed = time.time() - self.start_time
        total_processed = self.successful + self.failed
        
        self.logger.info("\n" + "="*60)
        self.logger.info("Processing Complete - Summary Statistics")
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
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Kwai-Keye Video Pair Comparison Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model_name', type=str, required=True, 
                       help='Model name (subfolder name under model folder)')
    parser.add_argument('--input_json', type=str, default='videos.json', 
                       help='Input JSON file path containing video pairs')
    parser.add_argument('--prompt_file', type=str, default='prompt_generate.txt', 
                       help='System prompt file path')
    parser.add_argument('--max_frames', type=int, default=32, 
                       help='Maximum frames per video')
    parser.add_argument('-t', '--thinking', action='store_true',
                       help='Enable thinking mode (uses higher temperature and more tokens)')
    
    return parser.parse_args()


def main():
    """Main entry point for the Kwai-Keye video processing script."""
    args = parse_args()
    
    print("="*60)
    print("Kwai-Keye Video Pair Comparison Analysis")
    print("Incremental write mode: Enabled")
    print("File lock protection: Enabled")
    print("="*60)
    
    config = {
        "model_name": args.model_name,
        "input_json_file": args.input_json,
        "prompt_file": args.prompt_file,
        "max_frames_per_video": args.max_frames,
        "thinking": args.thinking,
    }
    
    print(f"Configuration:")
    print(f"  - Model name: {config['model_name']}")
    print(f"  - Model path: {os.path.join(MODEL_FOLDER, config['model_name'])}")
    print(f"  - Input file: {config['input_json_file']}")
    
    # Choose output file name based on thinking mode
    suffix = "_thinking_results.json" if config['thinking'] else "_nothinking_results.json"
    print(f"  - Output file: {os.path.join(OUTPUT_FOLDER, config['model_name'] + suffix)}")
    print(f"  - Log directory: {os.path.join(LOG_FOLDER, config['model_name'])}")
    print(f"  - Max frames per video: {config['max_frames_per_video']}")
    print(f"  - Thinking mode: {'Enabled' if config['thinking'] else 'Disabled'}")
    print("="*60)
    
    try:
        processor = KwaiKeyeProcessor(config)
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
