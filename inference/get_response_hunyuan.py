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
import math
import re
from transformers import ARCHunyuanVideoProcessor, ARCHunyuanVideoForConditionalGeneration
from tqdm import tqdm
import argparse


# Configuration: these can be overridden via command line arguments or environment variables
MODEL_FOLDER = os.getenv("MODEL_FOLDER", "models")  
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "response")  
LOG_FOLDER = os.getenv("LOG_FOLDER", "logs")  


class VideoProcessor:
    """
    Main video processing class using local HunyuanVideo model.
    Processes pairs of videos and generates comparative analysis.
    """
    
    # Class-level file lock for thread-safe output file operations
    _output_file_lock = Lock()
  
    def __init__(self, config):
        """
        Initialize the video processor.
        
        Args:
            config (dict): Configuration dictionary containing:
                - model_name: Name of the model folder
                - input_json_file: Path to input JSON file with video pairs
                - prompt_file: Path to system prompt file
        """
        self.model_name = config.get('model_name')
        self.model_path = os.path.join(MODEL_FOLDER, self.model_name)
        self.input_json_file = config.get('input_json_file', 'input_videos.json')
        self.batch_size = 1  # HunyuanVideo only supports single video processing
        self.prompt_file = config.get('prompt_file', 'prompt_generate.txt')
        self.font_path = os.path.join(self.model_path, "ARIAL.TTF")
        
        # Set up output file path
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        self.output_file = os.path.join(OUTPUT_FOLDER, f"{self.model_name}_results.json")
        
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
        
        self.device = torch.device("cuda")
      
        # Initialize HunyuanVideo model
        self.logger.info(f"Loading HunyuanVideo model: {self.model_path}")
        
        try:
            self.model = ARCHunyuanVideoForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).eval()
            self.model.to(self.device)
            self.logger.info("✅ HunyuanVideo model loaded successfully")
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {e}")
            raise
      
        # Load processor
        try:
            self.processor = ARCHunyuanVideoProcessor.from_pretrained(
                self.model_path,
                font_path=self.font_path
            )
            self.logger.info("✅ Processor loaded successfully")
        except Exception as e:
            self.logger.error(f"❌ Processor loading failed: {e}")
            raise
      
        # Set generation configuration
        self.generation_config = dict(
            max_new_tokens=1024,
            do_sample=False,
        )
      
        self.logger.info(f"Configuration:")
        self.logger.info(f"  - Model name: {self.model_name}")
        self.logger.info(f"  - Model path: {self.model_path}")
        self.logger.info(f"  - Input file: {self.input_json_file}")
        self.logger.info(f"  - Output file: {self.output_file}")
        self.logger.info(f"  - Prompt file: {self.prompt_file}")
      
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
                
                    # Clean result for backup as well
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
  
    def _validate_video_files(self, video1_path, video2_path):
        """
        Validate video files exist and log size information.
        
        Args:
            video1_path (str): Path to first video
            video2_path (str): Path to second video
            
        Raises:
            FileNotFoundError: If either video file doesn't exist
        """
        if not os.path.exists(video1_path):
            raise FileNotFoundError(f"Video file not found: {video1_path}")
        if not os.path.exists(video2_path):
            raise FileNotFoundError(f"Video file not found: {video2_path}")
        
        try:
            size1_mb = os.path.getsize(video1_path) / (1024 * 1024)
            size2_mb = os.path.getsize(video2_path) / (1024 * 1024)
            self.logger.info(f"Video 1 size: {size1_mb:.2f}MB, Video 2 size: {size2_mb:.2f}MB")
            
            max_size_mb = 500
            if size1_mb > max_size_mb:
                self.logger.warning(f"⚠️ Video 1 is large ({size1_mb:.2f}MB), may affect processing speed")
            if size2_mb > max_size_mb:
                self.logger.warning(f"⚠️ Video 2 is large ({size2_mb:.2f}MB), may affect processing speed")
                
        except Exception as e:
            self.logger.warning(f"Unable to get file size info: {e}")
    
    def calculate_frame_indices(self, vlen: int, fps: float, duration: float) -> tuple:
        """
        Calculate video frame indices for sampling.
        For videos <=150s, sample at 1fps. For longer videos, sample to get 150 frames.
        
        Args:
            vlen (int): Total number of frames in video
            fps (float): Video frames per second
            duration (float): Video duration in seconds
            
        Returns:
            tuple: (frame_indices, sample_fps)
        """
        frames_per_second = fps

        if duration <= 150:
            interval = 1
            intervals = [
                (int(i * interval * frames_per_second), int((i + 1) * interval * frames_per_second))
                for i in range(math.ceil(duration))
            ]
            sample_fps = 1
        else:
            num_segments = 150
            segment_duration = duration / num_segments
            intervals = [
                (int(i * segment_duration * frames_per_second), int((i + 1) * segment_duration * frames_per_second))
                for i in range(num_segments)
            ]
            sample_fps = 1 / segment_duration

        frame_indices = []
        for start, end in intervals:
            if end > vlen:
                end = vlen
            frame_indices.append((start + end) // 2)

        return frame_indices, sample_fps

    def load_video_frames(self, video_path: str):
        """
        Load video frames from file.
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            tuple: (list of PIL Images, sample_fps)
        """
        video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=4)
        vlen = len(video_reader)
        input_fps = video_reader.get_avg_fps()
        duration = vlen / input_fps

        frame_indices, sample_fps = self.calculate_frame_indices(vlen, input_fps, duration)

        return [Image.fromarray(video_reader[idx].asnumpy()) for idx in frame_indices], sample_fps

    def build_prompt(self, prompt: str, num_frames: int):
        """
        Build model input prompt with video frame tokens.
        
        Args:
            prompt (str): Text prompt
            num_frames (int): Number of video frames
            
        Returns:
            str: Formatted prompt for model
        """
        video_prefix = "<image>" * num_frames
        return f"<|startoftext|>{video_prefix}\n{prompt}\n Output the thinking process in and final answer in <answer> </answer> tags, i.e., <answer> answer here </answer>.<sep>"

    def extract_answer(self, text):
        """
        Extract answer from model output.
        
        Args:
            text (str): Model output text
            
        Returns:
            str: Extracted answer (returns full text if no answer tags found)
        """
        return text  # If no answer tags found, return entire text

    def process_video_pair(self, video1_path, video2_path):
        """
        Process a pair of videos and generate comparative analysis.
        
        Args:
            video1_path (str): Path to first video (source)
            video2_path (str): Path to second video (destination)
            
        Returns:
            str: Model's analysis response
        """
        try:
            # Load frames from both videos
            video1_frames, sample_fps1 = self.load_video_frames(video1_path)
            video2_frames, sample_fps2 = self.load_video_frames(video2_path)
            
            # Combine video frames
            all_frames = video1_frames + video2_frames
            avg_sample_fps = (sample_fps1 + sample_fps2) / 2
            
            # Create silent audio
            duration = len(all_frames) / avg_sample_fps
            sr = 16000
            audio = np.zeros(int(duration * sr), dtype=np.float32)
            
            # Build complete prompt
            full_prompt = f"{self.system_prompt}\n\nSource video: [First {len(video1_frames)} frames]\nDestination video: [Next {len(video2_frames)} frames]"
            
            # Build model input
            prompt_text = self.build_prompt(full_prompt, len(all_frames))
            
            video_inputs = {
                "video": all_frames,
                "video_metadata": {
                    "fps": avg_sample_fps,
                },
            }
            
            audio_inputs = {
                "audio": audio,
                "sampling_rate": sr,
                "duration": float(duration),
            }
            
            # Process inputs
            inputs = self.processor(
                text=prompt_text,
                **video_inputs,
                **audio_inputs,
                return_tensors="pt",
            )
            
            # Ensure duration is integer type
            if 'duration' in inputs:
                inputs['duration'] = inputs['duration'].long()
            
            inputs = {
                k: (v.to(self.device, dtype=self.model.dtype) if v.dtype.is_floating_point else v.to(self.device))
                for k, v in inputs.items()
            }
            
            # Generate response
            outputs = self.model.generate(**inputs, **self.generation_config)
            output_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer
            answer = self.extract_answer(output_text)
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error processing video pair: {e}")
            raise
  
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
        self.logger.info("Starting processing (using local HunyuanVideo model)")
        self.logger.info(f"Incremental write mode: Enabled")
        self.logger.info("="*60)
      
        with tqdm(total=pending, desc="Processing progress") as pbar:
            for entry in pending_data:
                self.logger.info(f"\nProcessing video pair [Entry {entry['index']}]")
                
                max_retries = 3
                retry_count = 0
                success = False
                
                while retry_count < max_retries and not success:
                    try:
                        # Validate video files
                        self._validate_video_files(entry['video1_path'], entry['video2_path'])
                        
                        # Process video pair
                        response = self.process_video_pair(entry['video1_path'], entry['video2_path'])
                        
                        # Build result
                        result = {
                            "index": entry['index'],
                            "video1_path": entry['video1_path'],
                            "video2_path": entry['video2_path'],
                            "response": response
                        }
                        
                        # Save result
                        self._append_result_to_file(result)
                        self.processed_indices.add(entry['index'])
                        
                        self.successful += 1
                        self.logger.info(f"[Entry {entry['index']}] ✅ Processing successful and saved")
                        
                        success = True
                        pbar.update(1)
                        
                    except Exception as e:
                        retry_count += 1
                        self.logger.error(f"[Entry {entry['index']}] Processing failed (attempt {retry_count}/{max_retries}): {str(e)}")
                        
                        if retry_count < max_retries:
                            self.logger.warning(f"Retrying {retry_count}/{max_retries}...")
                            time.sleep(2)
                        else:
                            self.failed += 1
                            self.logger.error(f"[Entry {entry['index']}] ❌ Processing failed: {str(e)}")
                            
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
        description='Video Pair Comparison Analysis - Using local HunyuanVideo model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model_name', type=str, required=True, 
                        help='Model name (subfolder name under model folder)')
    parser.add_argument('--input_json', type=str, default='videos.json', 
                        help='Input JSON file path containing video pairs')
    parser.add_argument('--prompt_file', type=str, default='prompt_generate.txt', 
                        help='System prompt file path')
    
    return parser.parse_args()


def main():
    """Main entry point for the video processing script."""
    args = parse_args()
    
    print("="*60)
    print("Video Pair Comparison Analysis")
    print("Using local HunyuanVideo model")
    print("Incremental write mode: Enabled")
    print("File lock protection: Enabled")
    print("="*60)
    
    config = {
        "model_name": args.model_name,
        "input_json_file": args.input_json,
        "prompt_file": args.prompt_file,
    }
    
    print(f"Configuration:")
    print(f"  - Model name: {config['model_name']}")
    print(f"  - Model path: {os.path.join(MODEL_FOLDER, config['model_name'])}")
    print(f"  - Input file: {config['input_json_file']}")
    print(f"  - Output file: {os.path.join(OUTPUT_FOLDER, config['model_name'] + '_results.json')}")
    print(f"  - Log directory: {os.path.join(LOG_FOLDER, config['model_name'])}")
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
