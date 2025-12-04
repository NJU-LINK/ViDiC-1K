"""
Video Comparison Analysis Tool
Using OpenAI SDK with Incremental Write Mode
"""

import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import re
import logging
from datetime import datetime
from pathlib import Path
import traceback
import base64

# Configure standard output encoding
sys.stdout.reconfigure(encoding='utf-8')

# Set up logging configuration
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

# Use OpenAI SDK
try:
    from openai import OpenAI
except ImportError:
    logger.error("OpenAI SDK not installed")
    print("Please install openai SDK: pip install openai")
    sys.exit(1)

# Global locks
file_lock = Lock()
api_lock = Lock()


class VideoProcessor:
    """Main video processing class"""
    
    def __init__(self, config):
        """Initialize processor"""
        self.input_json_file = config.get('input_json_file', 'input_videos.json') 
        self.output_file = config.get('output_file', 'video_analysis_results.json')
        self.error_file = config.get('error_file', 'video_analysis_errors.json')
        self.checkpoint_file = config.get('checkpoint_file', 'processing_checkpoint.json')
        self.max_workers = config.get('max_workers', 2)
        self.max_pairs = config.get('max_pairs', None)
        self.api_delay = config.get('api_delay', 2)
        self.timeout = config.get('timeout', 300)
        self.model = config.get('model', 'your-model-name')
        self.api_key = config.get('api_key')
        self.max_file_size_mb = config.get('max_file_size_mb', 10)
        self.resume_from_checkpoint = config.get('resume_from_checkpoint', True)
        self.skip_failed = config.get('skip_failed', False)
        self.max_retries = config.get('max_retries', 3)
        
        # Create output directories if they don't exist
        for file_path in [self.output_file, self.error_file, self.checkpoint_file]:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        logger.info(f"OpenAI client configuration:")
        logger.info(f"  - Model: {self.model}")
        logger.info(f"  - Max file size: {self.max_file_size_mb}MB")
        logger.info(f"  - Input file: {self.input_json_file}")
        logger.info(f"  - Output file: {self.output_file}")
        
        # Statistics
        self.successful = 0
        self.failed = 0
        self.skipped_large_files = 0
        self.skipped_processed = 0
        self.skipped_failed = 0
        self.start_time = None
        
        # Load processing records
        self.checkpoint_data = self._load_checkpoint()
        self.processed_indices = set(self.checkpoint_data.get('successful_indices', []))
        
        # System prompt
        self.system_prompt = self._load_system_prompt()
        
        # Initialize or load existing result file
        self._initialize_output_file()
    
    def _initialize_output_file(self):
        """Initialize output file (supports incremental writing)"""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        logger.info(f"Output file exists with {len(existing_data)} historical records")
                    else:
                        with open(self.output_file, 'w', encoding='utf-8') as f:
                            json.dump([], f, ensure_ascii=False)
                        logger.info("Output file format error, reinitialized")
            except (json.JSONDecodeError, Exception) as e:
                backup_file = f"{self.output_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(self.output_file, backup_file)
                logger.warning(f"Failed to read output file, backed up to: {backup_file}")
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False)
        else:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False)
            logger.info("Created new output file")
    
    def _append_result_to_file(self, result):
        """Incrementally write single result to file"""
        with file_lock:
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    data = []
                
                data.append(result)
                
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                logger.debug(f"Successfully wrote result incrementally, total {len(data)} records")
                
            except Exception as e:
                logger.error(f"Incremental write failed: {e}")
                backup_file = f"{self.output_file}.incremental"
                try:
                    if os.path.exists(backup_file):
                        with open(backup_file, 'r', encoding='utf-8') as f:
                            backup_data = json.load(f)
                    else:
                        backup_data = []
                    
                    backup_data.append(result)
                    
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        json.dump(backup_data, f, ensure_ascii=False, indent=2)
                    
                    logger.warning(f"Result saved to backup file: {backup_file}")
                except Exception as e2:
                    logger.error(f"Backup file write also failed: {e2}")
    
    def _load_system_prompt(self):
        """Load system prompt"""
        prompt_path = "prompt_generate.txt"
        
        if not os.path.exists(prompt_path):
            error_msg = f"Error: Prompt file does not exist: {prompt_path}"
            logger.error(error_msg)
            print("\n" + "="*60)
            print(error_msg)
            print("Please create the prompt file before running!")
            print("="*60)
            sys.exit(1)
        
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            
            if not prompt:
                error_msg = f"Error: Prompt file is empty: {prompt_path}"
                logger.error(error_msg)
                print("\n" + "="*60)
                print(error_msg)
                print("Please add content to the prompt file!")
                print("="*60)
                sys.exit(1)
            
            logger.info(f"Successfully loaded system prompt file: {prompt_path}")
            logger.info(f"Prompt length: {len(prompt)} characters")
            
            return prompt
            
        except Exception as e:
            error_msg = f"Error: Failed to read prompt file: {e}"
            logger.error(error_msg)
            sys.exit(1)
    
    def _load_checkpoint(self):
        """Load checkpoint data"""
        if not self.resume_from_checkpoint:
            return {}
        
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                    logger.info(f"Checkpoint file loaded successfully")
                    return checkpoint
            except Exception as e:
                logger.warning(f"Failed to load checkpoint file: {e}")
                return {}
        return {}
    
    def _save_checkpoint(self, index, success=True):
        """Save processing checkpoint"""
        try:
            if success:
                self.processed_indices.add(index)
                if 'successful_indices' not in self.checkpoint_data:
                    self.checkpoint_data['successful_indices'] = []
                if index not in self.checkpoint_data['successful_indices']:
                    self.checkpoint_data['successful_indices'].append(index)
            
            self.checkpoint_data['statistics'] = {
                'last_update': datetime.now().isoformat(),
                'successful': self.successful,
                'failed': self.failed,
                'total_processed': len(self.processed_indices)
            }
            
            with file_lock:
                with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(self.checkpoint_data, f, ensure_ascii=False, indent=2)
                    
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def _append_error_to_file(self, error_info):
        """Incrementally write error info to error file"""
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
                    
            except Exception as e:
                logger.error(f"Failed to write error file: {e}")
    
    def load_input_data(self):
        """Load input data from JSON file"""
        logger.info(f"Loading input file: {self.input_json_file}")
        
        if not os.path.exists(self.input_json_file):
            raise FileNotFoundError(f"Input file does not exist: {self.input_json_file}")
        
        data_list = []
        
        try:
            with open(self.input_json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Handle different JSON formats
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
                        logger.warning(f"Item {idx} missing required video path fields")
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
            
            logger.info(f"Successfully loaded {len(data_list)} entries")
            return data_list
            
        except Exception as e:
            logger.error(f"Failed to load input file: {e}")
            raise
    
    @staticmethod
    def encode_video_to_base64(video_path):
        """Encode video file to base64"""
        with open(video_path, 'rb') as video_file:
            return base64.b64encode(video_file.read()).decode('utf-8')
    
    def prepare_video_content(self, video_path):
        """Prepare video content for API"""
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        logger.debug(f"File {os.path.basename(video_path)} ({file_size_mb:.2f}MB) using Base64 encoding")
        base64_video = self.encode_video_to_base64(video_path)
        
        video_ext = os.path.splitext(video_path)[1].lower().strip('.')
        mime_type = f"video/{video_ext}" if video_ext else "video/mp4"
        
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_video}"
            }
        }
    
    def process_single_entry(self, entry):
        """Process single data entry"""
        index = entry['index']
        video1_path = entry['video1_path']
        video2_path = entry['video2_path']
        
        logger.info(f"[Entry {index}] Starting processing")
        
        # Check if already processed
        if index in self.processed_indices:
            logger.info(f"[Entry {index}] Already processed, skipping")
            self.skipped_processed += 1
            return None
        
        try:
            # Validate files
            for video_path in [video1_path, video2_path]:
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video file not found: {video_path}")
            
            size1_mb = os.path.getsize(video1_path) / (1024 * 1024)
            size2_mb = os.path.getsize(video2_path) / (1024 * 1024)
            
            logger.info(f"[Entry {index}] Video 1: {size1_mb:.2f}MB, Video 2: {size2_mb:.2f}MB")
            
            # Prepare video content
            video1_content = self.prepare_video_content(video1_path)
            video2_content = self.prepare_video_content(video2_path)
            
            # Build messages
            messages = [
                {
                    "role": "system",  
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Source video:"},
                        video1_content,
                        {"type": "text", "text": "Destination video:"},
                        video2_content
                    ]
                }
            ]
            
            # API call
            with api_lock:
                logger.info(f"[Entry {index}] Calling API...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.7
                )
                
                time.sleep(self.api_delay)
            
            if not response or not response.choices:
                raise ValueError("API response is empty")
            
            response_content = response.choices[0].message.content
            
            if not response_content:
                raise ValueError("Response content is empty")
            
            # Build result
            result = {
                "index": index,
                "video1_path": video1_path,
                "video2_path": video2_path,
                "response": response_content,
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }
            
            # Immediately write result to file incrementally
            self._append_result_to_file(result)
            
            # Save success checkpoint
            self._save_checkpoint(index, success=True)
            
            self.successful += 1
            logger.info(f"[Entry {index}] ✅ Processing successful and saved")
            
            return result
            
        except Exception as e:
            self.failed += 1
            logger.error(f"[Entry {index}] ❌ Processing failed: {str(e)}")
            error_logger.error(f"[Entry {index}] Error details: {traceback.format_exc()}")
            
            # Save error info (incremental write)
            error_info = {
                "index": index,
                "video1_path": video1_path,
                "video2_path": video2_path,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            self._append_error_to_file(error_info)
            
            return None
    
    def process_all(self):
        """Process all data"""
        self.start_time = time.time()
        
        # Load input data
        data_list = self.load_input_data()
        
        if not data_list:
            logger.info("No data to process")
            return
        
        if self.max_pairs:
            data_list = data_list[:self.max_pairs]
            logger.info(f"Limited processing count to {self.max_pairs} entries")
        
        total = len(data_list)
        logger.info(f"Preparing to process {total} entries")
        
        logger.info("="*60)
        logger.info("Starting batch processing")
        logger.info(f"Model: {self.model}")
        logger.info(f"Concurrency: {self.max_workers}")
        logger.info(f"Incremental write mode: Enabled")
        logger.info("="*60)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_entry = {
                executor.submit(self.process_single_entry, entry): entry 
                for entry in data_list
            }
            
            for i, future in enumerate(as_completed(future_to_entry), 1):
                entry = future_to_entry[future]
                
                try:
                    result = future.result(timeout=self.timeout)
                    if result:
                        status = "✅ Success"
                    else:
                        status = "⏭️ Skipped"
                except Exception as e:
                    status = "❌ Exception"
                    logger.error(f"Task execution exception: {e}")
                
                elapsed = time.time() - self.start_time
                eta_seconds = (elapsed / i) * (total - i) if i > 0 else 0
                
                logger.info(
                    f"Progress: {i}/{total} | Entry {entry['index']} {status} | "
                    f"Success: {self.successful} | Failed: {self.failed} | "
                    f"ETA: {eta_seconds/60:.1f} min"
                )
        
        logger.info(f"✅ All results saved incrementally to: {self.output_file}")
        self.print_summary()
    
    def print_summary(self):
        """Print processing summary"""
        elapsed = time.time() - self.start_time
        total_processed = self.successful + self.failed
        
        logger.info("\n" + "="*60)
        logger.info("Processing Complete - Summary")
        logger.info("="*60)
        logger.info(f"Total time: {elapsed/60:.2f} minutes")
        logger.info(f"Total processed: {total_processed}")
        logger.info(f"Successful: {self.successful}")
        logger.info(f"Failed: {self.failed}")
        logger.info(f"Skipped: {self.skipped_processed}")
        
        if total_processed > 0:
            logger.info(f"Success rate: {self.successful/total_processed*100:.2f}%")
            logger.info(f"Average processing time: {elapsed/total_processed:.2f} sec/entry")
        
        logger.info(f"\nOutput files:")
        logger.info(f"  - Results: {self.output_file}")
        logger.info(f"  - Errors: {self.error_file}")
        logger.info(f"  - Checkpoint: {self.checkpoint_file}")


def main():
    """Main function"""
    print("="*60)
    print("Video Comparison Analysis Processing Tool")
    print("Incremental Write Mode: Enabled")
    print("="*60)
    
    # Configuration parameters
    config = {
        "input_json_file": "input_videos.json",
        "output_file": "output/analysis_results.json",
        "error_file": "output/analysis_errors.json",
        "checkpoint_file": "output/checkpoint.json",
        "max_workers": 2,
        "max_pairs": None,
        'api_delay': 2,
        'timeout': 300,
        'model': 'YOUR_MODEL_NAME',
        'api_key': "YOUR_API_KEY_HERE",
        'max_file_size_mb': 20,
        'resume_from_checkpoint': True,
        'skip_failed': False,
        'max_retries': 3
    }
    
    print(f"Configuration:")
    print(f"  - Input file: {config['input_json_file']}")
    print(f"  - Output file: {config['output_file']}")
    print(f"  - Error file: {config['error_file']}")
    print(f"  - Checkpoint file: {config['checkpoint_file']}")
    print(f"  - Model: {config['model']}")
    print(f"  - Concurrency: {config['max_workers']}")
    print("="*60)
    
    try:
        processor = VideoProcessor(config)
        processor.process_all()
        print("\n✅ Processing complete!")
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        print("\n⚠️ Processing interrupted by user")
    except Exception as e:
        logger.error(f"Program exception: {e}", exc_info=True)
        print(f"\n❌ Program exception: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
