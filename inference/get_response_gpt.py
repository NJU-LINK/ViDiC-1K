import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import logging
from datetime import datetime
import traceback
import base64
import cv2

sys.stdout.reconfigure(encoding='utf-8')

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

try:
    from openai import OpenAI
except ImportError:
    print("Please install: pip install openai")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("Please install: pip install opencv-python")
    sys.exit(1)

file_lock = Lock()
api_lock = Lock()


class VideoProcessor:
    def __init__(self, config):
        self.input_json_file = config.get('input_json_file', 'input_videos.json') 
        self.output_file = config.get('output_file', 'video_analysis_results.json')
        self.error_file = config.get('error_file', 'video_analysis_errors.json')
        self.checkpoint_file = config.get('checkpoint_file', 'processing_checkpoint.json')
        self.max_workers = config.get('max_workers', 2)
        self.max_pairs = config.get('max_pairs', None)
        self.api_delay = config.get('api_delay', 2)
        self.timeout = config.get('timeout', 300)
        self.model = config.get('model', 'gpt-4o-mini')
        self.api_key = config.get('api_key')
        self.resume_from_checkpoint = config.get('resume_from_checkpoint', True)
        self.max_retries = config.get('max_retries', 3)
        self.frame_interval_seconds = config.get('frame_interval_seconds', 0.5)
        self.max_frames_per_video = config.get('max_frames_per_video', 20)
        self.frame_quality = config.get('frame_quality', 85)
        self.max_frame_width = config.get('max_frame_width', 768)
        self.temp_frame_dir = config.get('temp_frame_dir', 'temp_frames')
        
        if not os.path.exists(self.temp_frame_dir):
            os.makedirs(self.temp_frame_dir)
        
        for file_path in [self.output_file, self.error_file, self.checkpoint_file]:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
        
        self.client = OpenAI(api_key=self.api_key)
        
        logger.info(f"="*80)
        logger.info(f"Configuration: Model={self.model}, FPS={1/self.frame_interval_seconds:.1f}")
        logger.info(f"="*80)
        
        self.successful = 0
        self.failed = 0
        self.skipped_processed = 0
        self.start_time = None
        
        self.checkpoint_data = self._load_checkpoint()
        self.processed_indices = set(self.checkpoint_data.get('successful_indices', []))
        self.system_prompt = self._load_system_prompt()
        self._initialize_output_file()
    
    def extract_frames_from_video(self, video_path):
        """Extract frames from video file"""
        frames = []
        temp_paths = []
        cap = None
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video: {os.path.basename(video_path)} - {total_frames} frames, {fps:.1f}fps, {duration:.1f}s")
            
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
                
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                temp_filename = f"{video_name}_f{i:03d}.jpg"
                temp_path = os.path.join(self.temp_frame_dir, temp_filename)
                
                cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.frame_quality])
                temp_paths.append(temp_path)
                
                with open(temp_path, 'rb') as f:
                    frame_base64 = base64.b64encode(f.read()).decode('utf-8')
                    frames.append(frame_base64)
            
            cap.release()
            logger.info(f"✅ Extraction completed: {len(frames)} frames")
            return frames, temp_paths
            
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            if cap:
                cap.release()
            for path in temp_paths:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            raise
    
    def prepare_frames_for_api(self, frames_base64_list, video_label=""):
        """Prepare frames for API request"""
        content = []
        if video_label:
            content.append({"type": "text", "text": f"\n{video_label} ({len(frames_base64_list)} frames):"})
        
        for frame_base64 in frames_base64_list:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}", "detail": "low"}
            })
        return content
    
    def cleanup_temp_frames(self, temp_paths):
        """Clean up temporary frame files"""
        for path in temp_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass
    
    def process_single_entry(self, entry):
        """Process a single video pair entry"""
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
        temp_paths_video1 = []
        temp_paths_video2 = []
        
        while retry_count < self.max_retries:
            try:
                for video_path in [video1_path, video2_path]:
                    if not os.path.exists(video_path):
                        raise FileNotFoundError(f"Video not found: {video_path}")
                
                logger.info(f"[Entry {index}] Extracting frames from video 1...")
                frames_video1, temp_paths_video1 = self.extract_frames_from_video(video1_path)
                
                logger.info(f"[Entry {index}] Extracting frames from video 2...")
                frames_video2, temp_paths_video2 = self.extract_frames_from_video(video2_path)
                
                content_video1 = self.prepare_frames_for_api(frames_video1, "Video A")
                content_video2 = self.prepare_frames_for_api(frames_video2, "Video B")

                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Video A:"},
                            *content_video1,
                            {"type": "text", "text": "Video B:"},
                            *content_video2,
                        ]
                    }
                ]
                
                with api_lock:
                    logger.info(f"[Entry {index}] Calling API...")
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=4096,
                        temperature=0.7
                    )
                    
                    if not response or not response.choices:
                        raise ValueError("API response is empty")
                    
                    response_content = response.choices[0].message.content
                    actual_tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0
                    time.sleep(self.api_delay)
                
                self.cleanup_temp_frames(temp_paths_video1)
                self.cleanup_temp_frames(temp_paths_video2)
                
                result = {
                    "index": index,
                    "video1_path": video1_path,
                    "video2_path": video2_path,
                    "frames_extracted": {"video1": len(frames_video1), "video2": len(frames_video2)},
                    "response": response_content,
                    "tokens_used": actual_tokens,
                    "timestamp": datetime.now().isoformat()
                }
                
                self._append_result_to_file(result)
                self._save_checkpoint(index, success=True)
                self.successful += 1
                logger.info(f"[Entry {index}] ✅ Success")
                return result
                
            except Exception as e:
                last_error = e
                retry_count += 1
                self.cleanup_temp_frames(temp_paths_video1)
                self.cleanup_temp_frames(temp_paths_video2)
                logger.error(f"[Entry {index}] ❌ Attempt {retry_count} failed: {e}")
                if retry_count < self.max_retries:
                    time.sleep(retry_count * 3)
        
        self.failed += 1
        error_info = {
            "index": index,
            "video1_path": video1_path,
            "video2_path": video2_path,
            "error": str(last_error),
            "timestamp": datetime.now().isoformat()
        }
        self._append_error_to_file(error_info)
        return None
    
    def _initialize_output_file(self):
        """Initialize output file"""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        logger.info(f"Output file already exists with {len(data)} entries")
                        return
            except:
                pass
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False)
    
    def _append_result_to_file(self, result):
        """Append result to output file"""
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
                logger.error(f"Failed to write result: {e}")
    
    def _load_system_prompt(self):
        """Load system prompt from file"""
        prompt_path = "prompt_generate.txt"
        if not os.path.exists(prompt_path):
            default_prompt = "Analyze two videos and compare their differences."
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(default_prompt)
            return default_prompt
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    
    def _load_checkpoint(self):
        """Load checkpoint data"""
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
        """Save checkpoint data"""
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
        """Append error information to error file"""
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
        """Load input data from JSON file"""
        if not os.path.exists(self.input_json_file):
            raise FileNotFoundError(f"Input file not found: {self.input_json_file}")
        
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
        """Run the batch processing"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting batch processing")
        logger.info(f"{'='*80}\n")
        
        self.start_time = time.time()
        
        try:
            data_list = self.load_input_data()
            if not data_list:
                logger.error("No data to process")
                return
            
            total_pairs = len(data_list)
            logger.info(f"Total {total_pairs} video pairs to process\n")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.process_single_entry, entry): entry 
                    for entry in data_list
                }
                for future in as_completed(futures):
                    try:
                        future.result(timeout=self.timeout)
                    except:
                        pass
            
            elapsed = time.time() - self.start_time
            logger.info(f"\n{'='*80}")
            logger.info(f"✅ Processing completed!")
            logger.info(f"Total: {total_pairs}, Success: {self.successful}, Failed: {self.failed}")
            logger.info(f"Time elapsed: {elapsed:.2f} seconds")
            logger.info(f"{'='*80}\n")
            
        except KeyboardInterrupt:
            logger.warning(f"\nInterrupted by user")
        except Exception as e:
            logger.error(f"\nError occurred: {e}")
        finally:
            if os.path.exists(self.temp_frame_dir):
                try:
                    for file in os.listdir(self.temp_frame_dir):
                        try:
                            os.remove(os.path.join(self.temp_frame_dir, file))
                        except:
                            pass
                except:
                    pass


def main():
    config = {
        'input_json_file': 'input_videos.json',
        'output_file': 'output/response.json',
        'error_file': 'output/errors.json',
        'checkpoint_file': 'output/checkpoint.json',
        'max_workers': 3,
        'max_pairs': None,
        'api_delay': 2,
        'timeout': 300,
        'model': 'gpt-4o',
        'api_key': "YOUR_API_KEY_HERE",
        'resume_from_checkpoint': True,
        'max_retries': 3,
        'frame_interval_seconds': 0.5,
        'max_frames_per_video': 35,
        'frame_quality': 90,
        'max_frame_width': 4096,
        'temp_frame_dir': 'temp_frames'
    }
    
    processor = VideoProcessor(config)
    processor.run()


if __name__ == "__main__":
    main()
