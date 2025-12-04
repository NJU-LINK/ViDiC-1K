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

# LLaVA相关导入
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, process_images
from llava.utils import disable_torch_init
import re

# 固定的文件夹路径配置
MODEL_FOLDER = ""
OUTPUT_FOLDER = "response"
LOG_FOLDER = "logs"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def process_video_to_image_list(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """处理视频并返回图像对象列表"""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    
    image_list = []
    for frame_index in frame_indices:
        frame = vr[frame_index].asnumpy()
        pil_image = Image.fromarray(frame).convert('RGB')
        processed_images = dynamic_preprocess(
            pil_image, 
            image_size=input_size,
            use_thumbnail=True,
            max_num=max_num
        )
        image_list.extend(processed_images)
    return image_list 


class LLaVAProcessor:
    """LLaVA 视频对比处理主类"""
    
    # 添加类级别的文件锁
    _output_file_lock = Lock()
    
    def __init__(self, config):
        """初始化处理器"""
        disable_torch_init()
        
        self.model_name = config.get('model_name', 'llava-v1.6-vicuna-7b')
        self.model_path = config.get('model_path')
        self.model_path = ''
        
        self.input_json_file = config.get('input_json_file', 'input_videos.json')
        self.prompt_file = config.get('prompt_file', 'prompt_generate.txt')
        self.thinking = config.get('thinking', False)
        self.max_frames_per_video = config.get('max_frames_per_video', 32)
        
        # 设置输出文件路径
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        if self.thinking:
            self.output_file = os.path.join(OUTPUT_FOLDER, f"{self.model_name}_thinking_results.json")
        else:
            self.output_file = os.path.join(OUTPUT_FOLDER, f"{self.model_name}_nothinking_results.json")
        
        # 设置日志
        self._setup_logging()
        
        # 验证输入文件存在
        if not os.path.exists(self.input_json_file):
            raise FileNotFoundError(f"输入JSON文件不存在: {self.input_json_file}")
        
        # 验证模型路径存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
        
        # 检测 GPU 数量
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            self.logger.error("❌ 未检测到GPU，此脚本需要GPU支持")
            raise RuntimeError("需要GPU才能运行此脚本")
        
        self.logger.info(f"✅ 检测到 {gpu_count} 个GPU")
        
        # 打印GPU信息
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            self.logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        
        # 初始化 LLaVA 模型
        self.logger.info(f"正在加载 LLaVA 模型: {self.model_path}")
        self.logger.info(f"启用思考模式: {self.thinking}")
        
        try:
            self.model_path = "/mnt/shared-storage-user/colab-share/liujiaheng/pjlab-oss/models/Llava/llava-v1.6/llava-v1.6-model"
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=self.model_path,
                model_base=None,
                model_name=get_model_name_from_path(self.model_path)
            )
            if next(self.model.parameters()).device.type == 'cuda':
                self.logger.info("✅ LLaVA 模型加载成功 (GPU)")
            else:
                self.logger.warning("⚠️ 模型在CPU上运行，性能可能受影响")
        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {e}")
            raise
        
        # 视频处理配置
        self.input_size = 448
        self.fps = 1.0
        self.max_num = 1
        
        # 生成配置
        if self.thinking:
            self.temperature = 0.6
            self.max_new_tokens = 8192
        else:
            self.temperature = 0.1
            self.max_new_tokens = 2048
        
        self.logger.info(f"配置信息:")
        self.logger.info(f"  - 模型名称: {self.model_name}")
        self.logger.info(f"  - 模型路径: {self.model_path}")
        self.logger.info(f"  - 每个视频最大帧数: {self.max_frames_per_video}")
        self.logger.info(f"  - 输入文件: {self.input_json_file}")
        self.logger.info(f"  - 输出文件: {self.output_file}")
        self.logger.info(f"  - 提示词文件: {self.prompt_file}")
        self.logger.info(f"  - 思考模式: {'启用' if self.thinking else '禁用'}")
        self.logger.info(f"  - 温度: {self.temperature}")
        self.logger.info(f"  - 最大生成tokens: {self.max_new_tokens}")
        
        # 统计信息
        self.successful = 0
        self.failed = 0
        self.skipped_processed = 0
        self.start_time = None
        
        # 从输出文件加载已处理的记录
        self.processed_indices = self._load_processed_indices()
        
        # 提示词
        self.system_prompt = self._load_system_prompt()
        
        # 如果启用思考模式，添加思考提示
        if self.thinking:
            R1_SYSTEM_PROMPT = """You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within 
 tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.

"""
            self.system_prompt = R1_SYSTEM_PROMPT + self.system_prompt
        
        # 初始化或加载已有的结果文件
        self._initialize_output_file()
    
    def _setup_logging(self):
        """设置日志配置"""
        log_dir = os.path.join(LOG_FOLDER, self.model_name)
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"processing_{timestamp}.log")
        error_log_file = os.path.join(log_dir, f"errors_{timestamp}.log")
        
        self.logger = logging.getLogger(f"LLaVAProcessor_{self.model_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.error_logger = logging.getLogger(f"error_logger_{self.model_name}")
        self.error_logger.setLevel(logging.ERROR)
        self.error_logger.handlers.clear()
        
        error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
        error_handler.setFormatter(logging.Formatter('%(asctime)s - [ERROR] - %(message)s'))
        self.error_logger.addHandler(error_handler)
    
    def _load_processed_indices(self):
        """从输出文件加载已处理的索引"""
        processed = set()
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if 'index' in item:
                                processed.add(item['index'])
                        self.logger.info(f"从输出文件加载了 {len(processed)} 条已处理记录")
            except Exception as e:
                self.logger.warning(f"加载已处理记录失败: {e}")
        return processed
    
    def _initialize_output_file(self):
        """初始化输出文件"""
        with self._output_file_lock:
            if os.path.exists(self.output_file):
                try:
                    with open(self.output_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        if isinstance(existing_data, list):
                            self.logger.info(f"输出文件已存在，包含 {len(existing_data)} 条历史记录")
                        else:
                            with open(self.output_file, 'w', encoding='utf-8') as f:
                                json.dump([], f, ensure_ascii=False)
                            self.logger.info("输出文件格式错误，已重新初始化")
                except (json.JSONDecodeError, Exception) as e:
                    backup_file = f"{self.output_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    os.rename(self.output_file, backup_file)
                    self.logger.warning(f"输出文件读取失败，已备份至: {backup_file}")
                    with open(self.output_file, 'w', encoding='utf-8') as f:
                        json.dump([], f, ensure_ascii=False)
            else:
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False)
                self.logger.info("创建新的输出文件")
    
    def _append_result_to_file(self, result):
        """增量写入单个结果到文件"""
        with self._output_file_lock:
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
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
                
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                self.logger.debug(f"成功增量写入结果，当前共 {len(data)} 条记录")
                
            except Exception as e:
                self.logger.error(f"增量写入失败: {e}")
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
                    
                    self.logger.warning(f"结果已保存到备用文件: {backup_file}")
                except Exception as e2:
                    self.logger.error(f"备用文件写入也失败: {e2}")
    
    def _load_system_prompt(self):
        """加载系统提示词"""
        prompt_path = self.prompt_file
        
        if not os.path.exists(prompt_path):
            error_msg = f"❌ 错误: 提示词文件不存在: {prompt_path}"
            self.logger.error(error_msg)
            print("\n" + "="*60)
            print(error_msg)
            print("请创建提示词文件后再运行程序！")
            print("="*60)
            sys.exit(1)
        
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            
            if not prompt:
                error_msg = f"❌ 错误: 提示词文件为空: {prompt_path}"
                self.logger.error(error_msg)
                print("\n" + "="*60)
                print(error_msg)
                print("请在提示词文件中添加内容！")
                print("="*60)
                sys.exit(1)
            
            self.logger.info(f"✅ 成功加载系统提示词文件: {prompt_path}")
            self.logger.info(f"提示词长度: {len(prompt)} 字符")
            
            return prompt
            
        except Exception as e:
            error_msg = f"❌ 错误: 读取提示词文件失败: {e}"
            self.logger.error(error_msg)
            sys.exit(1)
    
    def _log_error(self, error_info):
        """记录错误信息到日志"""
        self.error_logger.error(json.dumps(error_info, ensure_ascii=False, indent=2))
    
    def load_input_data(self):
        """从JSON文件加载输入数据"""
        self.logger.info(f"开始加载输入文件: {self.input_json_file}")
        
        if not os.path.exists(self.input_json_file):
            raise FileNotFoundError(f"输入文件不存在: {self.input_json_file}")
        
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
                        self.logger.warning(f"第{idx}项缺少必要的视频路径字段")
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
            
            self.logger.info(f"✅ 成功加载 {len(data_list)} 条数据")
            return data_list
            
        except Exception as e:
            self.logger.error(f"加载输入文件失败: {e}")
            raise
    
    def generate(self, image_files, qs: str):
        """LLaVA 生成函数"""
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images = image_files
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample= False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    
    def process_video_pair(self, entry):
        """处理单个视频对"""
        video1_path = entry['video1_path']
        video2_path = entry['video2_path']
        
        if not os.path.exists(video1_path):
            raise FileNotFoundError(f"视频文件未找到: {video1_path}")
        if not os.path.exists(video2_path):
            raise FileNotFoundError(f"视频文件未找到: {video2_path}")
        
        # 分别处理两个视频，每个视频采样32帧
        frame_images1 = process_video_to_image_list(
            video1_path, 
            input_size=self.input_size,
            max_num=self.max_num,
            num_segments=self.max_frames_per_video
        )
        frame_images2 = process_video_to_image_list(
            video2_path,
            input_size=self.input_size,
            max_num=self.max_num,
            num_segments=self.max_frames_per_video
        )
        
        # 合并所有帧
        all_frames = frame_images1 + frame_images2
        full_prompt = """ROLE & TASK\nYou are an objective video analyst. Your task is to compare a Source video and a Destination video, reporting only verifiable visual facts.\nCRITICAL RULES:\nRadical Objectivity: This is your most important rule. Describe only what is visually present. Do not use subjective words (e.g., "beautiful," "sad"), make assumptions, or interpret actions. If a detail is not 100% clear, you MUST omit it.\nEnglish Only: Your entire response must be in English.\nNo Timestamps: Do not mention specific timecodes or frame numbers.\nConcise & Factual: Be direct and avoid unnecessary embellishment. Ensure similarities and differences do not contradict each other.\nOUTPUT FORMAT\nStructure your response using the following categories. For each, list Similarities and Differences. Only include categories where there are noticeable things to report.\nsubject\nstyle\nbackground\ncamera\nmotion\nposition\nplayback technique\nANALYSIS GUIDELINES\nSubject: Describe the type, quantity, appearance, and pose of subjects.\nStyle: You MUST use only these terms: American comic style, Ukiyo-e, Anime, Pixel Art, Ghibli Style, Cyberpunk, Steampunk, Low Poly, Voxel Art, Minimalist, Flat Design, Retro, Oil Painting, Watercolor, Sketch, Graffiti, Ink Wash Painting, Black and White, Monochromatic, CG Rendering, realistic (un-stylized).\nBackground: Describe the environment, lighting, and key background objects.\nCamera: Describe the camera's angle, scale (e.g., medium shot), and movement.\nMotion: Describe the subjects' actions and movements.\nPosition: Describe the relative placement of subjects in the frame.\nPlayback Technique: You MUST use only these terms: slow-motion, fast-forward, reverse, forward motion.\n"""
        full_prompt += f"Source video: first {len(frame_images1)} frames\n"
        full_prompt += f"Destination video: next {len(frame_images2)} frames after Source video\n"
        full_prompt += "Please analyze these two videos and provide a comparison."
        try:
            response = self.generate(all_frames, full_prompt)
            # 如果启用了思考模式，提取最终答案
            if self.thinking and '</think>' in response:
                response = response.split('</think>', 1)[-1].strip()
            
            return response
        except Exception as e:
            self.logger.error(f"模型推理过程中发生错误: {str(e)}")
            raise e
    
    def process_all(self):
        """处理所有数据"""
        self.start_time = time.time()
        
        data_list = self.load_input_data()
        
        if not data_list:
            self.logger.info("没有需要处理的数据")
            return
        
        pending_data = [entry for entry in data_list if entry['index'] not in self.processed_indices]
        
        if not pending_data:
            self.logger.info("✅ 所有数据已处理完成")
            return
        
        total = len(data_list)
        pending = len(pending_data)
        
        self.logger.info(f"总数据: {total} 条")
        self.logger.info(f"已处理: {len(self.processed_indices)} 条")
        self.logger.info(f"待处理: {pending} 条")
        
        self.logger.info("="*60)
        self.logger.info("开始处理 (使用 LLaVA 模型)")
        self.logger.info(f"增量写入模式: 启用")
        self.logger.info("="*60)
        
        with tqdm(total=pending, desc="处理进度") as pbar:
            for entry in pending_data:
                self.logger.info(f"\n处理视频对 {entry['index']}")
                
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
                        self.logger.info(f"[Entry {entry['index']}] ✅ 处理成功并已保存")
                        
                        success = True
                        pbar.update(1)
                        
                    except Exception as e:
                        self.logger.error(f"处理错误: {str(e)}")
                        self.error_logger.error(f"处理错误详情: {traceback.format_exc()}")
                        retry_count += 1
                        
                        if retry_count < max_retries:
                            self.logger.warning(f"正在重试 {retry_count}/{max_retries}...")
                            time.sleep(2)
                        else:
                            self.failed += 1
                            self.logger.error(f"[Entry {entry['index']}] ❌ 处理失败")
                            
                            error_info = {
                                "index": entry['index'],
                                "video1_path": entry['video1_path'],
                                "video2_path": entry['video2_path'],
                                "error": str(e),
                                "timestamp": datetime.now().isoformat()
                            }
                            self._log_error(error_info)
                            pbar.update(1)
        
        self.logger.info(f"✅ 所有结果已保存到: {self.output_file}")
        self.print_summary()
    
    def print_summary(self):
        """打印处理摘要"""
        elapsed = time.time() - self.start_time
        total_processed = self.successful + self.failed
        
        self.logger.info("\n" + "="*60)
        self.logger.info("处理完成 - 统计摘要")
        self.logger.info("="*60)
        self.logger.info(f"总耗时: {elapsed/60:.2f} 分钟")
        self.logger.info(f"处理总数: {total_processed}")
        self.logger.info(f"成功: {self.successful}")
        self.logger.info(f"失败: {self.failed}")
        self.logger.info(f"跳过: {self.skipped_processed}")
        
        if total_processed > 0:
            self.logger.info(f"成功率: {self.successful/total_processed*100:.2f}%")
            self.logger.info(f"平均处理时间: {elapsed/total_processed:.2f} 秒/条")
        
        self.logger.info(f"\n输出文件: {self.output_file}")
        self.logger.info(f"日志目录: {os.path.join(LOG_FOLDER, self.model_name)}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='LLaVA 视频对比分析处理程序',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model_name', type=str, default='llava-v1.6-vicuna-7b',
                       help='模型名称')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型路径（如不指定则使用默认路径）')
    parser.add_argument('--input_json', type=str, default='videos.json', 
                       help='输入JSON文件路径')
    parser.add_argument('--prompt_file', type=str, default='prompt_generate.txt', 
                       help='系统提示词文件路径')
    parser.add_argument('--max_frames', type=int, default=16, 
                       help='每个视频最大帧数')
    parser.add_argument('-t', '--thinking', action='store_true',
                       help='启用思考模式')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("="*60)
    print("LLaVA 视频对比分析处理程序")
    print("增量写入模式：已启用")
    print("文件锁保护：已启用")
    print("="*60)
    
    config = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "input_json_file": args.input_json,
        "prompt_file": args.prompt_file,
        "max_frames_per_video": args.max_frames,
        "thinking": args.thinking,
    }
    
    print(f"配置信息:")
    print(f"  - 模型名称: {config['model_name']}")
    print(f"  - 输入文件: {config['input_json_file']}")
    
    # 根据思考模式选择输出文件名
    suffix = "_thinking_results.json" if config['thinking'] else "_nothinking_results.json"
    print(f"  - 输出文件: {os.path.join(OUTPUT_FOLDER, config['model_name'] + suffix)}")
    print(f"  - 日志目录: {os.path.join(LOG_FOLDER, config['model_name'])}")
    print(f"  - 每个视频最大帧数: {config['max_frames_per_video']}")
    print(f"  - 思考模式: {'启用' if config['thinking'] else '禁用'}")
    print("="*60)
    
    try:
        processor = LLaVAProcessor(config)
        processor.process_all()
        print("\n✅ 处理完成!")
    except KeyboardInterrupt:
        print("\n⚠️ 处理被用户中断")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
