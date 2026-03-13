import requests
import os
import json
import time
import traceback
import re
import threading
from threading import Lock, Thread, Semaphore
from queue import Queue
from datetime import datetime
import sys
import io

sys.stdout.reconfigure(encoding='utf-8')

API_KEY = ""
MODEL_URL = ""
MODEL_NAME = "gpt-5-mini"

file_lock = Lock()
console_lock = Lock()


def safe_print(msg):
    with console_lock:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")


def safe_post_json(url, headers, payload, max_retries=3, timeout=120):
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                wait_time = min(2 ** attempt * 5, 60)
                safe_print(f"⚠️  Rate limited, waiting {wait_time}s")
                time.sleep(wait_time)
            else:
                safe_print(f"❌ Request failed ({response.status_code}): {response.text[:100]}")
        except requests.exceptions.Timeout:
            safe_print(f"⚠️  Timeout on attempt {attempt}")
        except Exception as e:
            safe_print(f"⚠️  Exception on attempt {attempt}: {e}")
        
        if attempt < max_retries:
            time.sleep(min(2 ** attempt, 10))
    
    raise RuntimeError("Request failed after max retries")


def parse_judge_response(raw_text):
    if raw_text is None:
        return {"answer": "error", "explanation": "Empty response"}
    
    if isinstance(raw_text, dict):
        return {
            "answer": raw_text.get("answer", "error"),
            "explanation": raw_text.get("explanation", "")
        }
    
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)
    
    # Try to extract JSON code block
    match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            return {
                "answer": parsed.get("answer", "error"),
                "explanation": parsed.get("explanation", "")
            }
        except:
            pass
    
    # Try direct JSON parsing
    try:
        parsed = json.loads(raw_text)
        return {
            "answer": parsed.get("answer", "error"),
            "explanation": parsed.get("explanation", "")
        }
    except:
        pass
    
    # Regex extraction
    answer_match = re.search(r"answer\s*[:：]\s*(yes|no|unsure)", raw_text, re.I)
    reason_match = re.search(r"(explanation|because|reason)[:：]?\s*(.*)", raw_text, re.I | re.DOTALL)
    
    return {
        "answer": answer_match.group(1).lower() if answer_match else "error",
        "explanation": reason_match.group(2).strip() if reason_match else raw_text.strip()
    }


# ==================== File Operations ====================
def save_result(data, output_path):
    """Thread-safe save (caller must already hold lock)"""
    abs_path = os.path.abspath(output_path)
    temp_path = f"{output_path}.tmp"
    
    try:
        # Ensure directory exists
        dir_path = os.path.dirname(abs_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Write to temporary file
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        temp_size = os.path.getsize(temp_path)
        safe_print(f"💾 Temp file: {temp_size} bytes")
        
        # Atomic replace
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_path, output_path)
        
        # Verify
        if os.path.exists(output_path):
            final_size = os.path.getsize(output_path)
            safe_print(f"✅ Save successful: {final_size} bytes")
            return True
        else:
            safe_print(f"❌ Save failed: file does not exist")
            return False
            
    except Exception as e:
        safe_print(f"❌ Save exception: {e}")
        traceback.print_exc()
        return False


def load_progress(progress_file):
    """Load checkpoint progress"""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                safe_print(f"📂 Loaded progress: {len(data)} objects completed")
                return set(data)
        except Exception as e:
            safe_print(f"⚠️  Progress file corrupted: {e}")
    return set()


def save_progress(completed_objects, progress_file):
    """Save checkpoint progress"""
    try:
        dir_path = os.path.dirname(progress_file)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        temp_path = f"{progress_file}.tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(sorted(list(completed_objects)), f)
        
        if os.path.exists(progress_file):
            os.remove(progress_file)
        os.rename(temp_path, progress_file)
        
    except Exception as e:
        safe_print(f"⚠️  Progress save failed: {e}")


def load_clean_data(structured_path):
    """Load raw data and clear old results"""
    with open(structured_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cleaned_count = 0
    for item in data:
        for section in ["Similarities", "Differences"]:
            for question in item["structured_analysis"].get(section, []):
                if question.pop("answer", None) is not None:
                    cleaned_count += 1
                question.pop("explanation", None)
    
    if cleaned_count > 0:
        safe_print(f"🧹 Cleaned {cleaned_count} old results")
    
    return data


def restore_from_output(checklist_data, output_path, completed_objects):
    """Restore completed results from output file"""
    if not os.path.exists(output_path) or not completed_objects:
        return
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        restored_count = 0
        for item_index in completed_objects:
            if item_index < len(saved_data):
                checklist_data[item_index] = saved_data[item_index]
                for section in ["Similarities", "Differences"]:
                    for q in saved_data[item_index]["structured_analysis"].get(section, []):
                        if "answer" in q:
                            restored_count += 1
        
        safe_print(f"♻️  Restored {len(completed_objects)} objects, {restored_count} questions")
        
    except Exception as e:
        safe_print(f"⚠️  Restore failed: {e}")


# ==================== Core Processing Logic ====================
def process_question(question_text, model_description, section, headers, prompt_template):
    """Process a single question"""
    full_prompt = prompt_template.format(
        description=model_description,
        question=question_text
    )
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": f"You are a rigorous video analysis assistant. This is a {section} problem."
            },
            {
                "role": "user",
                "content": full_prompt
            }
        ]
    }
    
    try:
        response = safe_post_json(MODEL_URL, headers, payload)
        result = response.json()
        message_text = result["choices"][0]["message"]["content"]
        parsed = parse_judge_response(message_text)
        
        return {
            "answer": parsed.get("answer", "error"),
            "explanation": parsed.get("explanation", message_text.strip())
        }
    except Exception as e:
        return {
            "answer": "error",
            "explanation": f"Processing failed: {str(e)}"
        }


def process_object(item_index, item, response_dict, headers, prompt_template, semaphore):
    """Process all questions for one object"""
    key = (item["video1_path"], item["video2_path"])
    
    if key not in response_dict:
        safe_print(f"⚠️  [Object{item_index}] No matching description")
        model_description = ""
        has_description = False
    else:
        model_description = response_dict[key]
        has_description = True
    
    video_pair = f"{os.path.basename(item['video1_path'])} vs {os.path.basename(item['video2_path'])}"
    total_questions = sum(len(item["structured_analysis"].get(s, [])) for s in ["Similarities", "Differences"])
    
    safe_print(f"🔹 [Object{item_index}] Starting: {video_pair} ({total_questions} questions)")
    
    completed_questions = 0
    
    for section in ["Similarities", "Differences"]:
        for question in item["structured_analysis"].get(section, []):
            question_text = question.get("question", "")
            
            if not has_description:
                result = {
                    "answer": "error",
                    "explanation": "No model description"
                }
            else:
                with semaphore:
                    result = process_question(
                        question_text,
                        model_description,
                        section,
                        headers,
                        prompt_template
                    )
            
            question.update(result)
            completed_questions += 1
    
    safe_print(f"✅ [Object{item_index}] Completed: {video_pair} ({completed_questions}/{total_questions})")
    return True


def worker_thread(task_queue, checklist_data, response_dict, headers, prompt_template,
                  output_path, progress_file, completed_objects, semaphore):
    """Worker thread"""
    thread_name = threading.current_thread().name
    
    while True:
        try:
            item_index = task_queue.get(timeout=1)
        except:
            continue
        
        if item_index is None:
            safe_print(f"🛑 [{thread_name}] Received stop signal")
            task_queue.task_done()
            break
        
        try:
            item = checklist_data[item_index]
            
            # Process object
            process_object(item_index, item, response_dict, headers, prompt_template, semaphore)
            
            # Save immediately
            safe_print(f"💾 [{thread_name}] [Object{item_index}] Preparing to save")
            
            with file_lock:
                completed_objects.add(item_index)
                
                if save_result(checklist_data, output_path):
                    save_progress(completed_objects, progress_file)
                    safe_print(f"📊 Progress: {len(completed_objects)}/{len(checklist_data)}")
                else:
                    safe_print(f"❌ [{thread_name}] [Object{item_index}] Save failed")
                    completed_objects.discard(item_index)
            
        except Exception as e:
            safe_print(f"❌ [{thread_name}] [Object{item_index}] Exception: {e}")
            traceback.print_exc()
            
            # Mark as error and save
            with file_lock:
                try:
                    for section in ["Similarities", "Differences"]:
                        for question in item["structured_analysis"].get(section, []):
                            question.update({
                                "answer": "error",
                                "explanation": f"Exception: {str(e)}"
                            })
                    completed_objects.add(item_index)
                    save_result(checklist_data, output_path)
                    save_progress(completed_objects, progress_file)
                except Exception as save_err:
                    safe_print(f"❌ [{thread_name}] Error state save failed: {save_err}")
        
        finally:
            task_queue.task_done()
    
    safe_print(f"👋 [{thread_name}] Thread exiting")


# ==================== Main Function ====================
def judge_checklist_multithreaded(
    prompt_template,
    structured_path,
    response_path,
    output_path,
    num_threads=13,
    max_concurrent_requests=10
):
    """Multithreaded judgment main function"""
    
    safe_print("=" * 60)
    safe_print("🚀 Starting multithreaded judgment task")
    safe_print("=" * 60)
    
    progress_file = f"{output_path}.progress"
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_print(f"📁 Output directory: {os.path.abspath(output_dir)}")
    
    # Pre-create empty output file
    if not os.path.exists(output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([], f)
        safe_print(f"📄 Created empty output file")
    
    # Load data
    safe_print(f"📂 Loading raw data: {structured_path}")
    checklist_data = load_clean_data(structured_path)
    total_objects = len(checklist_data)
    safe_print(f"✅ Loaded {total_objects} objects")
    
    # Load progress
    completed_objects = load_progress(progress_file)
    
    # Restore data
    if completed_objects:
        restore_from_output(checklist_data, output_path, completed_objects)
    
    # Load model descriptions
    safe_print(f"📂 Loading model descriptions: {response_path}")
    with open(response_path, 'r', encoding='utf-8') as f:
        model_data = json.load(f)
    
    response_dict = {}
    for item in model_data:
        key = (item["video1_path"], item["video2_path"])
        response_text = item.get("response", "")
        if isinstance(response_text, dict):
            response_text = json.dumps(response_text, ensure_ascii=False)
        response_dict[key] = str(response_text)
    
    safe_print(f"✅ Loaded {len(response_dict)} descriptions")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # Create task queue
    task_queue = Queue()
    pending_objects = []
    
    for item_index in range(total_objects):
        if item_index not in completed_objects:
            task_queue.put(item_index)
            pending_objects.append(item_index)
    
    safe_print("=" * 60)
    safe_print(f"📊 Total objects: {total_objects}")
    safe_print(f"📊 Completed: {len(completed_objects)}")
    safe_print(f"📊 Pending: {len(pending_objects)}")
    safe_print("=" * 60)
    
    if not pending_objects:
        safe_print("✅ All objects completed")
        return
    
    # Start threads
    safe_print(f"🚀 Starting {num_threads} threads")
    safe_print(f"⚡ Max concurrent: {max_concurrent_requests}")
    
    semaphore = Semaphore(max_concurrent_requests)
    threads = []
    start_time = time.time()
    
    for i in range(num_threads):
        t = Thread(
            target=worker_thread,
            args=(task_queue, checklist_data, response_dict, headers, prompt_template,
                  output_path, progress_file, completed_objects, semaphore),
            name=f"Worker-{i+1}"
        )
        t.daemon = False
        t.start()
        threads.append(t)
    
    safe_print(f"✅ {num_threads} threads started")
    
    # Wait for tasks to complete
    safe_print("⏳ Waiting for tasks to complete...")
    task_queue.join()
    safe_print("✅ All tasks completed")
    
    # Send stop signal
    safe_print("🛑 Sending stop signals...")
    for _ in range(num_threads):
        task_queue.put(None)
    
    # Wait for threads to exit
    for t in threads:
        t.join(timeout=10)
        if t.is_alive():
            safe_print(f"⚠️  {t.name} did not exit normally")
    
    # Final save
    safe_print("💾 Performing final save...")
    with file_lock:
        save_result(checklist_data, output_path)
    
    elapsed_time = time.time() - start_time
    
    safe_print("=" * 60)
    safe_print("🎉 Task completed")
    safe_print(f"   Processed objects: {len(completed_objects)}")
    safe_print(f"   Total time: {elapsed_time:.1f}s")
    safe_print(f"   Output file: {os.path.abspath(output_path)}")
    
    # Verify file
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / 1024
        safe_print(f"✅ File verification passed ({file_size:.1f} KB)")
    else:
        safe_print(f"❌ Output file does not exist")
    
    safe_print("=" * 60)
    
    # Clean up progress file
    if os.path.exists(progress_file):
        try:
            os.remove(progress_file)
            safe_print(f"🗑️  Cleaned checkpoint file")
        except:
            pass


# ==================== Entry Point ====================
if __name__ == "__main__":
    PROMPT_TEMPLATE = (
        "Based on the description generated by the model, determine whether the answer to the following question should be \"yes\" or \"no\", and provide a brief reason.\n"
        "**Judgment Principles**:\n"
        "1. **Default to Same**: Unless the description explicitly states that there is a difference, you must default to considering it as the same, and answer the question based on this assumption.\n"
        "2. **Validating Differences**: To conclude that something is different, rely on explicit content or reasonable logical inference. Strictly avoid over-interpretation.\n"
        "3. **Handling Generalizations**: If the question uses broad or general adjectives (e.g., \"general\", \"overall\"), focus on the holistic content and main idea rather than specific details or minor discrepancies.\n"
        "Output format is a JSON object: {{\"answer\": \"yes/no\", \"explanation\": \"reason\"}}\n\n"
        "【Model Description】\n{description}\n\n"
        "【Question】\n{question}\n"
    )

    STRUCTURED_JSON = "checklist.json"
    MODEL_RESPONSE_JSON = "model_responses.json"
    OUTPUT_JSON = "judgment_results.json"

    try:
        judge_checklist_multithreaded(
            PROMPT_TEMPLATE,
            STRUCTURED_JSON,
            MODEL_RESPONSE_JSON,
            OUTPUT_JSON,
            num_threads=5,
            max_concurrent_requests=5
        )
    except KeyboardInterrupt:
        safe_print("\n⚠️  User interrupted")
    except Exception as e:
        safe_print(f"\n❌ Abnormal exit: {e}")
        traceback.print_exc()