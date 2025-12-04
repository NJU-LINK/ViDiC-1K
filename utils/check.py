import json
import io
import sys

# 保证输出 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_checklist(json_path):
    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_questions = 0
    simi_questions = 0
    difi_questions = 0

    simi_incorrect = []  # 存储不是 "no" 的位置
    difi_incorrect = []  # 存储不是 "yes" 的位置

    for idx, item in enumerate(data):
        structured = item.get("structured_analysis", {})
        similarities = structured.get("Similarities", [])
        differences = structured.get("Differences", [])

        simi_questions += len(similarities)
        difi_questions += len(differences)
        total_questions += len(similarities) + len(differences)

        # 检查 Similarities
        for q_idx, q in enumerate(similarities):
            ans = q.get("correct_answer", "").lower()
            if ans != "no":
                simi_incorrect.append({"data_index": idx, "question_index": q_idx, "answer": ans})

        # 检查 Differences
        for q_idx, q in enumerate(differences):
            ans = q.get("correct_answer", "").lower()
            if ans != "yes":
                difi_incorrect.append({"data_index": idx, "question_index": q_idx, "answer": ans})

    # 输出结果
    print(f"总问题数: {total_questions}")
    print(f"Similarities 问题数: {simi_questions}")
    print(f"Differences 问题数: {difi_questions}")

    if simi_incorrect:
        print(f"Similarities 中答案不是 'no' 的位置和答案: {simi_incorrect}")
    else:
        print("Similarities 答案全部为 'no'")

    if difi_incorrect:
        print(f"Differences 中答案不是 'yes' 的位置和答案: {difi_incorrect}")
    else:
        print("Differences 答案全部为 'yes'")

if __name__ == "__main__":
    json_file = r"C:\Users\wjt11\Desktop\checklist_final.json"
    check_checklist(json_file)
