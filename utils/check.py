import json
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_checklist(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_questions = 0
    simi_questions = 0
    difi_questions = 0

    simi_incorrect = []  
    difi_incorrect = []

    for idx, item in enumerate(data):
        structured = item.get("structured_analysis", {})
        similarities = structured.get("Similarities", [])
        differences = structured.get("Differences", [])

        simi_questions += len(similarities)
        difi_questions += len(differences)
        total_questions += len(similarities) + len(differences)

        # Check Similarities
        for q_idx, q in enumerate(similarities):
            ans = q.get("correct_answer", "").lower()
            if ans != "no":
                simi_incorrect.append({"data_index": idx, "question_index": q_idx, "answer": ans})

        # Check Differences
        for q_idx, q in enumerate(differences):
            ans = q.get("correct_answer", "").lower()
            if ans != "yes":
                difi_incorrect.append({"data_index": idx, "question_index": q_idx, "answer": ans})

    # Output results
    print(f"Total questions: {total_questions}")
    print(f"Similarities questions: {simi_questions}")
    print(f"Differences questions: {difi_questions}")

    if simi_incorrect:
        print(f"Similarities with answers not 'no': {simi_incorrect}")
    else:
        print("All Similarities answers are 'no'")

    if difi_incorrect:
        print(f"Differences with answers not 'yes': {difi_incorrect}")
    else:
        print("All Differences answers are 'yes'")

if __name__ == "__main__":
    json_file = r"C:\Users\wjt11\Desktop\ViDiC\ViDiC-1K\checklist\checklist_final.json"
    check_checklist(json_file)
