import json
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def calculate_global_scores(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_questions = 0
    total_correct = 0

    total_similarity_questions = 0
    total_similarity_correct = 0

    total_difference_questions = 0
    total_difference_correct = 0

    overall_class_total = {}
    overall_class_correct = {}

    for item in data:
        structured = item.get("structured_analysis", {})
        similarities = structured.get("Similarities", [])
        differences = structured.get("Differences", [])

        for q in similarities:
            cls = q.get("class", "unknown")
            total_similarity_questions += 1
            total_questions += 1
            overall_class_total[cls] = overall_class_total.get(cls, 0) + 1

            if q.get("answer", "").lower() == q.get("correct_answer", "").lower():
                total_similarity_correct += 1
                total_correct += 1
                overall_class_correct[cls] = overall_class_correct.get(cls, 0) + 1

        for q in differences:
            cls = q.get("class", "unknown")
            total_difference_questions += 1
            total_questions += 1
            overall_class_total[cls] = overall_class_total.get(cls, 0) + 1

            if q.get("answer", "").lower() == q.get("correct_answer", "").lower():
                total_difference_correct += 1
                total_correct += 1
                overall_class_correct[cls] = overall_class_correct.get(cls, 0) + 1

    total_score = total_correct / total_questions if total_questions > 0 else 0
    similarity_score = total_similarity_correct / total_similarity_questions if total_similarity_questions > 0 else 0
    difference_score = total_difference_correct / total_difference_questions if total_difference_questions > 0 else 0

    print(f"Total questions: {total_questions}")
    print(f"Total correct: {total_correct}")
    print(f"Overall total score: {total_score:.3f}")
    print(f"Overall similarity score: {similarity_score:.3f} ({total_similarity_correct}/{total_similarity_questions})")
    print(f"Overall difference score: {difference_score:.3f} ({total_difference_correct}/{total_difference_questions})")

    print("\nOverall class scores:")
    for cls in overall_class_total:
        correct = overall_class_correct.get(cls, 0)
        total = overall_class_total[cls]
        score = correct / total if total > 0 else 0
        print(f"  Class '{cls}': {score:.3f} ({correct}/{total})")


if __name__ == "__main__":
    json_file = r"D:\video_edit\temp_work\glm.json"
    calculate_global_scores(json_file)
