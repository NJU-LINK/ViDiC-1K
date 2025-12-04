import json

input_path = r"C:\Users\wjt11\Desktop\ViDiC\ViDiC-1K\response\example_response.json"
output_path = r"C:\Users\wjt11\Desktop\ViDiC\ViDiC-1K\response\example_response_sorted.json"

# 读取文件
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Step 1: 按原始 index 排序
data_sorted = sorted(data, key=lambda x: x["index"])

# Step 2: 重新编号 index 从 1 到 N（最多 1000）
for new_idx, item in enumerate(data_sorted, start=1):
    item["index"] = new_idx
    if new_idx >= 1000:
        break

# Step 3: 若原列表超过1000项，则截断到 1000
data_sorted = data_sorted[:1000]

# Step 4: 保存
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data_sorted, f, indent=4, ensure_ascii=False)

print("完成：列表已排序并重新编号为 1-1000。")
