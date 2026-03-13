<div align="center">
  <h1>
  <img src="assets/logo.png" width="200" alt="ViDiC Logo" style="vertical-align: middle; margin-bottom: 10px;"><br>
    ViDiC: 视频差异描述 (Video Difference Captioning)
  </h1>
  
  <p align="center">
    <a href="https://github.com/NJU-LINK/ViDiC-1K"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
    <a href="https://arxiv.org/abs/2512.03405"><img src="https://img.shields.io/badge/arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
    <a href="https://vidic-1k.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge" alt="Project Page"></a>
    <a href="https://huggingface.co/datasets/NJU-LINK/ViDiC-1K"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow?style=for-the-badge" alt="Dataset"></a>
  </p>

  <p align="center">
    <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
  </p>
</div>

---

## 📋 摘要

理解动态场景之间的视觉差异，需要对构成要素、空间位置以及时间维度的变化进行对比感知——这在现有的视觉-语言系统中仍是一个未被充分探索的能力。虽然先前的图像差异描述 (Image Difference Captioning, IDC) 工作使模型能够描述静态图像之间的语义变化，但这些方法无法捕捉动作的连续性、事件的演变或随时间变化的编辑一致性。

为了解决这个问题，我们推出了 **ViDiC (Video Difference Captioning)**，这是一项将差异描述扩展到视频领域的新任务。我们要介绍了 **ViDiC-1K** 基准测试，旨在评估多模态大语言模型 (MLLMs) 对视频对之间的相似点和不同点进行细粒度描述的能力。这一设定超越了传统的视频相似度或视频编辑指标，将重点放在了**对编辑的理解**而非编辑的执行上。

<p align="center">
  <img src="assets/page.png" width="800" alt="ViDiC 任务概览">
  <br>
  <em>图 1: ViDiC 任务示意图。模型必须生成涵盖七个类别的相似性和差异性详细描述，并对照细粒度的检查清单进行评估。</em>
</p>

## 🌟 主要特性

- **🎥 首个视频差异描述基准**: 一个统一的任务，要求对视频对进行描述性、对比性和时间性的理解。
- **📝 ViDiC-1K 数据集**: 精选的 1,000 个视频对，包含 3720 个用于对比的检查清单条目 (checklist items)。
- **🔍 双重清单评估 (Dual-Checklist Evaluation)**: 一个严格的评估框架，分别评估**相似性** (检查是否产生幻觉) 和**差异性** (检查感知能力)。
- **🤖 可扩展的大模型裁判 (LLM-as-a-Judge)**: 一个自动化、可解释的评估协议，使用 GPT-5-Mini 对照经过人工验证的真值 (Standard Ground Truth) 来量化事实准确性。

## 📈 基准统计

<p align="center">
  <img src="assets/stats.png" width="800" alt="Dataset Statistics">
</p>

- **视频对总数**: 1,000 (真实场景 & 合成数据)
- **评估维度**: 7 个类别 (主体、风格、背景、运镜、动作、位置、播放技术)
- **视频时长**: 主要集中在 2-12 秒
- **数据来源**: 筛选自 8 个以上的公共数据集 (如 VidDiffBench, LMArena) 以及自生成的合成数据 (Veo3 + 帧拼接)。

## 📰 最新动态
- 🤗 ViDiC-1K 数据集已经在 Hugging Face 上线。
- 🚀 评估代码和排行榜已发布。

## 📂 文件结构
```
ViDiC/
├── assets/ # Images for README
│   ├── page.png
│   └── stats.png
│
├── checklist/  # The annotion file
│   └── checklist.json
│
├── data/   # Video files  Get from the hugging face
|
├── inference/   # Inference scripts for popular models
│   ├── get_response_GLM.py
│   ├── get_response_gemini.py
|   └── ......
|
├── judge/   # judge with gpt5-mini
│   ├── judge.py
│
├── prompt/
│   ├── prompt_get_response.txt
│   └── prompt_judge.txt
│
├── response/
│   └── example_response.json
│
└── utils/
    └── calculate.py # get the score
```

## 📊 基准测试结果
### 总体模型表现

| Model | Param. | Avg. | Diff. | Sim. | Subject | Motion | Pos. | Backgr. | Cam. | Style | Tech. |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Human** 🧠 | - | 94.46 | 92.99 | 98.57 | 96.36 | 94.36 | 90.14 | 96.70 | 92.90 | 82.31 | 97.18 |
| **_Closed-Source_** |
| Gemini-2.5-Pro | 🔒 | **69.33** | **66.84** | 76.27 | **71.95** | **61.71** | **70.42** | **75.47** | 60.41 | 79.27 | **66.20** |
| Gemini-3.0-Flash | 🔒 | 65.81 | 60.04 | 81.87 | 66.17 | 57.78 | 68.31 | 69.31 | **63.88** | 77.44 | 61.97 |
| Gemini-2.5-Flash | 🔒 | 63.73 | 57.87 | 80.04 | 66.60 | 56.92 | 64.79 | 66.12 | 58.99 | **81.71** | 42.25 |
| GPT-5 | 🔒 | 62.26 | 62.03 | 62.90 | 62.63 | 56.79 | 68.05 | 74.03 | 49.75 | 61.18 | 40.62 |
| **_Open-Source_** |
| Qwen3-VL | 32B | 63.90 | 62.75 | 67.11 | 66.64 | 55.38 | 69.01 | 70.30 | 58.20 | 62.20 | 45.07 |
| Qwen3-VL 💡 | 8B | 57.57 | 49.43 | 80.24 | 59.70 | 48.03 | 59.86 | 62.27 | 54.73 | 71.95 | 26.76 |
| Qwen3-VL | 8B | 55.75 | 50.99 | 69.04 | 56.76 | 46.84 | 58.45 | 64.91 | 49.21 | 62.20 | 29.58 |
| InternVL-3.5 💡 | 38B | 53.62 | 47.64 | 70.26 | 54.48 | 43.42 | 53.17 | 64.36 | 49.21 | 54.27 | 26.76 |
| Mimo-VL-SFT 💡 | 7B | 51.26 | 41.20 | 79.33 | 51.72 | 39.32 | 49.30 | 55.78 | 53.31 | 71.95 | 26.76 |
| Qwen2.5-VL-Instruct | 72B | 46.22 | 38.04 | 69.01 | 45.00 | 35.10 | 47.54 | 53.74 | 45.34 | 62.80 | 23.94 |
| InternVL-3.5 | 38B | 45.85 | 36.83 | 70.98 | 45.11 | 40.00 | 45.94 | 51.71 | 42.74 | 61.59 | 21.13 |
| InternVL-3.5 💡 | 8B | 45.78 | 37.80 | 68.02 | 46.23 | 33.68 | 46.48 | 53.14 | 42.74 | 66.46 | 21.13 |
| Qwen2.5-VL-Instruct | 32B | 45.30 | 35.55 | 72.48 | 45.28 | 35.62 | 46.83 | 52.53 | 43.92 | 52.44 | 22.54 |
| Keye-VL-1.5 💡 | 8B | 45.24 | 30.94 | 85.12 | 43.99 | 35.89 | 45.55 | 50.72 | 45.76 | 63.98 | 21.43 |
| Mimo-VL-SFT | 7B | 43.09 | 33.27 | 70.47 | 45.67 | 32.82 | 43.31 | 44.88 | 45.58 | 51.83 | 22.54 |
| Qwen2.5-VL-Instruct | 7B | 38.68 | 25.90 | 74.31 | 35.95 | 32.53 | 35.21 | 41.52 | 43.44 | 57.32 | 22.54 |
| InternVL-3.5 | 8B | 38.18 | 29.34 | 62.83 | 39.33 | 30.48 | 38.03 | 43.89 | 33.12 | 54.88 | 18.31 |
| Keye-VL-1.5 | 8B | 38.12 | 28.94 | 63.74 | 38.51 | 31.53 | 34.52 | 43.72 | 35.52 | 51.55 | 21.43 |
| GLM-4.1V 💡 | 9B | 36.51 | 29.04 | 57.33 | 38.30 | 30.94 | 33.10 | 40.81 | 34.38 | 41.46 | 21.13 |
| Kimi-VL-A3B 💡 | 16B | 34.82 | 21.23 | 72.71 | 33.21 | 28.03 | 31.34 | 35.97 | 40.69 | 52.44 | 21.13 |
| InternVideo2.5 | 7B | 34.18 | 16.95 | 82.26 | 29.76 | 30.60 | 32.75 | 33.00 | 42.74 | 57.32 | 21.13 |
| LLaVA-V1.6-Vicuna | 7B | 25.19 | 0.58 | **93.79** | 17.89 | 25.98 | 22.18 | 17.16 | 43.69 | 49.39 | 22.54 |
| ViDiC-Qwen (Ours) | 7B | 50.43 | 41.72 | 74.69 | 50.37 | 38.70 | 52.11 | 57.38 | 48.73 | 68.10 | 26.76 |

*注：**Diff.** 衡量对变化的感知能力；**Sim.** 用于检查幻觉（准确率的逆反）。多模态大模型普遍在运镜 (Camera) 和播放技术 (Technique) 方面表现挣扎。*

**主要发现**

1. 📉 显著差距：描述时间上的差异（如动作、运镜）比描述静态属性（如风格、主体）要困难得多。
2. ⚖️ 权衡取舍：具有思维链（"Thinking"）能力的模型在差异检测上有所提升，但往往会在相同的区域产生差异幻觉（导致 Sim. 分数较低）。
3. 🚧 关键弱点：几乎所有模型在播放技术（例如倒放、慢动作）这一项上都严重失效。

## 📝 引用
如果您觉得 ViDiC 对您的研究有帮助，请考虑引用我们的论文：

```bibtex
  @misc{wu2025vidicvideodifferencecaptioning,
        title={ViDiC: Video Difference Captioning}, 
        author={Jiangtao Wu and Shihao Li and Zhaozhou Bian and Yuanxing Zhang and Jialu Chen and Runzhe Wen and An Ping and Yiwen He and Jiakai Wang and Jiaheng Liu},
        year={2025},
        eprint={2512.03405},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2512.03405}, 
  }
```

## 📄 许可协议
我们的数据集根据 CC-BY-NC-SA-4.0 许可证发布。

## 📧 联系方式
如有问题或反馈：

- 🐛 问题反馈: GitHub Issues
- 💬 讨论: Hugging Face Discussions
