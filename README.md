<div align="center">
  <h1>
    ViDiC: Video Difference Captioning
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

## 📋 Abstract

Understanding visual differences between dynamic scenes requires the comparative perception of compositional, spatial, and temporal changes—a capability that remains underexplored in existing vision-language systems. While prior work on Image Difference Captioning (IDC) has enabled models to describe semantic changes between static images, these approaches fail to capture motion continuity, event evolution, or editing consistency over time.

To address this, we introduce **ViDiC (Video Difference Captioning)**, a new task that extends difference captioning into the video domain. We present the **ViDiC-1K** benchmark, designed to evaluate the ability of Multimodal Large Language Models (MLLMs) to provide fine-grained descriptions of similarities and differences between video pairs. This formulation moves beyond traditional video similarity or video editing metrics, focusing instead on **edit understanding** rather than edit execution.

<p align="center">
  <img src="assets/page.png" width="800" alt="ViDiC Task Overview">
  <br>
  <em>Figure 1: Illustration of the ViDiC task. A model must generate captions detailing similarities and differences across seven categories, assessed against a fine-grained checklist.</em>
</p>

## 🌟 Key Features

- **🎥 First Video Difference Captioning Benchmark**: A unified task requiring descriptive, comparative, and temporal understanding of video pairs.
- **📝 ViDiC-1K Dataset**: 1,000 curated video pairs annotated with over 4,000 comparative checklist items.
- **🔍 Dual-Checklist Evaluation**: A rigorous framework evaluating **Similarity** (checking for hallucinations) and **Difference** (checking for perception) separately.
- **🤖 Scalable LLM-as-a-Judge**: An automated, interpretable evaluation protocol using GPT-5-Mini to quantify factual accuracy against human-verified ground truths.

## 📈 Benchmark Statistics

<p align="center">
  <img src="assets/stats.png" width="800" alt="Dataset Statistics">
</p>

- **Total Pairs**: 1,000 (Real & Synthetic)
- **Total Checklist Items**: ~4,100 (1,056 Similarity / 3,051 Difference)
- **Evaluation Dimensions**: 7 Categories (Subject, Style, Background, Camera, Motion, Position, Playback Technique)
- **Video Duration**: Primarily 2-12 seconds
- **Data Sources**: Curated from 8+ public datasets (e.g., VidDiffBench, LMArena) and self-generated synthetic data (Veo3 + frame splicing).

## 📰 News
- 🤗 ViDiC-1K Dataset is available on Hugging Face.
- 🚀 Evaluation code and leaderboards is released.

## 📂 File Structure

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
│   └── LMArena
|   └── style
|   └── ......
|
├── inference/   # Inference scripts for popular models
│   ├── get_response_GLM.py
│   ├── get_response_gemini.py
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

## 📊 Benchmark Results
### Overall Model Performance

| Model | Param. | Avg. | Diff. | Sim. | Subject | Motion | Pos. | Backgr. | Cam. | Style | Tech. |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **_Closed-Source_** | | | | | | | | | | | |
| Gemini-2.5-Pro | 🔒 | 66.72 | 63.73 | 75.33 | 67.71 | 62.78 | 68.24 | 70.65 | 59.97 | 75.79 | 74.32 |
| GPT-5 | 🔒 | 62.94 | 57.32 | 79.17 | 61.52 | 57.78 | 65.31 | 69.15 | 57.39 | 77.60 | 54.66 |
| Gemini-2.5-Flash | 🔒 | 58.87 | 52.11 | 78.37 | 59.63 | 51.29 | 57.23 | 63.98 | 52.82 | 81.58 | 55.41 |
| Gemini-2.0-Flash | 🔒 | 53.71 | 50.26 | 63.66 | 58.90 | 48.71 | 57.86 | 57.11 | 47.30 | 55.79 | 18.92 |
| GPT-4o | 🔒 | 49.95 | 39.14 | 81.12 | 46.79 | 43.53 | 51.89 | 53.73 | 49.18 | 77.89 | 27.03 |
| **_Open-Source_** | | | | | | | | | | | |
| Qwen3-VL | 32B | 61.38 | 58.54 | 71.50 | 64.60 | 51.77 | 62.00 | 68.62 | 52.66 | 74.86 | 47.83 |
| Qwen3-VL | 8B | 53.23 | 50.44 | 63.20 | 58.66 | 43.33 | 52.33 | 63.49 | 40.92 | 66.28 | 11.59 |
| Mimo-VL-SFT | 7B | 52.59 | 46.51 | 70.17 | 54.39 | 46.55 | 51.25 | 57.31 | 48.37 | 67.71 | 25.33 |
| InternVL-3.5 💡 | 38B | 52.44 | 46.25 | 70.30 | 52.66 | 43.04 | 53.77 | 59.80 | 47.80 | 72.63 | 20.27 |
| InternVL-3.5 | 38B | 50.49 | 40.09 | 80.46 | 48.35 | 44.34 | 51.89 | 54.93 | 49.18 | 76.32 | 14.86 |
| Qwen2.5-VL-Instruct | 72B | 49.71 | 42.56 | 70.30 | 48.07 | 44.82 | 48.11 | 55.92 | 46.42 | 68.95 | 22.97 |
| Qwen2.5-VL-Instruct | 32B | 47.83 | 43.42 | 60.53 | 49.72 | 40.78 | 49.69 | 55.12 | 38.39 | 68.42 | 20.27 |
| InternVL-3.5 💡 | 8B | 45.01 | 41.18 | 56.07 | 46.79 | 37.06 | 45.60 | 54.03 | 35.76 | 61.58 | 17.57 |
| InternVL-3.5 | 8B | 43.67 | 35.68 | 66.70 | 43.21 | 37.54 | 45.60 | 48.46 | 39.02 | 68.42 | 14.86 |
| GLM-4.1V 💡 | 9B | 40.95 | 33.99 | 61.08 | 42.60 | 34.35 | 38.13 | 47.26 | 33.83 | 64.58 | 14.67 |
| Qwen2.5-VL-Instruct | 7B | 39.39 | 35.22 | 51.42 | 39.82 | 33.82 | 37.42 | 47.96 | 30.74 | 58.95 | 14.86 |
| Kimi-VL-A3B 💡 | 16B | 35.16 | 28.68 | 53.88 | 37.48 | 26.00 | 35.63 | 42.99 | 22.56 | 70.31 | 14.67 |
| InternVideo2.5 | 7B | 32.70 | 23.14 | 60.32 | 32.72 | 23.43 | 33.13 | 36.42 | 28.70 | 66.15 | 14.67 |
| Keye-VL-1.5 | 8B | 32.45 | 25.53 | 57.13 | 32.86 | 25.80 | 30.67 | 39.18 | 24.69 | 60.00 | 8.70 |
| Llama-3.2 | 11B | 19.43 | 5.23 | 61.01 | 14.48 | 20.31 | 17.84 | 13.44 | 29.56 | 40.00 | 11.70 |
| LLaVA-V1.6-Vicuna | 7B | 8.96 | 5.11 | 20.07 | 7.49 | 12.20 | 13.44 | 6.96 | 10.02 | 6.25 | 6.67 |


*Note: **Diff.** measures perception of changes; **Sim.** checks for hallucinations (inverse accuracy). MLLMs generally struggle with Camera and Playback Techniques.*

**Key Findings**
1. 📉 Significant Gaps: Describing temporal differences (Motion, Camera) is much harder than static attributes (Style, Subject).
2. ⚖️ Trade-off: "Thinking" models improve Difference detection but often hallucinate differences in identical areas (lower Similarity score).
3. 🚧 Critical Weakness: Almost all models fail significantly on Camera works and Playback Techniques (e.g., reverse, slow-motion).

## 📝 Citation

If you find ViDiC useful in your research, please consider citing our paper:

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

## 📄 License
Our dataset is released under the CC-BY-NC-SA-4.0 license.

## 📧 Contact
For questions and feedback:

- 🐛 Issues: GitHub Issues
- 💬 Discussions: Hugging Face Discussions

