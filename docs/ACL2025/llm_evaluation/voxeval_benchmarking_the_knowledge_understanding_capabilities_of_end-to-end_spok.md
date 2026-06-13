---
title: >-
  [论文解读] VoxEval: Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models
description: >-
  [ACL2025][LLM评测][spoken language model] 提出 VoxEval，首个支持端到端纯语音输入-输出评估的 SpeechQA 基准，涵盖 56 个学科、26 种输入音频条件，系统揭示了当前端到端语音大模型在知识理解和数学推理方面的严重不足。 1. 语音交互需求增长：自然人机交互通常以口语形式发…
tags:
  - "ACL2025"
  - "LLM评测"
  - "spoken language model"
  - "speech QA"
  - "benchmark"
  - "end-to-end evaluation"
  - "robustness"
---

# VoxEval: Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models

**会议**: ACL2025  
**arXiv**: [2501.04962](https://arxiv.org/abs/2501.04962)  
**代码**: [dreamtheater123/VoxEval](https://github.com/dreamtheater123/VoxEval)  
**领域**: LLM评测  
**关键词**: spoken language model, speech QA, benchmark, end-to-end evaluation, robustness

## 一句话总结

提出 VoxEval，首个支持端到端纯语音输入-输出评估的 SpeechQA 基准，涵盖 56 个学科、26 种输入音频条件，系统揭示了当前端到端语音大模型在知识理解和数学推理方面的严重不足。

## 背景与动机

1. **语音交互需求增长**：自然人机交互通常以口语形式发生，端到端语音大模型（SLM）成为该领域的关键方向，但其知识理解能力缺乏系统评估。
2. **现有基准不支持端到端评估**：已有 SpeechLLM 基准（AudioBench、VoiceBench 等）要么用文本问答配对音频输入，要么仅评估文本输出，无法反映真实语音-语音交互管道的性能。
3. **忽视输入音频条件多样性**：真实场景中说话人音色、语速、口音、环境噪声各异，但现有基准未系统评估 SLM 对这些变化的鲁棒性。
4. **从 S2T 到 S2S 存在性能断崖**：多项研究表明 SLM 从 speech-to-text 切换到 speech-to-speech 评估时性能显著下降，凸显端到端评估的必要性。
5. **数学推理在口语格式下难以评估**：数学表达式在书面形式简洁，但转写为口语（如"两千三百五十一"代替"2351"）后评估极具挑战，此前无相关基准。
6. **SLM 亟需面向知识理解的综合评估**：现有语音模型评估多聚焦语言建模（词汇/句法/语义）或副语言特征，缺乏面向世界知识理解的全面基准。

## 方法详解

### 整体框架：VoxEval 基准构建

- **功能**：基于 MMLU 测试集构建纯语音格式的知识理解 QA 基准，包含 13,938 个独有语音问答对，覆盖 56 个学科（排除含代码片段的高中计算机科学），每个问题通过 26 种输入音频条件扩展为 153,318 个变体。
- **为什么**：MMLU 学科结构全面（STEM/社科/人文）、被广泛用于评测文本 LLM，但此前仅以文本形式评估 SLM 的附加文本处理能力，无法反映语音处理能力；同一数据的口语版本难度远高于书面版本。
- **怎么做**：将 MMLU 四选一题目拼接为自然语言序列（"问题…请从ABCD中选择…选项A…选项B…"），连同答案一并通过 OpenAI TTS API 合成语音。评估时用 Whisper-large-v3 将 SLM 的语音回答转录为文本后做字符串匹配计算准确率。

### 关键设计 1：多样化输入音频条件

- **功能**：系统构建 26 种输入音频条件，包括 6 种说话人、5 种语言变体（填充词/发音错误/不流畅/自我纠正/非母语）、2 种副语言变体（语调偏移/语速变化）和多种音频质量变体（高斯噪声/彩噪/背景音乐/混响/各类滤波器等）。
- **为什么**：SLM 在语义内容不变而音频特征改变时应保持一致的回答准确率，这对事实类问题尤为关键，但此前无基准系统测试此鲁棒性。
- **怎么做**：说话人使用 OpenAI TTS 的 6 种声音（alloy/echo/fable 等）；语言变体通过 GPT-4o 修改原始问题文本再 TTS 合成；副语言变体通过音频增强（±5 半音音调偏移、0.5x-2x 语速变化）实现；音频质量变体用 audiomentations 库添加噪声/混响/滤波。所有变体均基于 alloy 说话人。

### 关键设计 2：口语数学推理评估

- **功能**：首次将数学推理引入 SLM 评估，将书面数学表达式转换为口语形式，并设计直接回答（Direct Answer）与链式思维（Chain-of-Thought）两种评估模式。
- **为什么**：数学推理是 AI 辅导等应用的核心能力，人们经常以口语形式表达数学问题，但现有 TTS 系统无法正确读出数学表达式（阿拉伯数字/运算符/括号等）。
- **怎么做**：两步法——先用 GPT-4o few-shot 提示将书面数学转为口语（如"4÷(2+8)"→"four divided by the sum of two and eight"），再用 TTS 合成。CoT 模式在问题前加语音提示"Please explain your reasoning step by step"，评估时分"截断"（只取最后答案段）和"不截断"（含完整推理链）两种。

## 实验关键数据

### 实验 1：整体知识理解性能

| 模型 | Alloy 准确率 | 最佳准确率 | 随机基线 |
|------|-------------|-----------|---------|
| SpeechGPT | 0.01% | 0.02% | 25% |
| TWIST | 4.80% | 5.58% | 25% |
| SPIRIT-LM | 20.84% | 20.96% | 25% |
| Moshi | 12.16% | 12.98% | 25% |
| GLM-4-Voice | **37.63%** | **38.15%** | 25% |
| Whisper+Llama-3.1-8B | 55.25% | 55.73% | 25% |

**发现**：(1) 多数 SLM 未超过 25% 随机猜测基线，仅 GLM-4-Voice 超过，说明当前 SLM 在指令遵循和知识理解方面严重不足；(2) 级联系统（Whisper+Llama）大幅领先所有端到端 SLM，差距达 17+ 个百分点。

### 实验 2：音频条件鲁棒性

| 条件 | GLM-4-Voice | Moshi | SPIRIT-LM |
|------|------------|-------|-----------|
| 标准（Alloy） | 37.63% | 12.16% | 20.84% |
| 音调偏移（Pitch） | 33.45%（↓4.2） | 6.09%（↓6.1） | 17.88%（↓3.0） |
| 语速变化（Speed） | 34.69%（↓2.9） | 10.13%（↓2.0） | 19.11%（↓1.7） |
| 背景噪声（Noise） | 36.95%（↓0.7） | 10.18%（↓2.0） | 19.50%（↓1.3） |

**发现**：(1) 音调偏移是对 SLM 影响最大的条件，Moshi 准确率近乎腰斩；(2) 不同说话人间性能差异可达 1-4 个百分点，fable 声音通常最难；(3) 数学推理方面，所有模型准确率极低（最高约 27%），CoT 提示未能提升甚至降低了部分模型性能，表明 SLM 缺乏口语格式下的逐步推理能力。

## 亮点

- **首个端到端语音QA基准**：输入输出均为语音，填补了 SLM 评估在端到端设置下的空白
- **系统化鲁棒性评估**：26 种音频条件的设计覆盖面广，直接揭示了 SLM 对音调/语速/噪声的脆弱性
- **首次口语数学推理评估**：创新性地将书面数学转为口语格式并引入 CoT 评估，开辟了新评估维度
- **基准设计合理**：基于 MMLU 构建保证了学科全面性和与文本基线的可比性
- **数据规模大**：13,938 个基础 QA 对 × 26 种条件 = 153,318 个评估样本，统计充分

## 局限与展望

- **评估指标依赖 ASR**：最终答案需 Whisper 转录后字符串匹配，ASR 错误会引入噪声，非端到端的"真正"评估
- **仅四选一 MCQ 格式**：真实语音交互远比选择题复杂，缺乏开放式问答评估
- **数据源单一**：完全基于 MMLU，继承了 MMLU 本身的偏差和局限（如学科覆盖不均、部分题目争议）
- **TTS 合成语音与真人语音有差距**：所有音频均由 TTS 生成，可能无法完全代表真实人类语音的自然变异
- **未评估商业闭源 SLM**：如 GPT-4o 的语音模式等

## 与相关工作的对比

### vs VoiceBench (Chen et al., 2024)
VoiceBench 评估 SpeechLLM 在不同音频条件下的表现，但仅基于文本输出评估（非端到端），且不含数学推理。VoxEval 首次实现输入输出均为语音的端到端评估，音频条件更丰富（26 vs VoiceBench 的有限变体），且开创性地纳入数学推理。

### vs AudioBench (Wang et al., 2024)
AudioBench 包含 5,196 个知识理解问题，但 QA 对为文本格式，仅适合评估 S2T-LLM。VoxEval 的 13,938 个纯语音 QA 对专为端到端 SLM 设计，并通过 153,318 个变体系统评估鲁棒性。

### vs MMLU (Hendrycks et al., 2021)
VoxEval 构建于 MMLU 之上，将其从文本扩展到语音模态。实验表明语音版本对模型的挑战远大于文本版本（GLM-4-Voice 在 VoxEval 上约 37% vs 对应文本 LLM 在 MMLU 上 60%+），验证了跨模态评估的必要性。同时 VoxEval 排除了含代码片段的学科，体现了对语音格式适配的细致考量。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个端到端语音 QA 基准 + 口语数学推理评估
- 实验充分度: ⭐⭐⭐⭐ — 5 个 SLM + 1 级联基线，26 种音频条件，分析详尽
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，图表丰富，结构合理
- 价值: ⭐⭐⭐⭐ — 为 SLM 社区提供急需的评估工具，有效暴露关键短板

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](../../ACL2026/llm_evaluation/e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)
- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)
- [\[ACL 2025\] SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science](seedbench_a_multi-task_benchmark_for_evaluating_large_language_models_in_seed_sc.md)
- [\[ACL 2025\] CodeMEnv: Benchmarking Large Language Models on Code Migration](codemenv_benchmarking_large_language_models_on_code_migration.md)
- [\[ACL 2025\] Exposing Numeracy Gaps: A Benchmark to Evaluate Fundamental Numerical Abilities in Large Language Models](exposing_numeracy_gaps_a_benchmark_to_evaluate_fundamental_numerical_abilities_i.md)

</div>

<!-- RELATED:END -->
