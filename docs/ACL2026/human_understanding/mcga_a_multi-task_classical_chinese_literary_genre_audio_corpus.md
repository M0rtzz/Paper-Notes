---
title: >-
  [论文解读] MCGA: A Multi-task Classical Chinese Literary Genre Audio Corpus
description: >-
  [ACL 2026][人体理解][古典文学语音语料库] 本文构建了首个面向中国古典文学的大规模（119小时、22000条样本）全版权音频语料库 MCGA，涵盖赋、诗、文、词、曲五大文体和六项语音任务（ASR/S2TT/SEC/SQA/SU/SR），并通过评测 10 个多模态大模型揭示了当前模型在古典文学语音理解上的显著不足。
tags:
  - ACL 2026
  - 人体理解
  - 古典文学语音语料库
  - 多模态大语言模型
  - 语音情感分析
  - 跨模态一致性
  - 中国古典文学研究
---

# MCGA: A Multi-task Classical Chinese Literary Genre Audio Corpus

**会议**: ACL 2026  
**arXiv**: [2601.09270](https://arxiv.org/abs/2601.09270)  
**代码**: https://github.com/yxduir/MCGA  
**领域**: 语音与自然语言处理/中国古典文学  
**关键词**: 古典文学语音语料库, 多模态大语言模型, 语音情感分析, 跨模态一致性, 中国古典文学研究

## 一句话总结

本文构建了首个面向中国古典文学的大规模（119小时、22000条样本）全版权音频语料库 MCGA，涵盖赋、诗、文、词、曲五大文体和六项语音任务（ASR/S2TT/SEC/SQA/SU/SR），并通过评测 10 个多模态大模型揭示了当前模型在古典文学语音理解上的显著不足。

## 研究背景与动机

**领域现状**：多模态大语言模型（MLLM）的快速发展为中国古典文学研究（CCS）带来了新的可能性。然而，现有研究主要集中在文本（ACLUE、WenMind 等）和视觉（Oracle-Bench、MCS-Bench 等）模态，古典文学的语音维度几乎完全空白。这一领域缺少高质量的领域专用音频语料库，导致 MLLM 在古典中文语音理解方面的能力无法被系统评估和提升。

**现有痛点**：(1) 现有中国文化数据集大多只涉及文本或图像模态，没有平行的古典文学语音数据；(2) 少数涉及中文语音的资源主要面向现代汉语，无法覆盖古典文学特有的修辞、典故和语音韵律特征；(3) 版权问题一直困扰着开源 CCS 音频数据集的建设——互联网上的朗诵音频往往受版权限制，无法自由分发用于研究。

**核心矛盾**：多模态大模型具备了强大的文本和视觉理解能力，但在古典中文语音理解这一维度上完全缺失评测基础设施。没有语音语料库就无法评测，无法评测就无法推动模型在该领域的进步。

**本文目标**：构建一个覆盖多文体、多任务、全版权的古典中文文学音频语料库，建立系统的评测框架，全面评估当前 MLLM 在古典文学语音理解上的能力。

**切入角度**：从"文体多样性"和"任务多样性"两个维度出发——文体上涵盖中国文学史上最重要的五种文体（赋、诗、文、词、曲），任务上设计从基础（ASR）到高级（语音推理 SR）的六级递进任务体系。

**核心 idea**：招募 28 名母语者人工录制所有音频并获得版权转让，利用 LLM 生成问答对并经三重验证确保质量，构建一个同时支持 6 项语音任务和 4 项文本任务的平行语料库。

## 方法详解

### 整体框架

MCGA 语料库的构建分为三个阶段：(1) 数据收集与预处理——从网络收集公共领域的古典文学文本及拼音，清洗后将文本分段以限制录音时长在 30 秒以内；(2) 人工录音——28 名母语者按照统一录音规范完成全部文本的录制，并经过 MLLM 检查和人工检查两轮质量控制；(3) 文本数据构建——利用 DeepSeek-V3.2 基于每个片段的完整文学上下文生成多任务的问答对，再通过 DeepSeek-V3.2、GPT-5-mini 和 Gemini-3-Flash 三重验证过滤无效样本。

### 关键设计

1. **六级递进任务体系**:
    - 功能：覆盖从低级感知到高级推理的完整语音理解能力谱
    - 核心思路：设计六项核心语音任务——ASR（自动语音识别）、S2TT（语音到文本翻译）、SEC（语音情感描述）、SQA（口语问答）、SU（语音理解）、SR（语音推理）。其中 ASR 测试基础转录能力，S2TT 要求古文到现代英文的跨语言翻译，SEC 要求模型识别说话人特征并分析逐句情感，SQA 是开放式事实性问答，SU 和 SR 分别测试基于语音内容的理解和需要外部知识的推理能力。同时利用平行文本数据支持 MT、QA、LU、LR 四项文本任务。
    - 设计动机：单一任务无法全面评估 MLLM 的古典文学理解能力，递进式任务设计可以精确定位模型在不同认知层次上的瓶颈。

2. **情感描述保真度指标（ECF）**:
    - 功能：为古典文学语音情感描述任务提供细粒度的自动评估指标
    - 核心思路：ECF 由三个子指标组成——ECF-P（人物识别，0-2 分，检测年龄和性别识别准确性，每错一项扣 1 分）、ECF-G（全局情感基调，0-3 分，评估整体情感氛围描述的丰富度和准确性）、ECF-S（逐句情感保真度，0-5 分，评估逐句转录和情感分析质量，每个情感错误扣 1 分，出现幻觉则直接 0 分）。最终归一化到百分制。
    - 设计动机：现有语音情感评测指标主要针对现代语音的情感分类（如快乐/悲伤），无法捕捉古典文学朗诵中的复杂情感层次——如一首送别诗中交织的不舍、豁达和壮志。

3. **跨模态一致性指标（CMC）**:
    - 功能：量化 MLLM 在语音和文本两种输入模态下的表现一致性
    - 核心思路：$CMC = \frac{1}{3}\left(\frac{SQA}{QA} + \frac{SU}{LU} + \frac{SR}{LR}\right) \times 100$，即对 SQA/SU/SR 三个语音任务与其对应文本任务（QA/LU/LR）的得分比值取平均。CMC 值越接近 100，说明模型的语音理解能力与文本理解能力越一致。
    - 设计动机：一个理想的 MLLM 在接受语音输入和文本输入时应给出一致的答案。CMC 可以揭示模型是否真正"理解"了语音内容，还是仅依赖文本通道的能力。

### 损失函数 / 训练策略

微调实验使用 Qwen2.5-Omni-7B 作为基座模型，采用 LoRA（$r=8, \alpha=32$）在 MCGA 训练集上训练 3 个 epoch，使用 AdamW 优化器，学习率 $1 \times 10^{-4}$，在 4 块 A100 GPU 上完成。

## 实验关键数据

### 主实验

| 模型 | ASR (CER↓) | S2TT (LLM-B↑) | SEC (ECF↑) | SQA (F1↑) | SU (Acc↑) | SR (Acc↑) | 总分↑ |
|------|-----------|---------------|------------|----------|----------|----------|------|
| GPT-4o-mini-Audio | 20.6 | 43.5 | 5.7 | 30.6 | 74.8 | 70.2 | 304.2 |
| Gemini-3-Flash | 6.1 | 74.0 | 54.0 | 48.7 | 86.6 | 83.7 | 440.9 |
| Qwen2.5-Omni-7B | 10.1 | 49.7 | 37.0 | 43.5 | 81.3 | 79.3 | 380.7 |
| Qwen3-Omni-30B | 4.4 | 67.6 | 58.4 | 51.5 | 86.9 | 82.9 | 442.9 |
| Step-Audio-2-mini | 9.9 | 41.9 | 36.8 | 45.2 | 80.5 | 80.4 | 374.9 |
| Phi-4-Multimodal | 59.6 | 27.5 | 12.7 | 24.5 | 50.6 | 54.4 | 210.1 |

Qwen3-Omni 总分最高（442.9），在 ASR、SEC、SQA、SU 上领先；Gemini-3-Flash 在 S2TT 和 SR 上表现最好，体现闭源模型在英文生成和推理上的优势。

| 模型 | 诗 CER | 词 CER | 曲 CER | 赋 CER | 文 CER |
|------|--------|--------|--------|--------|--------|
| Qwen3-Omni-30B | 3.8 | 2.8 | 4.1 | 6.2 | 4.3 |
| Qwen2.5-Omni-7B | 9.9 | 7.5 | 8.9 | 14.8 | 8.8 |
| Qwen-Omni-MCGA (微调) | 2.8 | 3.1 | 7.8 | 5.3 | 4.1 |

### 消融实验

| 配置 | ASR CER↓ | S2TT↑ | SEC↑ | SQA↑ | SU↑ | SR↑ |
|------|---------|-------|------|------|-----|-----|
| Qwen2.5-Omni-7B (原始) | 10.1 | 49.7 | 37.0 | 43.5 | 81.3 | 79.3 |
| Qwen-Omni-MCGA (微调) | — | — | — | — | — | — |

微调后的 Qwen-Omni-MCGA 在诗和文的 ASR 上超越了 30B 参数的 Qwen3-Omni（CER 2.8 vs 3.8），证明了 MCGA 作为训练资源的高价值。

### 关键发现

- **赋是最难的文体**：所有模型在赋（Fu）上的 CER 均最高，源于赋的华丽修辞、频繁用典和大量语气词。
- **SEC 是最难的任务**：即使最强的 Qwen3-Omni 在 SEC 上也仅得 58.4 分，GPT-4o-mini-Audio 因安全协议拒绝回答情感分析请求，仅得 5.7 分。
- **数据一致性高**：训练集/验证集/测试集的 CER 差异仅 0.1（Qwen3-Omni），证明录音质量控制有效。
- **开源模型追平闭源**：Qwen3-Omni 总分（442.9）超过 Gemini-3-Flash（440.9），开源模型在中文古典领域已达到竞争力水平。
- **小模型微调收益巨大**：7B 参数的 Qwen2.5-Omni 经 MCGA 微调后，在部分文体 ASR 上超越 30B 的 Qwen3-Omni。

## 亮点与洞察

- **填补领域空白**：MCGA 是首个专门面向古典中文文学的大规模全版权音频语料库，真正解决了该领域音频数据从无到有的问题。所有 22000 条音频均由原始录制者签署版权转让协议，彻底解决了开源语音数据集的知识产权困境。
- **ECF 指标设计精巧**：将语音情感评估分解为人物识别、全局基调和逐句保真度三个层次，既适配古典文学朗诵的特殊性，又保持了自动评估的可操作性。
- **CMC 指标的洞察力**：通过语音/文本任务得分比值来衡量跨模态一致性，可以清晰暴露模型"依赖文本通道而非真正理解语音"的问题。
- **文体维度的分析**：发现"赋最难、词最易"的规律，为后续针对特定文体优化模型提供了方向。

## 局限与展望

- 语料库仅包含标准普通话录音，未涵盖方言朗诵或吟唱等传统古典文学表演形式。
- SEC 评估依赖 LLM 评委（DeepSeek API），主观情感判断的自动评估本身是开放问题。
- 训练实验仅在 Qwen2.5-Omni-7B 上验证，未覆盖其他基座模型的微调效果。
- 缓存文件截断导致 SQA/SU/SR 的详细分析和 CMC 指标的具体实验数据未能完整获取。
- 未来可扩展到古典文学吟唱、戏曲等更丰富的音频形态。

## 相关工作与启发

- **vs ACLUE/WenMind**: 这些基准仅覆盖文本模态，MCGA 首次将古典文学评测扩展到语音维度。
- **vs MCS-Bench/Oracle-Bench**: 这些多模态基准侧重文本+视觉，MCGA 填补了文本+语音的空白。
- **vs LibriSpeech/Common Voice**: 通用语音数据集面向现代语言，无法处理古典汉语的用典、修辞和音韵特征。
- **vs CII-Bench**: CII-Bench 关注中文文化常识的图文理解，MCGA 聚焦古典文学的深度语音理解和情感分析。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个古典中文文学全版权音频语料库，填补明确的领域空白
- 实验充分度: ⭐⭐⭐⭐ 10 个模型、6 项任务、5 种文体的全面评测，分析维度丰富
- 写作质量: ⭐⭐⭐⭐ 结构清晰，指标定义严谨，数据呈现充分
- 价值: ⭐⭐⭐⭐ 对推动古典文学数字化研究和 MLLM 语音能力评测有重要意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification](odutqa-mdc_a_task_for_open-domain_underspecified_tabular_qa_with_multi-turn_dial.md)
- [\[ACL 2026\] ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](researchbench_benchmarking_llms_in_scientific_discovery_via_inspiration-based_ta.md)
- [\[ACL 2026\] SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning](samora_semantic-aware_mixture_of_lora_experts_for_task-adaptive_learning.md)
- [\[CVPR 2025\] Two by Two: Learning Multi-Task Pairwise Objects Assembly for Generalizable Robot Manipulation](../../CVPR2025/human_understanding/two_by_two_learning_multi-task_pairwise_objects_assembly_for_generalizable_robot.md)
- [\[ICCV 2025\] SAMO: A Lightweight Sharpness-Aware Approach for Multi-Task Optimization with Joint Global-Local Perturbation](../../ICCV2025/human_understanding/samo_a_lightweight_sharpness-aware_approach_for_multi-task_optimization_with_joi.md)

<!-- RELATED:END -->
