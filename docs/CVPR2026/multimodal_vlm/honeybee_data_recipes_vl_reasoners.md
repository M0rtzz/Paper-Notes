---
title: >-
  [论文解读] HoneyBee: Data Recipes for Vision-Language Reasoners
description: >-
  [CVPR 2026][多模态][VLM推理] 系统性地研究了 VL 推理训练数据的设计空间——数据来源选择、干预策略筛选、图像/问题/CoT 三维度缩放——基于洞察构建了 250 万样本的 HoneyBee 数据集，3B VLM 在 MathVerse 上超越 SOTA 7.8pp，并提出共享 Caption 解码的测试时缩放策略节省 73% token。
tags:
  - CVPR 2026
  - 多模态
  - VLM推理
  - CoT数据策划
  - 数据缩放
  - 视觉推理
  - 测试时缩放
---

# HoneyBee: Data Recipes for Vision-Language Reasoners

**会议**: CVPR 2026  
**arXiv**: [2510.12225](https://arxiv.org/abs/2510.12225)  
**代码**: [数据集](https://huggingface.co/datasets/facebook/HoneyBee)  
**领域**: 多模态VLM / 数据工程  
**关键词**: VLM推理, CoT数据策划, 数据缩放, 视觉推理, 测试时缩放

## 一句话总结

系统性地研究了 VL 推理训练数据的设计空间——数据来源选择、干预策略筛选、图像/问题/CoT 三维度缩放——基于洞察构建了 250 万样本的 HoneyBee 数据集，3B VLM 在 MathVerse 上超越 SOTA 7.8pp，并提出共享 Caption 解码的测试时缩放策略节省 73% token。

## 研究背景与动机

**领域现状**：VLM 在推理任务上越来越依赖高质量 CoT 训练数据，社区已涌现多个 VL 推理数据集（Math-LLaVA、LLaVA-CoT、R1-OneVision 等）。

**现有痛点**：

1. 对"什么样的 VL 推理数据是好数据"缺乏系统性理解——各工作用不同来源、过滤策略和生成模型，无法公平比较各因素的独立贡献
2. VL 推理数据的缩放行为（该缩放图片数、问题数还是 CoT 数？）从未被系统探索
3. 许多看似合理的数据增强策略实际效果未经严格验证

**核心矛盾**：社区在盲目拼凑和扩大数据集，但不清楚每个设计决策对最终推理性能的真正贡献。

**本文目标** 系统性地理解并优化 VL 推理 CoT 数据集的构建过程，给出可复现的"数据配方"。

**切入角度**：三阶段控制变量实验——上下文策划（选数据源）→ 数据干预（增强/过滤）→ 大规模缩放（图像/问题/CoT 三维度）。

**核心 idea**：通过严格控制变量的实验，揭示"数据来源选择 >> 数据增强策略"、"大部分看似合理的干预反而有害"、"三维度缩放未饱和"三大发现，并据此构建最优数据集。

## 方法详解

### 整体框架

三阶段数据策划流水线：**上下文策划**（固定 CoT 生成器对比 6 个数据源排名）→ **数据干预**（测试 8 种感知/推理增强策略）→ **大规模缩放**（沿图像/问题/CoT 三轴扩展）。最终整合为 HoneyBee 数据集（250 万样本 = 150 万 VL + 104 万纯文本推理），用于 SFT 训练 PLM 系列 VLM。

### 关键设计

1. **上下文来源排名**

    - 功能：在统一实验框架下公平对比不同数据源对 VL 推理性能的影响
    - 核心思路：固定 CoT 生成器（Llama-4-Scout）和训练流程，每个数据源限制 50K 样本。对比 6 个来源：ViRL、Math-LLaVA、R1-OneVision、Cauldron、PixMo、MMK12
    - 关键发现：来源间性能差距高达 11.4pp（ViRL 最优 vs MMK12 最差）；**混合数据源不如单一最优源**——Top-2/Top-4 混合反而性能下降
    - 设计动机：数据来源选择的影响远大于任何后续干预策略

2. **数据干预策略筛选**

    - 功能：严格测试 8 种感知/推理增强策略，找出真正有效的干预
    - 核心思路：测试视觉扰动、文字丰富图像、感知冗余过滤、浅层感知过滤、Caption-and-Solve（感知增强）和纯文本推理混入、干扰项增加、长度/难度过滤（推理增强）
    - 关键发现：大部分干预**反而降低性能**（视觉扰动 -1.6pp、文字丰富图像 -1.3pp、感知冗余过滤 -3.6pp、难度均衡 -5.5pp）。仅两个策略稳定有效：(a) **Caption-and-Solve**（训练时先生成图像描述再解题，+3.3pp）；(b) **混入纯文本推理数据**（+7.5pp）
    - 设计动机：不要想当然地做数据增强——需要严格 A/B 测试

3. **三维数据缩放**

    - 功能：分别研究缩放图像数量、每图问题数（合成新问题）、每题 CoT 数量的效果
    - 核心思路：对 39K 真实图片，生成 16 个 CoT/题、14 个合成新问题/图，经 majority voting 过滤后获得 150 万 VL 样本 + 104 万纯文本推理
    - 关键发现：三个维度同时缩放都能持续提升性能，且在 250 万规模仍未饱和

### 损失函数 / 训练策略

- 标准 SFT：最大化 log p(C_j | I_j, Q_j)，全参数微调 PLM（含视觉编码器和 LLM 骨干）
- 训练 5 个 epoch 取最优 checkpoint
- 后续可接一轮 GRPO 强化学习进一步提升（+1.9pp）
- Caption-and-Solve 训练格式：模型先输出图像描述 C，再输出解题过程 S

## 实验关键数据

### 主实验

| 模型 | Average (5 tasks) | MathVerse | MathVista | MathVision | We-Math |
|------|-----------------|-----------|-----------|------------|---------|
| Qwen2.5-VL-3B | 42.6 | 35.0 | 58.9 | 23.7 | 49.2 |
| PLM-HoneyBee-3B | **46.2** | **42.8** | **61.2** | **29.9** | **59.3** |
| 提升 | +8.4% | +22.3% | +3.9% | +26.2% | +20.5% |

| 模型规模 | Average | 对比 |
|---------|---------|------|
| PLM-HoneyBee-1B | 超 InternVL-3-1B 28pp | 极小模型也获益 |
| PLM-HoneyBee-8B | 49.8 | 超 Qwen2.5-VL-7B (48.5) +2.7% |

### 消融实验

| 因素 | 发现 | 影响幅度 |
|------|------|---------|
| 数据来源选择 | ViRL 最优 vs MMK12 最差 | 11.4pp 差距 |
| 混合数据源 | Top-2/Top-4 混合不如单一最优源 | -0.5~-1.5pp |
| Caption-and-Solve | 独立生成 (I->C, (I,Q)->S) 最好 | +3.3pp |
| 纯文本推理混入 | OpenThoughts3 re-annotated | +7.5pp |
| 视觉扰动 | 反而有害 | -1.6pp |
| 难度均衡 | 反而有害 | -5.5pp |
| 50K->250K->2.5M | 持续提升未饱和 | +4.8pp |
| 共享 Caption 解码 | 64次采样共享一次 caption | token 数 -73%，性能不降 |

### 关键发现

- **数据来源选择影响远大于任何干预策略**：来源差距 11.4pp vs 最佳干预 +7.5pp
- **大多数看似合理的数据增强策略反而有害**：视觉扰动、文字丰富图像、感知冗余过滤等均下降
- **Caption-and-Solve 的成功关键**：将感知（图像描述）和推理（解题）解耦为两个独立生成过程
- **数据缩放三维度均未饱和**：继续投入资源扩大数据集仍有收益
- **3B/8B 模型实验高度相关**：为小模型做数据选择的结论可推广到大模型
- **共享 Caption 解码**节省 73% token 且性能不降

## 亮点与洞察

1. 非常扎实的数据工程研究，控制变量做得极好，类似 OpenThoughts 但聚焦多模态
2. "大部分看似合理的数据增强其实无效"对社区有很强的警示价值
3. **共享 Caption 解码**非常实用：训练时先描述图片再解题，推理时复用描述供多次采样
4. 250 万规模仍未饱和的发现指明了继续投入的方向

## 局限与展望

1. 仅关注数学推理类任务，对 VQA、图像理解等通用 VL 任务的数据策划规律未探索
2. 仅使用单一教师模型（Llama-4-Scout），不同教师模型是否有不同最优配方未知
3. 单图场景，多图推理和视频推理的数据策划未涉及
4. 数据来源排名可能依赖评估基准——换一组 evaluation 排名可能不同

## 相关工作与启发

- **vs OpenThoughts**：OpenThoughts 系统研究了文本推理数据设计空间，HoneyBee 将同样理念扩展到多模态，加入图像维度缩放和感知增强策略
- **vs LLaVA-CoT / R1-OneVision**：各自提出了 VL 推理数据集但缺乏公平对比，HoneyBee 在统一框架下比较并发现 ViRL 最优
- **vs Math-LLaVA / MAVIS**：关注特定数学视觉场景数据构建，HoneyBee 更全面地覆盖多种来源和干预策略
- **启发**：Caption-and-Solve "先描述再推理"的策略可能对 VL 导航、具身智能等其他领域同样有效

## 评分

- 新颖性: ⭐⭐⭐⭐ 方法论本身不新（数据工程+scaling），但系统性实验设计和反直觉发现有很高价值
- 实验充分度: ⭐⭐⭐⭐⭐ 极其充分，多模型、多数据集、多策略全面对比，控制变量严格
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，图表精良，实验设置透明，是数据工程论文的范本
- 价值: ⭐⭐⭐⭐⭐ 对 VL 推理数据研究有很强指导意义，数据集开源且规模大

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework](scaling_test-time_robustness_of_vision-language_models_via_self-critical_inferen.md)
- [\[CVPR 2026\] Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models](video-only_tom_enhancing_theory_of_mind_in_multimodal_large_language_models.md)
- [\[CVPR 2026\] Beyond Static Artifacts: A Forensic Benchmark for Video Deepfake Reasoning in Vision Language Models](beyond_static_artifacts_a_forensic_benchmark_for_video_deepfake_reasoning_in_vis.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)

<!-- RELATED:END -->
