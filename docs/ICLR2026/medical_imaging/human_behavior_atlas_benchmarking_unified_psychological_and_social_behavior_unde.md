---
title: >-
  [论文解读] Human Behavior Atlas: Benchmarking Unified Psychological and Social Behavior Understanding
description: >-
  [ICLR 2026][医学图像][行为理解基准] 构建 Human Behavior Atlas——首个覆盖情感、认知、病理和社会过程四大维度的大规模多模态行为理解统一基准（101K+ 样本），并训练三种 OmniSapiens-7B 模型变体验证其在多任务训练和迁移学习中的有效性。
tags:
  - ICLR 2026
  - 医学图像
  - 行为理解基准
  - 心理与社会行为
  - 多模态学习
  - 统一模型
  - 情感计算
---

# Human Behavior Atlas: Benchmarking Unified Psychological and Social Behavior Understanding

**会议**: ICLR 2026  
**arXiv**: [2510.04899](https://arxiv.org/abs/2510.04899)  
**代码**: 待审稿后公开  
**领域**: 多模态 / 行为理解  
**关键词**: 行为理解基准, 心理与社会行为, 多模态学习, 统一模型, 情感计算

## 一句话总结

构建 Human Behavior Atlas——首个覆盖情感、认知、病理和社会过程四大维度的大规模多模态行为理解统一基准（101K+ 样本），并训练三种 OmniSapiens-7B 模型变体验证其在多任务训练和迁移学习中的有效性。

## 研究背景与动机

利用智能系统感知心理和社会行为——即通过可观察行为和社会交互表现出来的情感、认知和病理状态——一直是 AI 领域的核心挑战。现有工作的主要问题：

**碎片化**: 每个任务（情感分析、抑郁检测、动作识别等）都有专门的数据集和单任务系统，缺乏跨任务的可扩展性和迁移能力

**格式不一致**: 数据集在输入表示（预提取特征 vs. 原始信号）、输出格式（主观标注 vs. 分类标签）和评估协议上高度异构

**重复投入**: 每个任务需要独立的架构设计、数据收集和训练流程，造成大量资源浪费

**缺乏统一模型**: 社区在训练能够同时理解情感、认知、病理和社会行为的统一模型方面进展有限

Human Behavior Atlas 旨在通过标准化数据格式、统一评估指标来填补这一空白，推动通用行为理解模型的发展。

## 方法详解

### 整体框架

Human Behavior Atlas 的构建遵循五步流程：（1）定义行为分类体系；（2）收集对齐的多模态数据集；（3）统一数据格式为 prompt-target 形式；（4）标准化评估指标；（5）提取行为描述符增强基准。

### 关键设计

1. **行为分类体系（四大维度）**:

    - **情感状态 (Affective States)**: 情绪和情绪化——从短期感受（愤怒、快乐）到持续性情感
    - **认知状态 (Cognitive States)**: 注意力、推理、惊讶、决策等内在心理过程
    - **病理 (Pathology)**: 心理和精神疾病状态——抑郁、焦虑等
    - **社会过程 (Social Processes)**: 社会互动和交流行为——幽默、意图、合作等
    - 一个任务可对应多个维度（如情绪识别涉及情感和认知）

2. **数据集收集与统一（13个公开数据集）**:

    - 覆盖 10 项行为任务: 情感极性 (SEN)、情绪识别 (EMO)、社会推理 (SOC)、意图识别 (INT)、非语言交流 (NVC)、幽默检测 (HUM)、讽刺检测 (SAR)、焦虑检测 (ANX)、抑郁检测 (DEP)、PTSD 检测 (PTSD)
    - 总计 101,964 个样本，跨文本、音频、视频三种模态
    - 统一为 prompt-target 格式: prompt 引用可用模态，target 为自由文本或离散标签集
    - 连续输出（如 PHQ-9 分数）按原始论文指南离散化

3. **标准化评估框架**:

    - SEN: 二元加权 F1（正/负情感）
    - EMO: 各类别加权准确率的均值
    - HUM/SAR/ANX/DEP/PTSD: 加权 F1
    - SOC/INT/NVC: LLM-Judge 准确率（GPT-5-nano 评判生成回答是否匹配参考答案）
    - 统一了情绪标签（合并 joy/happiness，区分 positive/negative surprise）

4. **行为描述符提取**:

    - **视觉**: MediaPipe 提取面部特征点和体姿关键点
    - **音频**: OpenSMILE (ComParE 2016) 提取韵律特征（音高、能量、谱属性）
    - **文本**: Whisper v3 Large 转录缺失的文本

5. **三种 OmniSapiens-7B 模型变体**:

    - **OmniSapiens-7B SFT**: 基于 Qwen2.5-Omni-7B 的监督微调，使用倒数第二层表示经分类头和解码头处理不同任务类型
    - **OmniSapiens-7B BAM**: 在 SFT 冻结后附加 Behavioral Adapter Module（残差式适配器），整合行为描述符。公式：$h_{\text{adapt}} = h_{\text{penult}} + \alpha \cdot z_f$，其中 $z_f$ 由行为描述符经 FFN 处理得到
    - **OmniSapiens-7B RL**: 使用 GRPO（Group Relative Policy Optimization）强化学习训练，统一使用解码头和推理链格式 `<think>...</think>\boxed{answer}`

### 训练策略

- SFT: 5 epochs, LoRA ($r=32, \alpha=64$), 学习率 $1 \times 10^{-4}$, batch size 512
- BAM: 4 epochs, 冻结骨干，只训练适配器和头, 隐藏维度 256
- RL: 10 epochs, 学习率 $1 \times 10^{-6}$, 组采样 $n=5$, 复合奖励函数（准确率 + 格式 + 语义相似度）

## 实验关键数据

### 多任务训练主结果

| 模型 | EMO | HUM | INT | PTSD | ANX | DEP | SEN | SAR | SOC | NVC |
|------|-----|-----|-----|------|-----|-----|-----|-----|-----|-----|
| Gemma-3-4B | .550 | .597 | .227 | .499 | .601 | .463 | .738 | .529 | .191 | .023 |
| Qwen2.5-Omni-7B | .583 | .543 | .254 | .760 | .793 | .714 | .672 | .656 | .254 | .069 |
| HumanOmniV2-7B | .597 | .638 | .263 | .824 | .527 | .654 | .742 | .395 | .282 | .093 |
| **OmniSapiens-7B SFT** | **.631** | .532 | .256 | **1.00** | .909 | .733 | **.768** | .624 | .257 | .121 |
| **OmniSapiens-7B BAM** | **.645** | **.644** | .177 | **1.00** | **.909** | **.789** | **.786** | **.795** | .201 | **.162** |
| **OmniSapiens-7B RL** | .573 | **.639** | **.486** | .968 | .919 | .772 | .396 | .647 | **.304** | .133 |

SFT 和 BAM 在 10 项中 8 项超越通用多模态 LLM。

### 迁移学习实验

| 数据集 | OmniSapiens-7B SFT | Qwen2.5-Omni-7B | 提升 |
|--------|-------------------|----------------|------|
| MOSEI (SEN) | 0.724 | 0.612 | +18.3% |
| MELD (EMO) | 0.711 | 0.684 | +3.95% |
| DAIC-WOZ (DEP) | 0.749 | 0.579 | +29.4% |
| MUStARD (SAR) — 新任务 | 0.658 | 0.473 | **+39.1%** |

### 行为描述符效果 (BAM vs SFT)

| 任务 | SFT | BAM | 变化 |
|------|-----|-----|------|
| NVC | 0.12 | 0.16 | +33.0% |
| SAR | 0.62 | 0.80 | +29.0% |
| HUM | 0.53 | 0.64 | +21.0% |
| DEP | 0.73 | 0.79 | +8.2% |
| SOC | 0.26 | 0.20 | -23.1% |
| INT | 0.26 | 0.18 | -30.8% |

### 关键发现

- **SFT vs RL 的互补性**: SFT 在结构化分类任务上更强，RL 在开放式生成任务（INT、SOC）上更优，体现了两种训练策略的互补性
- **行为描述符的选择性收益**: BAM 在依赖细微面部/语音线索的任务（NVC、SAR、HUM）上大幅提升，但在需要推理的任务（SOC、INT）上反而下降，说明描述符应选择性使用而非全局应用
- **预训练支撑的语用识别能力**: 在讽刺检测任务上，OmniSapiens-7B 能识别语用线索（如 Chandler 的反讽），而 Qwen2.5-Omni-7B 默认预测"无讽刺"（93.2% 预测率）
- **跨任务迁移**: 即使在预训练中未见过 SAR 任务，在 Human Behavior Atlas 上预训练也能提升 39.1% 的迁移效果

## 亮点与洞察

1. **系统性基准构建方法论**: 论文不仅提供数据集，更提出了构建"行为图谱"的方法论框架——从分类体系定义、数据标准化、评估指标统一到模型评测，可推广到自闭症等特定领域
2. **端到端与特征工程的结合**: 通过残差式 BAM 适配器实现行为描述符的非侵入式整合，既不破坏骨干表示，又能按需增强特定任务
3. **RL 在行为理解中的潜力**: OmniSapiens-7B RL 展示了强化学习在需要推理的社会理解任务中的独特优势，暗示未来混合训练策略的方向
4. **数据来源多样性**: 数据集源自北美、欧洲和亚洲的多个地区，具有一定的文化多样性

## 局限与展望

1. **样本量不均衡**: 各任务数据量差异悬殊（CMU-MOSEI 31K vs DAIC-WOZ 189），可能影响多任务训练的平衡
2. **评估依赖 LLM Judge**: SOC/INT/NVC 使用 GPT-5-nano 评判，其一致性和偏差未充分分析
3. **缺乏真实场景验证**: 所有数据来自实验室或影视剧场景，与真实自然交互存在差距
4. **情绪标签合并的主观性**: 将 joy/happiness 合并、拆分 surprise 的决策缺乏严格的理论依据
5. **模型规模受限**: 仅测试 7B 参数模型，未探索规模缩放效应
6. **隐私与伦理**: 使用真实人类行为数据涉及隐私和知情同意问题，论文讨论不够深入

## 相关工作与启发

- **eMotions (Wu et al., 2025)**: 短视频情感分析数据集，但仅覆盖情绪识别单一任务
- **HumanOmni (Zhao et al., 2025)**: 以人为中心的理解数据集，但主要针对人体场景理解而非心理行为
- **PaLI / BLIP / Kosmos**: 大规模多模态预训练的范例，证明了多任务预训练的泛化能力
- **Affective Computing (Picard, 2000)**: 情感计算的开创性工作，本文将其视野扩展到认知、病理和社会维度

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个覆盖四大行为维度的统一基准，方法论框架有推广价值
- 实验充分度: ⭐⭐⭐⭐⭐ — 三种模型变体、多任务+迁移+描述符消融，分析全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，数据呈现直观，但部分细节需查阅附录
- 价值: ⭐⭐⭐⭐ — 填补统一行为理解基准的空白，为社区提供重要研究基础设施

<!-- RELATED:START -->

## 相关论文

- [Unlocking Multi-Site Clinical Data: A Federated Approach to Privacy-First Child Autism Behavior Analysis](../../CVPR2026/medical_imaging/unlocking_multi-site_clinical_data_a_federated_approach_to_privacy-first_child_a.md)
- [MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding](../../ICML2025/medical_imaging/medxpertqa_benchmarking_expert-level_medical_reasoning_and_understanding.md)
- [DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction](dm4ct_benchmarking_diffusion_models_for_computed_tomography_reconstruction.md)
- [Boosting Medical Visual Understanding From Multi-Granular Language Learning](boosting_medical_visual_understanding_from_multi-granular_language_learning.md)
- [Unified Biomolecular Trajectory Generation via Pretrained Variational Bridge](unified_biomolecular_trajectory_generation_via_pretrained_variational_bridge.md)

<!-- RELATED:END -->
