---
title: >-
  [论文解读] MMBoundary: Advancing MLLM Knowledge Boundary Awareness through Reasoning Step Confidence Calibration
description: >-
  [ACL 2025][多模态][多模态大模型] 提出 MMBoundary 框架，通过对 MLLM 推理链中每一步进行置信度校准（而非仅对整体回答），结合文本+跨模态自奖励信号与强化学习，显著降低多模态置信度校准误差（平均 7.5%）并提升任务性能（最高 8.3%）。
tags:
  - ACL 2025
  - 多模态
  - 多模态大模型
  - 置信度校准
  - 推理链
  - 知识边界
  - 强化学习
  - 幻觉缓解
---

# MMBoundary: Advancing MLLM Knowledge Boundary Awareness through Reasoning Step Confidence Calibration

**会议**: ACL 2025  
**arXiv**: [2505.23224](https://arxiv.org/abs/2505.23224)  
**代码**: [GitHub](https://github.com/Zhitao-He/MMBoundary)  
**领域**: multimodal_vlm  
**关键词**: 多模态大模型, 置信度校准, 推理链, 知识边界, 强化学习, 幻觉缓解

## 一句话总结

提出 MMBoundary 框架，通过对 MLLM 推理链中每一步进行置信度校准（而非仅对整体回答），结合文本+跨模态自奖励信号与强化学习，显著降低多模态置信度校准误差（平均 7.5%）并提升任务性能（最高 8.3%）。

## 研究背景与动机

多模态大语言模型（MLLM）在跨模态推理方面取得了显著进展，但其回答的可靠性仍然存疑。现有的置信度估计方法只关注**整体回答**的置信度，忽略了推理链中**每一步**可能出现的错误。这导致以下问题：

**幻觉雪球效应**：感知阶段的错误（如将"鼓"误识为"盾"）会沿推理链传播和放大

**高置信度错误**：由于推理的逻辑连贯性，即使答案错误，模型仍可能对整体回答给出高置信度

**缺乏不确定性表达**：模型无法在推理过程中主动标示不确定的步骤，错失自我修正的机会

核心动机：让 MLLM 像人类一样，在推理的每一步都能表达置信度，从而实现推理链的自我修正。

## 方法详解

MMBoundary 框架分为两个阶段：

### 阶段一：置信度表达热身（SFT）

**1. 内部置信度估计模块**

融合四种互补的不确定性估计方法，对模型生成的每个句子计算综合置信度分数：

- **Length-normalized log probability (ULNLP)**：计算生成 token 的平均负对数概率
- **Mean Token Entropy (UMTE)**：计算每个 token 分布的平均熵
- **TokenSAR (UTSAR)**：按 token 与整体文本的相关性加权计算负对数概率
- **CLIPScore (UCLIPS)**：评估生成句子与输入图像的跨模态相关性

最终加权融合：$U_{Final} = w_0 U_{LNLP} + w_1 U_{MTE} + w_2 U_{TSAR} + w_3 U_{CLIPS}$

根据 $U_{Final}$ 的分布（均值 $\mu$、标准差 $\sigma$），将置信度分为 5 级：fully confident → uncertain。

**2. 置信度分数-陈述映射**

为每个置信度等级预设自然语言陈述池（如"I'm quite sure about this"→高置信），实现分数与自然语言表达的双向映射。SFT 阶段随机选取对应陈述插入到模型原始回答中构造训练数据；RL 阶段通过 sentence encoder + 余弦相似度实现陈述→分数的反向映射。

**3. 监督微调**

在构造的数据上微调模型，使其学会为每个生成句子自然地附加置信度陈述，输出格式为 $[z_1, c_1, z_2, c_2, \dots, z_T, c_T]$。

### 阶段二：强化学习校准（RL with PPO）

设计三个互补的奖励函数：

- **知识准确度奖励 $R_{KA}$**：评估生成内容是否与参考推理链一致，确保内容可靠性
- **期望校准奖励 $R_{EC}$**：衡量表达的置信度与真实正确性之间的相关性，扩展至句子级别
- **置信度自校准奖励 $R_{CS}$**：确保模型外部表达的置信度与内部估计的置信度一致

总奖励：$R = \alpha R_{KA} + \beta R_{EC} + \gamma R_{CS}$，使用 PPO 优化。

## 实验关键数据

### 表1：主实验结果（LLaVA-NEXT 7B）

| 方法 | A-OKVQA ECE↓ | A-OKVQA MECE↓ | A-OKVQA Acc↑ | A-OKVQA F1↑ | ScienceVQA MECE↓ | CulturalVQA MECE↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| DPV | 0.563 | 0.582 | 0.650 | 0.512 | 0.611 | 0.650 |
| SaySelf | 0.345 | 0.384 | 0.734 | 0.603 | 0.462 | 0.437 |
| Conf-CSR | 0.408 | 0.437 | 0.785 | 0.618 | 0.503 | 0.513 |
| RCE | 0.361 | 0.394 | 0.788 | 0.620 | 0.475 | 0.453 |
| **MMBoundary** | **0.316** | **0.304** | **0.835** | **0.661** | **0.392** | **0.361** |

MMBoundary 在所有数据集上全面领先：MECE 平均降低 7.5%，CulturalVQA（OOD）上准确率提升 8.3%。

### 表2：人工评估结果（1-10 分，3 位研究生评估）

| 方法 | A-OKVQA Avg | ScienceVQA Avg | CulturalVQA Avg |
|------|:---:|:---:|:---:|
| Multisample | 4.47 | 4.85 | 4.75 |
| SaySelf | 7.08 | 6.98 | 6.83 |
| RCE | 6.90 | 7.19 | 6.62 |
| **MMBoundary** | **7.75** | **7.94** | **7.69** |

在 Faithful、Concise、Granular 三个维度上均显著优于所有基线，Kappa 值 0.79 表明标注一致性高。

### 消融实验要点

- 去掉 RL 阶段：MECE 平均下降 9.2%，说明 RL 校准至关重要
- 四种不确定性方法均有贡献，其中 TokenSAR 对置信度校准影响最大
- 分数-陈述映射（S-S Mapping）带来平均 4.8% 的提升
- 内部置信度估计（ICE）与自一致性方法（SCE）差异极小（平均 <0.06），但无需多次采样

## 亮点

1. **细粒度置信度校准**：首次将置信度估计从整体回答级别推进到推理链的每一步，实现了句子级的知识边界感知
2. **文本+视觉的多信号融合**：创新性地结合三种文本不确定性方法和 CLIPScore 跨模态约束，无需多次采样即可高效估计内部置信度
3. **SFT+RL 两阶段训练**：三个精心设计的奖励函数（知识对齐 + 外部校准 + 内部自校准）协同工作，兼顾任务性能和置信度精度
4. **推理链自修正**：低置信度步骤在推理时触发自我修正，有效阻断幻觉的传播

## 局限性

1. 内部不确定性方法能否准确反映模型对输出的真实置信度仍需更多研究验证
2. 依赖 GPT-4o 生成推理链标注数据，引入额外成本和潜在的标注偏差
3. 仅在 LLaVA-NEXT 7B 和 Qwen2VL 7B 上验证，对更大规模模型的适用性未知
4. 置信度分为 5 级的粒度划分基于经验（均值加减标准差），缺乏理论最优性保证
5. 当前仅在 VQA 任务上评估，更复杂的多模态推理场景（如视频理解、多轮对话）需进一步验证

## 相关工作对比

| 方法类别 | 代表工作 | MMBoundary 的优势 |
|---------|---------|-----------------|
| 整体置信度估计 | SaySelf, Multisample | 推理链每步都有置信度，避免高置信度错误 |
| 基于自一致性 | SC, Multisample | 无需多次采样，单次前向即可估计置信度 |
| 基于 DPO 优化 | Conf-CSR | 三奖励 PPO 同时优化知识对齐和双重校准 |
| 后验置信度 | RCE | 生成过程中实时表达置信度，而非事后评估 |
| 直接提示 | DPV, DPS | 通过 SFT+RL 训练获得校准能力，非零样本 |

## 评分

- 新颖性: ⭐⭐⭐⭐ — 推理步级别的置信度校准是一个有意义的新视角，多信号融合设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ — 三个数据集（含 OOD）、8 个基线、详实消融、人工评估、统计检验齐全
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表直观，方法描述完整
- 价值: ⭐⭐⭐⭐ — 解决了多模态推理中幻觉传播的实际痛点，框架具有较好的通用性

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World](craftext_benchmark_advancing_instruction_following_in_complex_multimodal_open-en.md)
- [\[ACL 2025\] COLING-UniA at SciVQA 2025: Few-Shot Example Retrieval and Confidence-Informed Ensembling for Multimodal Large Language Models](coling-unia_at_scivqa_2025_few-shot_example_retrieval_and_confidence-informed_en.md)
- [\[ACL 2025\] Activation Steering Decoding: Mitigating Hallucination in Large Vision-Language Models through Bidirectional Hidden State Intervention](activation_steering_decoding_mitigating_hallucination_in_large_vision-language_m.md)
- [\[ACL 2025\] GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art](godbench_a_benchmark_for_multimodal_large_language_models_in_video_comment_art.md)
- [\[ACL 2025\] CoSyn: Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation](cosyn_code_guided_synthetic_data.md)

<!-- RELATED:END -->
