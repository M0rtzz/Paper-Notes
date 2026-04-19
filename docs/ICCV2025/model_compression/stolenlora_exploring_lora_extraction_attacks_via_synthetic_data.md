---
title: >-
  [论文解读] StolenLoRA: Exploring LoRA Extraction Attacks via Synthetic Data
description: >-
  [ICCV 2025][模型压缩][LoRA extraction] StolenLoRA 首次提出针对 LoRA 自适应模型的模型提取攻击方向，利用 LLM 驱动的 Stable Diffusion 生成高质量合成数据替代真实数据集搜索，并设计基于分歧的半监督学习（DSL）策略通过选择性查询最大化信息增益，仅需 10k 次查询即可达到高达 96.60% 的攻击成功率，揭示了 LoRA 适配模型的严重安全漏洞。
tags:
  - ICCV 2025
  - 模型压缩
  - LoRA extraction
  - model extraction attack
  - PEFT
  - synthetic data
  - 扩散模型
  - 半监督学习
  - 提示学习
---

# StolenLoRA: Exploring LoRA Extraction Attacks via Synthetic Data

**会议**: ICCV 2025  
**arXiv**: [2509.23594](https://arxiv.org/abs/2509.23594)  
**代码**: 待确认  
**领域**: 模型安全 / 模型提取攻击 / LoRA / 参数高效微调  
**关键词**: LoRA extraction, model extraction attack, PEFT, synthetic data, Stable Diffusion, disagreement-based semi-supervised learning, LLM-driven prompting

## 一句话总结
StolenLoRA 首次提出针对 LoRA 自适应模型的模型提取攻击方向，利用 LLM 驱动的 Stable Diffusion 生成高质量合成数据替代真实数据集搜索，并设计基于分歧的半监督学习（DSL）策略通过选择性查询最大化信息增益，仅需 10k 次查询即可达到高达 96.60% 的攻击成功率，揭示了 LoRA 适配模型的严重安全漏洞。

## 研究背景与动机

LoRA（Low-Rank Adaptation）已成为大规模预训练模型高效微调的主流方法，但其轻量和紧凑的特性带来了新的安全隐患：

**LoRA 参数的脆弱性**：LoRA 仅微调少量低秩矩阵，模型的核心知识依赖公开的预训练基础模型。这意味着攻击者只需复制 LoRA 适配部分即可获得完整模型功能，门槛远低于传统全模型提取。

**现有提取方法的局限**：
   - **样本选择方法**（KnockoffNets/ActiveThief）：需要搜索庞大数据集寻找域内样本，对 LoRA 模型推理速度慢的情况计算开销巨大，且难以找到领域特定的合适样本。
   - **GAN 生成方法**（DFME）：GAN 在生成高维数据（如 224×224 图像）时困难重重，需要数百万次查询，且易遭遇 mode collapse。

**预训练模型公开可得**：大量预训练 ViT 模型公开发布在 Hugging Face 等平台，攻击者可轻松获取与受害者相同或相似的 base model，进一步降低了攻击门槛。

**研究空白**：传统模型提取研究关注整个模型的功能复制，但 LoRA 场景下只需提取紧凑的适配参数，这一新方向尚未被充分探索。

## 方法详解

### 问题定义：LoRA 提取

给定受害模型 F = F_base + ΔF（其中 F_base 公开，ΔF 为 LoRA 调整），攻击者目标是训练替代模型 F' = F'_base + ΔF' 使其功能尽可能接近 F。

**两种攻击场景**：
- **同 backbone 场景 (IB)**：攻击者使用与受害者相同的预训练模型 F_base。
- **跨 backbone 场景 (XB)**：攻击者使用不同的预训练模型 G_base（如受害者用 ImageNet-21k 监督预训练 ViT，攻击者用 MAE 自监督预训练 ViT）。

### 阶段一：LLM 驱动的数据合成

攻击者通常可获取模型的部署上下文信息（功能描述、目标类别名），以此为起点合成训练数据：

1. **LLM 驱动提示生成**：
    - 给定目标类名集合 C = {c_1, ..., c_n}，用 GPT-4o mini 生成多样化的图像描述。
    - 提示模板 T = [Subject, Background, Angle/Pose, Lighting, Style]。
    - 每个类别生成 m 个不同的提示变体：p_{i,j} = LLM(c_i, T, ω_j)。

2. **图像合成**：
    - 用 SDXL-Turbo 模型（仅需 4 步采样）对每个提示生成一张图像。
    - 完整合成数据集：X = ∪_i {SD(p_{i,j})}。
    - 合成图像自带类别伪标签（来自生成提示的类别信息）。

### 阶段二：高效查询与训练

**方案 A：随机学习 (StolenLoRA-Rand)**

直接用合成数据集查询受害模型 API，用交叉熵损失训练替代模型的 LoRA 参数。作为基线方案。

**方案 B：基于分歧的半监督学习 (StolenLoRA-DSL)**

DSL 通过选择性查询和标签精炼迭代提升攻击效率：

1. **初始化**：生成初始合成数据集 X_0，用伪标签（来自生成提示）训练初始替代模型 F'_0。
2. **迭代过程（每轮 t）**：
    - 生成 β·b_t 个新候选样本 X_cand^t。
    - **分歧过滤**：用当前替代模型 F'_t 对每个候选预测类别 ĉ 和置信度 p̂：
     - 若 ĉ = 伪标签 c(x) 且 p̂ ≥ τ（阈值 0.95）：样本置信度高，加入 X_conf^t，使用伪标签直接训练，无需查询。
     - 否则：样本不确定，加入 X_uncer^t。
    - **选择性查询**：从 X_uncer^t 中选取置信度最低的 b_t 个样本查询受害模型获取真实标签。
    - **训练**：合并 X_conf^t（伪标签）和 X_query^t（真实标签）训练更新替代模型。
3. **标签精炼 (Label Refining)**：
    - 用 EMA 更新软标签：q^(i+1) = μ·q^(i) + (1-μ)·p^(i+1)。
    - 用软标签的交叉熵损失训练，缓解伪标签噪声和分布偏移。
4. 循环直到查询预算耗尽。

### 关键设计思想

- **聚焦信息增益最大化**：只查询替代模型"不确定"的样本，避免浪费查询预算在已经掌握的简单样本上。
- **合成数据 + 伪标签减少查询依赖**：大部分高置信样本直接用伪标签训练，实际查询量远少于合成数据量。
- **EMA 标签精炼的双重作用**：既抑制了伪标签噪声，又桥接了合成数据与真实分布的偏移。

## 实验关键数据

### 主要结果（10k 查询预算）

#### 同 backbone 场景 (IB)

| 方法 | CUBS200 Acc/ASR | Caltech256 Acc/ASR | Indoor67 Acc/ASR | Food101 Acc/ASR | Flowers102 Acc/ASR |
|------|---:|---:|---:|---:|---:|
| KnockoffNets | 70.95/80.54 | 85.69/90.71 | 79.18/92.59 | 82.53/90.62 | 76.24/77.28 |
| ActiveThief | 72.33/82.11 | 86.92/92.01 | 77.46/90.57 | 80.34/88.22 | 77.51/78.56 |
| DFME | 0.56/0.64 | 0.48/0.51 | 2.71/3.17 | 1.62/1.78 | 2.81/2.85 |
| E³ | 71.94/81.67 | 82.36/87.18 | 81.43/95.22 | 79.65/87.46 | 80.67/81.77 |
| **StolenLoRA-Rand** | **75.35/85.54** | 87.62/92.75 | 82.38/96.33 | 79.00/86.75 | **93.74/95.01** |
| **StolenLoRA-DSL** | 73.23/83.13 | **89.30/94.53** | **82.61/96.60** | **80.57/88.47** | 87.46/88.65 |

#### 跨 backbone 场景 (XB)

| 方法 | CUBS200 Acc/ASR | Caltech256 Acc/ASR | Indoor67 Acc/ASR |
|------|---:|---:|---:|
| KnockoffNets | 6.77/7.69 | 41.34/43.76 | 41.79/48.87 |
| ActiveThief | 15.42/17.50 | 42.89/45.40 | 36.04/42.14 |
| E³ | 17.55/19.92 | 48.17/50.99 | 55.85/65.31 |
| **StolenLoRA-Rand** | **45.70/51.88** | 51.75/54.78 | 59.18/69.20 |
| **StolenLoRA-DSL** | **50.14/56.92** | **65.01/68.82** | **65.07/76.09** |

- IB 场景最高 ASR 达 96.60%（Indoor67），仅用 10k 查询
- XB 场景大幅超越所有基线，CUBS200 上 StolenLoRA-DSL 的 ASR 是 KnockoffNets 的 7.4 倍
- DFME（GAN-based）在 LoRA 提取中完全失效（ASR < 3.17%），验证了 GAN 方法不适用于此场景

### 硬标签场景

仅提供 one-hot 预测（无概率分布）时，StolenLoRA 仍保持竞争力：
- IB 场景性能小幅下降（如 CUBS200 从 75.35% → 67.00%）
- XB 场景某些数据集反而略有提升（Caltech256 从 51.75% → 56.69%），硬标签起到了正则化效果

### 消融实验要点

| 组件 | IB CUBS200 | XB CUBS200 |
|------|---:|---:|
| Full StolenLoRA-DSL | 73.23 | 50.14 |
| - Template | 72.33 | 42.21 |
| - LLM | 降低显著 | 降低更显著 |
| - DSL (→ Random) | 降低 | 降低明显 |
| GPT-4o → Llama-3.1-8B | 轻微下降 | 下降更多 |

- LLM 驱动提示是核心，去除后在 XB 场景下降最严重
- 置信度阈值 τ = 0.95 为最优值，过高 (0.99) 排除了有价值样本

## 亮点与洞察

- **新的安全方向**：首次系统定义并研究 LoRA 提取攻击，揭示了 PEFT 方法带来的新安全面。LoRA 的紧凑性本是优势，在安全视角下却成为漏洞。
- **LLM + 扩散模型的攻击数据供应链**：用 LLM 生成多样化文本描述 → SDXL-Turbo 合成图像 → 自带伪标签，完全绕过了对真实数据集的依赖，攻击成本极低。
- **DSL 的精巧查询优化**：不是均匀查询，而是聚焦在替代模型"困惑"的样本上，实现了查询预算的最大化利用。10k 查询即可达到 96.60% ASR。
- **跨 backbone 攻击的可行性**：即使攻击者不知道确切的预训练模型，仍可实现有效提取（XB 场景 ASR 高达 76.09%），这使得攻击在实际场景中更具威胁性。
- **EMA 标签精炼的双重作用**：既抑制了伪标签噪声，又桥接了合成数据与真实分布的偏移，一个简单机制解决两个问题。

## 局限与展望

- 实验仅在 ViT-Base 上验证，未涉及更大的 ViT 变体或非 ViT 架构（如 CNN、混合架构）。
- 仅考虑图像分类任务的 LoRA 提取，更复杂的任务（检测、分割、NLP）中 LoRA 提取的难度和可行性未知。
- LoRA rank 固定为 r=4，不同 rank 对提取难度的影响未系统分析。
- 防御策略（多样化 LoRA 部署）仅作为初步探索，缺乏更全面的防御方案。
- 合成数据质量依赖 LLM 和 SD 模型，对于高度专业化的领域（如医学影像）可能难以生成足够准确的域内数据。
- 未考虑受害模型 API 的速率限制、查询检测等实际部署中的防护措施。
- 攻击者需要知道目标类别名才能生成合成数据，对全黑盒场景有一定限制。

## 相关工作与启发

- **vs. KnockoffNets**：KnockoffNets 搜索 CC3M 等大规模数据集选择查询样本，需要遍历 3M 图像。StolenLoRA 的合成策略更高效且无需真实数据集。
- **vs. DFME**：DFME 用 GAN 生成查询数据，在 LoRA 场景完全失败（ASR < 3.17%），因为 GAN 难以生成 224×224 高质量域内图像。StolenLoRA 用 Stable Diffusion 彻底解决了这一限制。
- **vs. E³**：E³ 基于语义相似度从真实数据集中筛选样本，StolenLoRA 在所有场景中均超越 E³，且不依赖真实数据集。
- **对 PEFT 安全研究的警示**：LoRA 的广泛部署（Hugging Face 上大量共享的 LoRA 适配器）需要配套的知识产权保护机制。未来可探索 LoRA 水印、差分隐私训练、输出扰动等防御手段。
- **合成数据在安全研究中的新角色**：合成数据不仅可用于数据增强和训练，还可被武器化用于攻击场景，安全社区需要关注这一双面性。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次定义 LoRA 提取问题，LLM+SD 合成数据链和 DSL 策略均为本文原创贡献
- 实验充分度: ⭐⭐⭐⭐ 5 个数据集 × 2 种场景 × 4 个对比方法 + 消融 + 硬标签 + 超参分析
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，IB/XB 场景划分合理，算法伪代码规范完整
- 价值: ⭐⭐⭐⭐ 揭示了 PEFT 时代的新型安全威胁，对 LoRA 生态的安全实践有直接指导意义

<!-- RELATED:START -->

## 相关论文

- [WildChat-50m: A Deep Dive Into the Role of Synthetic Data in Post-Training](../../ICML2025/model_compression/wildchat-50m_a_deep_dive_into_the_role_of_synthetic_data_in_post-training.md)
- [OuroMamba: A Data-Free Quantization Framework for Vision Mamba](ouromamba_a_data-free_quantization_framework_for_vision_mamba.md)
- [Toward Efficient Inference Attacks: Shadow Model Sharing via Mixture-of-Experts](../../NeurIPS2025/model_compression/toward_efficient_inference_attacks_shadow_model_sharing_via_mixture-of-experts.md)
- [Predictive Data Selection: The Data That Predicts Is the Data That Teaches](../../ICML2025/model_compression/predictive_data_selection_the_data_that_predicts_is_the_data_that_teaches.md)
- [Partial Forward Blocking: A Novel Data Pruning Paradigm for Lossless Training Acceleration](partial_forward_blocking_a_novel_data_pruning_paradigm_for_lossless_training_acc.md)

<!-- RELATED:END -->
