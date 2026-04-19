---
title: >-
  [论文解读] Adaptive Multi-prompt Contrastive Network for Few-shot Out-of-distribution Detection
description: >-
  [ICML2025][LLM/NLP][OOD检测] 提出 AMCN（Adaptive Multi-prompt Contrastive Network），通过生成三类自适应文本 prompt（可学习 ID prompt、标签固定 OOD prompt、标签自适应 OOD prompt）并结合类别自适应阈值，在仅有少量 ID 标注样本的条件下实现高质量 OOD 检测，显著超越现有 few-shot OOD 检测方法。
tags:
  - ICML2025
  - LLM/NLP
  - OOD检测
  - few-shot学习
  - 提示学习
  - CLIP
  - 对比学习
  - 自适应阈值
---

# Adaptive Multi-prompt Contrastive Network for Few-shot Out-of-distribution Detection

**会议**: ICML2025  
**arXiv**: [2506.17633](https://arxiv.org/abs/2506.17633)  
**代码**: GitHub（作者声明开源）  
**领域**: llm_nlp  
**关键词**: OOD检测, few-shot学习, prompt learning, CLIP, 对比学习, 自适应阈值

## 一句话总结

提出 AMCN（Adaptive Multi-prompt Contrastive Network），通过生成三类自适应文本 prompt（可学习 ID prompt、标签固定 OOD prompt、标签自适应 OOD prompt）并结合类别自适应阈值，在仅有少量 ID 标注样本的条件下实现高质量 OOD 检测，显著超越现有 few-shot OOD 检测方法。

## 研究背景与动机

OOD 检测旨在识别与训练分布不一致的异常样本，防止模型对未知类别样本做出错误预测，在自动驾驶等安全关键场景中至关重要。然而，现有大多数 OOD 检测方法依赖大量 ID（in-distribution）样本进行训练，严重限制了实际应用。

本文瞄准一个更具挑战性的设定：**few-shot OOD 检测**——每个 ID 类别仅有少量（如 1 或 8 个）标注样本。该设定面临三大核心挑战：

**背景敏感性**：少样本训练容易导致模型对图像背景产生理解偏差。例如猫主要拍摄于室内、狗主要拍摄于室外，模型可能依赖背景而非物体本身进行判断。

**类别数增加时性能下降**：随着 ID 类别增多，ID-OOD 边界变得更加复杂，类别特征重叠加剧，模型难以学习细粒度差异。

**过拟合风险**：有限样本下模型容易过拟合，泛化能力严重受限。

此外，作者发现不同类别具有**不同程度的多样性**（multi-diversity），例如"猫"类的样本多样性远高于"牛"类，但已有方法忽略了这一类间多样性差异，使用统一阈值进行 OOD 判定，效果受限。

## 方法详解

### 整体框架

AMCN 基于 CLIP 视觉-语言预训练模型，包含三个核心模块：

1. **Adaptive Prompt Generation（自适应 Prompt 生成）**：为 ID 分类生成三类文本 prompt
2. **Prompt-based Multi-diversity Distribution Learning（基于 Prompt 的多样性分布学习）**：学习各类别的类内/类间分布，生成自适应阈值
3. **Prompt-guided OOD Detection（Prompt 引导的 OOD 检测）**：通过 ID-OOD 分离损失和多 prompt 对比学习实现精确检测

### 关键设计 1：三类自适应 Prompt

为弥补 OOD 样本缺失和 ID 样本稀缺的问题，利用 CLIP 的文本-图像连接能力，设计三类 prompt：

- **可学习 ID Prompt (LIP)**：$f_{lip}^i = [W_1][W_2]\dots[W_{N_{IP}}][y_i]$，其中前缀 token 可学习，标签名固定为 ID 类别名
- **标签固定 OOD Prompt (LFOP)**：$f_{lfop}^i = [M_1]\dots[M_{N_{lfop}}][o_i]$，前缀可学习，标签固定为外部数据集的 OOD 类别名（如 CIFAR-100 中不与 ID 重叠的类别"chair"）
- **标签自适应 OOD Prompt (LAOP)**：$f_{laop}^i = [H_1]\dots[H_{N_{laop}}][o_i']$，前缀和标签均可学习，标签由 OOD 类初始化后自由优化

三类 prompt 共享文本编码器参数，训练时将所有 OOD prompt 作为负样本，与 ID 图像特征进行多 prompt 对比学习。ID 分类损失 $\mathcal{L}_C$ 使用加权交叉熵形式，通过温度缩放余弦相似度拉近 ID prompt 与 ID 图像、推远 OOD prompt。

### 关键设计 2：类别自适应阈值与分布学习

不同类别多样性不同（如 ImageNet-1k 中"cat"和"ox"的 T-SNE 分布差异显著），统一阈值无法适应所有类别。AMCN 为每个类别学习独立的阈值：

- 计算类分布得分 $\mathbb{S}_c(x_i) = \exp(o_c(x_i)) / (\tau_0 + \mathcal{M}_c^{pse})$
- 估计类内均值 $\mu_c$ 和标准差 $\sigma_c$
- 定义 **P-score** 自适应阈值：$P_c = \lambda \cdot \mu_c + (1-\lambda) \cdot \sigma_c$
- 当 $\mathbb{S}_c(x_i) > P_c$ 时判为 pseudo-OOD；若样本对所有类均为 pseudo-OOD，则为真实 OOD

同时引入 pseudo-OOD 分布 $\mathcal{M}_c^{pse}$ 的动量更新机制，在推理过程中在线自适应调整。

### 关键设计 3：Prompt 引导的 ID-OOD 分离

为明确 ID 与 OOD prompt 特征之间的边界，设计两个关键损失：

- **ID-OOD 分离损失** $\mathcal{L}_2$：在单位超球面上约束 OOD prototype 与 ID 图像的欧氏距离大于 ID prototype 与 ID 图像的距离，确保 OOD 特征被推远
- **OOD 对齐损失** $\mathcal{L}_3$：约束 LAOP（可学习标签）与 LFOP（固定标签）的特征原型在归一化空间中对齐，防止 LAOP 偏离 OOD 语义空间、滑向 ID 语义

### 训练策略

总损失由四部分组成：$\mathcal{L} = \mathcal{L}_1 + \alpha_1 \mathcal{L}_2 + \alpha_2 \mathcal{L}_3 + \alpha_3 \mathcal{L}_4$

| 损失 | 作用 | 说明 |
|------|------|------|
| $\mathcal{L}_C$ | ID 分类 | 加权多 prompt 对比学习 |
| $\mathcal{L}_I^1$ | 类内分布归一化 | 平衡各类 ID/OOD 得分比例 |
| $\mathcal{L}_I^2$ | 类间分布归一化 | 跨类平衡 ID/OOD 分布 |
| $\mathcal{L}_2$ | ID-OOD 分离 | 确保 OOD prototype 远离 ID 图像 |
| $\mathcal{L}_3$ | OOD 对齐 | 约束 LAOP 与 LFOP 语义对齐、远离 ID |
| $\mathcal{L}_4$ | OOD 检测对比学习 | 最小化 ID-OOD prompt 相似度 |

其中 $\mathcal{L}_1 = \mathcal{L}_C + \mathcal{L}_I^1 + \mathcal{L}_I^2$。超参设置：$\alpha_1=0.4, \alpha_2=0.2, \alpha_3=0.8$，优化器 AdamW，学习率 0.003，batch size 64，token 长度 16，训练 100 epoch。LFOP 和 LAOP 数量均设为 50。

## 实验关键数据

### 主实验：Few-shot OOD 检测性能对比（ImageNet-1k 为 ID）

| 方法 | Shot | Texture FPR95↓ | Texture AUROC↑ | Places FPR95↓ | Places AUROC↑ | SUN FPR95↓ | SUN AUROC↑ | iNat FPR95↓ | iNat AUROC↑ | Avg FPR95↓ | Avg AUROC↑ |
|------|------|--------|---------|--------|---------|--------|---------|--------|---------|--------|---------|
| KNN | Full | 64.35 | 85.67 | 39.61 | 91.02 | 35.62 | 92.67 | 29.17 | 94.52 | 42.19 | 90.97 |
| NPOS | Full | 46.12 | 88.80 | 45.27 | 89.44 | 43.77 | 90.44 | 16.58 | 96.19 | 37.94 | 91.22 |
| GL-MCM | 0 | 57.93 | 83.63 | 38.85 | 89.90 | 30.42 | 93.09 | 15.16 | 96.71 | 35.59 | 90.83 |
| SCT | 1 | 48.87 | 86.66 | 32.81 | 91.23 | 23.52 | 94.58 | 19.16 | 95.70 | 31.09 | 92.04 |
| **AMCN** | **1** | **39.16** | **89.88** | **32.76** | **92.78** | **23.26** | **94.85** | **18.84** | **96.18** | **30.87** | **92.47** |
| SCT | 8 | 40.35 | 91.82 | 38.77 | 92.41 | 23.48 | 94.77 | 18.65 | 95.82 | 32.32 | 93.53 |
| **AMCN** | **8** | **38.31** | **93.43** | **32.45** | **93.96** | **23.17** | **95.89** | **18.17** | **96.89** | **30.56** | **94.29** |

- 1-shot 设定下，AMCN 在 Texture 上 FPR95 比 SCT 降低 **9.71%**，AUROC 提升 **3.22%**
- 8-shot 设定下，AMCN 平均 AUROC 达 94.29%，超越所有 fully-supervised 基线
- AMCN 甚至在 1-shot 设定下超越了使用全部训练数据的 KNN、ViM、ODIN 等方法

### 消融实验：各模块贡献（Texture 数据集）

| M1 (Prompt生成) | M2 (分布学习) | M3 (OOD检测) | 1-shot FPR95↓ | 1-shot AUROC↑ | 8-shot FPR95↓ | 8-shot AUROC↑ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ✗ | ✔ | ✔ | 41.36 | 82.59 | 40.82 | 86.24 |
| ✔ | ✗ | ✔ | 40.95 | 83.24 | 40.26 | 86.88 |
| ✔ | ✔ | ✗ | 40.35 | 83.19 | 39.72 | 87.92 |
| ✔ | ✔ | ✔ | **39.16** | **89.88** | **38.31** | **93.43** |

三个模块缺一不可，移除任一模块均导致 AUROC 下降 5-7 个百分点，说明自适应 prompt、分布学习和 ID-OOD 分离三者形成了有效协同。

### 自适应阈值 vs 固定阈值（iNaturalist）

| 阈值类型 | 1-shot FPR95↓ | 1-shot AUROC↑ | 8-shot FPR95↓ | 8-shot AUROC↑ |
|:---:|:---:|:---:|:---:|:---:|
| Fixed | 20.70 | 93.83 | 20.51 | 94.16 |
| **Adaptive** | **18.84** | **96.18** | **18.17** | **96.89** |

自适应阈值带来 2.35%/2.73% 的 AUROC 提升，证实了类级别决策边界的必要性。

## 亮点与洞察

1. **三类 prompt 的互补设计精巧**：LIP 负责 ID 表征学习，LFOP 引入人类先验 OOD 知识，LAOP 探索潜在 OOD 语义空间，三者从不同角度构建 ID-OOD 分离边界
2. **类级自适应阈值是核心贡献**：认识到不同类的多样性差异并以 P-score 建模，融合类内均值和标准差，比全局固定阈值提升显著
3. **无需 OOD 训练数据**：完全不需要 OOD 图像样本，仅通过文本 prompt 构造 OOD 代理，充分发挥了 CLIP 跨模态能力
4. **1-shot 即超越 fully-supervised 基线**：仅用 1 个标注样本的性能已超过使用全部训练数据的 KNN/ViM/ODIN，展示了方法的高效性
5. **超参不敏感**：$\alpha_1, \alpha_2, \alpha_3$ 的敏感性分析表明模型在较大范围内保持稳定性能

## 局限性

1. **依赖 CLIP 预训练质量**：方法基于 CLIP-ViT-B/16，若目标域与 CLIP 预训练数据分布差异大（如医学影像），性能可能受限
2. **OOD 标签来源需要先验知识**：LFOP 需要从其他数据集获取 OOD 类别名称，这一步仍需人工选择，自动化程度不足
3. **仅在图像分类任务验证**：未在目标检测、语义分割等更复杂的视觉任务或非视觉模态上验证
4. **计算开销未充分讨论**：同时训练三类 prompt 和多个损失函数的训练成本相比简单方法（如 MCM、GL-MCM）更高
5. **pseudo-OOD 分布的在线更新稳定性**：动量更新机制在极端 few-shot（如 1-shot）下可能不够稳定，论文未深入分析收敛性

## 相关工作

- **OOD 检测**：ODIN (Liang et al., 2018)、ViM (Wang et al., 2022)、KNN (Sun et al., 2022)、NPOS (Tao et al., 2023)、MCM (Ming et al., 2022)，从分类、密度、距离、重构四类方法演进
- **Few-shot OOD 检测**：LoCoOp (Miyai et al., 2023)、CoOp (Zhou et al., 2022)、SCT (Yu et al., 2024) 利用 prompt learning 做少样本 OOD 检测，但忽略类间多样性差异
- **Prompt Learning**：从 NLP 的 GPT/BERT prompt 到视觉领域的 CoOp/CoCoOp，但多数假设封闭集，不适用于 OOD 检测场景
- **CLIP 与 OOD**：GL-MCM、SeTAR 利用 CLIP 做零样本 OOD 检测，本文进一步扩展到 few-shot 并引入 OOD prompt 构造

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|:---:|------|
| 新颖性 | 4 | 三类 prompt + 自适应阈值的组合设计新颖 |
| 技术深度 | 4 | 多损失函数设计完整，分布学习机制有理论动机 |
| 实验充分性 | 4 | 4 个 OOD 数据集 + 多 shot 设定 + 丰富消融 |
| 写作质量 | 3 | 整体清晰但公式符号较多，部分记号前后不一致 |
| 实用价值 | 3 | 方法有效但依赖 CLIP 和 OOD 标签先验 |
| **总分** | **3.6** | 扎实的 few-shot OOD 检测工作，核心贡献明确 |

<!-- RELATED:START -->

## 相关论文

- [Out-of-distribution Generalisation is Hard: Evidence from ARC-like Tasks](../../NeurIPS2025/llm_nlp/out-of-distribution_generalisation_is_hard_evidence_from_arc-like_tasks.md)
- [SEC-Prompt: SEmantic Complementary Prompting for Few-Shot Class-Incremental Learning](../../CVPR2025/llm_nlp/sec-promptsemantic_complementary_prompting_for_few-shot_class-incremental_learni.md)
- [PromptMoE: Generalizable Zero-Shot Anomaly Detection via Visually-Guided Prompt Mixing of Experts](../../AAAI2026/llm_nlp/promptmoe_generalizable_zero-shot_anomaly_detection_via_visually-guided_prompt_m.md)
- [HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation](../../ACL2025/llm_nlp/hygenar_an_llm-driven_hybrid_genetic_algorithm_for_few-shot_grammar_generation.md)
- [AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection](../../ECCV2024/llm_nlp/adaclip_adapting_clip_with_hybrid_learnable_prompts_for_zero.md)

<!-- RELATED:END -->
