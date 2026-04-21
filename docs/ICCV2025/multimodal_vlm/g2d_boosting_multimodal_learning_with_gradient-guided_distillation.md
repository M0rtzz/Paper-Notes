---
title: >-
  [论文解读] G2D: Boosting Multimodal Learning with Gradient-Guided Distillation
description: >-
  [ICCV 2025][多模态][modality imbalance] 提出G2D（Gradient-Guided Distillation），通过融合单模态教师到多模态学生的特征蒸馏+logit蒸馏损失，并结合基于单模态教师置信度分数的Sequential Modality Prioritization（SMP）梯度调制策略，解决多模态学习中的模态不平衡问题，在CREMA-D上实现85.89%准确率、超越所有专注模态不平衡的SOTA方法。
tags:
  - ICCV 2025
  - 多模态
  - modality imbalance
  - 知识蒸馏
  - gradient modulation
  - sequential modality prioritization
---

# G2D: Boosting Multimodal Learning with Gradient-Guided Distillation

**会议**: ICCV 2025  
**arXiv**: [2506.21514](https://arxiv.org/abs/2506.21514)  
**代码**: [GitHub](https://github.com/rakib062/G2D)  
**领域**: 多模态VLM / 模态不平衡 / 知识蒸馏  
**关键词**: modality imbalance, knowledge distillation, gradient modulation, sequential modality prioritization, multimodal fusion

## 一句话总结
提出G2D（Gradient-Guided Distillation），通过融合单模态教师到多模态学生的特征蒸馏+logit蒸馏损失，并结合基于单模态教师置信度分数的Sequential Modality Prioritization（SMP）梯度调制策略，解决多模态学习中的模态不平衡问题，在CREMA-D上实现85.89%准确率、超越所有专注模态不平衡的SOTA方法。

## 研究背景与动机

**模态不平衡现象**：多模态联合训练中，一个模态主导优化过程，其他模态被抑制——称为"模态竞争"或"模态懒惰"。这导致(i)多模态性能反不如单模态，或(ii)弱模态特征在联合训练中退化。

**CREMA-D上的典型案例**（图1）：
   - 音频单独训练达61.69%，在多模态训练中降到只有59.95%（影响不大）
   - **视频**单独训练达76.48%，但在多模态联合训练中骤降至27.42%（几乎崩溃）
   - 联合多模态模型仅67.47%，远低于视频单模态的76.48%

**现有方法的局限**：
   - **梯度调制**（OGM-GE, AGM）：动态调整弱模态梯度，但需要精细的超参数调节
   - **特征重平衡**（MLA, MMPareto）：调整各模态贡献，但无法完全消除不平衡
   - **知识蒸馏**（UMT, UME）：用单模态教师指导多模态学生，但选择哪种蒸馏方式需要经验调节

**核心insight**：弱模态的根本问题是优化不充分——在联合训练中，强模态快速收敛导致梯度信号主要服务于强模态。解决方案不是"削弱强模态"，而是"给弱模态专属的无干扰训练阶段"。

## 方法详解

### 整体框架
G2D包含三个核心组件：(1) 独立预训练的单模态教师 $\{T^m\}_{m=1}^k$，(2) 联合训练的多模态学生 $S$，(3) 融合蒸馏损失 $\mathcal{L}_{\text{G2D}}$ + SMP梯度调制策略。

### 关键设计1：G2D损失函数

融合三类损失：

**(1) 多模态学生损失 $\mathcal{L}_S$**：标准交叉熵（分类）或MSE（回归），使用融合后的多模态特征预测标签。

**(2) 特征蒸馏损失 $\mathcal{L}_{\text{feat}}$**：L2距离约束学生编码器的模态特征与教师编码器的模态特征对齐：

$$\mathcal{L}_{\text{feat}}^m = \mathbb{E}_{x \sim \mathcal{D}}\left[\|\phi_s^m(x^m; \theta_s^m) - \phi_t^m(x^m; \theta_t^m)\|^2\right]$$

**(3) Logit蒸馏损失 $\mathcal{L}_{\text{logit}}$**：用KL散度让多模态学生的输出分布逼近各单模态教师的分布：

$$\mathcal{L}_{\text{logit}}^m = \mathbb{E}_{x \sim \mathcal{D}}\left[\text{KL}(\sigma(l_t^m) \| \sigma(l_s))\right]$$

**总G2D损失**：

$$\mathcal{L}_{\text{G2D}} = \mathcal{L}_S + \alpha \sum_{m=1}^{k} \mathcal{L}_{\text{feat}}^m + \beta \sum_{m=1}^{k} \mathcal{L}_{\text{logit}}^m$$

特征蒸馏保留模态特定表示，logit蒸馏对齐决策边界，两者互补。

### 关键设计2：模态置信度量化（Scoring Module）

利用单模态教师的batch-wise平均softmax概率作为模态置信度：

$$\rho_t^m = \frac{1}{|\mathcal{B}^m|} \sum_{(x_i^m, y_i^m) \in \mathcal{B}^m} \text{Softmax}(l_t^m(x_i^m; \theta^m))[y_i^m]$$

置信度高的模态为"强势模态"，低的为"弱势模态"。关键优势：使用**单模态**教师的置信度，不受联合训练中模态不平衡的影响。

### 关键设计3：Sequential Modality Prioritization（SMP）

**核心假设**：给弱势模态专属的无干扰训练阶段可以缓解模态不平衡。

**具体策略**：
1. 根据教师置信度排序模态：$\pi_t[1]$（最弱）到 $\pi_t[k]$（最强）
2. 训练分阶段进行：前 $\tau_1$ 个epoch只训练最弱模态，接下来 $\tau_2$ 个epoch训练第二弱模态，最后所有模态联合训练
3. 通过梯度调制系数 $\kappa_q^m$ 控制哪些模态参与梯度更新：

$$\theta_{q+1}^m = \theta_q^m - \eta \cdot \kappa_q^m \cdot \frac{\partial \mathcal{L}_{\text{G2D}}}{\partial \theta_q^m}$$

其中 $\kappa_q^m = 1$ 表示该模态参与训练，$\kappa_q^m = 0$ 表示冻结。

这是**完全抑制**策略——不是用连续权重削弱强模态（如OGM-GE的 $1 - \tanh(x)$），而是直接将强模态梯度置零，确保弱模态获得完全的优化机会。

## 实验

### 数据集
- **CREMA-D**：音频-视频情感识别，6类
- **AV-MNIST**：音频-视频数字分类，10类
- **VGGSound**：音频-视频事件分类，309类
- **UR-Funny**：文本-视觉-音频幽默检测，2类
- **IEMOCAP**：音视频文本情感识别
- **MIS-ME**：土壤图像+气象表格回归（首次评估模态不平衡的回归场景）

### 主实验结果（双模态音视频）

| 方法 | CREMA-D Multi | AV-MNIST Multi | VGGSound Multi |
|------|--------------|----------------|----------------|
| Joint-Train | 67.47 | 69.77 | 50.97 |
| AGM | 78.48 | 72.14 | 47.11 |
| OGM-GE | 58.60 | 24.53 | 37.96 |
| MLA | 79.70 | 65.32 | 51.65 |
| ReconBoost | 83.62 | 72.14 | 52.74 |
| DLMG | 67.61 | 72.33 | 53.78 |
| UMT (KD baseline) | 67.61 | 72.33 | 53.78 |
| **G2D (本文)** | **85.89** | **73.03** | **53.82** |

G2D在CREMA-D上大幅领先（+2.27 vs ReconBoost），将视频模态在多模态训练中的性能从27.42%提升到72.72%。

### 三模态实验（UR-Funny）

| 模态组合 | Joint-Train | OGM-GE | MMPareto | ReconBoost | UMT | **G2D** |
|---------|-------------|--------|----------|------------|-----|---------|
| A-V Multi | 61.57 | 61.87 | 61.27 | 62.07 | 60.46 | **62.98** |
| A-TXT Multi | 62.17 | 62.47 | 62.88 | 61.06 | 62.47 | **63.28** |
| A-V-TXT Multi | 62.58 | 63.68 | 62.88 | 61.37 | 63.38 | **65.49** |

G2D在三模态场景下同样有效，且不会过度抑制强势模态（如文本）。

### 消融实验

| SMP对不同方法的增益 | 无SMP | 有SMP | 增益 |
|---|---|---|---|
| Joint-Train on CREMA-D | 67.47 | 80.78 | +13.31 |
| UMT on CREMA-D | 67.61 | 82.39 | +14.78 |
| G2D loss on CREMA-D | 78.63 | 85.89 | +7.26 |

| 完全抑制 vs 部分抑制 | CREMA-D | AV-MNIST | VGGSound | UR-Funny |
|---|---|---|---|---|
| 部分抑制（OGM-GE式） | 81.99 | 72.83 | 51.16 | 63.68 |
| **完全抑制（SMP）** | **85.89** | **73.03** | **53.82** | **65.49** |

### 关键发现
1. SMP对所有方法都有效——即使用在vanilla joint-training上也能带来+13个百分点的提升
2. 完全梯度抑制一致优于部分抑制，支持"给弱模态充分的无干扰训练"假设
3. G2D首次在回归任务（MIS-ME）上验证了模态不平衡的存在和缓解
4. Late fusion在G2D框架下表现最优，因为它保留了独立的单模态表示

## 亮点与洞察
1. **SMP策略简单但极其有效**：完全冻结强模态、只训练弱模态的策略，比精细的梯度权重调节更有效。这说明弱模态需要的不是more gradient，而是undisturbed gradient
2. **知识蒸馏的恰当使用**：特征蒸馏+logit蒸馏结合监督损失，既保持单模态的最优表示，又优化多模态目标
3. **通用性强**：适用于2模态/3模态、分类/回归、多种融合方式，且SMP可以即插即用到其他方法

## 局限性
1. SMP的 $\tau_j$ 超参数在不同数据集上需要调节（CREMA-D上最优为150个epoch训练弱模态）
2. 需要额外预训练多个单模态教师模型，增加了总体训练开销
3. 在AV-MNIST等模态不平衡较轻的数据集上提升有限（73.03 vs 72.76）
4. 未在大规模预训练模型（如CLIP, LLaVA）上验证

## 相关工作
- **梯度调制**：OGM-GE, AGM, PMR
- **特征重平衡**：MLA, MMPareto, ReconBoost
- **知识蒸馏**：UMT, UME
- **模态不平衡分析**：MSES, MSLR

## 评分
- 新颖性：3/5（KD框架+梯度调制的组合较为自然，SMP策略简单但effective）
- 技术深度：3/5（方法清晰但缺乏理论分析，SMP为什么有效需要更深入的解释）
- 实验充分度：5/5（6个数据集、10+baseline、大量消融、回归任务、fusion对比）
- 写作质量：4/5（结构清晰，图表丰富）

<!-- RELATED:START -->

## 相关论文

- [CovMatch: Cross-Covariance Guided Multimodal Dataset Distillation with Trainable Text Encoder](../../NeurIPS2025/multimodal_vlm/covmatch_crosscovariance_guided_multimodal_dataset_distillat.md)
- [Controlling Multimodal LLMs via Reward-guided Decoding](controlling_multimodal_llms_via_rewardguided_decoding.md)
- [BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models](basic_boosting_visual_alignment_with_intrinsic_refined_embeddings_in_multimodal_.md)
- [Boosting MLLM Reasoning with Text-Debiased Hint-GRPO](boosting_mllm_reasoning_with_text-debiased_hint-grpo.md)
- [KeepLoRA: Continual Learning with Residual Gradient Adaptation](../../ICLR2026/multimodal_vlm/keeplora_continual_learning_with_residual_gradient_adaptation.md)

<!-- RELATED:END -->
