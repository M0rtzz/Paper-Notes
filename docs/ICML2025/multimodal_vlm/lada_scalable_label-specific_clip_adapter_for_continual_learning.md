---
title: >-
  [论文解读] LADA: Scalable Label-Specific CLIP Adapter for Continual Learning
description: >-
  [ICML2025][多模态VLM][CLIP] 提出 LADA（Label-specific ADApter），通过在冻结 CLIP 图像编码器后追加轻量级的**类别特定记忆向量**，将所有已学任务的判别信息浓缩到统一特征空间，**彻底消除推理阶段的参数选择步骤**，在 X-TAIL 持续学习设定下取得 SOTA。
tags:
  - "ICML2025"
  - "多模态VLM"
  - "CLIP"
  - "持续学习"
  - "Label-Specific Adapter"
  - "特征蒸馏"
  - "类增量学习"
  - "跨域增量学习"
---

# LADA: Scalable Label-Specific CLIP Adapter for Continual Learning

**会议**: ICML2025  
**arXiv**: [2505.23271](https://arxiv.org/abs/2505.23271)  
**代码**: [MaolinLuo/LADA](https://github.com/MaolinLuo/LADA)  
**领域**: CLIP持续学习 / 参数高效微调  
**关键词**: CLIP, 持续学习, Label-Specific Adapter, 特征蒸馏, 类增量学习, 跨域增量学习

## 一句话总结

提出 LADA（Label-specific ADApter），通过在冻结 CLIP 图像编码器后追加轻量级的**类别特定记忆向量**，将所有已学任务的判别信息浓缩到统一特征空间，**彻底消除推理阶段的参数选择步骤**，在 X-TAIL 持续学习设定下取得 SOTA。

## 研究背景与动机

### 问题定义

跨域任务无关增量学习（X-TAIL）：模型依次从 $K$ 个来自不同领域的任务 $\{\mathcal{T}^1, \dots, \mathcal{T}^K\}$ 学习，推理时**不提供任务标识**，需同时识别已学类别和未见类别。

### 现有方法的不足

**Prompt 方法**（L2P、DualPrompt、S-Prompts）：推理时需从 prompt 池中**选择**对应 prompt → 选择错误直接降性能

**MoE-Adapter**（Yu et al. 2024）：预定义 adapter 数量 → 需知总任务数；推理时同样需要**选择**激活哪些 adapter

**全参微调**（ZSCL）：更新预训练参数 → 前向遗忘（forward forgetting）严重，零样本泛化退化

**分类器扩展**（RAIL）：依赖原始 CLIP 判断任务是否已学 → 误差传播

核心痛点：**参数选择（parameter selection）**——现有方法在推理时都需要一个额外步骤来决定用哪组参数提取特征，这种选择天然容易出错。

### LADA 的动机

设计一种**无需参数选择**的 adapter：将所有任务的判别信息浓缩到统一的 label-specific 特征中，推理时直接使用全部记忆向量，消除选择步骤。

## 方法详解

### 整体框架

LADA 由两个核心模块构成：

1. **文本编码器微调框架**：冻结旧任务文本特征，仅优化当前任务文本特征作为分类器
2. **可伸缩 Label-Specific Adapter**：在冻结的 CLIP 图像编码器之后追加类别特定记忆向量

### 模块一：文本编码器微调

对当前任务 $\mathcal{T}^k$，将旧任务 $\mathcal{T}^1, \dots, \mathcal{T}^{k-1}$ 的文本特征 $\boldsymbol{t}$ 冻结，仅更新 $\mathcal{T}^k$ 的文本特征。分类损失为：

$$\mathcal{L}(\boldsymbol{t}; k) = \sum_{i=1}^{k} \sum_{j=1}^{M^i} \hat{\mathcal{L}}(\boldsymbol{t}; k, i, j)$$

对当前任务：使用真实图像计算 softmax 交叉熵（Eq.3）。

对旧任务：无法访问原始图像，通过 **$\lambda$ 个聚类中心**（蒸馏原型 $\boldsymbol{p}_j^i$）替代（Eq.4），用原型与所有类文本特征的内积计算分类损失。

### 模块二：Label-Specific Adapter（核心创新）

**Step 1：构造 label-specific 特征**

对任务 $\mathcal{T}^k$ 的第 $j$ 类，用 k-means 对该类图像特征聚类得到 $\lambda_1$ 个聚类中心 $\boldsymbol{W}_j^k \in \mathbb{R}^{\lambda_1 \times d}$。

定义 label-specific 特征映射：

$$\varphi^k(\boldsymbol{i}) = [\boldsymbol{W}_1^k \boldsymbol{i}, \dots, \boldsymbol{W}_{M^k}^k \boldsymbol{i}]$$

所有任务的特征拼接为最终表示 $\varphi(\boldsymbol{i}) = [\varphi^1(\boldsymbol{i}), \cdots, \varphi^k(\boldsymbol{i})]$，**任务增加时特征自然扩展**。

**Step 2：固定分类器**

使用基于近邻的固定分类器 $h$，将 label-specific 特征映射为分类 logits：

$$(h \circ \varphi)(\boldsymbol{i})_j^i = \phi(\boldsymbol{W}_j^i \boldsymbol{i}) \cdot \boldsymbol{1}$$

其中 $\phi(x) = \exp(-\beta(1-x))$ 将内积转为非负值，$\beta$ 控制锐度。

**Step 3：训练策略**

- 冻结旧任务参数 $\boldsymbol{W}^1, \dots, \boldsymbol{W}^{k-1}$，仅更新 $\boldsymbol{W}^k$
- 当前任务：直接用真实样本计算 softmax 交叉熵（Eq.7）
- 旧任务：用蒸馏原型计算分类损失（Eq.8），防止新类参数干扰旧类判别

**Step 4：分布保持训练（Distribution-Preserved Training）**

仅用聚类中心不够，改用 **GMM（高斯混合模型）** 拟合旧任务各类特征分布：

$$\{\pi_j^i(l), \boldsymbol{p}_j^i(l), \boldsymbol{\Sigma}_j^i(l)\}_{l=1}^{\lambda_2} = \text{GMM}(\mathcal{D}_j^i)$$

增强原型：

$$\tilde{\boldsymbol{p}}_j^i(l) = \boldsymbol{p}_j^i(l) + \boldsymbol{e} \cdot \sqrt{\frac{\text{Tr}(\boldsymbol{\Sigma}_j^i(l))}{d}}$$

其中 $\boldsymbol{e}$ 为高斯噪声，方差由协方差矩阵的迹控制。通过加权损失（Eq.10）在旧任务上做分布保持训练。

### 关键设计优势

| 设计 | 作用 |
|------|------|
| 记忆向量放在图像编码器**之后** | 无需反向传播到 CLIP 编码器，训练高效 |
| 冻结旧任务 $\boldsymbol{W}$ | 天然防止后向遗忘 |
| 全部记忆向量参与推理 | 消除参数选择，避免误分配 |
| GMM + 噪声增强原型 | 用少量存储保持旧任务分布信息 |

## 实验关键数据

### X-TAIL 16-shot 设定

| 方法 | Transfer | Average | Last |
|------|----------|---------|------|
| Zero-shot CLIP | – | 57.7 | – |
| ZSCL | 59.0 | 60.0 | 63.4 |
| MoE-Adapters | 56.0 | 63.0 | 70.5 |
| Dual-RAIL | – | 71.3 | 82.3 |
| **LADA (Ours)** | **61.5** | **72.7** | **83.1** |

### X-TAIL Full-shot 设定

LADA 同样在 full-shot 上超越所有基线，Transfer / Average / Last 三指标均 SOTA。

### 关键观察

- Transfer 指标（未见任务泛化）：LADA 保持与 zero-shot CLIP 接近的泛化能力，大幅领先 MoE-Adapters（61.5 vs 56.0）
- Last 指标（最终性能）：LADA 在 10 个数据集学完后达到 83.1%，比 Dual-RAIL 高 0.8%
- Average 指标（过程均值）：72.7%，比 Dual-RAIL 高 1.4%

## 亮点与洞察

1. **消除推理阶段参数选择**：这是相对于 prompt-based 和 MoE 方法的核心突破——不需要"先猜用哪组参数"，所有记忆向量共同参与
2. **可伸缩性好**：每新增一个任务只加 $M^k \times \lambda_1$ 个向量（每类几个聚类中心），参数增长线性且轻量
3. **训练高效**：adapter 位于图像编码器之后，梯度不回传到 CLIP → GPU 显存占用小
4. **GMM 分布保持**：将旧任务的特征分布压缩为 GMM 参数（均值+协方差+权重），比存储原始特征或图像远更高效
5. **设计简洁**：没有复杂的路由机制、门控网络或注意力模块，纯粹基于内积和 k-means/GMM

## 局限与展望

1. **存储随类别数线性增长**：每个类需要 $\lambda_1$ 个记忆向量 + $\lambda_2$ 个 GMM 分量，类别数极多时（如 10,000 类）存储可能成为瓶颈
2. **GMM 假设**：用 GMM 拟合特征分布假设了局部高斯性，对高度非凸分布可能不够准确
3. **依赖 CLIP 冻结特征质量**：如果 CLIP 预训练表示对某些领域（如医学图像）本身就不好，LADA 的提升空间有限
4. **缺少对超大规模任务的验证**：实验最多 10 个任务顺序学习，实际场景可能有数百个任务
5. **固定分类器**：分类器 $h$ 基于 $\phi = \exp(-\beta(1-x))$ 的设计较为简单，可能限制了表达能力

## 相关工作与启发

- **RAIL / Dual-RAIL**（Xu et al. 2024）：扩展分类器维度但冻结特征 → 与 LADA 互补，LADA 改进特征端
- **MoE-Adapters**（Yu et al. 2024）：混合适配器但需参数选择 → LADA 消除了这一步
- **ZSCL**（Zheng et al. 2023）：知识蒸馏防遗忘但更新预训练参数 → LADA 完全冻结 CLIP
- **原型增强**（Zhang et al. 2023）：用原型代替存储样本 → LADA 的 GMM 分布保持是其升级版
- 启发：**将类别判别信息编码为可学习向量（memory units）** 的范式值得在其他持续学习场景中推广

## 评分

- 新颖性: ⭐⭐⭐⭐ — label-specific memory unit 消除参数选择的思路新颖
- 实验充分度: ⭐⭐⭐⭐ — X-TAIL 16-shot/full-shot 全面对比，消融实验充分
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，公式推导完整，图示直观
- 价值: ⭐⭐⭐⭐ — 对 CLIP 持续学习领域有实际推动，方法即插即用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning](../../AAAI2026/multimodal_vlm/harnessing_textual_semantic_priors_for_knowledge_transfer_and_refinement_in_clip.md)
- [\[NeurIPS 2025\] Continual Multimodal Contrastive Learning](../../NeurIPS2025/multimodal_vlm/continual_multimodal_contrastive_learning.md)
- [\[CVPR 2025\] NLPrompt: Noise-Label Prompt Learning for Vision-Language Models](../../CVPR2025/multimodal_vlm/nlprompt_noise-label_prompt_learning_for_vision-language_models.md)
- [\[CVPR 2026\] CLIP-Free, Label-Free, Unsupervised Concept Bottleneck Models](../../CVPR2026/multimodal_vlm/clip-free_label_free_unsupervised_concept_bottleneck_models.md)
- [\[ICLR 2026\] Enhanced Continual Learning of Vision-Language Models with Model Fusion](../../ICLR2026/multimodal_vlm/enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)

</div>

<!-- RELATED:END -->
