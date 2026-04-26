---
title: >-
  [论文解读] Prompting Language-Informed Distribution for Compositional Zero-Shot Learning
description: >-
  [ECCV 2024][Compositional Zero-Shot Learning] 本文提出 PLID 方法，利用 LLM 生成的句子级类别描述构建语言知识驱动的高斯分布，配合视觉-语言原语分解和随机 logit 融合，在组合零样本学习（CZSL）任务上取得 SOTA。
tags:
  - ECCV 2024
  - Compositional Zero-Shot Learning
  - CLIP
  - LLM
  - 提示学习
  - Primitive Decomposition
---

# Prompting Language-Informed Distribution for Compositional Zero-Shot Learning

**会议**: ECCV 2024  
**arXiv**: [2305.14428](https://arxiv.org/abs/2305.14428)  
**代码**: [https://github.com/Cogito2012/PLID](https://github.com/Cogito2012/PLID)  
**领域**: LLM/NLP  
**关键词**: Compositional Zero-Shot Learning, CLIP, LLM, Distribution Prompting, Primitive Decomposition

## 一句话总结

本文提出 PLID 方法，利用 LLM 生成的句子级类别描述构建语言知识驱动的高斯分布，配合视觉-语言原语分解和随机 logit 融合，在组合零样本学习（CZSL）任务上取得 SOTA。

## 研究背景与动机

**领域现状**：组合零样本学习（CZSL）要求从已见组合（如 sliced potatoes、red tomatoes）泛化到未见组合（如 sliced tomatoes）。近期工作基于 CLIP 的 prompt tuning，性能大幅超越传统视觉方法。

**现有痛点**：
   - **多样性不足**：CSP 等硬 prompt 方法对每个类只用一个 prompt（如 "a photo of [state][object]"），无法捕获类内视觉差异
   - **信息量不足**：ProDA 等分布式 prompt 方法虽引入多个 prompt 增加多样性，但 prompt 缺乏语言语义信息，对细粒度组合类别区分力有限
   - **原语纠缠**：视觉原语（state 和 object）天然耦合（如 tomatoes 天然与 red 相关），现有方法要么不解耦，要么仅在文本侧解耦

**核心矛盾**：prompt 的多样性和信息量需要同时满足——ProDA 有多样性但无信息量，CSP 有一定信息量但无多样性

**本文目标**：如何让 CLIP 的类别文本表示既多样又富含信息，且支持视觉-语言双侧的原语分解

**切入角度**：用 LLM 为每个组合类生成多条句子描述 → 构建类别高斯分布 → 同时在组合和原语空间做分布对齐

**核心 idea**：用 LLM 生成的描述性句子作为类分布的支撑点（DSP），通过 soft prompt 学习语言知识驱动的类分布，实现多样且有信息量的零样本组合识别。

## 方法详解

### 整体框架

输入图像 → CLIP 视觉编码器 + VFE 增强 → 得到图像特征 $\mathbf{v}$
类别名 → CLIP 文本编码器 + soft prompt → 得到类均值 $\mathbf{q}_y$
LLM 生成 M 条描述 → CLIP 编码 → DSP $\mathbf{D}^{(y)}$ → TFE 增强 → 类均值 $\mathbf{t}_y$
→ 构建组合/状态/物体三级高斯分布 → 分布对齐损失
→ VLPD 原语分解 → 随机 logit 融合 → 最终预测

### 关键设计

1. **语言知识驱动的分布 (LID)**：

    - 功能：为每个组合类 $y = (s, o)$ 构建基于 LLM 描述的高斯分布
    - 核心思路：用 LLM 生成 M 条描述 $S^{(y)} = \{S_1^{(y)}, ..., S_M^{(y)}\}$，经 CLIP 文本编码器得到 $\mathbf{D}^{(y)} \in \mathbb{R}^{M \times d}$。用 TFE（单层 cross-attention）将 $\mathbf{D}^{(y)}$ 融入类嵌入 $\mathbf{q}_y$ 得到增强均值 $\mathbf{t}_y = \Psi_{\text{TFE}}(\mathbf{q}_y, \mathbf{D}^{(y)})$。将 $\mathbf{t}_y + \mathbf{D}^{(y)}$ 作为分布支撑点，假设服从 $\mathcal{N}(\mathbf{t}_y, \boldsymbol{\Sigma}_y)$。训练目标最小化 NLL 上界：
    $\mathcal{L}_y(\mathbf{x}, y) = -\log \frac{\exp(h_y / \tau)}{\sum_{k=1}^{C} \exp((h_k + h_{k,y}^{(m)}) / \tau)}$
   其中 pairwise margin $h_{k,y}^{(m)} = \mathbf{v}^\top \mathbf{A}_{k,y} \mathbf{v} / (2\tau)$ 由协方差差分 $\mathbf{A}_{k,y} = \boldsymbol{\Sigma}_{kk} + \boldsymbol{\Sigma}_{yy} - \boldsymbol{\Sigma}_{ky} - \boldsymbol{\Sigma}_{yk}$ 决定
    - 设计动机：通过最小化该损失，自然地最小化类内方差（$\boldsymbol{\Sigma}_{yy}$）并最大化类间分离度（$\boldsymbol{\Sigma}_{ky}$），实现类分布的自动优化

2. **视觉-语言原语分解 (VLPD)**：

    - 功能：将图像特征 $\mathbf{v}$ 通过两个并行网络 $f_s, f_o$ 分解为状态和物体特征
    - 核心思路：
    $h_s = \cos(f_s(\mathbf{v}), \frac{1}{|\mathcal{Y}_s|}\sum_{y \in \mathcal{Y}_s} \mathbf{t}_y), \quad h_o = \cos(f_o(\mathbf{v}), \frac{1}{|\mathcal{Y}_o|}\sum_{y \in \mathcal{Y}_o} \mathbf{t}_y)$
   文本侧的原语嵌入通过分组平均组合嵌入获得（同 state 的所有组合取均值得 state 嵌入，同理 object）
    - 设计动机：不同于 DFSP 只做文本分解，VLPD 同时在视觉和文本两侧分解，实验表明双侧分解效果更好

3. **随机 logit 混合融合 (SLM)**：

    - 功能：在直接的组合预测 $h_y$ 和重组合预测 $h_y^{(rc)} = h_s + h_o$ 之间做随机加权融合
    - 核心思路：
    $\tilde{h}_y = (1 - \lambda) h_y + \lambda h_y^{(rc)}, \quad \lambda \sim \text{Beta}(a, b)$
   训练时从 Beta 分布采样 $\lambda$，测试时用期望 $\lambda = a/(a+b)$
    - 设计动机：Beta 随机性引入正则化效果；logit 级融合避免了 softmax 概率混合时在大类别空间丢失类间关系信息的问题

### 损失函数 / 训练策略

总训练损失为组合级 + 状态级 + 物体级的分布对齐损失之和：
$$\mathcal{L} = \mathcal{L}_y + \mathcal{L}_s + \mathcal{L}_o$$

三者均采用相同形式的 NLL 上界损失，且各自在对应的语义空间构建高斯分布。训练中仅优化 soft prompt $\mathbf{p}_{1:L}$、原语嵌入 $[\mathbf{s}][\mathbf{o}]$、TFE/VFE 参数和 $f_s, f_o$，CLIP 编码器保持冻结。

## 实验关键数据

### 主实验

**Closed-World Setting (H / AUC)**

| 数据集 | 指标 | PLID | DFSP (前SOTA) | CSP | 提升 |
|--------|------|------|-------------|-----|------|
| MIT-States | H / AUC | **39.0 / 22.1** | 37.3 / 20.6 | 36.3 / 19.4 | +1.7 / +1.5 |
| UT-Zappos | H / AUC | **52.4 / 38.7** | 47.2 / 36.0 | 46.6 / 33.0 | +5.2 / +2.7 |
| C-GQA | H / AUC | **27.9 / 11.0** | 27.1 / 10.5 | 20.5 / 6.2 | +0.8 / +0.5 |

**Open-World Setting (H / AUC)**

| 数据集 | 指标 | PLID | DFSP (前SOTA) | 提升 |
|--------|------|------|-------------|------|
| MIT-States | H / AUC | **20.4 / 7.3** | 19.3 / 6.8 | +1.1 / +0.5 |
| UT-Zappos | H / AUC | **46.6 / 30.8** | 44.0 / 30.3 | +2.6 / +0.5 |
| C-GQA | H / AUC | **10.6 / 2.5** | 10.4 / 2.4 | +0.2 / +0.1 |

### 消融实验

**主要组件消融 (MIT-States, AUC_cw)**

| 配置 | AUC_cw | AUC_ow | 说明 |
|------|--------|--------|------|
| (a) Baseline (均值池化) | 18.56 | 5.56 | 无分布建模 |
| (b) + LID | 20.43 | 6.50 | 分布建模带来 +1.87 |
| (c) + LID + FE | 21.09 | 6.95 | 特征增强有效 |
| (d) + LID + FE + OPT-1.3B | 21.67 | 7.01 | 更好 LLM 略有提升 |
| (e) + LID + FE + OPT + PDF | **22.12** | **7.34** | 原语分解融合 进一步提升 |

**VLPD 分解策略消融**

| 分解模态 | 融合方式 | AUC_cw | 说明 |
|----------|---------|--------|------|
| 仅文本分解 | 无融合 | 20.98 | DFSP 策略 |
| 文本+视觉 | 确定性融合 | 21.90 | 双侧分解更好 |
| 文本+视觉 | 随机融合 | **22.12** | Beta 随机性是正则化 |

### 关键发现

- ProDA（非信息化分布）远不如 PLID（信息化分布），说明仅有多样性不够，语言信息量是关键
- 小 LLM（OPT-1.3B）就够用，GPT-3.5 和 Mistral-7B 等大模型反而没有提升——说明 LLM 描述的质量而非规模是决定因素
- M=64 条描述、N=8 个图像增强视角是最优超参；描述数过少时分布不够丰富，过多时改善边际递减
- Beta(1,9) 效果最好，说明直接学习的组合预测应占主导，重组合预测起辅助校准作用
- tSNE 可视化清晰显示：分布学习后类内更紧凑、类间更分离

## 亮点与洞察

- **LLM 描述 = 类分布支撑点**：将 LLM 生成的多样描述转化为高斯分布的支撑点，这是一种新颖的「描述即分布」范式，可推广到任何零样本识别任务
- **NLL 上界自带方差优化**：通过推导 NLL 上界，自然得到最小化类内方差 + 最大化类间距离的训练信号，无需额外设计对比损失
- **参数效率**：只需一组 soft prompt，远小于 ProDA 需要学的大量 prompt 集合
- **logit 融合 > 概率融合**：在大类别空间（C-GQA 有 278K 类）中，softmax 概率归一化会丢失类间关系信息，logit 级融合更合理

## 局限与展望

- 当类别数极大时（C-GQA 278K 类），协方差矩阵 $\mathbf{A} \in \mathbb{R}^{d \times C \times C}$ 的计算需要分组近似才可行
- LLM 描述的质量对性能有影响，但如何自动评估和筛选描述质量尚未探讨
- 仅验证了 ViT-L/14 backbone；更小的 CLIP 模型或更新的 VLM 是否同样有效未知
- 图像侧的 VFE 使用多视角增强，N 过大时有过拟合倾向
- 开放世界设置下性能提升幅度有限，说明对极大搜索空间的应对还有改进空间

## 相关工作与启发

- **vs CSP**：CSP 用硬 prompt + 可学习原语嵌入，简单高效但缺乏多样性和信息量；PLID 在其基础上引入分布建模 + LLM 描述
- **vs DFSP**：DFSP 做文本侧原语分解 + 概率融合；PLID 改为视觉+文本双侧分解 + logit 融合，在所有数据集上均优于 DFSP
- **vs ProDA**：ProDA 学多组 soft prompt 构建分布，但 prompt 无语义信息且参数量大；PLID 用 LLM 描述替代，参数更少效果更好

## 评分

- 新颖性: ⭐⭐⭐⭐ LLM 描述构建类分布的思路新颖实用，分布建模的理论推导扎实
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、两种设定、极其丰富的消融（LID 层级、LLM 选择、FE 设计、融合策略、超参数、可视化）
- 写作质量: ⭐⭐⭐⭐⭐ 公式推导清晰，motivation-method-experiment 逻辑链完整，图表信息量大
- 价值: ⭐⭐⭐⭐ CZSL 领域的实质性推进，「描述即分布」范式有通用性

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] Learning to Obstruct Few-Shot Image Classification over Restricted Classes](learning_to_obstruct_few-shot_image_classification_over_restricted_classes.md)
- [\[NeurIPS 2025\] ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data](../../NeurIPS2025/llm_pretraining/zeus_zero-shot_embeddings_for_unsupervised_separation_of_tabular_data.md)
- [\[ACL 2025\] AutoDS: Autonomous Data Selection with Zero-shot Generative Classifiers for Mathematical Texts](../../ACL2025/llm_pretraining/autonomous_data_selection_with_zero-shot_generative_classifiers_for_mathematical.md)
- [\[ICML 2025\] When Can In-Context Learning Generalize Out of Task Distribution?](../../ICML2025/llm_pretraining/when_can_in-context_learning_generalize_out_of_task_distribution.md)
- [\[ACL 2025\] Training Dynamics Underlying Language Model Scaling Laws: Loss Deceleration and Zero-Sum Learning](../../ACL2025/llm_pretraining/training_dynamics_underlying_language_model_scaling_laws_loss_deceleration_and_z.md)

<!-- RELATED:END -->
