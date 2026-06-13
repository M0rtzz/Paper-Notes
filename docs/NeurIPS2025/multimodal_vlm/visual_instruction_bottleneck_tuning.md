---
title: >-
  [论文解读] Visual Instruction Bottleneck Tuning
description: >-
  [NeurIPS 2025][多模态VLM][信息瓶颈] 首次将信息瓶颈（IB）原理应用于多模态大语言模型的端到端指令微调，提出Visual Instruction Bottleneck Tuning（Vittle），在LLM内部插入轻量瓶颈层学习最小充分表征，在30种分布偏移场景下一致提升鲁棒性，同时不牺牲标准基准性能。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "信息瓶颈"
  - "多模态大语言模型"
  - "分布偏移鲁棒性"
  - "指令微调"
  - "表征学习"
---

# Visual Instruction Bottleneck Tuning

**会议**: NeurIPS 2025  
**arXiv**: [2505.13946](https://arxiv.org/abs/2505.13946)  
**作者**: Changdae Oh, Jiatong Li, Shawn Im, Sharon Li (University of Wisconsin–Madison)  
**代码**: [deeplearning-wisc/vittle](https://github.com/deeplearning-wisc/vittle)  
**领域**: 多模态VLM  
**关键词**: 信息瓶颈, 多模态大语言模型, 分布偏移鲁棒性, 指令微调, 表征学习  

## 一句话总结

首次将信息瓶颈（IB）原理应用于多模态大语言模型的端到端指令微调，提出Visual Instruction Bottleneck Tuning（Vittle），在LLM内部插入轻量瓶颈层学习最小充分表征，在30种分布偏移场景下一致提升鲁棒性，同时不牺牲标准基准性能。

## 研究背景与动机

### 问题背景
多模态大语言模型（MLLM）尽管在标准基准上取得优异性能，但在面对分布偏移时表现脆弱——包括图像的亮度/对比度变化、文本中的拼写错误等微小扰动，以及长尾分布样本。这与人类智能形成鲜明对比：人类能够将大量感知输入压缩为简洁的抽象表征，在对低级表面特征保持不变性的同时对高级抽象特征保持敏感性。

### 已有工作的不足
- **数据中心方法**：收集更多指令数据（如SVit、LLaVA-1.5等），需要大量标注成本
- **模型中心方法**：扩大模型规模或使用更强的骨干网络（如Eagle、Qwen2-VL），计算成本高昂
- **已有IB工作**：信息瓶颈主要在小规模分类任务中探索，或仅在投影层上冻结LLM骨干进行IB训练，未实现端到端
- **表征层面分析**：Oh等人(2025)发现MLLM的内部表征空间中，扰动样本与干净样本的嵌入距离过远，缺乏对表面变化的不变性

### 核心动机
从表征学习视角出发，不扩展数据或模型，而是通过信息瓶颈原理正则化MLLM的内部表征，使其丢弃输入特有的冗余信息、仅保留与响应相关的任务信息，从而在不变性与敏感性之间取得更好的平衡。

## 方法详解

### 信息瓶颈目标
给定多模态输入$X=(X_v, X_t)$、期望输出$Y$和中间表征$Z=f(X)$，IB目标为：

$$\max_f \text{IB}_f(X,Y) := I(Z,Y) - \beta \cdot I(Z,X)$$

其中$I(Z,Y)$保留任务相关信息，$I(Z,X)$中的冗余信息被压缩。

### 变分下界推导
直接优化IB不可行。本文针对MLLM的自回归多模态结构推导了专门的变分下界：

1. **压缩项上界**：利用因果遮罩的性质$p(z_v|x_v,x_t)=p(z_v|x_v)$，将$I(Z,X)$分解为视觉和文本两个KL散度项：
$$I(Z,X) \leq \mathbb{E}_{x_v}[D_{\text{KL}}(p(z_v|x_v)\|r(z_v))] + \mathbb{E}_{x_v,x_t}[D_{\text{KL}}(p(z_t|x_v,x_t)\|r(z_t))]$$

2. **预测项下界**：引入变分近似$q(y|z)$替代真实后验$p(y|z)$：
$$I(Z,Y) \geq \mathbb{E}_{x,y}[\mathbb{E}_{z|x}[\log q(y|z)]]$$

3. **最终目标**：
$$\mathcal{L}_\beta = \frac{1}{N}\sum_{i=1}^N \mathbb{E}_{z|x^i}[\log q(y^i|z)] - \beta(D_{\text{KL}}(p(z_v|x_v^i)\|r(z_v)) + D_{\text{KL}}(p(z_t|x_v^i,x_t^i)\|r(z_t)))$$

### Vittle架构实现
- **瓶颈层位置**：在LLM的第$l$层（默认第24层/共32层，即顶部25%）之后插入
- **后验分布建模**：为视觉和文本token各使用一个MLP $g_{\phi}:\mathbb{R}^d \to \mathbb{R}^{2d}$，输出均值$\mu$和方差$\sigma^2$，定义对角高斯后验
- **采样与插值**：通过重参数化技巧采样$\tilde{z}=\mu+\sigma\odot\epsilon$，然后与原始表征插值$\hat{z}=(1-\alpha)z+\alpha\tilde{z}$，$\alpha$按余弦调度递增至0.5
- **先验分布**：两种变体——Vittle (F)使用固定标准高斯$\mathcal{N}(0,I)$；Vittle (L)使用可学习高斯$\mathcal{N}(\mu_\psi, \sigma_\psi^2 \cdot I)$
- **推理时**：使用确定性后验均值$\tilde{z}=\mu$，取$\hat{z}=(z+\tilde{z})/2$
- **超参数**：$\beta=0.1/d$（$d$为隐藏维度），额外参数仅增加1.5%

### 理论支撑：EMID上界
本文证明Vittle的学习目标与EMID（有效互信息差异，衡量MLLM在分布偏移下的鲁棒性退化）的上界相关联。EMID上界可分解为输出熵与表征分布差异（JSD）的乘积与求和形式，而Vittle通过压缩表征来减少干净样本与扰动样本之间的JSD，从而降低EMID。

## 实验关键数据

### 实验1：扰动鲁棒性（LB-COCO及其27种变体）

在LB-COCO上施加27种扰动（9种视觉、9种文本、9种联合），使用GPT-4o评判相对偏好分数。

| 方法 | Clean | V Pert. | T Pert. | J Pert. |
|------|-------|---------|---------|---------|
| Baseline | 77.8 | 73.4 | 72.2 | 62.3 |
| LoRA | 73.4 | 70.4 | 62.7 | 39.7 |
| Weight Decay | 74.1 | 72.1 | 73.0 | 59.5 |
| **Vittle (L)** | 76.7 | 73.9 | 73.0 | 62.7 |
| **Vittle (F)** | 76.1 | **74.2** | **74.1** | **64.4** |

Vittle (F)在文本扰动和联合扰动上分别提升+1.9和+2.1，显著优于参数空间正则化方法（LoRA和Weight Decay）。

### 实验2：跨任务与跨架构验证

**长尾分布开放式QA**（相对偏好分数）：

| 方法 | LB-Wild | LB-Wilder | WV-Bench |
|------|---------|-----------|----------|
| Baseline | 51.6 | 156.9 | 60.0 |
| Vittle (L) | **54.6** | **168.8** | **60.4** |
| Vittle (F) | 52.2 | 166.1 | 59.7 |

**通用基准（封闭式QA）**：

| 方法 | SciQA | MMMU | MME | MMStar | Avg. |
|------|-------|------|-----|--------|------|
| Baseline | 64.6 | 35.6 | 69.7 | 33.7 | 50.9 |
| Vittle (L) | 64.7 | 35.3 | 70.5 | 33.7 | 51.1 |
| Vittle (F) | **65.4** | 34.5 | 70.1 | 33.5 | 50.9 |

**跨架构泛化（POPE幻觉检测）**：

| 骨干网络 | 方法 | POPE Clean | POPE Shifts Avg. |
|---------|------|------------|-----------------|
| LLaVA-Mini | Baseline | 79.37 | 77.39 |
| LLaVA-Mini | Vittle (F) | **81.07** | **78.32** |
| LLaVA++ (Llama3-8B) | Baseline | 84.60 | 80.54 |
| LLaVA++ (Llama3-8B) | Vittle (F) | **85.87** | **84.08** |

**表征空间分析（27种LB-COCO扰动平均）**：

| 方法 | JSD (↓) | EMID (↓) |
|------|---------|----------|
| Baseline | 0.068 | 0.026 |
| Vittle (L) | 0.048 | 0.021 |
| Vittle (F) | **0.047** | 0.025 |

Vittle将干净-扰动样本的JSD从0.068降至0.047（降幅31%），验证了表征压缩确实增强了不变性。

## 亮点

- **开创性视角**：首次将信息瓶颈原理引入MLLM端到端指令微调，提出从表征压缩角度增强鲁棒性的新范式，与传统数据/模型扩展路线形成互补
- **理论与实践统一**：推导了适配自回归多模态架构的IB变分下界，并建立了与EMID鲁棒性度量的理论联系，两种先验变体各有适用场景
- **极低成本的一致增益**：仅增加1.5%参数和20%训练时间，推理时间几乎不变，却在30种分布偏移、45个数据集、多种MLLM架构上一致提升鲁棒性
- **定性分析令人信服**：PCA可视化和余弦距离直方图直观展示了Vittle如何将扰动样本拉近干净样本，形成更紧凑的表征空间

## 局限与展望

- **依赖标注质量**：IB目标以响应$Y$作为"充分"信息的锚点，但LLM生成的指令数据通常存在噪声，噪声标注下IB的优势可能减弱
- **OCR能力略有下降**：信息压缩在增强高级语义鲁棒性的同时可能损害细粒度字符识别能力
- **不保证反事实/跨域泛化**：对于视觉-语言先验冲突的反事实样本或完全不同的域，IB单独无法保证泛化
- **仅验证了7B-13B规模**：未在更大规模（如70B+）或闭源模型上验证
- **先验选择缺乏自适应机制**：Vittle (F)在扰动场景更优而Vittle (L)在长尾场景更优，但无法自动切换
- **瓶颈层位置选择**：默认top 25%层，未充分探索多层级或自适应位置策略

## 与相关工作的对比

- **Alemi et al. (2017) VIB**：经典VIB仅用于小规模分类模型，本文是首个将IB扩展到大规模自回归多模态模型的工作
- **Bai et al. (2025)**：在投影层上做IB训练并冻结LLM骨干，仅浅层适配；本文直接修改LLM内部结构实现端到端IB
- **ROSS & LIT（信息最大化方向）**：与Vittle的信息压缩设计理念相反，在POPE幻觉检测上有效但在开放式QA上效果差，说明压缩比最大化更具通用性
- **LoRA / Weight Decay（参数空间正则化）**：LoRA在扰动下性能大幅下降（联合扰动从62.3降至39.7），权重空间正则化无法替代表征空间的信息控制
- **Oh et al. (2025) EMID**：提出MLLM鲁棒性的信息论度量，本文在此基础上证明Vittle能降低EMID上界并通过实验验证

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次将IB原理端到端集成到MLLM指令微调，理论推导与架构设计均属全新贡献
- 实验充分度: ⭐⭐⭐⭐⭐ — 45个数据集、30种分布偏移、多种MLLM架构、多项消融，实验规模和覆盖面极为全面
- 写作质量: ⭐⭐⭐⭐⭐ — 动机清晰、理论自洽，定量与定性分析互相呼应，图表设计精良
- 价值: ⭐⭐⭐⭐ — 提供了实用的鲁棒性增强方案，但仍限于7B-13B规模且OCR能力轻微损失

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning to Instruct for Visual Instruction Tuning](learning_to_instruct_for_visual_instruction_tuning.md)
- [\[NeurIPS 2025\] CoIDO: Efficient Data Selection for Visual Instruction Tuning via Coupled Importance-Diversity Optimization](coido_efficient_data_selection_for_visual_instruction_tuning_via_coupled_importa.md)
- [\[ICML 2025\] Parrot: Multilingual Visual Instruction Tuning](../../ICML2025/multimodal_vlm/parrot_multilingual_visual_instruction_tuning.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](../../ICCV2025/multimodal_vlm/metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[ACL 2025\] Enhancing Multimodal Continual Instruction Tuning with BranchLoRA](../../ACL2025/multimodal_vlm/branchlora_continual_instruction.md)

</div>

<!-- RELATED:END -->
