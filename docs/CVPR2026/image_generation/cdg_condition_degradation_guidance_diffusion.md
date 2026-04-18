---
title: >-
  [论文解读] Guiding Diffusion Models with Semantically Degraded Conditions (CDG)
description: >-
  [CVPR 2026][图像生成][扩散模型引导] 提出 Condition-Degradation Guidance (CDG)，用语义退化的条件 $\boldsymbol{c}_{\text{deg}}$ 替代 CFG 中的空提示 $\emptyset$，将引导从"好 vs 空"转变为"好 vs 几乎好"的精细化对比，从而在无需训练的前提下显著提升扩散模型的组合生成精度。
tags:
  - CVPR 2026
  - 图像生成
  - 扩散模型引导
  - 条件退化
  - 文本到图像
  - 组合生成
  - 注意力分析
---

# Guiding Diffusion Models with Semantically Degraded Conditions (CDG)

**会议**: CVPR 2026  
**arXiv**: [2603.10780](https://arxiv.org/abs/2603.10780)  
**代码**: [GitHub](https://github.com/Ming-321/Classifier-Degradation-Guidance)  
**领域**: 图像生成  
**关键词**: 扩散模型引导、条件退化、文本到图像、组合生成、注意力分析  

## 一句话总结

提出 Condition-Degradation Guidance (CDG)，用语义退化的条件 $\boldsymbol{c}_{\text{deg}}$ 替代 CFG 中的空提示 $\emptyset$，将引导从"好 vs 空"转变为"好 vs 几乎好"的精细化对比，从而在无需训练的前提下显著提升扩散模型的组合生成精度。

## 研究背景与动机

Classifier-Free Guidance (CFG) 是现代文本到图像扩散模型的基石，通过无条件预测与有条件预测的外推来增强生成质量。然而 CFG 的核心问题在于其依赖**语义空虚的空提示** $\emptyset$：

1. **几何纠缠问题**：条件 $\boldsymbol{c}$ 与 $\emptyset$ 之间的语义距离太大，导致引导信号将内容生成与风格/结构混合在一起，产生纠缠的梯度信号
2. **组合失败**：CFG 在复杂任务中经常失败，包括文字渲染错误、空间关系混乱、属性绑定不精确
3. **现有改进的局限**：过程修正方法（如 APG、TCFG）仍保留 $\boldsymbol{c}$ vs $\emptyset$ 的对比框架，只是做事后修正；负样本重构方法要么语义盲目（随机噪声），要么依赖昂贵的外部模型（VLM 生成负样本），没有利用提示自身 token 嵌入的内在语义结构

作者的核心观察是：如果用一个语义接近的退化条件 $\boldsymbol{c}_{\text{deg}}$ 替代 $\emptyset$，可以实现**共模抑制效应**——两个语义邻居共享的法向分量在相减时自动消除，只留下纯粹的语义修正信号。

## 方法详解

### 整体框架

CDG 的引导公式为：

$$D_\theta^{\text{CDG}}(\boldsymbol{x}_\sigma; \sigma, \boldsymbol{c}) = D_\theta(\boldsymbol{x}_\sigma; \sigma, \boldsymbol{c}) + (w-1)(D_\theta(\boldsymbol{x}_\sigma; \sigma, \boldsymbol{c}) - D_\theta(\boldsymbol{x}_\sigma; \sigma, \boldsymbol{c}_{\text{deg}}))$$

核心流程：
1. 从文本编码器的自注意力图中提取 token 重要性（Weighted PageRank）
2. 基于重要性将 token 分为**内容 token**（编码对象语义）和**上下文聚合 token**（编码全局上下文）
3. 通过分层退化策略构建退化条件 $\boldsymbol{c}_{\text{deg}}$
4. 用 $\boldsymbol{c}_{\text{deg}}$ 替代 CFG 中的空提示进行引导

### 关键设计

1. **Token 功能二分法与 Weighted PageRank 分析**：在 transformer 文本编码器中，token 自然分为两类——内容 token（如 "minecraft"、"cooking"）携带细粒度语义，上下文聚合 token（padding 和特殊 token）通过注意力吸收全局上下文信息。作者将自注意力矩阵建模为图，使用 WPR 算法计算每个 token 的重要性分数，验证了内容 token 的重要性显著高于上下文聚合 token 这一二分结构。

2. **分层退化策略 (Stratified Degradation)**：引入统一退化比率 $R_{\text{deg}} \in [0,2]$，通过 $r_{\text{content}} = \min(R_{\text{deg}}, 1.0)$ 和 $r_{\text{CtxAgg}} = \max(R_{\text{deg}}-1.0, 0)$ 映射到两类 token 的退化比率。这确保了内容 token 优先于上下文聚合 token 被退化。$R_{\text{deg}}=1.0$ 是一个自然的"语义边界"：$[0,1]$ 区间移除细粒度语义，$(1,2]$ 区间移除粗粒度全局语义。默认 $R_{\text{deg}}=1.0$ 时所有内容 token 被退化，无需 WPR 计算，开销几乎为零。

3. **掩码插值构建退化条件**：基于重要性排序生成二值掩码 $\boldsymbol{m}$，通过 $\boldsymbol{c}_{\text{deg}} = \boldsymbol{m} \odot \boldsymbol{c} + (1-\boldsymbol{m}) \odot \emptyset$ 在原始条件和空条件之间进行掩码插值。退化条件保留了全局语义骨架（上下文聚合 token），同时失去了细粒度语义细节（内容 token），实现了精确的"好 vs 几乎好"对比。

### 损失函数 / 训练策略

CDG 是**免训练**的即插即用模块：
- 掩码 $\boldsymbol{m}$ 仅在第一个去噪步骤计算一次，后续步骤复用
- 引入干预块索引 $\lambda_{\text{block}}$ 指定从哪个 transformer 块提取注意力图
- 在 $\lambda_{\text{block}}$ 处触发掩码构建，后续所有块使用 $\boldsymbol{c}_{\text{deg}}$
- 无需外部模型或额外训练

### 几何分析

作者从流形假设出发提出两个度量来解释 CDG 的优越性：
- **几何解耦度**：衡量引导信号与主去噪子空间的正交性，CDG 全程保持近乎完美的正交性
- **干扰能量比**：衡量引导信号投影到去噪子空间的能量比，CDG 的干扰极小

CDG 的共模抑制效应使得 $\boldsymbol{c}$ 和 $\boldsymbol{c}_{\text{deg}}$ 共享的法向分量相消，留下纯语义修正信号。

## 实验关键数据

### 主实验

| 模型 | 方法 | FID↓ | CLIP Score↑ | Aesthetic↑ | VQA Score↑ |
|------|------|------|-------------|------------|------------|
| SD3 | CFG | 35.69 | 31.73 | 5.66 | 91.44 |
| SD3 | **CDG** | **34.05** | **32.00** | **5.70** | **92.40** |
| SD3.5 | CFG | 34.56 | 31.85 | 6.21 | 91.94 |
| SD3.5 | **CDG** | **33.07** | **31.96** | **6.26** | **92.61** |
| FLUX.1 | CFG | 38.55 | 31.20 | 6.06 | 90.31 |
| FLUX.1 | **CDG** | **37.11** | **31.21** | **6.15** | **90.62** |
| Qwen | CFG | 42.45 | 32.11 | 2.57 | 93.66 |
| Qwen | **CDG** | **39.02** | **32.31** | 2.54 | **93.93** |

GenAI-Bench 组合推理（SD3.5）：CDG 在 Differentiation 上提升 +3.64，Comparison +2.36。

### 消融实验

| 重要性排序 | 分层退化 | FID↓ | VQA Score↑ |
|-----------|---------|------|------------|
| WPR | ✓ | 33.89 | 92.21 |
| 随机 | ✓ | 34.17 | 92.27 |
| WPR | ✗ | 35.06 | 86.31 |
| 反向WPR | ✗ | 50.73 | 80.10 |
| 随机 | ✗ | 47.02 | 83.55 |

**分层退化是性能的主要驱动力**：两种分层变体（前两行）大幅优于所有非分层变体（后三行），VQA 提升 5.9-12.2 分。

### 关键发现

1. **分层退化比 WPR 排序更重要**：在分层框架下，WPR 与随机排序性能相当，但 WPR 提供了理论基础和 $R_{\text{deg}}=1.0$ 边界的确定性依据
2. **CDG 在 FLUX 上的提升较小**：FLUX 使用了引导蒸馏，减少了对推理时引导的依赖
3. **CFG* 验证实验**证实了内容/上下文聚合 token 的二分法——内容 token 移除导致 CLIP Score 急剧下降，而上下文聚合 token 移除的影响更温和
4. **计算效率极高**：一次性计算策略仅增加 3.6% 开销；默认 $R_{\text{deg}}=1.0$ 时开销几乎为零

## 亮点与洞察

- 揭示了 transformer 文本编码器中内容/上下文聚合 token 的功能二分法，这不是特定架构的特性而是 transformer 编码器的基本属性
- "好 vs 几乎好"的引导范式比"好 vs 空"在几何上更优——引导信号与去噪方向正交，避免了能量浪费
- 通过共模抑制效应的类比优雅解释了 CDG 的工作原理
- 即插即用、零训练、近零开销，实用价值极高

## 局限与展望

- CDG 在已使用引导蒸馏的模型（如 FLUX）上改进较小
- $R_{\text{deg}}$ 虽然默认 1.0 效果好，但不同任务/风格可能需要微调
- 目前仅验证了文本到图像场景，视频生成等其他模态待探索
- WPR 分析虽然提供了理论洞察，但实际使用中结论是分层退化本身才是关键，分析工具的必要性有待讨论

## 相关工作与启发

- **APG/TCFG**：在 CFG 框架内的几何修正，治标不治本
- **PAG/SEG**：在模型内部机制层面扰动注意力/能量曲率，与 CDG 正交互补
- **Autoguidance**：使用弱模型提供负信号，需额外模型调优
- **启发**：CDG 的思路可以推广到其他条件生成框架——构建自适应、语义感知的负样本是实现精确语义控制的关键原则

## 评分

- **新颖性**: 8/10 — 从 token 功能二分法出发构建退化条件的思路新颖，几何分析深入
- **实验充分度**: 9/10 — 四个模型、多个 benchmark、全面的消融和机制验证
- **写作质量**: 9/10 — 逻辑清晰，几何分析与实验观察紧密结合
- **价值**: 8/10 — 即插即用的实用方案，为扩散模型引导设计提供了新的原则性框架
