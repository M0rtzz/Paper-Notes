---
title: >-
  [论文解读] AutoPP: Towards Automated Product Poster Generation and Optimization
description: >-
  [AAAI2026][product poster generation] 提出 AutoPP，首个将商品海报自动生成与基于 CTR 反馈的自动优化统一到一个框架中的流水线，通过 unified design module 联合设计背景/文字/排版，element rendering module 高效可控地生成海报，并利用 Isolated DPO (IDPO) 实现元素级别的点击率优化。
tags:
  - AAAI2026
  - product poster generation
  - CTR optimization
  - 扩散模型
  - DPO
  - 多模态
---

# AutoPP: Towards Automated Product Poster Generation and Optimization

**会议**: AAAI2026  
**arXiv**: [2512.21921](https://arxiv.org/abs/2512.21921)  
**代码**: [JD-GenX/AutoPP](https://github.com/JD-GenX/AutoPP)  
**领域**: recommender  
**关键词**: product poster generation, CTR optimization, diffusion model, DPO, multimodal generation

## 一句话总结

提出 AutoPP，首个将商品海报自动生成与基于 CTR 反馈的自动优化统一到一个框架中的流水线，通过 unified design module 联合设计背景/文字/排版，element rendering module 高效可控地生成海报，并利用 Isolated DPO (IDPO) 实现元素级别的点击率优化。

## 背景与动机

- 商品海报需要将产品、文字和背景巧妙组合以吸引用户点击，但手工制作和迭代优化非常耗时耗力
- 现有方法存在明显的自动化瓶颈：
    - PAID 采用四阶段流水线（提示词→排版→背景→文字渲染），其中文字属性（字体、颜色）依赖人工规则，限制了自动化且破坏视觉和谐
    - PosterMaker 虽然利用 SD3 + ControlNet 实现了背景与文字的同步渲染，但仍需用户为每张海报单独设计排版和卖点文案，无法高效大规模生产
- 在线优化方面，CG4CTR 和 CAIG 仅优化背景元素的 CTR，忽略了文字和排版对点击率的影响，且粗粒度的联合优化无法归因到具体元素

## 核心问题

1. **生成端**：如何仅凭基础产品信息（产品图 + 候选文案）就自动生成高质量海报，不需要人工设计排版、撰写卖点、指定文字属性？
2. **优化端**：如何将 CTR 改进精确归因到海报的具体元素（背景/文字/排版），实现细粒度的元素级优化，而非整体粗粒度调整？

## 方法详解

### 整体框架

AutoPP 由**生成器**和**优化器**两大部分组成：

### 1. Unified Design Module（统一设计模块）

- 使用 MLLM（基于 LLaVA 初始化）联合生成三个关键元素：背景提示词 $b$、选定文案 $T^*$、排版 $l$
- 输入：产品图 $I_{\text{product}}$ + 候选文案集合 $T$
- 通过自回归方式建模联合分布：$\pi(y|I_{\text{product}}, X_{\text{instr}}) = \prod_i p(y_i | I_{\text{product}}, X_{\text{instr}}, y_{<i})$
- 相比分散的多模型方案，单一模型联合推理保证了设计一致性

### 2. Element Rendering Module（元素渲染模块）

- 基于 FLUX.1 dev，将产品图和字形图编码为 condition tokens
- 关键创新：**Decomposed Attention (DA)** 机制替代 MM-DiT 中的 full attention
    - **Condition Self-Attention**：字形和产品 tokens 各自独立进行自注意力，捕获元素内部依赖
    - **Image-Condition Cross-Attention**：Query = [prompt tokens; noise tokens]，Key/Value = [所有类型 tokens 拼接]，实现跨模态信息交换
- token 机制的优势：无需像素级对齐，对字形图与目标图的空间偏差鲁棒
- 训练损失：flow matching loss + OCR perceptual loss（利用 PaddleOCRv4 backbone 中间特征强制文字区域清晰度，$\lambda=0.1$）

### 3. Systematic Element Replacement（系统化元素替换策略）

- 从生成的海报出发，每次只替换一个元素（保持其余不变）来创建变体：
    - 背景替换：用 GPT-4o 基于原提示词生成不同背景描述
    - 文字替换：从候选文案集中选等长替代文案
    - 排版替换：由统一设计模块重新生成排版
- 变体海报在京东平台进行随机展示实验，收集 CTR 反馈

### 4. Isolated Direct Preference Optimization (IDPO)

- 标准 DPO 对整体输出做粗粒度对齐，无法区分各元素的贡献
- IDPO 引入**元素感知权重**：$w_i = \sum_{c \in \{b, T^*, l\}} \alpha_c \cdot \mathbb{I}(y_i \in c)$
- 被替换的元素权重 $\alpha=5$，未变元素 $\alpha=1$，使 CTR 反馈精确引导最具影响力的元素
- 加权后的对数似然归一化：$\log \pi^w(y|I,X) = \frac{\sum_i w_i \log p(y_i|I,X,y_{<i})}{\sum_i w_i}$

### AutoPP1M 数据集

- **生成子集**：100 万张高质量商品海报（1:1，≥800×800），来自京东平台，经美学过滤、模糊检测、水印去除等清洗
- **优化子集**：5 万组配对比较，通过 10 天随机展示实验收集，每张海报至少 50 位用户浏览，共 111.8 万用户参与，配对间 CTR 差异≥1%

## 实验关键数据

### 海报生成（离线评估，500 张海报）

| 方法 | FID↓ | CLIP-T↑ | Alignment↓ | Overlap↓ | MIoU↑ |
|------|------|---------|------------|----------|-------|
| P&R | 104.05 | 27.21 | 0.014 | 0.024 | 0.203 |
| PAID | 83.55 | 28.92 | 0.013 | 0.041 | 0.215 |
| GPT-4o | 63.47 | 29.58 | 0.009 | 0.018 | 0.140 |
| **AutoPP** | **60.71** | **29.75** | **0.007** | **0.011** | **0.256** |

### 文字渲染质量

| 方法 | Sen. Acc↑ | NED↓ | FID↓ | CLIP-T↑ |
|------|-----------|------|------|---------|
| PosterMaker | 57.87 | 21.93 | 49.76 | 30.43 |
| **AutoPP** | **65.19** | **12.94** | **43.19** | **30.49** |

### 在线 CTR 优化（京东 1 周实验，10000 个产品）

- AutoPP (IDPO)：CTR 相对提升 **+4.49%**
- AutoPP (标准 DPO)：+3.10%
- CG4CTR / CAIG：CTR 为负（因仅优化背景，忽略文字和排版）

### 效率

- DA 机制在 800×800 分辨率下减少 MM-DiT block 18% GFLOPs，1024×1024 下减少 24%
- 不引入额外参数（PosterMaker +1.6B，Flux-ControlNet +4.2B）

### 数据量影响

- Reward Accuracy 随数据量增长：10K→51.20%，30K→67.19%，50K→75.99%

## 亮点

1. **端到端全自动化**：从基础产品信息到最终优化海报，完全无需人工输入排版、文案属性或手动调整
2. **IDPO 的细粒度归因**：通过系统化的单元素替换 + 元素感知权重的 DPO，首次实现了将 CTR 改进精确归因到孤立元素
3. **Decomposed Attention**：在不增加参数的前提下，通过将 full attention 分解为 condition SA + image-condition CA，降低了长序列的计算开销
4. **大规模工业验证**：AutoPP1M 是目前最大的商品海报数据集，在线实验有超百万真实用户参与
5. **跨语言泛化**：主要以中文训练，但在英文、日文、韩文上展现了涌现的跨语言生成能力

## 局限与展望

- CTR 优化使用所有用户的聚合数据，可能忽略少数群体偏好，未来可探索**个性化偏好学习**
- 设计模块和渲染模块仍是分离的两阶段，未来可整合为**单一自回归模型**并用 RLHF 统一优化
- 元素替换策略依赖 GPT-4o 生成背景变体，引入了外部模型依赖
- 仅在京东平台验证，跨平台泛化性未知

## 与相关工作的对比

| 方法 | 全自动生成 | 文字渲染 | 排版设计 | CTR 优化 | 元素归因 |
|------|:---:|:---:|:---:|:---:|:---:|
| PAID | ✗（需手工文字规则） | 规则渲染 | 自动 | ✗ | ✗ |
| PosterMaker | ✗（需用户提供排版和卖点） | SD3+ControlNet | 手动 | ✗ | ✗ |
| CG4CTR | - | - | - | ✓（仅背景） | ✗ |
| CAIG | - | - | - | ✓（仅背景） | ✗ |
| **AutoPP** | **✓** | **Token+DA** | **自动** | **✓（全元素）** | **✓（IDPO）** |

## 启发与关联

- IDPO 的元素隔离优化思路可推广到其他**多元素组合优化**场景（如广告创意、网页设计、推荐 feed 卡片）
- Decomposed Attention 的 condition SA + cross-attention 分解策略适用于任何多条件控制生成任务
- 系统化元素替换 + 偏好优化的范式可应用于其他需要在线 A/B 测试反馈的场景
- 将 MLLM 用于联合设计（同时输出布局、文案选择、背景描述）的做法值得在其他多步设计任务中借鉴

## 评分

- 新颖性: ⭐⭐⭐⭐ — IDPO 的元素级归因优化和全自动化流水线设计新颖，但各子模块（MLLM 设计、FLUX 渲染）基于成熟架构
- 实验充分度: ⭐⭐⭐⭐⭐ — 离线+在线双验证，百万用户规模实验，消融实验完整
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图示丰富，方法描述完整
- 价值: ⭐⭐⭐⭐⭐ — 强工业落地价值，京东实际部署，0.5% CTR 提升即有显著商业回报
