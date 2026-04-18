---
title: >-
  [论文解读] TextPecker: Rewarding Structural Anomaly Quantification for Enhancing Visual Text Rendering
description: >-
  [CVPR 2026][图像生成][visual text rendering] 提出 TextPecker——一种即插即用的结构异常感知 RL 策略，通过构建字符级结构异常标注数据集训练结构感知识别器，替代传统 OCR 的噪声奖励信号，联合优化语义对齐和结构保真度，在多个文本到图像模型（FLUX、SD3.5、Qwen-Image）上显著提升视觉文本渲染质量。
tags:
  - CVPR 2026
  - 图像生成
  - visual text rendering
  - structural anomaly
  - reinforcement-learning
  - reward model
  - OCR
---

# TextPecker: Rewarding Structural Anomaly Quantification for Enhancing Visual Text Rendering

**会议**: CVPR 2026  
**arXiv**: [2602.20903](https://arxiv.org/abs/2602.20903)  
**代码**: [GitHub](https://github.com/CIawevy/TextPecker)  
**领域**: image_generation  
**关键词**: visual text rendering, structural anomaly, reinforcement-learning, reward model, OCR

## 一句话总结

提出 TextPecker——一种即插即用的结构异常感知 RL 策略，通过构建字符级结构异常标注数据集训练结构感知识别器，替代传统 OCR 的噪声奖励信号，联合优化语义对齐和结构保真度，在多个文本到图像模型（FLUX、SD3.5、Qwen-Image）上显著提升视觉文本渲染质量。

## 研究背景与动机

**视觉文本渲染（VTR）仍是 T2I 生成的关键挑战**：即使是先进模型（如 FLUX、GPT-4o、BAGEL）也频繁产生扭曲、模糊、错位或缺字等结构异常。

**OCR/MLLM 作为评估器存在根本缺陷**：现有评估和 RL 优化流程依赖 OCR 模型或 MLLM 识别生成文本再计算编辑距离奖励。然而这些模型无法感知细粒度结构异常，表现为两类失败：(a) **误解读**：过度依赖语言先验"纠正"结构缺陷，忽略笔画缺失/错位等字形级缺陷；(b) **不可见**：直接忽略严重模糊/扭曲区域，当作不存在。

**评估器盲区导致误导性奖励**：OCR 的"自动纠错"会压低编辑距离 $N_e$、虚高奖励分数 $S$，导致 RL 优化方向偏离。即使是高度优化的 Qwen-Image、Seedream4.0 仍难以渲染结构忠实的文本。

**结构异常标注数据匮乏**：缺少字符级结构异常标注的训练数据，尤其是中文字符因二维空间组合和 8000+ 字符量带来组合爆炸。

## 方法详解

### 整体框架

TextPecker 采用 GRPO（Group Relative Policy Optimization）框架，核心改进在用结构感知复合奖励替代 OCR 奖励。流程：
1. 从参考策略模型 $\pi_{\theta_{\text{ref}}}$ 采样 $G$ 个候选输出 $\{o_i\}_{i=1}^G$
2. 结构感知识别器提取细粒度生成文本并标记结构异常字符
3. 计算联合奖励 $\mathcal{R}_i$（语义对齐 + 结构质量）
4. 归一化为组相对优势 $A_i$，通过 KL 散度约束优化策略模型 $\pi_\theta$

### 关键设计 1：结构质量分数 $\mathcal{S}_Q$

- **功能**：量化生成文本中结构异常字符的比例，并通过缩放因子放大对罕见但严重缺陷的惩罚。
- **公式**：

$$\mathcal{S}_Q = \text{clip}\left(1 - \omega \frac{N_a}{N_P},\ 0,\ 1\right)$$

其中 $N_P$ 是生成文本总字符数，$N_a$ 是被标记为结构异常的字符数，$\omega > 1$ 是缩放因子（实验中 $\omega=5$）。
- **设计动机**：对于强生成器，结构错误虽稀少但视觉上极为刺眼。$\omega$ 放大罕见错误的惩罚力度，防止策略因偶发缺陷获得高分。

### 关键设计 2：语义对齐分数 $\mathcal{S}_E$

- **功能**：在词级别进行匈牙利匹配，计算目标文本与生成文本的归一化编辑距离，并惩罚未匹配词。
- **公式**：

$$\mathcal{S}_E = 1 - \frac{\sum_{(t_i, p_j) \in \mathcal{M}} \text{NED}(t_i, p_j) + \text{Penalty}(\mathcal{T}, \mathcal{P}, \mathcal{M})}{\max(|\mathcal{T}|, |\mathcal{P}|)}$$

其中 $\mathcal{T}$、$\mathcal{P}$ 分别为目标和生成文本词集，$\mathcal{M}$ 是基于 NED 的匈牙利最优配对，$\text{Penalty}(\cdot)$ 统计未匹配词数。
- **设计动机**：生成文本词序可能与 prompt 不一致，需词级匹配而非简单字符串比较；惩罚多余/缺失词确保全面评估。

### 关键设计 3：复合奖励 $\mathcal{R}$

$$\mathcal{R} = w_E \mathcal{S}_E + w_Q \mathcal{S}_Q, \quad w_E + w_Q = 1$$

实验中 $w_E = w_Q = 0.5$，联合优化语义准确性和结构保真度。

### 关键设计 4：结构感知数据构建

三步流水线构建字符级结构异常标注数据集（共 1.4M 样本）：

1. **文本图像生成**：使用多个 T2I 模型（AnyText、SD1.5、SD3.5、FLUX、Seedream3.0、Qwen-Image 用于英文；Cogview4、Kolors、Seedream3.0、Qwen-Image 用于中文）生成大规模文本图像。中文 prompt 从 WanJuan1.0 采样，结合 Qwen3-235B 生成字体风格描述。
2. **结构异常标注**：先用 OCR 获取初步识别结果，标注员逐字符标记结构缺陷（模糊、扭曲、缺笔画、多余笔画），严重粘连字符用占位符标记。
3. **合成数据增强**：引入笔画编辑合成引擎，对中文字符执行三种笔画级操作：
    - **笔画删除**：移除部分笔画子集
    - **笔画交换**：交换不相交笔画对的位置（对齐质心）
    - **笔画插入**：从其他字符采样笔画插入

合成的异常和正常字符通过 SynthTIGER 渲染引擎放置到多样背景和布局上。

| 数据类型 | 级别 | 样本数 | 占比 |
|---------|------|--------|------|
| 人工标注 | Box | 559.6K | 39.32% |
| 人工标注 | Image | 131.1K | 9.21% |
| 合成异常文本 | Box | 452.5K | 31.80% |
| 合成异常文本 | Image | 100.0K | 7.03% |
| 合成正常文本 | Box | 150.0K | 10.54% |
| 合成正常文本 | Image | 30.0K | 2.10% |
| **合计** | – | **1.4M** | 100% |

### RL 优化基座

- 基于 Flow-GRPO 将 GRPO 扩展到 rectified-flow 设定，通过注入随机性将确定性动力学转为随机微分方程：

$$dx_t = \left(v_t + \frac{\sigma_t^2}{2t}(x_t + (1-t)v_t)\right)dt + \sigma_t\,dw_t$$

- 识别器骨干：Qwen3-VL-8B 和 InternVL3-8B，支持边界框级输入，全参数微调 2 个 epoch。

## 实验结果

### 结构异常感知（TSAP）与标准文本识别（CTR）

| 方法 | 英文 TSAP F1 | 英文 CTR Recall | 中文 TSAP F1 | 中文 CTR Recall |
|-----|-------------|----------------|-------------|----------------|
| PP-OCRv5 | 0.000 | 0.720 | 0.024 | 0.921 |
| GOT-OCR-2.0 | 0.000 | 0.610 | 0.008 | 0.853 |
| GPT-5 | 0.170 | 0.556 | 0.226 | 0.758 |
| Qwen3-VL-8B | 0.032 | 0.807 | 0.017 | 0.943 |
| InternVL3-8B | 0.183 | 0.759 | 0.153 | 0.927 |
| **TextPecker (InternVL3)** | **0.870** | **0.944** | **0.927** | **0.962** |
| **TextPecker (Qwen3-VL)** | **0.862** | **0.918** | **0.925** | **0.972** |

- 现有 OCR 和 MLLM 在 TSAP 上几乎完全失败（F1 ≈ 0），TextPecker 达到 0.87+ F1。
- TextPecker 同时提升标准文本识别能力，CTR Recall 超过 0.94。

### VTR RL 优化

- **FLUX**：相比基线 Sem. +38.3%、Qua. +31.6%；相比 OCR 奖励，GenTextEval Sem. +11.7%。
- **Qwen-Image 中文渲染**：语义对齐 +8.7%、结构保真度 +4.0%，达到新 SOTA。
- **SD3.5-M**：Qua. 从 0.671 提升至 0.959，Sem. 从 0.265 提升至 0.506。

## 消融实验

- 移除合成数据增强后中文识别性能显著下降，验证笔画编辑引擎对中文结构异常覆盖的必要性。
- 仅用人工标注数据训练时模型对未见异常类型泛化性差。
- $\omega=5$ 在缩放因子消融中取得最优平衡。

## 优点与局限

**优点**：
- 首次系统性识别 VTR 中结构异常感知的关键瓶颈，为评估和优化提供全新视角
- 即插即用，无需修改生成器架构，适用于任意 T2I 模型
- 笔画编辑合成引擎巧妙解决中文字符结构异常的组合爆炸问题
- 在已高度优化的 Qwen-Image 上仍取得显著提升

**局限**：
- 数据标注成本较高（559.6K box 级标注）
- 结构感知识别器基于 8B 参数 VLM，推理开销较大
- 主要验证中英文，其他文字系统（如阿拉伯文、日文假名）未覆盖

## 个人评价

⭐⭐⭐⭐

这篇论文对 VTR 领域的关键痛点（OCR 评估器的结构盲区）进行了深入分析和有效解决。从"OCR 和 MLLM 在 TSAP 上 F1 ≈ 0"这一发现出发，构建数据集→训练识别器→设计复合奖励→RL 优化的完整链路非常流畅。笔画编辑合成引擎的设计体现了对中文字符特性的深入理解。在已经高度优化的 Qwen-Image 上仍能取得 +8.7% 语义+4% 结构提升，充分说明方法的实用价值。不足之处在于标注成本较高且推理开销较大，但作为一项填补评估空白的工作，贡献突出。
