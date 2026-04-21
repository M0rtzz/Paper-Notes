---
title: >-
  [论文解读] Localizing and Mitigating Memorization in Image Autoregressive Models
description: >-
  [ICML2025][图像生成][图像自回归模型] 利用改进的UnitMem指标定位图像自回归模型（VAR/RAR）中的记忆化神经元，发现不同架构的记忆化分布模式存在显著差异，并通过缩小高记忆化神经元权重实现了大幅降低可提取训练数据量（VAR-d30从672降至110张）且对生成质量影响可控的隐私缓解方案。
tags:
  - ICML2025
  - 图像生成
  - 图像自回归模型
  - 记忆化定位
  - 隐私保护
  - UnitMem
  - 数据提取攻击
---

# Localizing and Mitigating Memorization in Image Autoregressive Models

**会议**: ICML2025  
**arXiv**: [2509.00488](https://arxiv.org/abs/2509.00488)  
**作者**: Aditya Kasliwal, Franziska Boenisch, Adam Dziedzic  
**代码**: 未公开  
**领域**: image_generation  
**关键词**: 图像自回归模型, 记忆化定位, 隐私保护, UnitMem, 数据提取攻击  

## 一句话总结

利用改进的UnitMem指标定位图像自回归模型（VAR/RAR）中的记忆化神经元，发现不同架构的记忆化分布模式存在显著差异，并通过缩小高记忆化神经元权重实现了大幅降低可提取训练数据量（VAR-d30从672降至110张）且对生成质量影响可控的隐私缓解方案。

## 研究背景与动机

### 问题背景
图像自回归（IAR）模型——如Visual Autoregressive Modeling（VAR）和Randomized Autoregressive（RAR）——已在图像生成质量和速度方面达到SOTA水平，超越了扩散模型等其他框架。然而，这类模型容易记忆训练数据，带来严重的隐私风险：被记忆的敏感数据可能被恶意提取或无意泄露。

### 已有工作的不足
- 记忆化现象在**扩散模型**中已被广泛研究（Carlini et al., 2023; Somepalli et al., 2023），但IAR模型的记忆化研究仍处于早期
- Kowalczuk et al. (2025) 已证明IAR模型甚至比扩散模型表现出更强的记忆化倾向，但缺乏对记忆化**定位**的深入分析
- 已有的记忆化定位方法（UnitMem）针对的是视觉编码器，尚未被系统性地应用于IAR这类迭代式生成架构
- 缺乏从记忆化定位到实际隐私缓解的端到端验证

### 核心动机
理解记忆化在IAR模型内部的**空间分布规律**，是构建实用隐私缓解策略的前提。本文旨在回答：(1) IAR模型中哪些组件负责存储训练数据？(2) 不同架构的记忆化模式有何差异？(3) 能否通过定向干预高记忆化组件来降低数据提取风险？

## 方法详解

### 整体框架

本文方法分为三个阶段：**记忆化度量** → **模式分析** → **干预缓解**。

1. **UnitMem指标适配**：将原本为视觉编码器设计的UnitMem指标改造以适配IAR模型的迭代生成特性
2. **记忆化定位分析**：在VAR（分层多尺度）和RAR（逐token）两种架构上系统性地定位记忆化神经元
3. **权重缩放干预**：对高记忆化神经元进行权重缩放，验证定位准确性并实现隐私缓解

### 关键设计1：UnitMem指标的IAR适配

UnitMem指标量化单个神经元$u$的记忆化程度：

$$\text{UnitMem}_{\mathcal{D}'}(u) = \frac{\mu_{max,u} - \mu_{-max,u}}{\mu_{max,u} + \mu_{-max,u}}$$

其中$\mu_{max,u}$为单元$u$对特定数据点$x_k$的最大激活值，$\mu_{-max,u}$为$u$对其余所有数据点的平均激活值。值越高表示该神经元对特定训练样本的"选择性"越强，即记忆化程度越高。

**GELU适配**：原始UnitMem针对ReLU设计（激活值非负），但VAR和RAR使用GELU激活函数（可产生负值）。本文使用激活值的**绝对值**来计算$\mu_{max,u}$和$\mu_{-max,u}$，确保激活幅度而非符号决定记忆化得分。

**Teacher-Forced推理**：IAR模型在生成过程中迭代使用相同组件。为避免错误累积影响度量准确性，在计算UnitMem激活值时采用teacher-forcing：每一步输入真实的前序数据而非模型自身预测。

**分析聚焦fc1层**：VAR和RAR的每个block包含attention层和两个全连接层（fc1、fc2）。由于fc1使用GELU激活而fc2无激活函数，UnitMem分析聚焦于fc1层的神经元。

### 关键设计2：架构特异性的记忆化定位

**VAR（分层多尺度架构）**：
- VAR跨10个尺度层级式生成图像，相同transformer blocks在每个尺度重复使用
- 本文以**尺度+block位置**二维方式计算UnitMem，揭示记忆化在尺度和深度间的分布
- **发现**：在粗尺度（低分辨率）下记忆化集中在初始blocks，随着尺度变精细逐渐转移至更深layers

**RAR（逐token自回归架构）**：
- RAR使用随机排列的token序列和双向注意力逐token生成
- 记忆化以block为粒度分析
- **发现**：记忆化集中在中间和后期blocks

### 关键设计3：权重缩放干预

对识别出的高UnitMem分数神经元进行干预验证：
- 将目标百分比的高记忆化神经元权重**缩放为原来的一半**
- 评估干预后：(1) 可提取训练图像数量变化；(2) FID等生成质量指标变化
- 若干预有效降低提取量且质量影响可控，则验证了定位准确性

## 实验关键数据

### 实验设置
- 模型：VAR-d16、VAR-d30（最小/最大配置）、RAR-Base、RAR-XXL（最小/最大配置）
- 训练数据：ImageNet-1k
- UnitMem计算子集：ImageNet-1k训练集的1%（每类均匀采样），经验证1%/5%/10%/20%子集得到相似的记忆化模式
- 每个数据点进行10次不同数据增强的前向传播取平均激活

### 主结果：数据提取缓解效果

| 模型 | 干预前可提取图像数 | 干预后可提取图像数 | 减少比例 | FID影响 |
|------|------------------|------------------|---------|--------|
| VAR-d16 | — | — | 显著减少 | 可控 |
| VAR-d30 | 672 | 110 | **83.6%** | 有限影响 |
| RAR-Base | — | — | 显著减少 | 可控 |
| RAR-XXL | 75 | 26 | **65.3%** | 有限影响 |

VAR-d30的提取量从672张骤降至110张（减少83.6%），RAR-XXL从75张降至26张（减少65.3%），表明定位方法准确识别了记忆化关键神经元。

### 消融实验：记忆化分布模式对比

| 架构类型 | 生成方式 | 记忆化集中区域 | 记忆化随深度变化 |
|---------|---------|--------------|----------------|
| VAR（分层多尺度） | 逐尺度预测 | 粗尺度→初始blocks；细尺度→深层blocks | 随尺度精细化向深层迁移 |
| RAR（逐token） | 随机排列token序列 | 中间和后期blocks | 集中在处理后期阶段 |

该对比揭示了架构设计对记忆化分布的决定性影响：分层架构的记忆化随分辨率动态迁移，而标准自回归架构的记忆化则稳定集中在后期处理阶段。

### 补充实验：UnitMem子集大小鲁棒性

| ImageNet-1k子集比例 | 记忆化模式一致性 |
|--------------------|----------------|
| 1% | 基准模式 |
| 5% | 与1%一致 |
| 10% | 与1%一致 |
| 20% | 与1%一致 |

验证了使用1%子集即可充分捕捉记忆化模式，大幅降低计算开销。

## 亮点与洞察

- **首次系统性定位IAR记忆化**：将UnitMem从视觉编码器扩展到生成式IAR模型，揭示了记忆化的架构特异性分布规律
- **发现架构决定记忆化模式**：VAR的分层设计导致记忆化随尺度动态迁移（粗→初始，细→深层），RAR的顺序设计导致记忆化后置——这一洞察为模型设计提供了隐私視角的参考
- **实用的隐私缓解方案**：仅通过权重缩放（无需重训练）即可减少80%+的可提取数据，且生成质量损失可控，具有极高的实用价值
- **轻量高效的度量方案**：单次前向传播+1%数据即可定位记忆化，扩展性极佳
- **GELU适配方案简洁有效**：通过取绝对值巧妙解决了UnitMem对非ReLU激活函数的兼容问题

## 局限与展望

- **仅覆盖两种IAR架构**：仅分析VAR和RAR，未涵盖其他IAR模型（如LlamaGen等），结论的普适性有待验证
- **干预策略较简单**：仅使用固定的0.5权重缩放，未探索自适应缩放、裁剪、微调等更精细的干预方式
- **仅在ImageNet-1k上验证**：缺乏在其他数据集（如LAION、COCO）上的交叉验证
- **缺乏与其他隐私保护方法的对比**：未与差分隐私（DP-SGD）、机器遗忘（Machine Unlearning）等方法进行系统对比
- **UnitMem的理论支撑有限**：激活值选择性与记忆化的因果关系未被严格证明，可能存在混淆因素
- **生成质量评估维度单一**：主要依赖FID，未评估多样性（如IS、Precision/Recall）和语义一致性等指标
- **未分析记忆化的类别分布**：是否某些类别更容易被记忆？类别不平衡对记忆化的影响未探讨

## 相关工作与启发

- **UnitMem (Wang et al., 2024a)**：本文核心工具，原始工作在视觉编码器上证明记忆化单元可分布式存在且随层深度变化。本文将其成功扩展到生成模型
- **Kowalczuk et al. (2025)**：证明IAR模型的记忆化倾向甚至超过扩散模型，本文数据提取实验方法直接沿用自该工作
- **Carlini et al. (2023)**：扩散模型数据提取攻击的开创性工作，本文将类似的安全审计思路应用于IAR模型
- **Maini et al. (2023)**：研究大语言模型中的记忆化定位，方法论上对本文有启发
- **VAR (Tian et al., 2024)**：分层自回归模型的代表作，本文发现其多尺度设计引入了独特的记忆化动态迁移现象
- **RAR (Yu et al., 2024)**：随机排列自回归模型，其双向注意力机制下记忆化更集中在后期处理阶段

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统性定位IAR模型记忆化并揭示架构特异性模式，具有明确的创新贡献
- 实验充分度: ⭐⭐⭐ — 核心实验设计合理且验证充分，但覆盖的模型和数据集较有限，缺乏与其他方法的对比
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，问题motivate充分，方法与验证的逻辑链条完整
- 价值: ⭐⭐⭐⭐ — 提供了实用的零成本隐私缓解方案和架构级别的记忆化洞察，对模型安全部署有直接意义

<!-- RELATED:START -->

## 相关论文

- [Understanding and Mitigating Memorization in Diffusion Models for Tabular Data](understanding_and_mitigating_memorization_in_diffusion_models_for_tabular_data.md)
- [Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes](understanding_and_mitigating_memorization_in_generative_models_via_sharpness_of_.md)
- [Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability](../../ICLR2026/image_generation/detecting_and_mitigating_memorization_in_diffusion_models_through_anisotropy_of_.md)
- [Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots](hierarchical_masked_autoregressive_models_with_low-resolution_token_pivots.md)
- [BitMark: Watermarking Bitwise Autoregressive Image Generative Models](../../NeurIPS2025/image_generation/bitmark_watermarking_bitwise_autoregressive_image_generative_models.md)

<!-- RELATED:END -->
