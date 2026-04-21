---
title: >-
  [论文解读] Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Models and Small Edge Models
description: >-
  [ICCV 2025][图像生成][文本到图像生成] 提出 RouteT2I，首个面向文本到图像生成的边缘-云模型路由框架，通过多维质量度量、Pareto 相对优越性和双门控 token 选择 MoE 架构，在控制成本的同时最大化图像生成质量。
tags:
  - ICCV 2025
  - 图像生成
  - 文本到图像生成
  - 模型路由
  - 边缘-云协同
  - 混合专家
  - 多指标质量评估
---

# Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Models and Small Edge Models

**会议**: ICCV 2025  
**arXiv**: N/A (CVF OpenAccess)  
**代码**: 无  
**领域**: image_generation  
**关键词**: 文本到图像生成, 模型路由, 边缘-云协同, 混合专家, 多指标质量评估

## 一句话总结

提出 RouteT2I，首个面向文本到图像生成的边缘-云模型路由框架，通过多维质量度量、Pareto 相对优越性和双门控 token 选择 MoE 架构，在控制成本的同时最大化图像生成质量。

## 研究背景与动机

### 核心问题
大型文本到图像（T2I）模型（如 SD3.5，80亿参数）生成质量优秀但部署成本极高（每百万请求 65K 美元）。轻量级边缘模型成本低但复杂提示下质量较差。**关键观察**：并非所有提示都需要大模型——简单提示下小模型可能产生同等甚至更好的结果（如图像中名词数量少时）。

### 为什么现有 LLM 路由方法不能直接迁移？

**图像质量评估困难**：不同于文本有明确答案，图像质量缺乏统一标准，受颜色、清晰度、物体完整性等多因素影响

**输出空间远大于输入空间**：一个文本提示可对应无数张图像，使得生成前预测质量极为困难

**单目标优化不够**：已有 LLM 路由方法通常只优化单一质量指标，无法适应图像质量的模糊性和多维性

### 核心洞察
名词数量是影响请求复杂度的一个直觉因子。实验表明（Fig. 3）：随着提示中名词数量增加，大模型的胜率提高、质量差距增大。但即使简单提示下，大模型也不总是更好——这就需要一个智能路由机制。

## 方法详解

### 整体框架

RouteT2I 包含三个核心部分：
1. **多维质量度量体系**（§4）：定义图像质量的多维评估方式
2. **路由模型**（§5.1）：基于双门控 token 选择 MoE 的 Transformer，预测 Pareto 相对优越性
3. **路由策略**（§5.2）：基于预测的质量差距和预设成本约束进行路由决策

**优化目标**：在云服务路由率上界 $\rho_r$ 约束下，最大化整体生成图像质量：
$$\max_{R(X)} Q(I_c) + (1-R(X))Q(I_e) \quad \text{s.t.} \quad P\{R(X)=1\} \leq \rho_r$$

### 关键设计

#### 1. 多维对比质量度量（§4）

**核心思想**：利用文本-图像的关联性，通过正负文本对来度量图像质量维度。

**单一指标的对比质量**：
$$q(I, m) = \sigma(\text{CLIP}(I, m^+) - \text{CLIP}(I, m^-))$$

其中 $m=(m^+, m^-)$ 是正负文本对。例如清晰度指标：正文本 "High definition photo"，负文本 "Low definition photo"。

**10 维质量评估**：
$$Q(I) = [q(I, m_i) | i=1,2,...,10]$$

涵盖 Definition、Detail、Clarity、Sharpness、Harmony、Realism、Color、Consistency、Layout、Integrity 十个维度，结合了真实照片质量因子和生成图像特有因子（如真实感、物体完整性）。

**为什么要用对比方法？** 相比仅用正面提示衡量质量，对比方法通过评估正面还是负面质量更主导，提供了更鲁棒可靠的评估。

#### 2. Pareto 相对优越性（PRS）

**为什么不能直接比较多维质量？** 实际中很难找到所有指标都最优的 Pareto 最优图像。因此放松约束：允许某些指标略差，只要在其他指标上显著超越。

**归一化质量距离**：
$$D_i(I_e, I_c) = \sigma\left(\frac{q(I_e, m_i) - q(I_c, m_i)}{\Gamma|\mu_i(I_e) - \mu_i(I_c)|}\right)$$

其中温度参数 $\Gamma$ 和 sigmoid 函数用于调节分布，区分相近质量并防止中心化。

**PRS 定义**：
$$\text{PRS}(I_e, I_c) = \sum_{i=1}^{N} w_i D_i(I_e, I_c)$$

PRS 偏离 0.5 的程度表明边缘/云模型的质量优势。PRS > 0.5 说明边缘更好，< 0.5 说明云更好。

#### 3. 双门控 Token 选择 MoE（核心架构创新）

**设计动机**：T2I 生成过程中，提示作为 token 序列通过交叉注意力与图像交互，不同 token 对图像质量的影响不同。路由模型需要识别关键 token 并评估其正/负影响。

**Token 选择门控**：

$$A = \text{Softmax}(T \cdot E^T)$$

其中 $T \in \mathbb{R}^{n \times d}$ 是 token 表示，$E \in \mathbb{R}^{k \times d}$ 是专家嵌入（每个专家对应一个质量指标）。通过 Top-K 选择与每个专家最相关的 token。

**为什么需要 token 选择？** 在 T2I 生成中，不同 token（名词、形容词等）对不同质量指标的影响差异巨大。例如表示颜色的 token 主要影响 Color 指标，而名词主要影响 Integrity 指标。选择关键 token 可以减少无关 token 的干扰。

**双门控设计**：引入正门控 $G^+$ 和负门控 $G^-$，分别选择对质量有正面和负面影响的 token：
$$T_i^o[t] = T[t] \cdot P_i^o \cdot S_i, \quad o \in \{+, -\}$$

其中 $P_i^+, P_i^- \in \mathbb{R}^{d \times l}$ 是将 token 投影到低维正/负表示空间的矩阵，$S_i \in \mathbb{R}^{l \times h}$ 是共享评分矩阵。由于 $l \ll h, d$，参数量从 $O(hd)$ 降至 $O(l(h+d))$。

**正负对比**：
$$\hat{T}[t] = \sigma(T^+[t] - T^-[t])$$

通过对比判断每个 token 的主导影响（正面还是负面），消除预测中的歧义。

**多头预测**：模型包含多个预测头，每个头输出一个质量指标的预测，增强噪声抵抗和鲁棒性。

### 路由策略

设置 PRS 阈值 $\alpha$，PRS 低于阈值的提示路由到云端（云模型质量显著更好），高于阈值的留在边缘：
$$\max_{\alpha \leq 1/2} P\{\text{PRS}(I_e, I_c) < \alpha | I_e, I_c \in \mathcal{I}_e, \mathcal{I}_c\} \leq \rho_r$$

上界 $\alpha \leq 1/2$ 确保不会把边缘模型更优的提示错误路由到云端。

## 实验关键数据

### 主实验

**设置**：SD3（云）+ SD2.1（边缘），COCO2014 数据集，50% 路由率。

**多维质量对比**（路由率 50%）：

| 方法 | Definition | Detail | Integrity | Δ P(%) |
|------|-----------|--------|-----------|--------|
| Edge Only | 0.6251 | 0.6685 | 0.4690 | - |
| Cloud Only | 0.6337 | 0.6847 | 0.4972 | - |
| Random | 0.6294 | 0.6766 | 0.4831 | 40.00 |
| RouteLLM-BERT | 0.6347 | 0.6792 | 0.4866 | 71.51 |
| Hybrid LLM | 0.6327 | 0.6784 | 0.4864 | 73.49 |
| ZOOTER | 0.6350 | 0.6796 | 0.4854 | 77.95 |
| **RouteT2I** | **0.6350** | **0.6786** | **0.4865** | **83.97** |

RouteT2I 在 10 个质量指标中的 6 个上优于所有基线，整体性能提升达到云模型增益的 **83.97%**。

**成本节省**（Δ P 目标下的云调用率节省）：

| 方法 | Δ P=40% | Δ P=50% | Δ P=60% |
|------|---------|---------|---------|
| RouteLLM-BERT | 56.15% | 51.39% | 46.92% |
| ZOOTER | 69.28% | 65.76% | 60.81% |
| **RouteT2I** | **71.81%** | **70.24%** | **66.61%** |

### 消融实验

| 配置 | Δ w(%) @ p=40% | @ p=50% | @ p=80% |
|------|----------------|---------|---------|
| w/o Multi-Metric | 27.37 | 22.81 | 19.92 |
| w/o Token Selection | 27.82 | 23.05 | 19.24 |
| w/o Dual-Gate | 27.22 | 22.09 | 21.62 |
| **RouteT2I (完整)** | **30.60** | **25.81** | **21.94** |

### 关键发现

1. **多维优化至关重要**：移除多维质量优化后 40% 路由率下性能下降 3.23%，因为单一指标无法全面评估图像质量
2. **Token 选择门控在高路由率下更重要**：80% 路由率下移除导致约 2% 下降，说明在大部分请求需路由时筛选关键 token 更关键
3. **双门控在中等路由率下最有效**：50% 路由率附近效果最显著，此时正负影响的区分对路由决策影响最大
4. **跨模型对泛化性**：在 18 种云-边模型组合上都有效，质量差距大的模型对（如 SD3-SD1.5）改进更明显
5. **可超越纯云模型**：某些情况下路由后的整体质量甚至超过完全使用云模型

## 亮点与洞察

1. **首创 T2I 路由问题**：将 LLM 路由的思想推广到图像生成领域，问题定义清晰完整
2. **正负对比质量度量**：用 CLIP 的正负文本对衡量图像质量维度，既优雅又有效
3. **PRS 设计精巧**：松弛 Pareto 最优性，允许在部分指标妥协但整体更优，符合实际需求
4. **MoE 与 T2I 生成过程对齐**：将专家与质量指标对齐、模拟交叉注意力中 token 的不同影响，架构设计有深刻的 domain insight
5. **实用价值高**：在 40% 相对性能提升目标下，减少 71.81% 的云调用，具有显著的成本节约效果

## 局限与展望

1. **预测路由 vs 非预测路由**：当前方案不运行边缘模型就做路由决策，但非预测路由（先运行边缘再决定是否重试云端）在某些场景可能更优
2. **质量指标依赖 CLIP**：CLIP 的对比质量度量可能不完全对齐人类偏好，可以结合 ImageReward 等人类对齐指标
3. **训练数据规模**：路由模型需要大量的 prompt-quality 对进行训练，数据收集成本较高
4. **动态模型池**：当前固定为两个模型的路由，扩展到多模型级联路由更具挑战性
5. **隐私问题**：将提示发送到云端涉及用户隐私，未讨论隐私保护机制

## 相关工作与启发

- **RouteLLM (2024)**：LLM 路由框架，使用 BERT 分类器或矩阵分解预测质量
- **Hybrid LLM**：允许弱模型在质量差距在阈值内时成功，节省成本但有质量妥协
- **ZOOTER**：预测候选模型输出的归一化质量，基于相对质量进行路由
- **启发**：多指标质量评估 + 对比度量的思路可推广到其他生成任务（视频生成、3D 生成等）的路由

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ （首个 T2I 路由框架，问题新颖，架构设计原创性强）
- 实验充分度: ⭐⭐⭐⭐⭐ （18 种模型对、多种基线、丰富消融、含人类评估）
- 写作质量: ⭐⭐⭐⭐ （问题阐述清晰，公式推导完整，但篇幅较长）
- 价值: ⭐⭐⭐⭐⭐ （实际商业部署价值极高，成本节约显著）

<!-- RELATED:START -->

## 相关论文

- [CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models](compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)
- [Diffusion Adaptive Text Embedding for Text-to-Image Diffusion Models](../../NeurIPS2025/image_generation/diffusion_adaptive_text_embedding_for_texttoimage_diffusion.md)
- [Enhancing Reward Models for High-quality Image Generation: Beyond Text-Image Alignment](enhancing_reward_models_for_high-quality_image_generation_beyond_text-image_alig.md)
- [Preference Adaptive and Sequential Text-to-Image Generation](../../ICML2025/image_generation/preference_adaptive_and_sequential_text-to-image_generation.md)
- [PLA: Prompt Learning Attack against Text-to-Image Generative Models](pla_prompt_learning_attack_against_text-to-image_generative_models.md)

<!-- RELATED:END -->
