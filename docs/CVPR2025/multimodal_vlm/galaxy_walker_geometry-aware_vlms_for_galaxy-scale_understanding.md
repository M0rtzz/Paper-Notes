---
title: >-
  [论文解读] Galaxy Walker: Geometry-aware VLMs For Galaxy-scale Understanding
description: >-
  [CVPR 2025][多模态VLM][视觉语言模型] 提出 Galaxy-Walker，首个几何感知的视觉语言模型框架，通过在欧几里得、球面和双曲三种空间上进行随机游走生成几何提示（Geometry Prompt），配合混合几何专家适配器（Geometry Adapter）…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "视觉语言模型"
  - "非欧几何"
  - "天文学"
  - "混合专家"
  - "黎曼流形"
---

# Galaxy Walker: Geometry-aware VLMs For Galaxy-scale Understanding

**会议**: CVPR 2025  
**arXiv**: [2503.18578](https://arxiv.org/abs/2503.18578)  
**代码**: 无  
**领域**: 物理  
**关键词**: 视觉语言模型, 非欧几何, 天文学, 混合专家, 黎曼流形

## 一句话总结
提出 Galaxy-Walker，首个几何感知的视觉语言模型框架，通过在欧几里得、球面和双曲三种空间上进行随机游走生成几何提示（Geometry Prompt），配合混合几何专家适配器（Geometry Adapter），在星系属性估计（$R^2$ 最高达 0.91）和形态分类任务（F1 提升 +0.17）上大幅超越通用 VLM 和领域专用模型。

## 研究背景与动机

**领域现状**：现代 VLM（GPT-4o、Claude 3.5 等）在视觉问答上表现优异，但其核心架构（patch embedding、卷积骨干、自注意力机制）完全构建在欧几里得空间中。天文学 ML 已从传统监督学习演进到 AstroCLIP 等跨模态模型。

**现有痛点**：将 VLM 应用于天文学分析时性能严重下降——GPT-4o 等在星系属性估计中 $R^2 < 0.6$，形态分类中 F1 仅 0.4-0.7。这是因为宇宙的几何结构天然包含非欧几何：行星轨道涉及球面空间，黑洞涉及双曲空间，而现有 VLM 无法表达这些几何特性。

**核心矛盾**：宇宙在不同尺度上展现出丰富的几何多样性——局部是平坦的欧几里得空间，星系层级关系适合双曲空间表示，全局相似性适合球面空间——但 VLM 的 patch 嵌入和 FFN 层都假定平面距离，忽略了球面/双曲距离关系。

**本文目标**：设计一种几何感知的 VLM 框架，能同时处理欧几里得、球面和双曲空间中的天文特征。

**切入角度**：在两个层面引入几何感知：(1) 输入层——通过在多类几何空间的物理图上进行随机游走生成几何 token 作为提示；(2) 特征层——用混合几何专家替换标准 FFN，根据 token 特性自适应路由到不同几何空间的专家。

**核心 idea**：在 VLM 的输入端注入多空间几何先验（通过 Riemannian 图上的随机游走），在特征处理端用欧/球/双曲三种 FFN 专家处理不同几何特性的空间各向异性。

## 方法详解

### 整体框架
Galaxy-Walker 基于预训练 VLM 构建，包含两个核心组件：(1) Geometry Prompt 模块——从星系的物理坐标（赤经/赤纬）出发，分别在欧几里得、球面和双曲空间建图并通过 Riemannian GraphSAGE 学习几何特征 token；(2) Geometry Adapter 模块——在 VLM 的 transformer 块中每隔 $k$ 层插入混合几何专家 FFN，配合门控网络路由 token。输出端设置 Numeric Head（回归）和 LM Head（分类）两个并行头。

### 关键设计

1. **几何提示（Geometry Prompt）**:

    - 功能：将多空间几何先验注入 VLM 的输入层
    - 核心思路：从星系的物理坐标出发，通过投影和指数映射 $\mathbf{V}_\mathbb{M} = exp_o^c(proj(\mathbf{V}_{phy}))$ 得到三种空间的坐标。在每种空间中用 KNN 建图，然后通过两层 Riemannian GraphSAGE 学习几何特征：第一层 $\mathcal{F}_{\mathbb{E} \to \mathbb{M}}$ 将欧几里得特征映射到目标流形，第二层 $\mathcal{F}_{\mathbb{M} \to \mathbb{M}}$ 在流形上做消息传递。最终生成三组几何 token $\mathbf{P}_\mathbb{E}, \mathbf{P}_\mathbb{H}, \mathbf{P}_\mathbb{S}$ 作为视觉提示
    - 设计动机：欧几里得空间捕获星系的局部邻近关系，球面空间捕获全局拓扑相似性，双曲空间捕获层级演化关系——三者互补才能完整描述宇宙几何

2. **几何适配器（Geometry Adapter）**:

    - 功能：在特征处理层面适配不同几何空间的各向异性
    - 核心思路：设计三种 FFN 专家：(a) 欧几里得专家 $\mathcal{F}_E$ 使用标准 FFN；(b) 球面专家 $\mathcal{F}_S$ 对输出做归一化并乘以可学习曲率 $\kappa$，保证输出在单位球面上；(c) 双曲专家 $\mathcal{F}_H$ 在 Poincaré 球的对数/指数映射下做变换。通过门控网络 $G$ 自适应路由：$y = \sum_{i \in \{E,S,H\}} G_i(x) \cdot \mathcal{F}_i(x)$
    - 设计动机：不同天文特征对应不同几何性质——行星轨道角度关系适合球面处理，黑洞附近的引力层级适合双曲处理，常规图像特征用欧几里得处理

3. **两阶段训练**:

    - 功能：高效训练几何感知模块
    - 核心思路：Stage 1 独立训练 Geometry Prompt 模块（用星系属性估计任务学习几何表示）；Stage 2 冻结注意力块，仅训练 Geometry Adapter FFN 层、投影层和 Numeric Head
    - 设计动机：分阶段训练降低了优化难度，保留了预训练 VLM 的语言建模能力

### 损失函数 / 训练策略
总损失 $\mathcal{L} = \mathcal{L}_{LM} + \lambda \mathcal{L}_{reg}$，$\mathcal{L}_{LM}$ 为语言建模损失，$\mathcal{L}_{reg}$ 为回归任务的 Smooth L1 损失。使用 fp32 精度处理模态输入，L2 归一化后应用可学习缩放因子。训练数据包含 84K 星系的属性估计和 ~200K 的形态分类样本。

## 实验关键数据

### 主实验

| 方法 | 星系属性估计 $R^2$ | 形态分类 F1 |
|------|-------------------|------------|
| GPT-4o | < 0.6 | 0.4-0.7 |
| Claude 3.5 | < 0.6 | 0.4-0.7 |
| AstroCLIP | 中等 | 中等 |
| Galaxy-Walker | **0.52-0.91** | **+0.17 F1** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅欧几里得专家 | 较低 | 缺失非欧几何信息 |
| 仅球面+双曲 | 提升 | 非欧几何有独立贡献 |
| 全部三种专家 | 最优 | 三种空间互补 |
| 无 Geometry Prompt | 下降 | 几何先验注入是关键 |
| 无门控路由 | 下降 | 均匀混合不如自适应路由 |

### 关键发现
- 通用 VLM 在天文任务上性能极差（$R^2 < 0.6$），说明几何感知是必要的
- 三种几何空间各有侧重：双曲空间对层级结构（如 BAR 特征）提升最大，球面空间对全局形态识别贡献大
- Galaxy-Walker 在挑战性特征（如 BAR、SAC）上 F1 提升达 +0.17，显示非欧几何建模的价值
- 仅训练 Adapter 参数（冻结注意力块），计算效率高且不破坏预训练表示

## 亮点与洞察
- **将黎曼几何引入 VLM 架构**：不仅是天文学的应用创新，更提供了一种在 VLM 中处理非欧几何数据的通用范式。医学影像（球面表面）、社交网络（层级结构）等领域也可能受益
- **Riemannian GraphSAGE 做几何提示**：在图神经网络上做跨流形的消息传递（$\mathcal{F}_{\mathbb{E} \to \mathbb{M}}$），是几何深度学习和 VLM 的优雅结合
- **MoE 架构的新应用**：传统 MoE 按语义/任务路由，本文按几何空间路由——将 MoE 的天然多样性机制映射到物理空间的多样性

## 局限与展望
- 实验仅在天文学领域验证，框架在其他需要非欧几何的领域（医学球面、分子图结构）的效果未知
- 几何空间限定为三种（欧/球/双曲），未探索更一般的黎曼流形或混合曲率空间
- 门控网络的路由决策缺乏可解释性——不清楚哪些天文特征被路由到了哪种几何专家
- 训练数据规模（84K 星系）在天文学标准下较小，更大规模训练可能带来进一步提升

## 相关工作与启发
- **vs AstroCLIP**: AstroCLIP 做跨模态星系特征交互但局限于欧几里得空间。Galaxy-Walker 扩展到多几何空间，性能显著超越
- **vs GeoCode/GeoGPT4V**: 它们通过数据增强增强 VLM 的几何感知，但仍在欧几里得框架内。Galaxy-Walker 在架构层面引入非欧几何
- **vs 混合曲率空间学习 (κ-GCN)**: κ-GCN 在图上做混合曲率学习，Galaxy-Walker 将类似思想引入 VLM 的 FFN 层，联合视觉-语言-几何三模态

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在 VLM 中系统引入非欧几何建模，开辟了全新方向
- 实验充分度: ⭐⭐⭐ 天文学实验充分，但仅限一个领域，缺少跨领域验证
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述详细，图表直观
- 价值: ⭐⭐⭐⭐ 对天文学 AI 有直接应用价值，对 VLM 几何建模有重要启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding](video-xl_extra-long_vision_language_model_for_hour-scale_video_understanding.md)
- [\[CVPR 2025\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)
- [\[ICCV 2025\] HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?](../../ICCV2025/multimodal_vlm/hrscene_how_far_are_vlms_from_effective_high-resolution_image_understanding.md)
- [\[CVPR 2025\] MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders](move-kd_knowledge_distillation_for_vlms_with_mixture_of_visual_encoders.md)
- [\[CVPR 2025\] Global-Local Tree Search in VLMs for 3D Indoor Scene Generation](global-local_tree_search_in_vlms_for_3d_indoor_scene_generation.md)

</div>

<!-- RELATED:END -->
