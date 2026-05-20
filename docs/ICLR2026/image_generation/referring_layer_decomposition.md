---
title: >-
  [论文解读] Referring Layer Decomposition
description: >-
  [ICLR 2026][图像生成][图层分解] 提出 Referring Layer Decomposition (RLD) 任务，根据用户提供的灵活提示（空间/文本/混合）从单张 RGB 图像中预测完整的 RGBA 图层，并构建了包含 111 万样本的 RefLade 数据集和自动评估协议。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "图层分解"
  - "RGBA 层"
  - "多模态引用输入"
  - "数据引擎"
  - "RefLayer"
---

# Referring Layer Decomposition

**会议**: ICLR 2026  
**arXiv**: [2602.19358](https://arxiv.org/abs/2602.19358)  
**代码**: [https://yaojie-shen.github.io/project/RLD/](https://yaojie-shen.github.io/project/RLD/)  
**领域**: 图像分解 / 图像编辑  
**关键词**: 图层分解, RGBA 层, 多模态引用输入, 数据引擎, RefLayer

## 一句话总结

提出 Referring Layer Decomposition (RLD) 任务，根据用户提供的灵活提示（空间/文本/混合）从单张 RGB 图像中预测完整的 RGBA 图层，并构建了包含 111 万样本的 RefLade 数据集和自动评估协议。

## 研究背景与动机

现代生成模型通常将图像作为整体进行处理，缺乏对单个场景元素的显式表示，使得选择性操纵、跨编辑一致性维护和语义对齐困难重重。图像图层（RGBA 格式的透明视觉单元）提供了更直观的框架，类似于 Photoshop 中的图层工作流。

现有方法的局限：
- MuLAn：数据规模有限（44K 图像），成功率仅 36%
- Text2Layer：只能分离前景/背景两层
- LayerDecomp：依赖合成监督，需要目标掩码

RLD 任务的核心创新在于支持多种用户提示（点、框、掩码、文本），实现按需提取目标 RGBA 图层。

## 方法详解

### 整体框架

RLD 包含三大组件：
1. **RefLade 数据集**：111 万图像-图层-提示三元组
2. **自动评估协议**：保持度、完成度、忠实度三轴评估
3. **RefLayer 基线模型**：基于 Stable Diffusion 3 的条件生成

### 数据引擎（6 阶段流水线）

1. **预过滤**：基于规则剔除低质量图像（保留图像 86.1% 适用率）
2. **场景理解**：集成封闭集检测、开放词汇检测和 MLLM 定位
3. **图层完成**：重建被遮挡的物体区域
4. **后处理**：细化掩码，预测 alpha 遮罩
5. **提示生成**：生成空间/文本/多模态提示
6. **后过滤**：评估 RGBA 图层的保真度、真实性和语义一致性

成功率从 MuLAn 的 36% 提升到 70%。

### 评估协议（三维度）

**保持度 $\mathcal{S}_{\text{vis}}$**：原始可见区域的 LPIPS 相似度

$$\mathcal{S}_{\text{vis}} = \mathbb{E}_{(p,g)\sim\mathcal{D}}[\text{LPIPS}(g_{\text{rgb}} \odot g_v, p_{\text{rgb}} \odot g_v)]$$

**完成度 $\mathcal{S}_{\text{gen}}$**：基于 CLIP 特征的方向相似度

$$\mathcal{S}_{\text{gen}} = \mathbb{E}[\cos(f(g_{\text{rgb}}) - f(g_{\text{rgb}} \odot g_v), f(p_{\text{rgb}}) - f(g_{\text{rgb}} \odot g_v))]$$

**忠实度 $\mathcal{S}_{\text{fid}}$**：将预测图层 alpha 混合到背景后计算 FID

**HPA 综合分数**：基于人类偏好 Elo 排名的归一化加权平均，与人类判断强相关。

### RefLayer 模型

- 基于 Stable Diffusion 3 构建
- VAE 编码器编码原始图像和位置提示
- 轻量卷积层压缩通道
- 双解码器：标准 RGB 解码器 + 自定义 alpha 解码器

**提示编码策略**：将所有空间提示统一为彩色 RGB 图像格式：
- 蓝色画布 → 背景
- 绿色区域 → 边界框
- 红色区域 → 掩码
- 高斯热图 → 点

## 实验

### 数据集统计

| 数据集 | 任务 | #图像 | #类别 | #实例 | 遮挡率 |
|--------|------|-------|-------|-------|--------|
| MuLAn | LD | 44,860 | 759 | 101,269 | 7.7% |
| **RefLade** | **RLD** | **430,488** | **12K** | **871,829** | **60.8%** |

### 评估协议验证

HPA 分数与人类 ELO 排名强相关，而单独的 $\mathcal{S}_{\text{vis}}$、$\mathcal{S}_{\text{fid}}$、$\mathcal{S}_{\text{gen}}$ 均无法一致地反映人类偏好。

### 质量评估

- 74.7% 的前景图层和 70.2% 的背景图层达到质量阈值
- 人工标注历时 43 天，由 9 名专业标注员完成
- 精心筛选获得 59K 高质量图像和 110K 验证图层

### 关键发现

- 粗粒度提示（单个点）可能导致粗粒度输出，而精确提示产生准确的物体级图层
- RefLayer 展现出强零样本泛化能力
- 多粒度提示系统支持从粗到细的灵活控制

## 亮点

- 首次定义了基于多模态引用输入的图层分解任务
- 数据引擎设计系统全面，将成功率从 36% 提升到 70%
- 评估协议与人类偏好高度对齐，解决了评估瓶颈
- RefLade 数据集规模远超现有同类（430K vs MuLAn 的 44K）

## 局限性

- 数据引擎依赖多个外部模型（检测/分割/补全），级联错误不可避免
- 人工标注成本高（43 天 × 9 人）
- 评估协议中 Ground Truth 图层本身可能不完美

## 相关工作

- **图像理解与编辑**：检测、分割、修复、alpha matting 等
- **组合图像表示**：MuLAn、Text2Layer 等 RGBA 图层方法
- **参考表达分割**：SAM 等可提示分割方法仅输出掩码，不重建遮挡内容

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 任务定义新颖，填补了研究空白
- 数据贡献：⭐⭐⭐⭐⭐ — 百万级数据集 + 数据引擎 + 人工标注
- 评估：⭐⭐⭐⭐ — 三维度评估协议对齐人类偏好
- 实用性：⭐⭐⭐⭐ — 对图像编辑和合成有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition](../../CVPR2026/image_generation/from_inpainting_to_layer_decomposition_repurposing_generative_inpainting_models_.md)
- [\[CVPR 2025\] Generative Image Layer Decomposition with Visual Effects](../../CVPR2025/image_generation/generative_image_layer_decomposition_with_visual_effects.md)
- [\[ICLR 2026\] Generalization of Diffusion Models Arises with a Balanced Representation Space](generalization_of_diffusion_models_arises_with_a_balanced_representation_space.md)
- [\[CVPR 2026\] Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](../../CVPR2026/image_generation/pluggable_pruning_with_contiguous_layer_distillation_for_diffusion_transformers.md)
- [\[ICLR 2026\] Generate Any Scene: Scene Graph Driven Data Synthesis for Visual Generation Training](generate_any_scene_scene_graph_driven_data_synthesis_for_visual_generation_train.md)

</div>

<!-- RELATED:END -->
