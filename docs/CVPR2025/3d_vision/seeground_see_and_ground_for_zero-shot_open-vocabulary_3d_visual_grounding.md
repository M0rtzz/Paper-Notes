---
title: >-
  [论文解读] SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding
description: >-
  [CVPR 2025][3D视觉][3D视觉定位] 本文提出 SeeGround，一个免训练的零样本 3D 视觉定位框架，通过将 3D 场景表示为查询对齐的渲染图像和空间增强文本描述的混合形式，利用 2D 视觉语言模型实现了在 ScanRefer 上超越之前零样本方法 7.7% 的精度。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D视觉定位
  - 零样本
  - 视觉语言模型
  - 视角自适应
  - 开放词汇
---

# SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding

**会议**: CVPR 2025  
**arXiv**: [2412.04383](https://arxiv.org/abs/2412.04383)  
**代码**: https://seeground.github.io  
**领域**: 3D视觉  
**关键词**: 3D视觉定位, 零样本, 视觉语言模型, 视角自适应, 开放词汇

## 一句话总结

本文提出 SeeGround，一个免训练的零样本 3D 视觉定位框架，通过将 3D 场景表示为查询对齐的渲染图像和空间增强文本描述的混合形式，利用 2D 视觉语言模型实现了在 ScanRefer 上超越之前零样本方法 7.7% 的精度。

## 研究背景与动机

**领域现状**：3D 视觉定位（3DVG）旨在根据文本描述在 3D 场景中定位目标物体，是增强现实和机器人感知的关键技术。监督方法如 ScanRefer、BUTD-DETR 等在标注数据集上表现优异，但依赖大量 3D 标注数据。

**现有痛点**：（1）监督方法受限于预定义类别和标注数据集，无法扩展到开放词汇场景；（2）大规模 3D 场景标注成本极高，限制了可扩展性；（3）近期的零样本方法如 LLM-Grounder 和 ZSVG3D 仅用文本描述 3D 场景，忽略了颜色、纹理、状态等关键视觉信息——这些信息难以用语言精确表达。

**核心矛盾**：2D VLM 训练在大规模 2D 数据上拥有强大的开放词汇理解能力，但无法直接处理 3D 数据（点云、体素）；纯文本描述 3D 场景又丢失了丰富的视觉细节。

**本文目标**：构建一个桥梁，让 2D VLM 能"看到"并理解 3D 场景，实现零样本开放词汇 3D 视觉定位。

**切入角度**：将 3D 场景转换为 2D VLM 兼容的混合表示——查询对齐的渲染图像提供视觉线索，文本化的 3D 空间描述提供精确位置信息。

**核心 idea**：根据查询文本动态选择渲染视角（而非固定俯视图或多视角），在渲染图像上用视觉提示标注物体以建立 2D-3D 对应关系，让 VLM 同时看到视觉细节和读到空间关系来完成定位。

## 方法详解

### 整体框架

输入为 3D 场景点云和文本查询。首先用开放词汇 3D 检测器获取场景中所有物体的 3D 框和语义标签，存入 OLT（Object Lookup Table）。然后根据查询动态选择渲染视角生成 2D 图像，在图像上用视觉提示标注物体，将图像、空间描述和查询一起输入 VLM 输出目标物体 ID，最终从 OLT 检索其 3D 框。

### 关键设计

1. **视角自适应模块（Perspective Adaptation Module）**:

    - 功能：根据查询文本动态选择最合适的渲染视角
    - 核心思路：先用 VLM 解析查询 $\mathsf{Q}$，识别锚点物体 $\boldsymbol{A}$（如"花纹椅子"）和候选目标 $\mathcal{O}^{(C)}$。然后将虚拟相机初始放在场景中心，面向锚点物体，向后向上移动以获得合适的视野覆盖。用 $\text{look\_at\_view\_transform}$ 计算旋转和平移矩阵，渲染查询对齐的 2D 图像 $\mathbf{I} = \text{Render}(\mathcal{S}, \mathbf{R}_c, \mathbf{T}_c)$。
    - 设计动机：固定视角（俯视图或多视角）无法对齐查询的空间语义——"右边的窗户"需要从说话者视角而非俯视图才能正确理解。动态视角避免了冗余信息和遮挡问题。

2. **融合对齐模块（Fusion Alignment Module）**:

    - 功能：建立 2D 图像中的物体与 3D 空间描述之间的精确对应
    - 核心思路：将 OLT 中物体的 3D 框投影到渲染图像上，检测遮挡并过滤被遮挡物体。在可见物体上叠加视觉提示（编号标记），使 VLM 能明确将图像中看到的物体与文本描述中的 3D 位置关联起来。将带标记的图像、空间描述文本和查询一起输入 VLM 进行推理。
    - 设计动机：当场景有多个相似物体（如多把椅子）时，VLM 难以将文本中的"3号椅子在坐标(2.1, 3.4, 0.5)"与图像中的正确椅子对应。视觉提示标记显式建立了这种对应，是关键的"桥梁"。

3. **混合 3D 场景表示**:

    - 功能：创建同时包含视觉和空间信息的 VLM 兼容输入
    - 核心思路：将 3D 场景表示为 $(\mathbf{I}, \mathcal{T}) = \mathbf{F}(\mathcal{S}, \mathsf{Q}, \mathcal{OLT})$。文本部分 $\mathcal{T}$ 包含每个物体的 3D 框（中心、尺寸）和语义标签，提供精确的 3D 位置。图像部分 $\mathbf{I}$ 提供颜色、纹理、形状、状态等视觉线索。两者互补——文本弥补图像缺失的精确 3D 坐标，图像弥补文本无法描述的视觉细节。
    - 设计动机：纯文本方法无法区分颜色相近的物体（如"花纹椅子" vs "纯色椅子"），纯图像方法无法提供精确 3D 位置。混合表示让 VLM 两方面都能利用。

### 损失函数 / 训练策略

SeeGround 是免训练的零样本方法，不需要损失函数和训练过程。使用 GPT-4V/Claude 等闭源 VLM 或 LLaVA 等开源 VLM 作为推理引擎。物体检测只需对每个场景做一次，结果缓存在 OLT 中供所有查询复用。

## 实验关键数据

### 主实验

ScanRefer 验证集零样本 3DVG 性能（Acc@0.25）：

| 方法 | 监督 | Unique | Multiple | Overall |
|---|---|---|---|---|
| LLM-Grounder | 零样本 | 30.4 | 8.6 | 12.8 |
| ZSVG3D | 零样本 | 57.2 | 21.8 | 28.6 |
| **SeeGround** | **零样本** | **71.7** | **28.3** | **36.3** |
| WS-3DVG | 弱监督 | 53.8 | 22.5 | 28.5 |
| BUTD-DETR | 全监督 | 84.2 | 46.6 | 52.2 |

Nr3D 数据集：

| 方法 | Easy | Hard | Dep. | Indep. | Overall |
|---|---|---|---|---|---|
| ZSVG3D | 42.6 | 31.0 | 28.2 | 42.2 | 37.3 |
| **SeeGround** | **54.0** | **36.4** | **40.1** | **47.1** | **44.4** |

### 消融实验

| 配置 | ScanRefer Acc@0.25 | 说明 |
|---|---|---|
| 仅文本描述（无图像） | ~28-29 | 缺少视觉信息 |
| 固定俯视图 | ~30-32 | 遮挡和视角不对齐 |
| 多视角 | ~32-34 | 冗余且 VLM 难以整合 |
| **查询对齐视角 + 视觉提示** | **36.3** | 完整方法 |

### 关键发现

- 在 ScanRefer 上超越之前零样本 SOTA ZSVG3D 7.7%（36.3 vs 28.6），在 Nr3D 上超越 7.1%
- 零样本方法甚至超越了弱监督方法 WS-3DVG（36.3 vs 28.5）
- 查询对齐视角对含方向性描述的查询提升最大（如"左边"、"右边"）
- 即使文本输入不完整，视觉线索也能帮助正确定位——证明了多模态融合的鲁棒性

## 亮点与洞察

1. **"跨模态桥梁"设计**：将 3D 场景表示为图像+文本的混合格式，巧妙地让 2D VLM 能理解 3D 空间关系
2. **查询驱动视角选择**：不同于固定视角，根据查询文本动态调整渲染视角，这种"任务感知的观察策略"可迁移到 embodied AI 场景
3. **视觉提示的对应关系建立**：在图像上标注编号来建立 2D-3D 对应，简单有效地解决了多物体场景的歧义问题

## 局限与展望

- 依赖预训练 3D 检测器的质量——漏检或误检会直接影响最终定位
- VLM 的推理能力限制了复杂空间关系的理解（如嵌套引用）
- 渲染图像的质量受点云密度影响，稀疏点云可能生成不清晰的图像
- 未来可以引入多轮推理或 self-refinement 提升复杂查询的处理能力

## 相关工作与启发

- **vs LLM-Grounder**：LLM-Grounder 仅用 LLM 做文本推理，缺乏视觉信息；SeeGround 引入视觉模态大幅提升定位精度
- **vs ZSVG3D**：ZSVG3D 也是零样本但主要依赖文本形式的 3D 描述；SeeGround 的混合表示提供了更丰富的信息
- **vs Agent3D-Zero**：Agent3D-Zero 使用多视角 VLM 做 3D 问答；SeeGround 专注于定位任务，引入查询驱动视角更精准
- 启发："让 2D 模型理解 3D"的关键在于信息表示——不需要 3D 原生输入，只需要恰当的 2D 投影和空间文本描述

## 评分

- 新颖性: 7/10 — 混合表示和动态视角的思路直观但有效，但核心依赖 VLM 能力
- 实验充分度: 8/10 — ScanRefer 和 Nr3D 两个标准 benchmark，消融全面
- 写作质量: 8/10 — 框架图清晰，例子直观展示了方法优势
- 价值: 8/10 — 在零样本 3DVG 上大幅推进 SOTA，为免训练方法开辟了可行路径

<!-- RELATED:START -->

## 相关论文

- [GREAT: Geometry-Intention Collaborative Inference for Open-Vocabulary 3D Object Affordance Grounding](great_geometry-intention_collaborative_inference_for_open-vocabulary_3d_object_a.md)
- [Text-Guided Sparse Voxel Pruning for Efficient 3D Visual Grounding](text-guided_sparse_voxel_pruning_for_efficient_3d_visual_grounding.md)
- [Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)
- [ProxyTransformation: Preshaping Point Cloud Manifold with Proxy Attention for 3D Visual Grounding](proxytransformation_preshaping_point_cloud_manifold_with_proxy_attention_for_3d_.md)
- [JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](jopp-3d_joint_open_vocabulary_semantic_segmentation_on_point_clouds_and_panorama.md)

<!-- RELATED:END -->
