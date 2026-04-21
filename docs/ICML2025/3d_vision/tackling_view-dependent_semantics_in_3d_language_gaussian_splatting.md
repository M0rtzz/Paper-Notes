---
title: >-
  [论文解读] LaGa: Tackling View-Dependent Semantics in 3D Language Gaussian Splatting
description: >-
  [ICML 2025][3D视觉][3D Gaussian Splatting] 提出LaGa方法，通过3D场景分解建立跨视角语义连接、用自适应聚类+双因子重加权构建视角聚合语义表示，解决3D语言高斯中被忽视的视角依赖语义问题，在LERF-OVS上3D mIoU达64.0%（+18.7%）。
tags:
  - ICML 2025
  - 3D视觉
  - 3D Gaussian Splatting
  - 图像分割
  - view-dependent semantics
  - scene decomposition
  - CLIP
---

# LaGa: Tackling View-Dependent Semantics in 3D Language Gaussian Splatting

**会议**: ICML 2025  
**arXiv**: [2505.24746](https://arxiv.org/abs/2505.24746)  
**代码**: [GitHub](https://github.com/SJTU-DeepVisionLab/LaGa)  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, open-vocabulary segmentation, view-dependent semantics, scene decomposition, CLIP

## 一句话总结

提出LaGa方法，通过3D场景分解建立跨视角语义连接、用自适应聚类+双因子重加权构建视角聚合语义表示，解决3D语言高斯中被忽视的视角依赖语义问题，在LERF-OVS上3D mIoU达64.0%（+18.7%）。

## 研究背景与动机

**领域现状**：3D-GS的开放词汇场景理解主流做法是将CLIP等VLM的2D语义特征通过可微光栅化投影到3D高斯上。训练时优化3D特征使渲染的特征图与2D语义对齐，推理时渲染特征图做pixel-wise理解。

**现有痛点**：这些方法在2D渲染feature map上表现不错，但直接用3D特征做3D感知（如检索3D高斯）时性能急剧下降。原因在于它们忽视了一个根本问题：同一个3D物体从不同视角观察会呈现不同语义——作者称之为"视角依赖语义"（view-dependent semantics）。例如一本护照正面可以识别标题，但从背面或侧面看就完全不可辨认。

**核心矛盾**：简单将2D语义投影到3D高斯会导致每个高斯只继承了特定视角的语义，产生假阳性（噪声高斯被错误检索）和假阴性（目标高斯被遗漏），2D与3D理解之间存在根本性gap。

**本文目标**：如何在3D-GS框架下构建能够保留多视角语义信息的3D表示，实现直接3D场景理解而非依赖2D渲染。

**切入角度**：作者先做了两个定量分析：（1）语义相似度分布分析——同一物体不同视角的语义特征之间的intra-object相似度甚至低于不同物体之间的inter-object相似度；（2）语义检索完整性分析——约50%的2D语义特征无法完整检索对应的3D物体。这两个分析提供了view-dependent semantics存在的强证据。

**核心 idea**：通过3D场景分解将多视角2D mask聚合为3D物体来建立跨视角语义连接，再用自适应聚类提取代表性语义描述符并加权聚合，保留视角依赖的关键信息。

## 方法详解

### 整体框架

LaGa的pipeline分三步：（1）数据准备：用SAM提取2D mask，用CLIP提取对应语义特征；（2）3D场景分解：通过对比学习训练高斯亲和特征，用HDBSCAN聚类将多视角2D mask聚合为连贯的3D物体，建立跨视角语义连接；（3）视角聚合语义表示：对每个3D物体的多视角语义做K-means聚类提取代表性描述符，再用双因子加权策略抑制噪声。推理时直接在3D空间根据文本查询检索物体，无需渲染2D特征图。

### 关键设计

1. **对比学习驱动的3D场景分解**:

    - 功能：将多视角2D mask聚合为语义无关但结构连贯的3D物体，建立跨视角连接
    - 核心思路：为每个3D高斯训练一个亲和特征 $\mathbf{f}_\mathbf{g} \in \mathbb{R}^{C'}$，渲染到2D后通过masked average pooling得到每个mask的原型 $\hat{\mathbf{f}}_\mathbf{M}$。训练目标是同一mask内的特征聚拢、mask外的特征远离（对比损失）。训练后用HDBSCAN聚类所有mask原型，将同一3D物体的多视角mask归为一组 $\mathcal{S}_i$，再通过原型与高斯亲和特征的相似度将高斯分配到物体
    - 设计动机：2D分割主要捕获物体边界，跨视角稳定，不受高层语义变化影响。这使得场景分解可以不被视角依赖语义干扰，从而建立可靠的跨视角连接——低层分割稳定但高层语义不稳定

2. **自适应跨视角描述符提取**:

    - 功能：为每个3D物体提取能有效概括多视角语义变化的代表性描述符集合
    - 核心思路：对每个物体 $\mathcal{G}^{\mathcal{S}_i}$ 的多视角语义特征集 $\mathcal{V}^{\mathcal{S}_i}$ 做K-means聚类，聚类中心即为语义描述符。描述符个数 $N^{\mathcal{G}^{\mathcal{S}_i}}$ 通过轮廓系数（silhouette score）自适应确定——语义复杂度高的物体分配更多描述符
    - 设计动机：不同物体的语义复杂度差异很大（简单物体如墙壁 vs 复杂物体如有文字的护照），固定数量的描述符无法适配。自适应聚类确保每个物体都能用最精炼的方式保留其多视角语义变化

3. **双因子加权描述符聚合**:

    - 功能：推理时计算物体对文本查询的匹配度，同时抑制噪声描述符
    - 核心思路：给定文本查询 $\mathbf{q}$，物体级别的得分为 $\text{REL}(\mathcal{G}^{\mathcal{S}_i}, \mathbf{q}) = \max_\mathbf{d} \omega^\mathbf{d} \cdot \text{Rel}(\mathbf{d}, \mathbf{q})$。权重 $\omega^\mathbf{d}$ 由两个因子的乘积决定：（i）方向一致性——描述符与全局平均特征的余弦相似度，偏离主流语义的异常描述符（如书脊看起来像"刀"）被抑制；（ii）内部紧凑性——聚类中心的L2范数，语义一致的聚类中心范数大（成员方向一致平均后不抵消），不一致的范数小
    - 设计动机：不是所有视角的语义都同等可靠。方向一致性确保主流语义被优先关注，内部紧凑性确保只有高确信度的描述符获得高权重，两者互补地抑制噪声

### 损失函数

场景分解阶段使用对比损失：$\mathcal{L} = \sum_\mathbf{I} \sum_\mathbf{M} \sum_\mathbf{p} (1-2\mathbf{M}(\mathbf{p})) \max(\langle \hat{\mathbf{f}}_\mathbf{M}, \mathbf{F}_\mathbf{I}(\mathbf{p}) \rangle, 0)$。语义表示构建阶段无需额外训练，只是后处理。

## 实验关键数据

### 主实验：LERF-OVS数据集（3D mIoU %）

| 方法 | 类型 | Figurines | Teatime | Ramen | Waldo Kitchen | 均值 |
|------|------|-----------|---------|-------|---------------|------|
| SAGA | 3D | 36.2 | 19.3 | 53.1 | 14.4 | 30.7 |
| LangSplat | 3D | 25.9 | 35.6 | 29.3 | 33.5 | 31.1 |
| OpenGaussian | 3D | 61.1 | 59.1 | 29.2 | 31.9 | 45.3 |
| SuperGSeg* | 3D | 43.7 | 55.3 | 18.1 | 26.7 | 35.9 |
| **LaGa** | **3D** | **64.1** | **70.9** | **55.6** | **65.6** | **64.0** |
| N2F2 | 2D | 47.0 | 69.2 | 56.6 | 47.9 | 54.4 |
| OccamLGS* | 2D | 58.6 | 70.2 | 51.0 | 65.3 | 61.3 |

### 跨数据集验证

| 数据集 | 指标 | OpenGaussian | LaGa |
|--------|------|-------------|------|
| 3D-OVS | mIoU | — | 95.3 |
| ScanNet (19类) | mIoU / mAcc | 24.7 / 41.5 | **32.5 / 49.1** |
| ScanNet (15类) | mIoU / mAcc | 30.1 / 48.3 | **35.5 / 53.5** |
| ScanNet (10类) | mIoU / mAcc | 38.3 / 55.2 | **42.6 / 63.2** |

### 关键发现

- LaGa在3D理解上超越所有3D方法+18.7% mIoU，甚至超越最优2D方法（OccamLGS 61.3 → LaGa 64.0），首次让3D方法在此基准超越2D方法
- Waldo Kitchen场景提升最大（31.9 → 65.6），因物体多、视角变化大，OpenGaussian的刚性特征分配严重失效
- 3D-OVS上提升有限（95.3%），因前向场景、物体少，view-dependent semantics问题不显著
- ScanNet上也优于OpenGaussian，验证了大规模场景的泛化性

## 亮点与洞察

- "视角依赖语义"问题的发现和定量验证是核心贡献——用两个分析实验提供了强证据，让论文在problem formulation层面就很有说服力
- 方法本身是后处理（场景分解+聚类+加权），不修改3D-GS训练或推理pipeline，简洁且实用
- 对比学习做场景分解不受view-dependent semantics影响的洞察——低层分割跨视角稳定而高层语义不稳定，这种分层思维可迁移到其他3D理解任务

## 局限性

- 场景分解依赖SAM的2D mask质量，极端遮挡或透明物体可能失败
- 聚类步骤是离线后处理，不支持新视角或新物体的在线更新
- 外观极度相似的不同物体（如一排相同杯子）可能被错误分组
- 大规模户外场景和动态场景的效果未验证

## 相关工作与启发

- **vs OpenGaussian**: 核心区别在于OpenGaussian用规则选择代表视角为每个高斯分配单一CLIP特征，忽略了多视角信息。LaGa用聚类+加权保留多视角语义
- **vs LangSplat/N2F2**: 依赖渲染2D特征图，2D好但3D差。LaGa直接在3D空间操作，首次让3D理解超越2D方法
- **启发**：view-dependent semantics问题在NeRF和点云等其他3D表示中同样存在，聚类+重加权策略具有通用性

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 视角依赖语义问题的发现和分析是原创贡献
- 实验充分度: ⭐⭐⭐⭐ 三个数据集验证+定量分析
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机→定量分析→方法→实验的逻辑链清晰
- 价值: ⭐⭐⭐⭐⭐ +18.7% mIoU证明了view-dependent semantics是3D理解的关键瓶颈

<!-- RELATED:START -->

## 相关论文

- [CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting](../../ICCV2025/3d_vision/clip-gs_unifying_vision-language_representation_with_3d_gaussian_splatting.md)
- [EnvGS: Modeling View-Dependent Appearance with Environment Gaussian](../../CVPR2025/3d_vision/envgs_modeling_view-dependent_appearance_with_environment_gaussian.md)
- [Online Language Splatting](../../ICCV2025/3d_vision/online_language_splatting.md)
- [ReferSplat: Referring Segmentation in 3D Gaussian Splatting](refersplat_referring_segmentation_in_3d_gaussian_splatting.md)
- [ExtrinSplat: Decoupling Geometry and Semantics for Open-Vocabulary Understanding in 3D Gaussian Splatting](../../CVPR2026/3d_vision/extrinsplat_decoupling_geometry_and_semantics_for_open-vocabulary_understanding_.md)

<!-- RELATED:END -->
