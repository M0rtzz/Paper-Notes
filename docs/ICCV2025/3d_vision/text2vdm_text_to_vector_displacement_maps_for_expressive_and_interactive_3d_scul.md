---
title: >-
  [论文解读] Text2VDM: Text to Vector Displacement Maps for Expressive and Interactive 3D Sculpting
description: >-
  [ICCV 2025][3D视觉][VDM笔刷生成] 提出Text2VDM,首个从文本生成VDM雕刻笔刷的框架,通过Sobolev预条件网格变形和语义增强SDS损失解决子对象结构生成中的语义耦合问题。
tags:
  - ICCV 2025
  - 3D视觉
  - VDM笔刷生成
  - Score Distillation
  - 语义增强
  - 网格变形
  - 3D建模
---

# Text2VDM: Text to Vector Displacement Maps for Expressive and Interactive 3D Sculpting

**会议**: ICCV 2025  
**arXiv**: [2502.20045](https://arxiv.org/abs/2502.20045)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: VDM笔刷生成, Score Distillation, 语义增强, 网格变形, 3D建模

## 一句话总结

提出Text2VDM,首个从文本生成VDM雕刻笔刷的框架,通过Sobolev预条件网格变形和语义增强SDS损失解决子对象结构生成中的语义耦合问题。

## 研究背景与动机

专业3D资产创作依赖多样化的雕刻笔刷来添加表面细节和几何结构。**向量位移贴图(VDM)**是建模软件中的标准笔刷表示,每个像素存储3D位移向量,能创建复杂的表面细节(如裂纹、木纹)或几何结构(如耳朵、犄角)。

然而现有方法无法生成VDM笔刷:

**T2I模型** — VDM不是自然图像,难以直接生成

**3D生成方法** — 生成完整对象,无法控制子对象结构

**SDS损失的语义耦合** — 生成鹿角时会附带整个鹿头,生成龟壳时会包含龟尾巴

核心问题:**SDS的训练数据大多是完整对象图像,因此文本条件下的目标分布总是倾向于完整对象语义**。

## 方法详解

### 笔刷初始化

VDM表示为512×512三通道图像,三种初始化方式:
- **零值VDM** — 平面网格,默认设置
- **尖峰VDM** — 适合凸起几何结构
- **用户指定VDM** — 自定义体积和方向

### Sobolev预条件网格变形

通过网格拉普拉斯重参数化实现内在平滑:
$$v \leftarrow v - \eta(I + \lambda L)^{-1}\frac{\partial \mathcal{L}_{SE}}{\partial v}$$

其中 $L$ 是网格拉普拉斯,$\lambda=15$ 控制梯度扩散范围。相比直接添加拉普拉斯正则化,预条件框架在大变形下保持正确拓扑,减少三角形翻转。

### 语义增强SDS损失

标准SDS: $\nabla_\theta\mathcal{L}_{SDS} = \mathbb{E}[\omega(t)(\epsilon_\phi(\mathbf{x}_t; y, t) - \epsilon)\frac{\partial \mathbf{x}}{\partial \theta}]$

**CSD的语义抑制方法不可行** — 负提示的语义同样存在耦合,相减后产生无意义分布。

**本文的语义增强方法**: 使用Compel对提示词token进行加权混合:
$$e_w = e_0 + s \cdot (e - e_0)$$

语义聚焦的文本嵌入 $\epsilon_\phi^*$ 替代原始 $\epsilon_\phi$:
$$\nabla_\theta\mathcal{L}_{SE} = \mathbb{E}[\omega(t)(\epsilon_\phi^*(\mathbf{x}_t; y, t) - \epsilon)\frac{\partial \mathbf{x}}{\partial \theta}]$$

关键优势:Compel产生的语义聚焦嵌入**独立于时间变化**,比Attend-and-Excite更稳定。

## 实验

### 定量评估 (40个文本提示)

| 方法 | Geometry CLIP Score↑ | 网格自交叉率↓ |
|------|---------------------|-------------|
| Paint-it | 0.2375 | 19.42% |
| Text2Mesh | 0.2497 | 7.18% |
| TextDeformer | 0.2477 | 0.04% |
| **Text2VDM** | **0.2556** | **0.77%** |

### 用户研究 (32参与者)

| 方法 | 几何质量偏好↑ | 文本一致性偏好↑ |
|------|-------------|---------------|
| Paint-it | 3.1% | 1.7% |
| Text2Mesh | 18.3% | 27.3% |
| TextDeformer | 3.3% | 3.4% |
| **Text2VDM** | **75.3%** | **67.6%** |

### 关键发现

1. Text2VDM在CLIP Score和用户偏好上均显著领先
2. 语义增强SDS有效解决了语义耦合:鹿角不再附带耳朵和嘴巴,龟壳不再附带头和尾巴
3. Sobolev预条件保持了良好的网格拓扑,自交叉率仅0.77%
4. 生成的VDM可直接在Blender/ZBrush中使用

## 亮点与洞察

1. **新任务定义** — 文本到VDM笔刷生成是首创,与艺术工作流高度兼容
2. **语义耦合的深层分析** — 准确定位了SDS训练数据偏差导致的问题,且证明语义抑制路线不可行
3. **Compel加权的优雅方案** — 简单有效,独立于时间变化,比AaE更稳定
4. **实际可交互** — 生成的笔刷支持实时交互建模,不需要每次编辑都重新优化

## 局限性

- 可能出现多视角不一致(SDS的通病)
- 语义增强的权重 $s=1.1^2$ 需针对不同提示调整
- 优化过程仍需数分钟

## 相关工作

- **局部3D生成**: 3D Highlighter, 3D Paintbrush, FocalDreamer, MagicClay
- **SDS改进**: CSD, VSD, Attend-and-Excite
- **几何笔刷**: VDM概念已有但无自动生成方法

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (全新任务+语义增强SDS)
- 技术深度: ⭐⭐⭐⭐ (预条件优化+语义分析深入)
- 实验充分度: ⭐⭐⭐⭐ (定量+用户研究+消融)
- 实用价值: ⭐⭐⭐⭐⭐ (直接兼容主流建模软件)

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] GenVDM: Generating Vector Displacement Maps From a Single Image](../../CVPR2025/3d_vision/genvdm_generating_vector_displacement_maps_from_a_single_image.md)
- [\[ICCV 2025\] Advancing Text-to-3D Generation with Linearized Lookahead Variational Score Distillation](advancing_text-to-3d_generation_with_linearized_lookahead_variational_score_dist.md)
- [\[ICCV 2025\] SegmentDreamer: Towards High-Fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation](segmentdreamer_towards_high-fidelity_text-to-3d_synthesis_with_segmented_consist.md)
- [\[ICCV 2025\] ExCap3D: Expressive 3D Scene Understanding via Object Captioning with Varying Detail](excap3d_expressive_3d_scene_understanding_via_object_captioning_with_varying_det.md)
- [\[ICCV 2025\] Dynamic Point Maps: A Versatile Representation for Dynamic 3D Reconstruction](dynamic_point_maps_a_versatile_representation_for_dynamic_3d_reconstruction.md)

<!-- RELATED:END -->
