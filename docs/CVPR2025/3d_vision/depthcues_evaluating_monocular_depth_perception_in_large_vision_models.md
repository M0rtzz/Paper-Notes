---
title: >-
  [论文解读] DepthCues: Evaluating Monocular Depth Perception in Large Vision Models
description: >-
  [CVPR 2025][3D视觉][单目深度感知] 提出 DepthCues 基准，通过六个人类单目深度线索任务（高度、光影、遮挡、透视、大小、纹理梯度）系统评估 20 个大规模预训练视觉模型的深度感知能力，揭示了类人深度线索在现代视觉模型中的涌现现象。
tags:
  - CVPR 2025
  - 3D视觉
  - 单目深度感知
  - 视觉基础模型
  - 深度线索
  - 基准评测
  - 表征探测
---

# DepthCues: Evaluating Monocular Depth Perception in Large Vision Models

**会议**: CVPR 2025  
**arXiv**: [2411.17385](https://arxiv.org/abs/2411.17385)  
**代码**: [https://danier97.github.io/depthcues](https://danier97.github.io/depthcues)  
**领域**: 3D视觉  
**关键词**: 单目深度感知、视觉基础模型、深度线索、基准评测、表征探测

## 一句话总结

提出 DepthCues 基准，通过六个人类单目深度线索任务（高度、光影、遮挡、透视、大小、纹理梯度）系统评估 20 个大规模预训练视觉模型的深度感知能力，揭示了类人深度线索在现代视觉模型中的涌现现象。

## 研究背景与动机

大规模预训练视觉模型（如 DINOv2、Stable Diffusion）在单目深度估计任务中展现出惊人的能力，但这些模型在预训练阶段并未接受显式深度监督。一个核心问题是：**深度感知是如何在这些模型中涌现的？** 人类视觉系统依赖多种单目深度线索（elevation、light-shadow、occlusion、perspective、size、texture gradient）来感知深度，但现有基准仅评估模型的深度估计精度，未探究模型是否理解这些底层视觉线索。本文从人类视觉科学的角度出发，构建了专门的基准来弥补这一研究空白。

## 方法详解

### 整体框架

DepthCues 基准包含六个任务，每个任务对应一个人类单目深度线索。评估协议采用**特征探测（probing）**：冻结预训练模型的特征提取器 $\phi(\cdot)$，在其上训练轻量探测头 $g_\theta$ 来解决特定任务，通过探测性能衡量模型对该线索的理解程度。

### 关键设计

1. **六类深度线索任务设计**:
    - 功能：全面覆盖人类单目深度感知的核心线索
    - 核心思路：每个任务从已有数据集改造而来，分别测试高度（地平线估计）、光影（物体-阴影关联）、遮挡（物体是否被遮挡）、透视（消失点估计）、大小（3D物体体积比较）、纹理梯度（纹理面深度排序）
    - 设计动机：将抽象的"深度感知"分解为可量化的子能力，使分析更精细

2. **任务特定特征提取**:
    - 功能：从预训练模型中提取适合不同任务的特征表示
    - 核心思路：对于物体级任务（遮挡、光影、大小、纹理），使用掩码池化提取区域特征 $\mathbf{f} = \frac{\sum_{h,w} M_A \odot \text{up}(\phi(I))}{\sum_{h,w} M_A}$；对于全局任务（高度、透视），使用完整特征图
    - 设计动机：不同线索需要不同层级的空间信息，统一特征提取方式会损失任务相关信息

3. **多层级探测与评估**:
    - 功能：公平比较不同架构和预训练设置的模型
    - 核心思路：二分类任务使用 MLP 探测头，回归任务使用注意力探测头；对每个模型搜索最优层，5 次随机种子取均值
    - 设计动机：非线性探测器比线性探测器更适合捕获复杂的深度线索表征

### 损失函数 / 训练策略

- 二分类任务（光影、遮挡、大小、纹理梯度）：二元交叉熵损失
- 回归任务（高度、透视）：均方误差损失
- 采用超参数搜索确定最优探测层，每个任务独立训练

## 实验关键数据

### 主实验

| 模型 | 平均排名 | 深度估计 NYUv2 Acc(%) | 特点 |
|------|---------|---------------------|------|
| DepthAnythingv2 | 1 | 最强 | 专用深度模型，六项线索最全面 |
| DINOv2-b14 | Top-5 | 87.78 | 自监督模型，无深度监督也涌现线索 |
| Stable Diffusion | Top-5 | - | 生成模型也具备深度理解 |
| CLIP | 最末 | 43.78 | 视觉-语言模型深度线索最弱 |

### 消融实验

| 配置 | NYUv2 Acc(%) | DIW WHDR(%) | 说明 |
|------|-------------|-------------|------|
| DINOv2 原始 | 87.78 | 11.99 | 基线 |
| DINOv2+DC 微调 | 87.06 | 11.95 | 微调后略降 |
| concat(DINOv2, DINOv2+DC) | **88.46** | **11.72** | 拼接原始+微调特征最优 |
| CLIP 原始 | 43.78 | 35.25 | 基线 |
| concat(CLIP, CLIP+DC) | **44.32** | **33.53** | 显式注入线索提升深度感知 |

### 关键发现

- **类人深度线索在更大、更新的模型中更强**：DINOv2、Stable Diffusion 等自监督/生成模型展现出强深度线索
- **多视角模型在纹理梯度线索上更优**：CroCo、DUSt3R 等多视角模型在 texture-grad 上排名前四
- **DepthCues 性能与深度估计高度相关**：Spearman 相关性显著，验证了基准的有效性
- **没有模型在所有六项线索上都最优**：各模型各有短板
- **SigLIP 远强于 CLIP**：虽然都是图文匹配训练，但 SigLIP 用了 9 倍数据，线索理解显著更好

## 亮点与洞察

- 首次系统性地将人类视觉科学中的单目深度线索理论引入大模型分析领域
- 发现在 DepthCues 上微调可以增强模型的深度感知能力，即使只有极稀疏的标注（<35K 图像），暗示了一种不依赖密集深度标注的新方向
- 人类在测试集上达到 95%±1.48% 准确率，说明任务设计合理且可被人类解决

## 局限与展望

- 仅分析了公开预训练权重，未直接研究预训练目标和数据集对线索涌现的因果影响
- 未涵盖与自运动（ego-motion）和场景运动相关的深度线索
- 纹理梯度任务使用合成数据，可能与其他线索的相关性偏低
- DepthCues 数据规模有限，微调后需拼接原始特征才能提升，泛化存在局限

## 相关工作与启发

- 与 Probe3D、GeoMeter 等 3D 感知探测工作互补，但更聚焦于单目深度线索的系统分析
- 微调实验启发：可以考虑设计更大规模的线索学习数据集，或将线索学习作为深度估计的辅助预训练任务
- 为理解 Depth Anything V2、Marigold 等模型的成功提供了新视角

## 评分

- 新颖性: ⭐⭐⭐⭐ 从人类视觉科学角度分析模型深度感知的研究视角很新颖，但探测方法本身是标准做法
- 实验充分度: ⭐⭐⭐⭐⭐ 20 个模型、6 个任务、多项相关性分析和消融实验非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，分析深入，图表设计有效
- 价值: ⭐⭐⭐⭐ 为社区提供了有价值的基准和分析工具，但实际应用价值有待进一步挖掘

<!-- RELATED:START -->

## 相关论文

- [Vision-Language Embodiment for Monocular Depth Estimation](vision-language_embodiment_for_monocular_depth_estimation.md)
- [DSPNet: Dual-vision Scene Perception for Robust 3D Question Answering](dspnet_dual-vision_scene_perception_for_robust_3d_question_answering.md)
- [Perception Tokens Enhance Visual Reasoning in Multimodal Language Models](perception_tokens_enhance_visual_reasoning_in_multimodal_language_models.md)
- [Scalable Autoregressive Monocular Depth Estimation](scalable_autoregressive_monocular_depth_estimation.md)
- [Video Depth Without Video Models](video_depth_without_video_models.md)

<!-- RELATED:END -->
