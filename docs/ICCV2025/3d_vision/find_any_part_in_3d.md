---
title: >-
  [论文解读] Find Any Part in 3D
description: >-
  [ICCV 2025][3D视觉][3D部件分割] 提出Find3D，构建了一个由2D基础模型（SAM + Gemini）驱动的自动化3D数据标注引擎，生成210万个部件标注，训练出首个同时具备开放世界、跨类别、部件级和前馈推理能力的3D分割模型，零样本mIoU提升260%，推理速度比现有方法快6-300倍。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D部件分割
  - 开放世界
  - 数据引擎
  - 对比学习
  - 零样本泛化
---

# Find Any Part in 3D

**会议**: ICCV 2025  
**arXiv**: [2411.13550](https://arxiv.org/abs/2411.13550)  
**代码**: [项目主页](https://ziqi-ma.github.io/find3dsite/)  
**领域**: 3D视觉 / 3D部件分割  
**关键词**: 3D部件分割, 开放世界, 数据引擎, 对比学习, 零样本泛化

## 一句话总结

提出Find3D，构建了一个由2D基础模型（SAM + Gemini）驱动的自动化3D数据标注引擎，生成210万个部件标注，训练出首个同时具备开放世界、跨类别、部件级和前馈推理能力的3D分割模型，零样本mIoU提升260%，推理速度比现有方法快6-300倍。

## 研究背景与动机

- **3D基础模型为何还未出现**？关键瓶颈是数据稀缺
- **现有3D部件分割数据集的局限**：
    - ShapeNet-Part：仅16个类别、41种部件类型，所有椅子朝向相同
    - PartNet-E：45个类别但仅40种唯一部件类型，限于简单家居物体
    - 合计仅71种唯一部件类型
- **现有方法的问题**：
    - 2D聚合方法（PointCLIP、PartSLIP++）：缺乏3D几何信息，跨视角不一致，推理慢
    - 闭集方法（PointNeXt）：无法泛化到未见类别
    - 测试时优化方法（LERF、Feature3DGS）：需要逐场景优化，耗时数分钟
    - 蒸馏方法（PartDistill）：逐类别蒸馏，无法零样本推理
- **核心思路**：当解决数据挑战时，用简单通用的训练策略就能获得强大的模型

## 方法详解

### 整体框架

Find3D由两部分组成：
1. **可扩展数据引擎**：自动标注互联网上的3D资产
2. **开放世界3D部件模型**：基于对比学习的点云Transformer

### 数据引擎

自动标注Objaverse 3D资产的流程：

1. **多视角渲染**：对每个3D资产从不同相机角度渲染多视图
2. **最佳朝向选择**：用Gemini从10个朝向中选择最自然的视角
3. **SAM分割**：使用网格点提示对每个渲染图进行分割
4. **过滤**：丢弃过小（<350像素）、过大（>20%像素）或低置信度的mask
5. **Gemini标注**：将每个mask叠加到原图上，请Gemini命名对应部件
6. **合并**：相同标签的mask合并
7. **3D反投影**：基于投影几何，将mask映射到点云中的3D点
8. **文本嵌入**：用SigLIP嵌入标签文本，作为监督信号

**数据规模**：处理30K个Objaverse物体（761个类别），生成210万部件标注，包含124,615种唯一部件类型——是现有数据集的**1775倍**。

### 开放世界3D部件模型

**架构**：采用PT3（Point Transformer V3）架构，通过空间填充曲线将点云体素化和序列化。在Transformer最后一层添加4层MLP，将点特征对齐到SigLIP的768维嵌入空间。模型共46.2M参数。

**对比学习训练**：由于同一点可能有多个标签（位置、材质、功能等），且许多点未标注，故采用对比学习：

$$l_i = -\log \frac{\exp(f(C_i) \cdot T(\text{label}_i))}{\sum_{j=1}^{|\mathcal{B}|} \exp(f(C_i) \cdot T(\text{label}_j))}$$

其中 $f(C_i)$ 是标签对应点集的平均特征（池化去噪），$T(\text{label}_i)$ 是SigLIP文本嵌入。每个batch 64个物体，约3000个正样本对。

**推理方式**：对任意文本查询s，计算其SigLIP嵌入与每个点特征的余弦相似度，分配给相似度最高的查询。

**数据增强**：随机旋转（三轴）、缩放、翻转、抖动、色彩增强等，避免对姿态和颜色的过度依赖。

## 实验关键数据

### 主实验：开放世界方法对比（Objaverse-General + ShapeNet-Part）

| 方法 | Obj-Gen Seen mIoU | Obj-Gen Unseen mIoU | ShapeNet-Part Canonical mIoU | ShapeNet-Part Rotated mIoU |
|------|-------------------|---------------------|------------------------------|---------------------------|
| **Find3D** | **34.10** | **27.41** | **28.39** | **29.64** |
| PointCLIPV2 | 11.27 | 11.09 | 20.22 | 18.19 |
| PartSLIP++ | 15.03 | 10.43 | 6.46 | 6.03 |
| OpenMask3D | 11.93 | 10.31 | 10.37 | 14.56 |

关键发现：在零样本未见类别上，Find3D的mIoU提升**260%**（27.41 vs 11.09）。即使零样本评估，也超过了在ShapeNet-Part上训练的PointCLIPV2。

### 推理速度与模型属性对比

| 方法 | 时间 | 开放世界 | 跨类别 | 部件级 | 前馈推理 |
|------|------|---------|--------|--------|---------|
| **Find3D** | **0.9s** | ✓ | ✓ | ✓ | ✓ |
| PointCLIPV2 | 5.4s | ✓ | ✓ | ✓ | ✗ |
| PartSLIP++ | 174.3s | ✓ | ✗ | ✓ | ✗ |
| OpenMask3D | 296.5s | ✓ | ✓ | ✗ | ✗ |
| PointNeXt | 1.4s | ✗ | ✓ | ✓ | ✓ |

Find3D是唯一同时满足所有四个属性的方法，速度比其他开放世界方法快**6-300倍**。

### 泛化能力对比（闭集方法 ShapeNetPart-V2）

| 方法 | 训练数据 | ShapeNet-Part mIoU | ShapeNetPart-V2 mIoU |
|------|----------|-------------------|---------------------|
| PointNeXt | ShapeNet-Part | 80.44 | 28.70 (↓64%) |
| PartDistill | ShapeNet-Part | 63.9 | N/A |
| **Find3D** | 数据引擎 | 28.39 (零样本) | **42.15** |

关键发现：PointNeXt在域外数据上性能暴跌64%，Find3D的零样本性能反超其1.5倍。

### 数据规模分析

训练类别数与零样本mIoU呈清晰的正相关缩放趋势：
- 16类（ShapeNet规模）→ ~14% mIoU
- 45类（PartNet-E规模）→ ~18% mIoU
- 761类（Find3D完整数据）→ ~27% mIoU

### 鲁棒性对比

| 变化条件 | PointCLIPV2性能变化 | Find3D性能变化 |
|----------|-------------------|---------------|
| 改变查询提示 | ↓64% | ↓1% |
| 随机旋转物体 | ↓46% | ↑3% |
| 更换域(ShapeNetPart-V2) | ↓56% | ↑20% |

## 亮点与洞察

1. **数据引擎范式**：核心贡献不是模型架构创新，而是构建了从2D基础模型到3D标注的自动化pipeline，使数据规模提升1775倍
2. **规模即泛化**：通过数据缩放分析证明，泛化能力直接来源于训练数据的多样性，呼应了NLP/CV领域的缩放定律
3. **简单但有效的训练策略**：仅用对比学习+数据增强，无需逐类别微调、多pass推理或预定义part排序逻辑
4. **灵活的查询能力**：支持不同粒度（"limbs" vs "arms"+"legs"）和不同维度（身体部位 vs 衣物）的文本查询

## 局限性

- 点云体素采样分辨率（0.02）限制了对不突出的细粒度部件（如表面上的按钮）的识别
- 旋转等变性训练导致对称部件倾向于预测相同标签
- 仅依赖点云模态，缺少2D图像模态的细节信息
- 数据引擎依赖SAM和Gemini的质量，标注错误会传播到下游

## 相关工作与启发

- **与"scaling law"思想的呼应**：类似于NLP中GPT系列的成功，本文在3D领域验证了"数据规模→泛化能力"这一路径
- **数据引擎的可复制性**：使用现成的SAM+Gemini组合，pipeline清晰可复制，为3D其他任务提供了方法论参考
- **闭集→开放世界**的过渡经验：不应针对小数据集过度定制，而应追求数据多样性和训练通用性

## 评分 ⭐⭐⭐⭐⭐

非常出色的工作。数据引擎的构建思路具有广泛的启发意义，实验设计全面且有说服力。260%的mIoU提升和300倍的速度提升令人印象深刻。缩放分析进一步验证了数据引擎的价值。

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] From One to More: Contextual Part Latents for 3D Generation](from_one_to_more_contextual_part_latents_for_3d_generation.md)
- [\[ICCV 2025\] SAS: Segment Any 3D Scene with Integrated 2D Priors](sas_segment_any_3d_scene_with_integrated_2d_priors.md)
- [\[ICCV 2025\] WildSeg3D: Segment Any 3D Objects in the Wild from 2D Images](wildseg3d_segment_any_3d_objects_in_the_wild_from_2d_images.md)
- [\[CVPR 2025\] Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera](../../CVPR2025/3d_vision/depth_any_camera_zero-shot_metric_depth_estimation_from_any_camera.md)
- [\[ICCV 2025\] Self-supervised Learning of Hybrid Part-aware 3D Representations of 2D Gaussians and Superquadrics](self-supervised_learning_of_hybrid_part-aware_3d_representations_of_2d_gaussians.md)

<!-- RELATED:END -->
