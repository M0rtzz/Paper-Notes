---
title: >-
  [论文解读] Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera
description: >-
  [CVPR 2025][3D视觉][度量深度估计] 提出 Depth Any Camera (DAC) 框架，通过 ERP 统一表示、Pitch-aware 转换和 FoV 对齐等技术，实现仅用透视图像训练即可零样本泛化到鱼眼和360°相机的度量深度估计，在大视野数据集上 $\delta_1$ 精度提升高达50%。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "度量深度估计"
  - "零样本泛化"
  - "鱼眼相机"
  - "全景相机"
  - "等距矩形投影"
---

# Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera

**会议**: CVPR 2025  
**arXiv**: [2501.02464](https://arxiv.org/abs/2501.02464)  
**代码**: [https://yuliangguo.github.io/depth-any-camera](https://yuliangguo.github.io/depth-any-camera)  
**领域**: 3D视觉  
**关键词**: 度量深度估计, 零样本泛化, 鱼眼相机, 全景相机, 等距矩形投影

## 一句话总结

提出 Depth Any Camera (DAC) 框架，通过 ERP 统一表示、Pitch-aware 转换和 FoV 对齐等技术，实现仅用透视图像训练即可零样本泛化到鱼眼和360°相机的度量深度估计，在大视野数据集上 $\delta_1$ 精度提升高达50%。

## 研究背景与动机

现有度量深度基础模型（Metric3Dv2、UniDepth、ZoeDepth 等）虽在透视图像上泛化良好，但在大视野 (FoV) 相机（鱼眼、360°）上性能急剧下降。核心挑战包括：

1. **如何选择统一相机模型**来表示不同 FoV？
2. **如何利用透视训练数据**泛化到大 FoV 相机特有的高畸变区域？
3. **训练样本尺寸差异巨大**：不同 FoV 在统一空间中映射为差异极大的区域
4. **训练与测试分辨率不一致**

值得注意的是，UniDepth 尝试用网络学习球面转换来处理不同 FoV，但在大 FoV 上表现很差，说明深度学习网络在外推到训练域之外的数据空间时能力有限。

## 方法详解

### 整体框架

DAC 以等距矩形投影 (ERP) 作为统一图像表示，训练时将各种相机的透视图像转换为 ERP 补丁，测试时同样将大 FoV 图像映射到 ERP 空间进行推理。核心流程：训练图像 → Pitch-aware ERP 转换 + 数据增强 → FoV 对齐 → 多分辨率训练 → 模型推理。网络架构采用 iDisc，损失函数为 SIlog loss。

### 关键设计

1. **Pitch-Aware Image-to-ERP 转换**:
    - 功能：将透视图像高效转换为 ERP 补丁，并模拟大 FoV 相机特有的高畸变区域
    - 核心思路：利用 Gnomonic 几何的闭式映射，在 ERP 补丁的均匀网格点和输入图像坐标之间建立对应关系，通过 grid sampling 实现高效转换。关键创新在于将切面中心纬度 $\lambda_c$ 设为相机 pitch 角——当已知或可估计相机朝向时，透视数据可投影到 ERP 空间的不同纬度，模拟只有大 FoV 相机才能观察到的高畸变区域。还支持对 $\lambda_c$ 加噪声的 pitch 增强
    - 设计动机：神经网络在外推到训练数据空间之外时泛化能力有限，需要在训练时就模拟目标域的数据分布

2. **FoV 对齐**:
    - 功能：归一化不同 FoV 训练样本到统一的 ERP 补丁大小，解决尺寸差异问题
    - 核心思路：对每个输入图像应用特定的缩放增强 $s_\sigma = \text{FoV}_c / \text{FoV}_e$，使其 FoV 匹配预定义的 ERP 补丁 FoV。ERP 补丁 FoV 由 $\text{FoV}_e = H_e \pi / H_E$ 确定。这样单个预定义尺寸的 ERP 补丁可最大化包含内容信息，最小化背景填充浪费
    - 设计动机：HM3D 等数据集中相机 FoV 从36°到124°变化巨大，不做对齐会导致某些样本内容信息损失严重或计算浪费

3. **多分辨率训练**:
    - 功能：应对训练与测试分辨率不匹配问题
    - 核心思路：每个 ERP 补丁额外缩放到两个较低分辨率（原始的0.7和0.4倍），每个 batch 喂入三个不同分辨率的图像并求和损失。使模型学习尺度等变特征
    - 设计动机：大 FoV 测试图像的宽高比和分辨率可能与训练补丁差异很大，尤其在使用注意力模块时，不同数量的 image tokens 会影响性能

### 深度表示

- 使用欧几里得距离（从相机中心的距离）而非 Z-buffer 格式，因为后者在球面投影下会产生不准确的低深度值
- 深度缩放操作遵循 Metric3D 的 canonical model 范式

## 实验关键数据

### 主实验（零样本测试，大FoV数据集）

| 测试数据集 | 方法 | $\delta_1$↑ | Abs Rel↓ | 说明 |
|------|------|------|----------|------|
| Matterport3D (360°) | Metric3Dv2 | 0.429 | 0.279 | 室内670K训练 |
| | **DAC (Ours)** | **0.773** | **0.156** | **同数据训练，$\delta_1$提升80%** |
| Pano3D-GV2 (360°) | Metric3Dv2 | 0.506 | 0.261 | 室内670K训练 |
| | **DAC (Ours)** | **0.812** | **0.139** | **$\delta_1$提升60%** |
| ScanNet++ (鱼眼150°) | Metric3Dv2 | 0.649 | 0.192 | 室内670K训练 |
| | **DAC (Ours)** | **0.852** | **0.132** | **$\delta_1$提升31%** |
| KITTI360 (鱼眼180°) | Metric3Dv2 | 0.768 | 0.152 | 室外130K训练 |
| | **DAC (Ours)** | **0.786** | **0.156** | 轻微提升 |

### 消融实验（HM3D训练 → Pano3D-GV2/ScanNet++测试）

| 配置 | Pano3D $\delta_1$↑ | ScanNet++ $\delta_1$↑ | 说明 |
|------|---------|------|------|
| DAC (Full) | 0.725 | 0.654 | 完整框架 |
| w/o Pitch-Aware ERP | 0.491 | - | $\delta_1$下降32% |
| w/o Pitch Aug | 0.691 | - | 去除 pitch 增强 |
| w/o FoV Align | 0.408 | - | $\delta_1$下降44% |
| w/o Multi-Reso | 0.513 | - | $\delta_1$下降29% |

### 关键发现

- 在室内360°数据集上，DAC 比 Metric3Dv2（使用更大 backbone 和更多数据训练）$\delta_1$ 提升近 **50%**
- FoV 对齐是最关键的组件：去除后 $\delta_1$ 下降44%
- 室外 KITTI360 上提升较小，因为 LiDAR 点集中在低畸变区域，且训练数据 pitch 变化有限
- UniDepth 尽管使用网络学习球面转换，在大 FoV 上仍表现很差，证实了几何先验相比纯数据驱动的优势
- DAC 使用更轻量的 ResNet101 backbone 就超越了使用 DINOv2/ViT-L 的更大模型

## 亮点与洞察

- **训练数据不变，仅改变表示空间**: DAC 核心贡献是数据转换管线而非新网络架构，可广泛适配已有深度网络
- **Pitch-aware ERP 的域扩展效应**: 通过几何变换将透视训练数据"投射"到大 FoV 特有的高畸变区域，有效解决了外推问题
- **欧几里得距离 vs Z-buffer**: 在球面投影下使用欧几里得距离的必要性是重要的工程洞察
- **与 Metric3D 的互补**: DAC 不替换 Metric3D 的管线，而是提供更优的统一表示空间

## 局限与展望

- 室外场景提升有限，受限于训练数据 pitch 分布窄
- 未尝试与更强的 backbone (DINOv2 ViT-G) 结合
- 需要已知或可估计的相机参数（pitch 角）
- 每个 ERP 补丁的 Gnomonic 映射虽高效但仍增加数据预处理开销
- 未处理动态场景或视频深度估计

## 相关工作与启发

- Metric3D 通过透视图像的 canonical camera 归一化实现跨数据集训练，DAC 将此思路推广到 ERP 空间
- 与针对大 FoV 的特化方法（使用可变形卷积等）相比，DAC 不需要特殊网络结构
- 启发：表示空间的选择比网络架构更重要——几何先验驱动的数据增强可能是跨域泛化的关键

## 评分

- 新颖性: ⭐⭐⭐⭐ ERP 统一表示+pitch-aware 转换思路简洁有效，FoV对齐设计实用
- 实验充分度: ⭐⭐⭐⭐ 覆盖360°、鱼眼、室内外，消融清晰，但室外改进有限
- 写作质量: ⭐⭐⭐⭐ 图示丰富清晰，技术细节完整
- 价值: ⭐⭐⭐⭐⭐ 解决了深度基础模型跨相机泛化的实际痛点，具有很大实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Efficient Depth Estimation for Unstable Stereo Camera Systems on AR Glasses](efficient_depth_estimation_for_unstable_stereo_camera_systems_on_ar_glasses.md)
- [\[CVPR 2025\] UniK3D: Universal Camera Monocular 3D Estimation](unik3d_universal_camera_monocular_3d_estimation.md)
- [\[CVPR 2025\] SharpDepth: Sharpening Metric Depth Predictions Using Diffusion Distillation](sharpdepth_sharpening_metric_depth_predictions_using_diffusion_distillation.md)
- [\[CVPR 2025\] Zero-Shot Monocular Scene Flow Estimation in the Wild](zero-shot_monocular_scene_flow_estimation_in_the_wild.md)
- [\[CVPR 2025\] Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation](helvipad_a_real-world_dataset_for_omnidirectional_stereo_depth_estimation.md)

</div>

<!-- RELATED:END -->
