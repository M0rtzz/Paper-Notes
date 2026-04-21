---
title: >-
  [论文解读] MegaScenes: Scene-Level View Synthesis at Scale
description: >-
  [ECCV 2024][3D视觉][新视角合成] 从Wikimedia Commons互联网照片构建包含10万+SfM重建的大规模场景级3D数据集MegaScenes，并结合warp条件和位姿条件提升场景级新视角合成的位姿一致性。
tags:
  - ECCV 2024
  - 3D视觉
  - 新视角合成
  - 大规模数据集
  - 扩散模型
  - 位姿条件生成
  - SfM重建
---

# MegaScenes: Scene-Level View Synthesis at Scale

**会议**: ECCV 2024  
**arXiv**: [2406.11819](https://arxiv.org/abs/2406.11819)  
**代码**: https://megascenes.github.io (有)  
**领域**: 3D视觉  
**关键词**: 新视角合成, 大规模数据集, 扩散模型, 位姿条件生成, SfM重建

## 一句话总结

从Wikimedia Commons互联网照片构建包含10万+SfM重建的大规模场景级3D数据集MegaScenes，并结合warp条件和位姿条件提升场景级新视角合成的位姿一致性。

## 研究背景与动机

**领域现状**: 位姿条件扩散模型(如Zero-1-to-3、ZeroNVS)在新视角合成(NVS)上取得进展，但主要在物体级别(Objaverse)或物体中心的小规模场景数据集(DTU、CO3D)上训练

**现有痛点**: 缺乏大规模、多样化的场景级训练数据，导致已有方法无法泛化到真实in-the-wild场景；MegaDepth仅196个地标，CO3D/ACID/RealEstate10K覆盖类别有限

**核心矛盾**: 物体级数据集充足（Objaverse-XL数百万模型），但场景级数据集在规模、多样性、位姿分布上远远不足，成为场景级NVS的瓶颈

**本文目标**: 
   - 构建大规模场景级3D数据集弥补数据缺口
   - 解决现有NVS方法在场景级应用时位姿不准确的问题

**切入角度**: 利用Wikimedia Commons的800万+开放许可图片，通过SfM重建获得场景级3D信息；用warped image作为额外位姿条件提升一致性

**核心 idea**: 大规模互联网照片SfM重建 + warp图像条件化 = 高质量场景级NVS

## 方法详解

### 整体框架

MegaScenes工作包含两部分：(1) 数据集构建——从Wikimedia Commons收集图片并用COLMAP做SfM重建；(2) NVS方法改进——在ZeroNVS基础上增加warp图像条件和外参矩阵条件。

### 关键设计

1. **数据集构建Pipeline**:

    - **场景识别**: 利用Wikidata类别层次结构（如"bridges"、"religious buildings"）自顶向下识别Wikimedia Commons中的场景类别
    - **图片下载**: 下载每个场景类别下所有图片，通过子类别过滤去除不相关图片
    - **SfM重建与清洗**: 对每个场景用COLMAP提取SIFT特征并做词汇树匹配+稀疏重建；用Doppelgangers pipeline处理视觉歧义场景
    - 最终规模：约43万场景、900万图片、10万+SfM重建、200万注册图像

2. **训练数据挖掘**:

    - 光照一致性：利用元数据筛选拍摄时间差 < 3小时的图片对
    - 视觉重叠：要求至少50个共同SfM 3D点
    - 长宽比保持：将长边resize到256并pad短边，避免center crop丢失信息
    - 手动排查去除298个遮挡严重的场景
    - 最终获得 $2,086,036$ 对训练数据，来自32,259个场景

3. **Warp条件化**:

    - **核心洞察**: 位姿矩阵的数值编码不直观，模型需独自学习空间变换；而warped图像直接编码了像素该如何移动，与场景尺度天然对齐
    - **操作方式**: 用Depth-Anything估计单目深度 → 与COLMAP稀疏点云对齐 → 将参考图RGBD反投影成mesh → 从目标位姿渲染得到warped image
    - 将warped image与目标图、参考图concatenate输入扩散模型
    - 训练/推理时均可计算warp，单张图像推理时用单目深度确定场景尺度

4. **Warp + 位姿联合条件**:

    - 仅用warp条件存在两个问题：(a) 深度估计不准时warp质量差；(b) 仅有2D像素运动线索，模型难以理解3D结构
    - 因此额外保留ZeroNVS的外参矩阵条件（flatten后通过cross-attention），实现warp和位姿的互补
    - 外参保证3D几何一致性（如生成分隔墙、完整建筑），warp保证精确的像素对齐

### 损失函数 / 训练策略

- 基于Stable Diffusion扩散模型微调，使用标准扩散训练损失
- 位姿编码：将外参矩阵flatten + FOV一起作为cross-attention的key/value
- 参考图像的CLIP embedding也通过cross-attention注入，保持生成结果与参考的一致性
- 翻译尺度由参考图深度的第20百分位数确定

### 评估指标设计

提出"Masked"版本指标（Masked LPIPS/PSNR/SSIM）：仅在参考图warp到目标视角后有像素覆盖的区域评估，更合理地衡量生成一致性而非创造性生成区域。

## 实验关键数据

### 主实验 — MegaScenes测试集

| 方法 | LPIPS↓ | PSNR↑ | SSIM↑ | FID↓ | KID↓ |
|------|--------|-------|-------|------|------|
| Zero-1-to-3 (released) | 0.548 | 9.09 | 0.241 | 86.9 | 0.063 |
| ZeroNVS (released) | 0.616 | 7.47 | 0.151 | 69.1 | 0.049 |
| Zero-1-to-3 (MS) | 0.429 | 12.16 | 0.367 | 9.78 | 0.002 |
| ZeroNVS (MS) | 0.386 | 12.90 | 0.401 | 9.84 | 0.002 |
| SD-inpainting | 0.425 | 12.36 | 0.392 | 38.5 | 0.024 |
| **Ours** | **0.344** | **13.40** | **0.445** | **11.6** | **0.004** |

### 跨域泛化实验 — RealEstate10K

| 方法 | LPIPS↓ | PSNR↑ | SSIM↑ | FID↓ |
|------|--------|-------|-------|------|
| ZeroNVS (released) | 0.456 | 9.49 | 0.353 | 123.0 |
| ZeroNVS (MS) | 0.205 | 16.02 | 0.630 | 61.1 |
| **Ours** | **0.177** | **17.22** | **0.666** | **60.0** |

### 关键发现

- 仅在MegaScenes上微调就能显著提升性能：ZeroNVS在FID上从69.1降至9.84
- Zero-1-to-3 (MS)即超过ZeroNVS (released)，证明MegaScenes数据集本身的价值
- Warp + 位姿联合条件在所有4个测试集(MegaScenes/DTU/MipNeRF360/RE10K)上均为最优
- SD-inpainting虽然masked重建指标好（直接复制warp像素），但3D理解差，FID/KID远差于微调模型

## 亮点与洞察

- **数据驱动的洞察**: 互联网照片的多样性（光照、天气、设备、位姿分布）是提升泛化能力的关键，远优于受控环境下的数据集
- **Warp条件的巧妙设计**: 将几何信息"视觉化"为warped image，让扩散模型以图像理解的方式学习3D变换，比直接编码位姿矩阵更有效
- **Masked指标**: 解决了生成模型在非重叠区域自由生成导致重建指标失真的问题
- **工程可扩展性**: 基于Wikimedia Commons的开放许可数据，支持未来持续扩展

## 局限与展望

1. 仅使用了数据集的一小部分（47.5万/200万张图片），未利用文本caption等元数据
2. 依赖单目深度估计做warp，深度不准时效果下降
3. 无法处理大角度视角变化（如场景背后的视角）
4. 未考虑光照条件建模，仅通过元数据时间过滤绕过光照问题
5. 场景尺度推理时依赖单目深度的绝对尺度，可能不准确

## 相关工作与启发

- **MegaDepth** → MegaScenes: 同样利用互联网照片+SfM，但MegaScenes规模大100倍+
- **Zero-1-to-3/ZeroNVS**: 作为baseline，证明位姿条件扩散模型需要场景级数据
- **DL3DV-10K**: 并行工作，从视频创建场景数据集，但位姿多样性有限
- **启发**: 数据集规模和多样性对3D生成模型的重要性可能被低估；互联网照片是一个被低估的大规模3D数据源

## 评分

- **新颖性**: ⭐⭐⭐⭐ (数据集构建方法论扎实，warp条件设计简洁有效)
- **实验充分度**: ⭐⭐⭐⭐⭐ (4个测试集、多种baseline、定性+定量全面)
- **写作质量**: ⭐⭐⭐⭐ (结构清晰，动机阐述充分)
- **价值**: ⭐⭐⭐⭐⭐ (数据集对社区有长远价值，开源代码/数据/模型)

<!-- RELATED:START -->

## 相关论文

- [NGP-RT: Fusing Multi-Level Hash Features with Lightweight Attention for Real-Time Novel View Synthesis](ngp-rt_fusing_multi-level_hash_features_with_lightweight_attention_for_real-time.md)
- [Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis](generative_camera_dolly_extreme_monocular_dynamic_novel_view_synthesis.md)
- [Thermal3D-GS: Physics-induced 3D Gaussians for Thermal Infrared Novel-view Synthesis](thermal3d-gs_physics-induced_3d_gaussians_for_thermal_infrared_novel-view_synthe.md)
- [Analysis-by-Synthesis Transformer for Single-View 3D Reconstruction](analysis-by-synthesis_transformer_for_single-view_3d_reconstruction.md)
- [Efficient Depth-Guided Urban View Synthesis (EDUS)](efficient_depth-guided_urban_view_synthesis.md)

<!-- RELATED:END -->
