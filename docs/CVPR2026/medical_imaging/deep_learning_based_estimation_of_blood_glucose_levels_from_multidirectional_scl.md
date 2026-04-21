---
title: >-
  [论文解读] Deep Learning–Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging
description: >-
  [CVPR 2026][医学图像][血糖估计] 提出ScleraGluNet，通过5个注视方向的巩膜血管照片，用并行CNN提取方向特异性血管特征，再经MRFO特征筛选和Transformer跨视角融合，同时完成三类代谢状态分类（93.8%准确率）和空腹血糖连续估计（MAE=6.42 mg/dL, r=0.983）。
tags:
  - CVPR 2026
  - 医学图像
  - 血糖估计
  - 巩膜血管成像
  - 多视角学习
  - MRFO
  - Transformer
---

# Deep Learning–Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging

**会议**: CVPR 2026  
**arXiv**: [2603.12715](https://arxiv.org/abs/2603.12715)  
**代码**: 无  
**领域**: 医学图像  
**关键词**: 血糖估计, 巩膜血管成像, 多视角学习, MRFO, Transformer  

## 一句话总结

提出ScleraGluNet，通过5个注视方向的巩膜血管照片，用并行CNN提取方向特异性血管特征，再经MRFO特征筛选和Transformer跨视角融合，同时完成三类代谢状态分类（93.8%准确率）和空腹血糖连续估计（MAE=6.42 mg/dL, r=0.983）。

## 研究背景与动机

**糖尿病监测的痛点**：全球5.37亿糖尿病患者需要频繁监测血糖。实验室检测（FPG、HbA1c）准确但需抽血，不便于日常自测；指尖采血有痛感和感染风险，依从性差；CGM虽然方便但需皮下传感器植入，成本较高。非侵入式血糖监测具有重大临床需求。

**巩膜血管是天然的代谢窗口**：慢性高血糖会导致微血管重构——直径改变、迂曲度增加、灌注异常。相比视网膜成像需要专业眼底相机，**巩膜/结膜血管成像只需普通前段相机**即可完成，设备成本低、操作简单，适合远程医疗和大规模筛查。已有OCTA研究证实了巩膜微血管与糖尿病的关联。

**现有方法的不足**：（1）仅使用单一视角拍摄，但糖尿病引起的微血管改变在空间上是异质的——上方、下方、鼻侧、颞侧巩膜的血管异常程度不同，单视角会遗漏关键信息；（2）没有充分利用多视角特征间的互补关系。本文的核心思路是**多方向采集覆盖全巩膜+多视角深度融合架构**。

## 方法详解

### 整体框架

每位受试者采集5张巩膜照片（正视、上视、下视、鼻侧、颞侧）→ 图像预处理（ROI提取+CLAHE+Frangi血管增强）→ 5个独立参数CNN分支分别提取各方向特征 → MRFO特征精炼去除冗余 → Transformer跨视角自注意力融合 → 双头输出（分类+回归）。

### 关键设计

1. **多方向巩膜采集协议**:

    - 功能：标准化采集5个注视方向的巩膜照片，全面覆盖各象限血管
    - 核心思路：正视作为参考和中央区域，上/下/左/右视分别暴露下方/上方/颞侧/鼻侧巩膜。每位参与者产生5张照片，共445×5=2225张
    - 设计动机：糖尿病微血管病变在空间上非均匀分布，不同区域呈现不同程度的血管口径变化、迂曲度增加和灌注异常。消融实验确认多视角显著优于单视角

2. **并行CNN+MRFO特征精炼**:

    - 功能：5个独立参数CNN分支分别学习各方向的血管模式，再用MRFO算法筛选最相关特征
    - 核心思路：各分支共享架构但参数独立，提取方向特异性的血管形态特征（口径变化、迂曲度、分支复杂度）。MRFO（Manta Ray Foraging Optimization）是生物启发优化算法，在拼接特征中筛选最判别性子集，去除跨视角冗余
    - 设计动机：直接拼接5路特征会引入大量冗余维度，稀释判别信号。MRFO自动识别并保留与血糖状态最相关的特征子集

3. **Transformer跨视角融合+双任务头**:

    - 功能：用自注意力发现跨象限的长程血管模式关联，同时输出分类和回归结果
    - 核心思路：MRFO精炼后的特征送入Transformer，自注意力机制捕捉跨视角模式（如双侧不对称重构、跨象限的细微血管特征）。最终Transformer输出接分类头（3类softmax）和回归头（FPG连续值），用 $L = L_{\text{CE}} + L_{\text{MSE}}$ 联合训练
    - 设计动机：多任务学习让表征同时服务于分类和回归，提供互补监督信号

### 损失函数 / 训练策略

联合损失 $L = L_{\text{CE}} + L_{\text{MSE}}$，交叉熵用于代谢状态分类，MSE用于血糖值回归。Adam优化器，subject-wise 5-fold交叉验证（GroupKFold确保同一参与者所有图片在同一fold，避免数据泄漏）。95%置信区间通过参与者级别bootstrap重采样（1000次迭代）估计。

图像预处理：ROI提取 → 颜色/亮度归一化 → CLAHE对比度增强 → Frangi滤波增强管状血管结构。

## 实验关键数据

### 主实验

**数据集**：445名参与者（正常150，控制型糖尿病140，高血糖型155），Changsha Aier Eye Hospital。

| 任务 | 指标 | 结果 |
|------|------|------|
| 三类分类 | 整体准确率 | **93.8%** (95% CI: 91.8-95.4%) |
| | AUC (正常/控制/高血糖) | 0.971 / 0.956 / 0.982 |
| | 正常/控制/高血糖 F1 | 0.937 / 0.918 / 0.942 |
| 血糖回归 | MAE | **6.42 mg/dL** |
| | RMSE | 7.91 mg/dL |
| | Pearson r | 0.983 |
| | R² | 0.966 |
| | Bland-Altman偏差 | +1.45 mg/dL (±8.33~+11.23) |

### 消融实验

| 配置 | 分类准确率 | 回归MAE | 说明 |
|------|-----------|---------|------|
| 单视角CNN | 最低 | 最高 | 无多视角信息 |
| 多视角CNN(直接拼接) | 中等 | 中等 | 有冗余未处理 |
| +MRFO特征选择 | 较好 | 较低 | 去冗余有效 |
| **ScleraGluNet(完整)** | **93.8%** | **6.42** | 全组件最优 |

### 关键发现

- 5-fold交叉验证各折精度稳定在92.8%-94.6%（SD=0.7%），说明结果不依赖有利的数据划分
- Grad-CAM分析显示：正常组注意力弥漫且较弱；控制型聚焦于轻度血管变化区域；高血糖型对扩张/迂曲血管有强烈且跨方向一致的激活
- 误分类主要发生在相邻代谢类别之间（正常↔控制型），符合血糖是连续谱的临床现实

## 亮点与洞察

- 从巩膜血管估计血糖是一个全新的临床应用场景。巩膜成像设备远比视网膜眼底相机便宜，且不需要散瞳，非常适合远程医疗和社区筛查。r=0.983的回归精度和Bland-Altman分析结果接近临床可用水平。
- 多方向采集协议有坚实的生理学依据——微血管异常在空间上非均匀分布，消融实验验证了多视角的必要性。

## 局限与展望

- 单中心研究（445人，同一家医院），泛化性未经外部验证，设备和人群多样性不足
- 混淆因素未充分控制：高血压、吸烟、贫血等也会影响巩膜血管形态
- 仅针对空腹血糖，未纳入餐后血糖动态和纵向监测
- 论文写作质量偏低，部分段落有明显LLM生成痕迹

## 相关工作与启发

- **vs 视网膜生物标记研究（如Google retinal biomarker）**: 巩膜成像设备更简单、成本更低，适合大规模筛查场景，但视网膜成像的临床验证更充分
- **vs PPG/热成像血糖估计**: 巩膜成像直接可视化微血管结构，物理耦合更强，受环境影响更小

## 评分

- 新颖性: ⭐⭐⭐⭐ 从巩膜血管到血糖估计是全新临床场景，多方向采集有创意
- 实验充分度: ⭐⭐⭐ 单中心445人数据集，消融充分但缺少外部验证和更多baseline对比
- 写作质量: ⭐⭐ 明显LLM辅助痕迹，部分段落冗余不自然
- 价值: ⭐⭐⭐ 临床应用潜力大但需多中心验证确认可行性

<!-- RELATED:START -->

## 相关论文

- [Deep Learning Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging](../../CVPR2025/medical_imaging/deep_learning_based_estimation_of_blood_glucose_levels_from_multidirectional_scl.md)
- [MuViT: Multi-Resolution Vision Transformers for Learning Across Scales in Microscopy](muvit_multi-resolution_vision_transformers_for_learning_across_scales_in_microsc.md)
- [Can Natural Image Autoencoders Compactly Tokenize fMRI Volumes for Long-Range Dynamics Modeling?](can_natural_image_autoencoders_compactly_tokenize_fmri_volumes_for_long-range_dy.md)
- [VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)
- [OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)

<!-- RELATED:END -->
