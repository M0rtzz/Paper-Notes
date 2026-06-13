---
title: >-
  [论文解读] Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection
description: >-
  [ICCV 2025][关键点检测] 提出首个基于草图的跨模态少样本关键点检测框架，利用原型网络、网格定位器、原型域适应和去风格化网络，仅需少量带标注草图即可在真实照片中检测新类别的新关键点。 关键点检测是计算机视觉中的基础问题，广泛应用于姿态估计和地标检测。现有方法面临以下局限： - 依赖大量标注：热力图回归和直接回归方法…
tags:
  - "ICCV 2025"
  - "关键点检测"
  - "少样本学习"
  - "草图"
  - "跨模态"
  - "域适应"
---

# Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection

**会议**: ICCV 2025  
**arXiv**: [2507.07994](https://arxiv.org/abs/2507.07994)  
**代码**: [https://subhajitmaity.me/DYKp](https://subhajitmaity.me/DYKp)  
**领域**: Keypoint Detection / Few-Shot Learning  
**关键词**: 关键点检测, 少样本学习, 草图, 跨模态, 域适应

## 一句话总结

提出首个基于草图的跨模态少样本关键点检测框架，利用原型网络、网格定位器、原型域适应和去风格化网络，仅需少量带标注草图即可在真实照片中检测新类别的新关键点。

## 研究背景与动机

关键点检测是计算机视觉中的基础问题，广泛应用于姿态估计和地标检测。现有方法面临以下局限：
- **依赖大量标注**：热力图回归和直接回归方法需要大规模标注数据集
- **少样本场景受限**：现有少样本关键点方法局限于特定图像域，无法泛化到新关键点和未见类别
- **源数据可能不可获取**：实际场景中源域真实图像可能因隐私、伦理或稀缺性无法使用

**为什么选择草图？** 草图作为人类最自然的表达方式之一，具有独特优势：
- 容易获取：几笔即可画出物体轮廓并标注关键点
- 无需源域真实图像：实现 source-free 的少样本检测
- 实际意义：珍稀物种、隐私限制或遮挡严重的场景下，草图是唯一可行的参考

核心挑战包括：(1) 草图-照片的巨大域差异；(2) 关键点级别的跨模态嵌入对齐；(3) 用户画风差异导致的风格变化。

## 方法详解

### 整体框架

N-way K-shot 学习问题：用 K 个带标注的草图（support set）在 M 个真实照片（query set）中检测 N 个关键点。框架包含：
1. **图像编码器 F**：提取 support 边缘图和 query 照片的特征图
2. **关键点提取器 P**：通过高斯池化从特征图中提取关键点嵌入
3. **去风格化网络 Z**：将不同风格的 support 嵌入映射为风格无关表示
4. **原型构建**：平均风格无关的 support 嵌入得到关键点原型
5. **特征调制器 M**：原型与 query 特征的逐元素乘法产生相关特征
6. **描述符网络 D + 网格定位器 GBL**：多尺度网格分类 + 偏移回归定位关键点

### 关键设计

**高斯池化关键点提取**：
$$\mathcal{P}(f_k, \mathbf{u}_{k,n}) = \sum_{\mathbf{x}} \exp\left(\frac{-\|\mathbf{x} - \mathbf{u}_{k,n}\|_2^2}{2\xi^2}\right) \cdot f_k[\mathbf{x}]$$

无需硬边界即可提取具有足够区分度的局部上下文信息。

**网格定位器 GBL**：
- 将关键点定位分解为两个子问题：
    - **网格分类**：预测关键点所在的 $L_i \times L_i$ 网格块（cross-entropy loss）
    - **网格偏移回归**：在选定网格块内预测精确偏移（L1 loss）
- 使用多尺度网格 $L = \{8, 12, 16\}$，最终预测取各尺度均值
- 相比 FSKD 的不确定性建模，更简洁且适合草图的稀疏特性

**原型域适应**：
- 受 Tanwisuth et al. 的原型域适应启发，通过 transport loss 拉近 support 原型和 query 关键点嵌入
- 使用归一化距离相似度代替判别式类概率，更适合关键点定位任务
- 转为有监督设置，利用已知的关键点对应关系

**去风格化网络 Z**：
- 针对不同边缘检测器（PiDiNet、HED、Canny）生成的风格差异
- 采用多尺度通道注意力机制，融合全局上下文到局部关键点嵌入
- 风格损失最小化不同风格版本间的嵌入距离

### 损失函数 / 训练策略

总损失包含关键点定位、域适应和去风格化三部分（含辅助关键点版本）：

$$\mathcal{L}_{total} = \lambda_{KP}(\mathcal{L}_{KP} + \mathcal{L}_{KP\text{-aux}}) + \lambda_{DA}(\mathcal{L}_{DA} + \mathcal{L}_{DA\text{-aux}}) + \lambda_{style}(\mathcal{L}_{style} + \mathcal{L}_{style\text{-aux}})$$

超参设置：$\lambda_{KP} = 0.5$，$\lambda_{DA} = 0.001$，$\lambda_{style} = 0.001$，$\xi = 14$。

辅助关键点通过插值生成（在两个可见关键点间取 $t = \{0.25, 0.5, 0.75\}$），最多 18 个辅助关键点，显著增强训练。

编码器使用 ImageNet 预训练 ResNet50，训练 80000 个 episode，Adam 优化器 lr=0.0001。

## 实验关键数据

### 主实验

Animal Pose 数据集（1-shot），PCK@0.1 指标：

| 类别 | 关键点 | B-Vanilla | FSKD | **Proposed** |
|------|--------|-----------|------|------------|
| Seen | Base | 44.16 | 48.75 | **55.10** |
| Seen | Novel | 18.06 | 37.99 | **45.14** |
| Unseen | Base | 40.47 | 38.14 | **43.17** |
| Unseen | Novel | 17.39 | 33.92 | **39.00** |

在最困难的设置（未见类+新关键点）上超越 FSKD 约 5 个 PCK 点。

Animal Kingdom 数据集结果（5 类超类，1-shot）：

| 设置 | B-Vanilla | FSKD | **Proposed** |
|------|-----------|------|------------|
| Unseen Novel | 5.22 | 10.06 | **14.42** |

### 消融实验

各模块贡献（Unseen Novel，1-shot）：

| 方法 | w/o Aux | w/ Aux |
|------|---------|--------|
| B-Vanilla | 17.39 | 29.98 |
| B-DA (+域适应) | 18.31 | 31.76 |
| B-Style (+去风格化) | 18.97 | 32.51 |
| **B-Full** | 19.03 | **39.00** |

- 辅助关键点带来最大提升（+12~20 PCK），远超各子模块的独立贡献
- B-Full 在使用辅助关键点后提升最大（19.03 → 39.00），说明各模块协同效应显著

真实手绘草图泛化测试（Sketchy 数据库，30 张真实草图）：
- Unseen Base: 42.40% (↓0.77)
- Unseen Novel: 38.49% (↓0.51)
- 几乎无性能损失，验证了从合成边缘图到真实草图的良好泛化

### 关键发现

1. **B-Vanilla 基线极弱**：没有域适应和辅助关键点时，Novel 关键点性能极差（仅 17-18 PCK）
2. **辅助关键点是关键**：为所有模块提供额外训练数据，性能提升幅度远超任何单一模块
3. **多模态联合训练更优**：同时使用草图和照片作为 support 可达 46.54 PCK，超越仅用照片的 FSKD (44.75)
4. 合成边缘图训练 → 真实手绘草图测试的迁移能力出奇稳定

## 亮点与洞察

1. **首创 source-free 跨模态少样本关键点检测**：实际意义重大——珍稀物种、隐私场景等都能应用
2. **去风格化设计巧妙**：通过模拟不同边缘检测器的风格差异来适应真实草图的用户差异
3. **辅助关键点策略**的半监督增强效果惊人，为少样本任务提供了通用的数据增强范式
4. 证明了草图作为"唯一可行源数据"的可行性，开辟了新的研究方向

## 局限与展望

1. 使用合成边缘图（PiDiNet/HED/Canny）代替真实草图训练，实际用户草图差异可能更大
2. 仅在动物数据集上评估，人工制品或机械部件等领域的泛化性未验证
3. 1-shot 设置下精度仍有较大提升空间（最优 39.00 PCK vs. 传统方法在充分标注下的 70+）
4. GBL 的简化设计（去除不确定性建模）在某些场景下可能不如 FSKD 灵活
5. 编码器共享可能限制跨模态特征的解耦能力

## 相关工作与启发

- **FSKD (Lu et al.)**：少样本关键点检测的先驱工作，本文的主要基线。使用不确定性建模的 GBL
- **Prototypical Networks**：原型网络思想的自然延伸，从分类任务扩展到关键点定位
- **Tanwisuth et al.**：原型域适应方法，启发了本文的跨模态关键点对齐
- 启发：草图在 CV 中的应用从检索扩展到了结构化几何理解（关键点），还可拓展到分割、3D 重建等

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 3.5 |
| 实验充分性 | 4 |
| 写作质量 | 3.5 |
| 实用价值 | 4 |
| 总评 | 3.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Few-Shot Personalized Scanpath Prediction](../../CVPR2025/others/few-shot_personalized_scanpath_prediction.md)
- [\[ICML 2025\] Feedforward Few-shot Species Range Estimation](../../ICML2025/others/feedforward_few-shot_species_range_estimation.md)
- [\[ICCV 2025\] Is Meta-Learning Out? Rethinking Unsupervised Few-Shot Classification with Limited Entropy](is_meta-learning_out_rethinking_unsupervised_few-shot_classification_with_limite.md)
- [\[ACL 2025\] Zero-Shot Conversational Stance Detection: Dataset and Approaches](../../ACL2025/others/zero-shot_conversational_stance_detection_dataset_and_approaches.md)
- [\[ICML 2025\] Provably Improving Generalization of Few-Shot Models with Synthetic Data](../../ICML2025/others/provably_improving_generalization_of_few-shot_models_with_synthetic_data.md)

</div>

<!-- RELATED:END -->
