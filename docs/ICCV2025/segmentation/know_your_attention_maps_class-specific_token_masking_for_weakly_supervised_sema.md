---
title: >-
  [论文解读] Know Your Attention Maps: Class-specific Token Masking for Weakly Supervised Semantic Segmentation
description: >-
  [ICCV 2025][图像分割][弱监督语义分割] 提出一种端到端的弱监督语义分割方法，通过在 ViT 中引入多个 [CLS] token（每个类别一个）、对 [CLS] token 输出嵌入进行随机掩码以及剪枝冗余注意力头，直接利用自注意力图生成类别特定的伪分割掩码，无需额外的 CAM 模块。
tags:
  - ICCV 2025
  - 图像分割
  - 弱监督语义分割
  - Transformer
  - 注意力图
  - CLS token
  - 注意力头剪枝
---

# Know Your Attention Maps: Class-specific Token Masking for Weakly Supervised Semantic Segmentation

**会议**: ICCV 2025  
**arXiv**: [2507.06848](https://arxiv.org/abs/2507.06848)  
**代码**: [github.com/HSG-AIML/TokenMasking-WSSS](https://github.com/HSG-AIML/TokenMasking-WSSS)  
**领域**: 语义分割（弱监督）  
**关键词**: 弱监督语义分割, Vision Transformer, 注意力图, CLS token, 注意力头剪枝

## 一句话总结

提出一种端到端的弱监督语义分割方法，通过在 ViT 中引入多个 [CLS] token（每个类别一个）、对 [CLS] token 输出嵌入进行随机掩码以及剪枝冗余注意力头，直接利用自注意力图生成类别特定的伪分割掩码，无需额外的 CAM 模块。

## 研究背景与动机

- **弱监督语义分割 (WSSS)** 旨在仅使用图像级标签实现像素级分割，减少细粒度标注的需求
- 传统方法依赖 **CAM（Class Activation Maps）** 来高亮判别性区域并生成伪分割掩码，但 CAM 存在定位粗糙、只关注最显著区域的问题，且需要额外模块
- ViT 的自注意力机制天然提供可解释性，但存在两个关键挑战：
    - 无法将特定类别分配给单独的注意力头
    - 不同 [CLS] token 没有硬分配到对应类别的保证
- 本文希望弥合 ViT 能力与类别特定特征嵌入需求之间的差距

## 方法详解

### 整体框架

输入图像被分割成 patch 并投影为 token 序列，在序列前追加 C 个 [CLS] token（C 为类别数）和 1 个 [REG] token，经过 Transformer 编码器后，利用各 [CLS] token 的自注意力图生成伪分割掩码。

### 关键设计

1. **多 [CLS] Token 设计**: 将标准 ViT 的单个 [CLS] token 扩展为 C 个，每个 [CLS] token 对应一个语义类别。扩展后的嵌入序列为 $\mathbf{z}_0 = [\text{[CLS]}_1; \text{[CLS]}_2; \ldots; \text{[CLS]}_C; \mathbf{z}_0; \text{[REG]}]$。[REG] token 借鉴 Darcet 等人的工作，用于捕获全局上下文信息，防止 [CLS] token 的自注意力被"污染"。

2. **随机掩码策略（Random Masking of [CLS] Tokens）**: 训练时，随机选择 50% 不对应当前图像标签的 [CLS] token 进行输出嵌入掩码（置零），核心思路是，掩码后模型必须依靠剩余可用的 token 做出类别决策，从而强制每个 [CLS] token 学习到正确的类别分配关系。掩码函数为：$m(i) = 0$ 若 $i \in \mathcal{Y}$（真实标签集），否则以50%概率设为1。

3. **注意力头剪枝（Attention Head Pruning）**: 在每个注意力头上引入可学习的门控标量 $g_i$，修改 MSA 为 $\text{MSA}(Q,K,V) = \text{concat}_i(g_i \cdot \text{head}_i)W^O$。通过 Hard Concrete 分布对 $L_0$ 范数的随机松弛实现可微剪枝。正则化损失 $\mathcal{L}_{\text{reg}} = \sum_i(1 - P(g_i=0|\phi_i))$，设 $\lambda=0.01$ 约可剪枝约 2/3 的注意力头，去除冗余和噪声头，使剩余头更具可解释性。

### 损失函数 / 训练策略

- 总损失：$\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda \mathcal{L}_{\text{reg}}$
- $\mathcal{L}_{\text{cls}}$ 为二元交叉熵分类损失
- $\lambda = 0.01$ 控制剪枝程度
- 推理时：提取预测类别对应的 [CLS] token 的自注意力图，reshape 到图像空间维度后二值化，按类别概率从低到高合并，未分配像素用邻域最常见值填充

## 实验关键数据

### 主实验

**伪掩码质量（Pascal VOC 2012, mIoU %）**:

| 方法 | 类型 | Backbone | train | val |
|------|------|----------|-------|-----|
| ViT-PCM + CRF (ECCV'22) | 多阶段 | ViT-B† | 71.4 | 69.3 |
| ReCAM (CVPR'22) | 多阶段 | ResNet50 | 70.5 | - |
| I/C-CTI (CVPR'24) | 多阶段 | DeiT-S | 73.7 | - |
| AFA (CVPR'22) | 单阶段 | MiT-B1 | 68.7 | 66.5 |
| ToCo (CVPR'23) | 单阶段 | ViT-B | 72.2 | 70.5 |
| DuPL (CVPR'24) | 单阶段 | ViT-B | 75.1 | 73.5 |
| **Ours** | 单阶段 | ViT-B | 74.5 | **73.7** |

**最终分割结果（mIoU %）**:

| 方法 | Backbone | VOC val | VOC test | COCO val |
|------|----------|---------|----------|----------|
| ReCAM (CVPR'22) | DL-V2 | 68.4 | 68.2 | 45.0 |
| I/C-CTI (CVPR'24) | ResNet38 | 74.1 | 73.2 | 45.4 |
| AFA (CVPR'22) | MiT-B1 | 66.0 | 66.3 | 38.9 |
| ToCo (CVPR'23) | ViT-B | 69.8 | 70.5 | 41.3 |
| DuPL (CVPR'24) | ViT-B | 72.2 | 71.6 | 43.6 |
| **Ours** | ViT-B | 72.7 | **73.5** | 43.2 |

### 消融实验

**各组件效果（mIoU %）**:

| 组件 | MS COCO | VOC |
|------|---------|-----|
| w/o 随机掩码 | 41.9 | 71.6 |
| w/ 随机掩码 | **43.2** | **72.7** |
| w/o [REG] token | 42.8 | 72.3 |
| w/ [REG] token | **43.2** | **72.7** |
| w/o 注意力头剪枝 | 41.7 | 72.0 |
| w/ 注意力头剪枝 | **43.2** | **72.7** |

**掩码比例敏感性分析**: 掩码比例从 0% 到 100% 测试，50% 左右达到最佳平衡点，超过后性能持平或略降。

### 关键发现

- 在三个专业数据集上（DFC2020 遥感、EndoTect 医学、ADE20K 场景）均取得弱监督 SOTA，DFC2020 上甚至超越全监督方法（mIoU 67.2 vs 53.1）
- 随机掩码是最关键的组件：不使用掩码时虽然注意力形状准确，但类别分配错误
- [REG] token 改善了类别边界的清晰度，特别是"天空"和"建筑"等类别之间的边界
- 注意力头剪枝使伪掩码更平滑、噪声更少

## 亮点与洞察

- 核心思想优雅简洁：通过多 [CLS] token + 随机掩码 + 剪枝，无需额外模块即可端到端实现 WSSS
- 随机掩码策略巧妙：通过"排除法"迫使每个 [CLS] token 锁定到特定类别
- 在标注稀缺的专业领域（遥感/医学）上特别有效，减少了对大规模标注数据的依赖
- 在 DFC2020 遥感数据集上超越全监督方法，说明弱监督 + 好的注意力利用可以胜过粗糙的全监督

## 局限与展望

- 随着类别数增加，[CLS] token 数量线性增长，模型参数和计算复杂度随之上升，**可扩展性受限**
- ADE20K（150 类）等大规模类别场景下的效率是公开挑战
- 可以探索动态 token 分配策略来减少不必要的计算
- 未与 SAM 等基础模型结合，可能进一步提升质量

## 相关工作与启发

- 与 MCTformer (CVPR'22) 类似使用多个类别 token，但本文的掩码策略在类别分配上更加可靠
- [REG] token 的设计来自 Darcet 等人的 Vision Transformer Register 工作
- DINO 的自监督 ViT 也展示了自注意力的可解释性，本文在此基础上通过约束实现类别特定的注意力

## 评分

- **新颖性**: ⭐⭐⭐⭐ 多 CLS token + 随机掩码 + 剪枝的组合设计新颖且有效
- **实验充分度**: ⭐⭐⭐⭐ 5 个数据集（标准+专业领域），消融全面
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，图示直观
- **价值**: ⭐⭐⭐⭐ 对弱监督分割特别是标注稀缺的专业领域有实际价值
