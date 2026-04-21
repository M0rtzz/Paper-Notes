---
title: >-
  [论文解读] TeethGenerator: A Two-Stage Framework for Paired Pre- and Post-Orthodontic 3D Dental Data Generation
description: >-
  [ICCV 2025][医学图像][正畸数据生成] 提出 TeethGenerator，一个两阶段框架用于生成配对的正畸前后 3D 牙齿点云模型，Stage I 用 VQ-VAE+扩散模型生成矫正后牙齿形态，Stage II 用 Transformer 根据风格模型生成对应的矫正前牙齿排列。
tags:
  - ICCV 2025
  - 医学图像
  - 正畸数据生成
  - 3D牙齿模型
  - VQ-VAE
  - 扩散模型
  - 配对数据合成
---

# TeethGenerator: A Two-Stage Framework for Paired Pre- and Post-Orthodontic 3D Dental Data Generation

**会议**: ICCV 2025  
**arXiv**: [2507.04685](https://arxiv.org/abs/2507.04685)  
**代码**: [https://github.com/lcshhh/teeth_generator](https://github.com/lcshhh/teeth_generator)  
**领域**: 医学影像 / 3D 生成  
**关键词**: 正畸数据生成, 3D牙齿模型, VQ-VAE, 扩散模型, 配对数据合成

## 一句话总结

提出 TeethGenerator，一个两阶段框架用于生成配对的正畸前后 3D 牙齿点云模型，Stage I 用 VQ-VAE+扩散模型生成矫正后牙齿形态，Stage II 用 Transformer 根据风格模型生成对应的矫正前牙齿排列。

## 研究背景与动机

- 数字正畸中的牙齿排列网络（如 TANet、OrthoGAN、TADPM 等）需要大量**配对的正畸前后 3D 牙齿模型**进行训练
- 获取这些配对数据极其耗时：需要从真实患者多年治疗过程中扫描获得，且涉及隐私和商业问题
- 当前唯一公开的 3D 正畸数据集仅有 1060 对，且错颌类型分布不均（如前牙开咬仅 20 例）
- 现有 3D 形状生成方法（GAN/VAE/扩散）主要针对单物体生成，**无法处理牙齿模型的多实体结构**（24-32 颗分割牙齿需同时生成）
- 牙齿数据合成的四大挑战：多实例生成、分布匹配、正畸一致性（前后牙齿形态相同）、姿态多样性

## 方法详解

### 整体框架

TeethGenerator 分两个阶段：Stage I 生成矫正后牙齿模型（形态生成），Stage II 生成对应的矫正前牙齿模型（风格迁移）。实际输出为点云，网格仅用于可视化。

### 关键设计

1. **牙齿空间结构编码**: 按照 FDI 编号系统将 32 颗牙齿组织为 $2 \times 2 \times 8$ 的结构化网格，保留关键空间关系：

    - 邻接关系（同侧相邻牙齿）
    - 双侧对称性（同颌左右对称）
    - 咬合对称性（上下颌对应关系）
    - 每颗牙齿采样 128 个点，再划分为 $r^3$ 个体素（$r=4$），总体素布局为 $[2r, 2r, 8r]$

2. **Stage I - 牙齿形态生成模块**:

    - **VQ-VAE 重建**：基于 PVCNN 范式提取体素内特征（PointNet + max-pooling），通过 3D U-Net 编码器-解码器重建聚合牙齿点云 $\hat{P}_{post} \in \mathbb{R}^{K \times N \times C}$，额外 MLP 生成有效牙齿掩码（因牙齿数量$K$是固定的但实际数可变）
    - **潜空间扩散模型**：冻结 VQ-VAE 参数后，在潜空间训练扩散模型学习编码分布。采样时从高斯噪声去噪得到潜编码，再通过 VQ-VAE 解码器生成矫正后牙齿模型
    - 重建损失使用 Chamfer Distance：$L_{rec} = \sum_i \text{CD}(P_i, \hat{P}_{post_i})$

3. **Stage II - 牙齿风格生成模块**:

    - **风格提取器（$E_{style}$）**：对风格模型的每颗牙齿分别提取体素特征 → 均值池化+标准池化 → 两个 MLP → 加和得到每颗牙的风格编码（捕获局部细节和风格特征）
    - **形状提取器（$E_{shape}$）**：对整个牙齿模型全局划分体素（而非逐牙齿）→ max-pooling → 获取完整形状特征（用于碰撞避免的全局上下文）
    - **Transformer 预测变换参数**：风格特征作为 Transformer 输入，形状特征通过交叉注意力注入每层 → 输出 $\mathbf{x} \in \mathbb{R}^{K \times 9}$（每颗牙齿 3 维平移 + 6 维旋转参数）
    - 旋转矩阵 $R_i \in \mathbb{R}^{3 \times 3}$ 从 6D 旋转表示计算，保证正畸前后牙齿形态一致

### 损失函数 / 训练策略

- **Stage I**：
    - VQ-VAE 重建损失：Chamfer Distance
    - 扩散模型损失：标准去噪目标 $\mathcal{L} = \mathbb{E}\|\epsilon_\theta(\mathbf{x_t}, t) - \epsilon\|_2^2$，Cosine 噪声调度
    - batch size 32，500 epochs，AdamW lr=1e-3

- **Stage II**：
    - 距离损失：$\mathcal{L}_{dis} = \sum_i \|P_{pre} - P_{style}\|^2$（利用点对应关系直接计算L2）
    - 碰撞避免损失：$\mathcal{L}_{ca} = \sum_{(a,b) \in \mathcal{K}}((\frac{1}{1+d/s})^{12} - 2(\frac{1}{1+d/s})^6)$（Lennard-Jones 势函数，邻接牙齿对+上下颌对）
    - 总损失：$\mathcal{L} = \mathcal{L}_{dis} + \mathcal{L}_{ca}$
    - 12 层 Transformer，8 头注意力，batch size 64，300 epochs，AdamW lr=1e-4

## 实验关键数据

### 主实验 (表格)

**矫正后牙齿生成质量对比（1-NNA + 独特性，720 样本）：**

| 模型 | CD (%, ↓) | EMD (%, ↓) | $U_{CD}$ (%, ↑) |
|:---:|:---:|:---:|:---:|
| PointFlow | 97.62 | 83.88 | 62.22 |
| DPM | 89.25 | 74.50 | 75.69 |
| PVD | 84.87 | 78.12 | 49.58 |
| LION | 90.41 | 77.93 | 52.78 |
| DiT-3D | 95.75 | 82.01 | 34.03 |
| **TeethGenerator** | **69.50** | **71.88** | **96.25** |

**合成数据对下游牙齿排列任务的提升**（TANet backbone）：

训练数据中加入合成数据（$n \times 720$），性能随合成数据量增加持续提升直到 $n=10$ 后趋于收敛。

### 消融实验 (表格)

**体素化策略和空间结构消融（Stage I）：**

| 编号 | 空间结构 | 体素化 | CD ↓ | EMD ↓ | $U_{CD}$ ↑ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2×2×8 | 3³ | 82.63 | 81.50 | 92.64 |
| **2** | **2×2×8** | **4³** | **69.50** | **71.88** | **96.25** |
| 3 | 2×2×8 | 5³ | 77.75 | 83.23 | 95.56 |
| 4 | - | 无体素 | 89.59 | 84.13 | 21.11 |
| 5 | - | 全局8×8×32 | 78.16 | 76.91 | 89.03 |
| 6 | 1×1×32 | 4³ | 74.48 | 72.82 | 94.17 |

### 关键发现

- **TeethGenerator 大幅领先通用3D生成方法**：CD 从最佳 baseline 84.87 降至 69.50，独特性从 75.69% 提升至 96.25%
- **通用方法无法生成分割的多牙齿结构**：PointFlow、LION 等只生成无法区分个体牙齿的整体点云
- **体素化特征提取至关重要**：无体素化（No.4）独特性骤降至 21.11%（大量重复输出）
- **$2 \times 2 \times 8$ 空间结构优于 $1 \times 1 \times 32$**：利用双侧和咬合对称性显式建模交互关系
- **风格提取器需要局部细节**：交换 $E_{style}$ 和 $E_{shape}$ 的体素化策略后无法复制目标风格
- **合成数据可显著提升下游性能**：加入 10 倍合成数据后 TANet 的牙齿排列指标持续改善

## 亮点与洞察

- 首个**配对正畸前后牙齿数据生成框架**，直击数字正畸领域数据匮乏的核心瓶颈
- Stage II 的设计巧妙：预测变换参数（平移+旋转）而非直接生成点云，天然保证正畸前后牙齿形态一致
- FDI 编号系统的 $2 \times 2 \times 8$ 网格编码充分利用了牙齿的解剖学结构先验
- Lennard-Jones 势函数作为碰撞避免损失的设计源自分子动力学，在3D密集排列场景中非常合适
- 合成数据有效性验证表明高质量合成3D医学数据可以切实改善下游任务

## 局限与展望

- 数据集仅有 720 训练样本，模型泛化到更多错颌类型的能力有待验证
- Stage I 输出固定 32 颗牙齿 + 掩码策略处理可变牙齿数，在缺牙较多的病例中可能效果受限
- 网格重建依赖非刚性配准从真实数据库匹配，生成的网格多样性受限于参考库
- 未考虑牙根、牙龈等结构，仅生成牙冠点云
- Stage II 的风格模型需要从已有数据中选取，限制了风格多样性

## 相关工作与启发

- 与 TANet、TADPM 等下游牙齿排列方法形成互补：TeethGenerator 解决数据问题，后者解决排列问题
- VQ-VAE + 潜空间扩散的组合（类似 LION）在多实体 3D 生成中表现优异
- 配对数据合成的"先生成标准状态再根据风格迁移"的两阶段策略可推广到其他医学配对数据生成

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个正畸配对3D数据生成框架，问题定义新颖；但各组件（VQ-VAE、扩散模型、Transformer）均为成熟技术的组合
- **实验充分度**: ⭐⭐⭐⭐ 包含分布相似性、独特性、下游任务提升、体素化消融、风格迁移策略对比
- **写作质量**: ⭐⭐⭐⭐ 问题动机清晰，可视化结果有说服力（生成牙齿模型 vs 真实牙齿对比）
- **价值**: ⭐⭐⭐⭐⭐ 解决了数字正畸领域的关键数据瓶颈，合成数据已验证可提升下游任务性能，具有重要临床应用前景

<!-- RELATED:START -->

## 相关论文

- [GuideGen: A Text-Guided Framework for Paired Full-Torso Anatomy and CT Volume Generation](../../AAAI2026/medical_imaging/guidegen_a_text-guided_framework_for_paired_full-torso_anatomy_and_ct_volume_gen.md)
- [A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](../../AAAI2026/medical_imaging/a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)
- [Multimodal 3D Genome Pre-training](../../NeurIPS2025/medical_imaging/multimodal_3d_genome_pre-training.md)
- [Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training](boosting_vision_semantic_density_with_anatomy_normality_modeling_for_medical_vis.md)
- [MRGen: Segmentation Data Engine for Underrepresented MRI Modalities](mrgen_segmentation_data_engine_for_underrepresented_mri_modalities.md)

<!-- RELATED:END -->
