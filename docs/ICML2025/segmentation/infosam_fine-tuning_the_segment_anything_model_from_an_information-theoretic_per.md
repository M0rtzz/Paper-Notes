---
title: >-
  [论文解读] InfoSAM: Fine-Tuning the Segment Anything Model from An Information-Theoretic Perspective
description: >-
  [ICML2025][图像分割][SAM微调] 提出 InfoSAM，从信息论角度为 SAM 的参数高效微调（PEFT）设计了基于 Rényi 互信息的关系压缩与蒸馏框架，通过压缩伪不变信息、保留域不变关系来提升微调效果。
tags:
  - ICML2025
  - 图像分割
  - SAM微调
  - 信息瓶颈
  - 知识蒸馏
  - 参数高效微调
  - Rényi互信息
  - 域不变关系
---

# InfoSAM: Fine-Tuning the Segment Anything Model from An Information-Theoretic Perspective

**会议**: ICML2025  
**arXiv**: [2505.21920](https://arxiv.org/abs/2505.21920)  
**代码**: [InfoSAM project page](https://github.com/zhangyuanhong/InfoSAM)  
**领域**: segmentation  
**关键词**: SAM微调, 信息瓶颈, 知识蒸馏, 参数高效微调, Rényi互信息, 域不变关系

## 一句话总结
提出 InfoSAM，从信息论角度为 SAM 的参数高效微调（PEFT）设计了基于 Rényi 互信息的关系压缩与蒸馏框架，通过压缩伪不变信息、保留域不变关系来提升微调效果。

## 研究背景与动机
- **问题**: SAM 在通用分割上表现优异，但在医学影像、遥感、农业等专业领域表现不佳，需要 PEFT 适配
- **现有不足**: 现有 PEFT 方法（LoRA、Adapter 等）独立调整各模块参数，忽略了预训练模型中编码器-解码器之间的隐式关系；传统蒸馏方法聚焦逐层特征对齐，缺对模块间关系的指导
- **核心观察**: SAM 的海量预训练学到了域不变的结构关系（如几何轮廓），但微调容易破坏这些关系；同时并非所有关系都有益——颜色等伪不变特征会干扰泛化
- **研究目标**: 如何从预训练 SAM 中**提取**域不变关系？如何将其**迁移**到微调模型？

## 方法详解
InfoSAM 由两个互补的信息论目标组成，形成"压缩-蒸馏"框架：

### 1. 注意力关系模块（Relation Module）
- 输入：图像编码器嵌入 $z_i^T \in \mathbb{R}^{B \times H \times W \times D}$ 和 mask decoder 输出 token $z_m^T \in \mathbb{R}^{B \times N \times D}$
- 通过 LayerNorm + 线性投影得到 Q、K，计算注意力分数并加残差：
$$S_\alpha = \frac{QK^\top}{\sqrt{D}} + z_m^T \cdot {z_i^T}^\top$$
- 输出经 $\ell_2$ 归一化得到关系表示 $r^T = f^T(z_i^T, z_m^T; \theta)$

### 2. 关系压缩损失 $\mathcal{L}_r$（Intra-SAM）
- **目标**: 最小化 $\mathbf{I}_\alpha(z_i^T, z_m^T; r^T)$，作为信息瓶颈压缩伪不变信息
- 基于 Rényi α-熵（α=2），利用 Frobenius 范数避免特征值分解：
$$\mathcal{L}_r = -\log_2 \|G_r^T\|_F^2 + \log_2 \|G_{imr}^T\|_F^2$$
- 其中 $G_{imr}^T = G_i^T \circ G_m^T \circ G_r^T$（Hadamard 积），$G$ 为多项式核 Gram 矩阵

### 3. 跨模型蒸馏损失 $\mathcal{L}_d$（Inter-SAM）
- **目标**: 最大化教师关系 $r^T$ 与学生关系 $r^S$ 之间的互信息 $\mathbf{I}_\alpha(r^T; r^S)$
$$\mathcal{L}_d = \log_2 \|G_r^T\|_F^2 + \log_2 \|G_r^S\|_F^2 - \log_2 \|G_r^{TS}\|_F^2$$
- 教师和学生共享同一关系模块参数 $\theta$

### 4. 总损失函数
$$\mathcal{L} = \mathcal{L}_{ce} + \lambda_1 \mathcal{L}_r + \lambda_2 \mathcal{L}_d$$
- $\mathcal{L}_{ce}$ 为标准分割损失（weighted IoU + BCE）

## 实验关键数据

### 表1: PEFT 方法对比（SAM ViT-B，5个数据集 × 4域）

| 方法 | CAMO $S_\alpha$↑ | ISIC Jac↑ | Kvasir $S_\alpha$↑ | Leaf IoU↑ | Road IoU↑ |
|------|:-:|:-:|:-:|:-:|:-:|
| SAM (zero-shot) | 79.7 | 61.0 | 71.4 | 37.6 | 7.2 |
| LoRA | 87.7 | 87.8 | 93.0 | 71.4 | 59.0 |
| Adapter | 88.2 | 87.7 | 93.4 | 74.4 | 60.5 |
| SU-SAM | 88.3 | 87.8 | 93.8 | 74.7 | 60.2 |
| **Adapter+Ours** | **88.6** | **88.0** | **94.4** | **75.6** | **61.4** |

### 表2: 蒸馏方法对比（与 Adapter Student 对比）

| 方法 | Kvasir $S_\alpha$↑ | Leaf IoU↑ | Road IoU↑ |
|------|:-:|:-:|:-:|
| Student (无蒸馏) | 93.4 | 74.4 | 60.5 |
| TinySAM | 88.5 | 48.6 | 25.7 |
| MobileSAM | 92.5 | 71.9 | 59.2 |
| VID | 93.7 | 75.1 | 60.7 |
| **InfoSAM (Ours)** | **94.4** | **75.6** | **61.4** |

### 消融实验（Ablation）

| $\mathcal{L}_r$ | $\mathcal{L}_d$ | Kvasir $S_\alpha$ | Leaf IoU | Road IoU |
|:-:|:-:|:-:|:-:|:-:|
| ✗ | ✗ | 93.4 | 74.4 | 60.5 |
| ✓ | ✗ | 93.6 (+0.2) | 75.2 (+0.8) | 61.0 (+0.5) |
| ✓ | ✓ | **94.4 (+1.0)** | **75.6 (+1.2)** | **61.4 (+0.9)** |

- 两个损失都有正贡献，$\mathcal{L}_d$ 的主要作用体现在跨域蒸馏上
- 在 SAM2（Hiera-B+）上同样有效：Kvasir 94.5、Leaf 77.3、Road 61.3

## 亮点与洞察
- **首个信息论 SAM 适配框架**: 将信息瓶颈理论引入 SAM PEFT，思路新颖且理论扎实
- **不对齐特征而对齐关系**: 不做逐层特征匹配，而是提取并迁移编码器-解码器间的域不变关系，避免了教师在下游域表现差时蒸馏反而降性能的问题
- **即插即用**: 与 LoRA、Adapter 等 PEFT 方法正交，也与 SAM/SAM2 架构无关
- **伪不变信息过滤**: 通过信息瓶颈压缩掉颜色等域特定信息，只保留几何结构等域不变信息
- **Rényi α=2 简化计算**: 用 Frobenius 范数替代特征值分解，降低计算开销

## 局限与展望
- 提升幅度有限：在部分数据集（如 LoRA+Ours vs LoRA）改进约 0.5-1%，域不变关系的收益可能有天花板
- 当教师模型在目标域表现极差时（如 Road IoU 仅 7.2%），蒸馏仍能正向迁移，但幅度受限
- 仅验证了 box/point prompt 场景，未探索 text prompt 或全自动分割
- Rényi 熵阶 α 固定为 2，未探究不同 α 值对性能的影响
- 关系模块引入额外参数和计算，论文未详细分析其开销
- 只在中等规模数据集上验证，缺少超大规模数据集（如 SA-1B 子集）的实验
- 超参数 $\lambda_1, \lambda_2$ 的敏感性分析不够充分

## 相关工作与启发
- **信息瓶颈 + 蒸馏**的组合可推广到其他 foundation model（如 CLIP、DINOv2）的 PEFT
- 与 VID、IBD 等基于互信息的蒸馏方法不同，InfoSAM 关注的是**模块间关系**而非单层特征
- 域不变特征的思路源自域自适应分割（DAS），但首次用信息论量化并迁移

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次将信息瓶颈理论用于 SAM PEFT，理论推导完整
- 实验充分度: ⭐⭐⭐⭐ — 4 个领域 8 个数据集，SAM+SAM2，对比充分
- 写作质量: ⭐⭐⭐⭐ — 信息论公式清晰，图示直观
- 价值: ⭐⭐⭐⭐ — 提供了 SAM 微调的新视角，即插即用的蒸馏方案有实用价值
