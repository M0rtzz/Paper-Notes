---
title: >-
  [论文解读] RnG: A Unified Transformer for Complete 3D Modeling from Partial Observations
description: >-
  [CVPR 2026][3D视觉][3D重建] RnG 提出重构引导因果注意力（Reconstruction-Guided Causal Attention），将 Transformer 的 KV-Cache 重新解释为隐式 3D 表示，用单个前馈 Transformer 统一完成从无位姿稀疏图像到完整 3D 几何与外观的重建与生成，速度比扩散方法快 100 倍以上。
tags:
  - CVPR 2026
  - 3D视觉
  - 3D重建
  - novel view synthesis
  - Transformer
  - KV-Cache
  - feed-forward
---

# RnG: A Unified Transformer for Complete 3D Modeling from Partial Observations

**会议**: CVPR 2026  
**arXiv**: [2603.01194](https://arxiv.org/abs/2603.01194)  
**代码**: https://npucvr.github.io/RnG  
**领域**: 3D视觉  
**关键词**: 3D reconstruction, novel view synthesis, transformer, KV-Cache, feed-forward

## 一句话总结

RnG 提出重构引导因果注意力（Reconstruction-Guided Causal Attention），将 Transformer 的 KV-Cache 重新解释为隐式 3D 表示，用单个前馈 Transformer 统一完成从无位姿稀疏图像到完整 3D 几何与外观的重建与生成，速度比扩散方法快 100 倍以上。

## 背景与动机

### 核心问题
当前 3D 重建基础模型（如 VGGT、DUSt3R）能从少量图像恢复可见区域的几何结构，但**无法建模未观测区域**。新视角合成（NVS）方法虽然能渲染未知视角，但通常缺乏一致的 3D 结构，或依赖已知位姿/扩散模型导致推理缓慢。

### 已有方法局限

| 方法 | 无位姿推理 | 相机控制 | 生成未见区域 | 显式3D | 实时推理 |
|------|:---------:|:-------:|:----------:|:-----:|:-------:|
| VGGT | ✓ | N/A | ✗ | ✓ | ✓ |
| DUSt3R | ✓ | N/A | ✗ | ✓ | ✓ |
| LVSM | ✗ | ✓ | ✓ | ✗ | ✓ |
| LGM | ✗ | ✓ | ✗ | ✓ | ✓ |
| Matrix3D | ✓ | ✓ | ✓ | ✓ | ✗ |
| **RnG (Ours)** | **✓** | **✓** | **✓** | **✓** | **✓** |

Matrix3D 虽然实现了统一重建与生成，但基于扩散设计，生成单张新视角需 27 秒，无法满足实时交互需求。

### 核心洞察
3D 重建基础模型的 latent space 中可能已编码了比可见几何更完整的 3D 理解。如果能将视角条件化的神经渲染作为对模型潜空间的查询，就能同时激活重建与生成能力。与以往"用生成先验辅助重建"的方向相反，RnG 证明了**用重建先验驱动生成**同样可行且高效。

## 方法详解

### 整体框架

RnG 基于 VGGT 的架构和预训练权重构建，是一个前馈 Transformer，处理流程如下：

1. **输入编码**：源视角图像 $\{\mathbf{I}_s\}$ 经 DINOv2 Vision Transformer 提取 token；目标视角通过 Plücker 射线图编码并经线性层映射为 token
2. **联合处理**：源视角和目标视角的 token 拼接后，经 $L=24$ 层交替的全局注意力（Global Attention）和帧级注意力（Frame Attention）处理
3. **多头解码**：
    - 源视角 token → Camera Head → 估计相机位姿 $\{\hat{\mathbf{g}}_s\}$
    - 目标视角 token → RGB Head $\mathcal{D}_\text{RGB}$ → 新视角图像 $\hat{\mathbf{I}}_t$
    - 目标视角 token → Point Head $\mathcal{D}_\text{pmap}$ → 点图 $\hat{\mathbf{p}}_t$（显式几何）

为保持 VGGT 的已学知识，第一个源视角使用专属的相机和寄存器 token，其余源视角与目标视角共享相同 token。模型训练时固定第一个视角位姿为：

$$\hat{\mathbf{g}}_{s=1} = \left[I_{3\times3} \mid [0, 0, -1]^\top\right]$$

这隐式地定义了重建的世界坐标系。

### 关键设计一：重构引导因果注意力（Reconstruction-Guided Causal Attention）

这是 RnG 最核心的创新。设计动机在于：**重建应指导生成，但生成不应干扰重建**。

具体地，在全局注意力块中引入二值掩码 $M$ 控制信息流：

$$M_{i,j} = \begin{cases} 0 & \text{if } i \in \{s\} \text{ and } j \in \{t\} \\ 1 & \text{elsewhere} \end{cases}$$

其中 $\{s\}$ 和 $\{t\}$ 分别为源视角和目标视角 token 的索引。注意力计算变为：

$$\text{Out} = \text{softmax}\left(\frac{M \odot QK^\top}{\sqrt{d_k}}\right)V$$

**信息流方向**：
- 源视角 query 只能 attend 到源视角 key → 保证重建不受目标视角影响
- 目标视角 query 同时 attend 源视角和目标视角 key → 生成过程利用重建信息

这种设计的巧妙之处在于：网络对源视角和目标视角使用**共享参数**，但通过注意力掩码在功能层面解耦——源视角 token 负责感知与位姿估计（重建），目标视角 token 负责合成外观与几何（生成）。因此模型参数高效且可联合训练。

### 关键设计二：KV-Cache 作为隐式 3D 表示

因果注意力的另一关键特性是允许将缓存的 key/value token 重新解释为**隐式 3D 表示**——编码了场景几何和外观的潜空间记忆，与观测方向无关。这使推理过程可拆分为两个阶段：

**阶段一：重建与缓存**（~0.2 秒）

由于源视角 token 的注意力过程与目标视角 token 完全独立，模型可以仅用源视角完成一次重建，并缓存源视角在每层全局注意力块中的 key 和 value token：

$$K_s' = \text{Cache}(K_s), \quad V_s' = \text{Cache}(V_s)$$

**阶段二：生成与查询**（<0.1 秒/视角）

合成新视角时，无需再对源视角计算全局和帧注意力，直接从缓存中读取：

$$\text{Out}_t = \text{softmax}\left(\frac{Q_t \cdot [K_s'; K_t]^\top}{\sqrt{d_k}}\right)[V_s'; V_t]$$

经 $L$ 层处理后，目标视角 token 由两个 DPT Head 分别解码：

$$\hat{\mathbf{I}} = \mathcal{D}_\text{RGB}(\text{Out}_t), \quad \hat{\mathbf{P}} = \mathcal{D}_\text{pmap}(\text{Out}_t)$$

最终通过累积多个视角查询的点图来恢复完整 3D 结构，相当于一个**虚拟 3D 扫描仪**。

### 损失函数

多任务损失由三部分组成：

$$\mathcal{L} = \mathcal{L}_\text{RGB} + \lambda_\text{pmap}\mathcal{L}_\text{pmap} + \lambda_c\mathcal{L}_\text{cam}$$

**新视角图像损失**——MSE + 感知损失：

$$\mathcal{L}_\text{RGB} = |\mathbf{I}_t - \hat{\mathbf{I}}_t|_2 + \lambda_p \cdot \text{Perceptual}(\mathbf{I}_t, \hat{\mathbf{I}}_t)$$

**点图损失**——不确定性加权的 aleatoric uncertainty loss，Point Head 输出四通道（xyz + 不确定性 $\Sigma_t$）：

$$\mathcal{L}_\text{pmap} = \|\Sigma_t \odot (\mathbf{P}_t - \hat{\mathbf{P}}_t)\| + \|\Sigma_t \odot (\nabla\mathbf{P}_t - \nabla\hat{\mathbf{P}}_t)\| - \alpha \cdot \log\Sigma_t$$

**相机位姿损失**——Huber 损失：

$$\mathcal{L}_\text{cam} = \sum_s |\mathbf{g}_s - \hat{\mathbf{g}}_s|_\epsilon$$

超参数设置：$\lambda_\text{pmap}=0.2$，$\lambda_c=1$，$\lambda_p=0.5$，$\alpha=0.2$。

### 训练细节

- 训练数据：Objaverse 数据集（LVIS 子集 + LGM 筛选列表，共 113.5K 物体）
- 分辨率：$256 \times 256$，patch size = 8
- 硬件：8 × A800 GPU，总 batch size = 96
- 训练步数：40K steps
- 精度：bfloat16 + gradient checkpointing

## 实验

### 主实验结果（GSO 数据集）

| 指标类别 | 指标 | Matrix3D (unposed) | VGGT | LVSM (posed) | **RnG (Ours)** |
|---------|------|:------------------:|:----:|:------------:|:-------------:|
| 位姿 | RA@5↑ | 43.77 | 74.24 | — | **85.15** |
| 位姿 | RT@5↑ | 65.92 | 65.68 | — | **86.02** |
| 位姿 | AUC@30↑ | 66.39 | 77.23 | — | **86.94** |
| 源视角深度 | Rel↓ | 9.43 | 5.96 | — | **0.584** |
| 源视角深度 | a1↑ | 92.26 | 97.72 | — | **99.93** |
| 新视角深度 | Rel↓ | 9.96 | — | — | **0.717** |
| 新视角深度 | a1↑ | 90.28 | — | — | **99.85** |
| 新视角合成 | PSNR↑ | 18.74 | — | 27.52 | **26.28** |
| 新视角合成 | SSIM↑ | 0.786 | — | 0.902 | 0.891 |
| 新视角合成 | LPIPS↓ | 0.193 | — | 0.090 | 0.098 |
| 完整3D | CD↓ | 0.067 | 0.026 | — | **0.0067** |

**关键发现**：
- RnG 在所有重建指标上大幅超越 VGGT 和 Matrix3D，位姿估计 RA@5 从 74.24 提升到 85.15
- 源视角深度 Rel 误差（0.584）比 VGGT（5.96）降低一个数量级
- 作为无位姿方法，RnG 的新视角合成质量（PSNR 26.28）接近需要已知位姿的 LVSM（27.52）
- Chamfer Distance（0.0067）显著优于所有方法，证明多视角融合的 3D 几何高度一致

### 消融实验

| 模型变体 | RA@5↑ | PSNR↑ | LPIPS↓ | 说明 |
|---------|:-----:|:-----:|:------:|------|
| LVSM-100K | — | 27.52 | 0.090 | LVSM 最佳性能（需位姿）|
| LVSM-40K | — | 24.62 | 0.154 | 同等训练步数 |
| **Ours-40K** | **85.15** | **26.28** | **0.098** | 完整模型 |
| Ours-15K | 81.65 | 24.86 | 0.124 | 小数据集 |
| Ours-15K-scratch | 8.25 | 20.78 | 0.204 | 无预训练权重 |
| Ours-15K-w/o cam | — | 24.85 | 0.124 | 去除相机位姿监督 |
| Ours-15K-FullAttn | 82.72 | 24.86 | 0.119 | 全双向注意力 |

**消融发现**：
1. **重建先验至关重要**：从头训练（scratch）性能大幅下降，PSNR 下降 4，证明 VGGT 的预训练权重是关键驱动力
2. **训练效率优势**：Ours-15K 即超越 LVSM-40K，说明重建先验带来的数据效率提升
3. **因果注意力 vs 全注意力**：替换为双向注意力（FullAttn）性能几乎无变化，证明因果设计在不牺牲精度的前提下实现了架构优势
4. **位姿监督兼容**：去除 Camera Head 不影响生成质量，多任务学习中重建与生成不冲突

### 效率对比

KV-Cache 使推理大幅加速：推理时间从 213ms 降至 85ms，FLOPS 从 12.26T 降至 2.29T；对比 Matrix3D 的 27 秒/视角快 **300+ 倍**。

### 泛化能力

虽然仅用 4 张输入图训练，RnG 可直接泛化到任意数量输入。增加源视角时合成质量持续提升；对于具有对称结构的物体，甚至单张图像也能得到合理结果。

## 亮点与创新

- **统一框架**：首个基于前馈 Transformer 同时实现无位姿 3D 重建和新视角几何+外观生成的模型
- **因果注意力**：通过注意力掩码而非独立模块实现任务解耦，参数高效且设计优雅
- **KV-Cache 复用**：将 NLP 中的 KV-Cache 机制赋予全新语义——隐式 3D 表示，首次缓存后可高效多次查询
- **逆向知识迁移**：从重建到生成的知识迁移方向，与主流的"生成先验辅助重建"方向互补

## 局限性

1. **纹理细节不足**：作为确定性前馈模型，无法像扩散模型那样生成极为精细的纹理
2. **世界坐标系定义**：数据准备中相机均朝向世界原点，实际应用中手持设备采集需满足此假设
3. **多视角累积噪声**：完整 3D 需累积多个视角查询的点图，多视角几何融合可能引入噪声和冲突

## 评分

| 维度 | 分数 |
|------|:----:|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 综合推荐 | ⭐⭐⭐⭐⭐ |

> 将 KV-Cache 重新解释为隐式 3D 表示是一个漂亮的设计，重建驱动生成的范式为统一 3D 理解提供了实时可行的新路径。实验全面，在多个任务上均达到 SOTA，且推理速度比扩散方法快两个数量级。

<!-- RELATED:START -->

## 相关论文

- [GGPT: Geometry-Grounded Point Transformer](ggpt_geometry_grounded_point_transformer.md)
- [LitePT: Lighter Yet Stronger Point Transformer](litept_lighter_yet_stronger_point_transformer.md)
- [MimiCAT: Mimic with Correspondence-Aware Cascade-Transformer for Category-Free 3D Pose Transfer](mimicat_mimic_with_correspondence-aware_cascade-transformer_for_category-free_3d.md)
- [PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis](pr-iqa_partial-reference_image_quality_assessment_for_diffusion-based_novel_view.md)
- [RNG: Relightable Neural Gaussians](../../CVPR2025/3d_vision/rng_relightable_neural_gaussians.md)

<!-- RELATED:END -->
