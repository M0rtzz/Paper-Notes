---
title: >-
  [论文解读] MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection
description: >-
  [CVPR 2026][医学图像][零样本异常检测] 提出 MoECLIP，将 Mixture-of-Experts 引入零样本异常检测（ZSAD），通过冻结正交特征分离（FOFS）和等角紧框架（ETF）损失实现 patch 级别的动态专家路由与特化，在14个工业/医学基准上达到 SOTA。
tags:
  - CVPR 2026
  - 医学图像
  - 零样本异常检测
  - 混合专家
  - CLIP
  - LoRA
  - 专家特化
---

# MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection

**会议**: CVPR 2026  
**arXiv**: [2603.03101](https://arxiv.org/abs/2603.03101)  
**代码**: [有](https://github.com/CoCoRessa/MoECLIP)  
**领域**: 医学图像  
**关键词**: 零样本异常检测, 混合专家, CLIP, LoRA, 专家特化

## 一句话总结

提出 MoECLIP，将 Mixture-of-Experts 引入零样本异常检测（ZSAD），通过冻结正交特征分离（FOFS）和等角紧框架（ETF）损失实现 patch 级别的动态专家路由与特化，在14个工业/医学基准上达到 SOTA。

## 研究背景与动机

### 1. 领域现状
视觉异常检测（AD）用于识别偏离正常模式的区域，在工业缺陷检测和医学图像诊断中至关重要。传统无监督异常检测（UAD）仅从正常数据学习，但仍需要大量正常样本。零样本异常检测（ZSAD）利用 CLIP 等视觉-语言模型的强大泛化能力，无需目标类别的训练数据即可检测异常，成为新兴范式。

### 2. 痛点
CLIP 预训练目标是全局语义理解，不擅长检测局部异常。现有 ZSAD 方法（PromptAD、AnomalyCLIP、AdaCLIP、AA-CLIP）虽然通过提示学习、适配器等方式增强 patch 表示，但都采用**patch-agnostic 设计**：对所有 patch 施加相同的统一变换，忽略不同图像区域（物体部件、背景、异常区域）的独特特性。

### 3. 核心矛盾
需要对 CLIP 做异常检测特化，但又要保留其泛化能力；需要对不同 patch 差异化处理，但简单的多专家组合会产生**功能冗余**（experts 学到相似功能）。

### 4. 要解决什么
(1) 打破 patch-agnostic 的设计局限，实现 patch 级别的动态适配；(2) 解决 MoE 的专家功能冗余问题，确保每个专家真正特化。

### 5. 切入角度
将 MoE 架构与 LoRA 结合引入 ZSAD，在输入端和输出端同时施加约束来分离专家功能。

### 6. 核心 idea
用 MoE 架构动态路由每个 patch 到合适的 LoRA 专家；用 FOFS 在输入端正交分割特征空间，用 ETF loss 在输出端强制最大等角分离，双管齐下消除专家冗余。

## 方法详解

### 整体框架

MoECLIP 在 CLIP Vision Encoder（ViT-L/14-336）的多个层（第6、12、18、24层）输出端集成 MoE 模块，编码器权重完全冻结。每个 MoE 模块包含 K=4 个 LoRA 专家和一个线性路由器，采用 Top-2 路由策略。训练时在辅助数据集（VisA）上监督学习，测试时在完全未见过的类别上评估。

整体流程：输入图像 → CLIP ViT 提取多层 patch 特征 → 各层 MoE 模块动态路由+特征适配 → PAA 多尺度聚合 → 与文本特征计算相似度 → 输出异常图和异常分数。

### 关键设计

#### 1. MoE-based Feature Adaptation（MoE 特征适配）

**功能**：在每层为每个 patch 动态选择最优专家组合来适配其表示。

**核心思路**：每个 MoE 模块接收 patch 特征 $F_i^l \in \mathbb{R}^d$，路由器计算各专家的路由分数，选 Top-k 个专家加权输出残差 $F_{i,\text{expert}}^l$。关键技巧是对 MoE 输出做**范数归一化**（匹配原始特征的 $\ell_2$ 范数），然后通过加权残差连接（$\lambda_{\text{MoE}}=0.1$）与原始特征混合，避免范数失配导致训练不稳和泛化退化。

**设计动机**：LoRA 作为专家天然参数量少（rank=8），减少过拟合风险；范数归一化借鉴 AA-CLIP，保持 CLIP 表示空间的稳定性。

#### 2. FOFS（冻结正交特征分离）

**功能**：在 LoRA 专家的输入端强制分离特征子空间，消除输入级冗余。

**核心思路**：将 $d$ 维输入特征空间分成 $K$ 个不重叠子空间 $c_1, \dots, c_K$。每个专家的 LoRA 下投影矩阵 $A_n \in \mathbb{R}^{r \times d}$ 被构造为分块矩阵——仅第 $n$ 个子空间对应的列填充 QR 分解得到的正交矩阵 $Q_n$，其余列为零。这保证了 $A_n A_m^T = 0$（$n \neq m$），即专家间互相正交。关键：$A_n$ 被冻结不训练，仅 $B_n$ 可学习。

**设计动机**：(1) 物理上强制不同专家看到不同的特征维度，从初始化就防止冗余；(2) 冻结 $A$ 矩阵保留 CLIP 泛化能力、降低过拟合风险，受到近期 LoRA 研究的启发——随机初始化的正交 $A$ 矩阵性能可媲美学习得到的。

#### 3. ETF Loss（等角紧框架损失）

**功能**：在 LoRA 专家的输出端约束专家向量最大等角分离，消除输出级冗余。

**核心思路**：对每层每个 patch，计算 $K$ 个专家输出（$\ell_2$ 归一化后）的 Gram 矩阵，用 Frobenius 范数惩罚其与理想 ETF Gram 矩阵的差距。理想结构要求：对角线为 1（单位范数），非对角线为 $-1/(K-1)$（最大等角）。

**设计动机**：FOFS 只能约束输入端，可学习的 $B$ 矩阵仍可能让专家输出收敛到相似空间。ETF loss 作为互补机制，在输出端进一步强制分化。

#### 4. PAA（Patch 平均聚合）

**功能**：在训练阶段整合多尺度上下文信息，提升对不同大小异常的检测能力。

**核心思路**：将 patch 嵌入重塑为 2D 空间网格，对多个滑窗尺度 $s \in \{1, 3, 5\}$ 做均值池化，独立生成多组 patch 特征。参数量为零。

**设计动机**：ViT 固定 patch 尺寸无法有效检测不同尺度的异常；现有方法仅在测试阶段使用 patch 聚合，缺少训练阶段的多尺度感知。对医学数据集效果提升尤其显著。

#### 5. Depth-wise Adapter

**功能**：为图像级异常分数提供语义对齐的全局特征。

**核心思路**：借鉴 MobileNet，使用 1D 深度可分离卷积（Depthwise + Pointwise）处理最后一层 PAA 特征，再全局平均池化得到图像级向量 $V_{\text{image}}$，与文本特征计算余弦相似度得到异常分数。

### 损失函数 / 训练策略

总损失：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \mathcal{L}_{\text{ac}} + \lambda_{\text{etf}}\mathcal{L}_{\text{etf}} + \lambda_{\text{bal}}\mathcal{L}_{\text{bal}}$

- **分割损失** $\mathcal{L}_{\text{seg}}$：Focal + Dice Loss，作用于多层多尺度异常图
- **分类损失** $\mathcal{L}_{\text{ac}}$：BCE Loss，作用于图像级异常分数
- **ETF 损失**：$\lambda_{\text{etf}}=0.01$，约束专家输出等角
- **Balance 损失**：$\lambda_{\text{bal}}=0.01$，用路由概率的变异系数平方防止专家坍塌

训练配置：OpenCLIP ViT-L/14-336，图像 518×518，Adam $\text{lr}=5 \times 10^{-4}$，20 epoch，2×V100 16GB。

## 实验关键数据

### 主实验

在14个数据集（5工业+9医学）上与6个 SOTA 方法对比，训练集均为 VisA（评估 VisA 时用 MVTec-AD 训练）。

**表1：图像级异常分类（AUROC, AP）**

| 方法 | MVTec-AD | VisA | BTAD | RSDD | DTD-Syn | BrainMRI | HeadCT | LiverCT | RetinaOCT | **平均** |
|------|----------|------|------|------|---------|----------|--------|---------|-----------|----------|
| WinCLIP | (91.8,95.1) | (78.1,77.5) | (83.3,84.1) | (85.3,65.3) | (95.0,97.9) | (45.1,80.3) | (83.7,81.6) | (66.5,56.1) | (53.7,44.3) | (75.8,75.8) |
| AnomalyCLIP | (91.9,96.2) | (82.1,85.4) | (92.5,94.2) | (74.0,73.2) | (93.3,97.7) | (70.8,90.6) | (95.1,95.3) | (68.2,63.4) | (74.7,73.9) | (82.5,85.5) |
| AA-CLIP | (90.9,96.0) | (79.2,83.7) | (94.8,97.5) | (94.9,94.2) | (92.5,97.7) | (79.6,94.4) | (95.4,94.3) | (58.4,49.7) | (83.4,83.8) | (85.5,87.9) |
| Bayes-PFL | (92.2,96.1) | (86.8,89.3) | (93.0,96.7) | (91.3,89.7) | (93.5,97.7) | (81.9,94.5) | (95.4,93.2) | (61.7,55.2) | (83.7,81.8) | (86.6,88.2) |
| **MoECLIP** | **(93.9,96.8)** | **(83.6,86.2)** | **(93.1,98.0)** | **(95.3,95.1)** | **(95.5,98.6)** | **(88.5,97.1)** | **(96.6,94.5)** | **(74.0,64.6)** | **(85.5,84.9)** | **(89.6,90.6)** |

MoECLIP 图像级平均 AUROC 89.6%（+3.0%），AP 90.6%（+2.4%）。

**表2：像素级异常分割（AUROC, AP）- 部分数据**

| 方法 | MVTec-AD | BTAD | BrainMRI | ColonDB | ClinicDB | Kvasir | **平均** |
|------|----------|------|----------|---------|----------|--------|----------|
| AA-CLIP | (91.6,45.4) | (95.6,49.4) | (96.7,55.1) | (82.8,31.5) | (89.2,49.8) | (86.0,52.9) | (93.2,45.8) |
| Bayes-PFL | (91.9,48.4) | (95.6,48.6) | (95.7,42.9) | (82.9,30.7) | (88.2,49.1) | (85.6,53.4) | (93.2,44.3) |
| **MoECLIP** | **(92.5,45.7)** | **(96.8,50.4)** | **(97.3,61.3)** | **(85.4,34.8)** | **(89.7,49.9)** | **(88.1,57.6)** | **(94.3,47.5)** |

像素级平均 AUROC 94.3%（+1.1%），AP 47.5%（+1.7%）。医学数据集提升尤为显著（BrainMRI AP +6.2%，Kvasir AP +4.2%）。

### 消融实验

**表3：组件消融（Pixel AUROC, Image AUROC）**

| 配置 | MVTec-AD | DTD-Syn | HeadCT | ColonDB | 平均 |
|------|----------|---------|--------|---------|------|
| Vanilla CLIP | (38.4,74.1) | (33.9,71.6) | (-,56.5) | (49.5,-) | (40.6,67.4) |
| w/o FOFS & ETF | (91.6,91.7) | (97.8,93.1) | (-,94.4) | (84.1,-) | (91.2,93.1) |
| w/o FOFS | (92.0,92.8) | (98.3,93.9) | (-,95.0) | (85.3,-) | (91.9,93.9) |
| w/o ETF Loss | (92.2,92.7) | (98.2,93.4) | (-,96.1) | (84.6,-) | (91.7,94.1) |
| w/o Depth Adapter | (92.0,92.5) | (98.1,93.8) | (-,94.5) | (85.0,-) | (91.7,93.6) |
| w/o PAA | (92.1,92.8) | (98.1,94.7) | (-,93.1) | (81.9,-) | (90.7,93.5) |
| **MoECLIP (full)** | **(92.5,93.9)** | **(98.8,95.5)** | **(-,96.6)** | **(85.4,-)** | **(92.2,95.3)** |

### 关键发现

1. **FOFS 和 ETF 是互补的**：单独移除任一组件均导致性能下降，两者共同移除下降更大，证明输入端+输出端双重约束的必要性
2. **功能冗余量化**：专家间余弦相似度从原始 MoE 的 0.45 → +FOFS 后 0.24 → +ETF 后 0.02，近乎完全消除冗余
3. **PAA 对医学域至关重要**：移除 PAA 后 HeadCT 下降 3.5%、ColonDB 下降 3.5%，多尺度感知对医学异常检测影响大
4. **专家数量不是越多越好**：K=4 最优，K>4 反而因功能冗余导致性能下降
5. **跨域泛化**：仅在工业数据训练，MoE 专家依然能在医学数据上有效特化路由

## 亮点与洞察

1. **首次将 MoE 引入 ZSAD**：开创性地从 patch-agnostic 到 patch-specialized 的范式转变
2. **双端约束设计优雅**：FOFS 在输入端物理隔离子空间（冻结、零额外参数），ETF loss 在输出端用几何最优结构约束，两者正交互补
3. **可视化验证直观**：Grad-CAM 清晰展示 Expert 1 聚焦异常区、Expert 2 聚焦物体主体、Expert 3 聚焦背景，证明路由确实基于内容特性
4. **FOFS 冻结 A 矩阵**的巧妙之处：利用近期 LoRA 研究发现（随机正交 A ≈ 学习 A），同时获得正交分离+参数节省+过拟合抑制三重收益

## 局限与展望

1. **专家数量手动设定**：K=4 是经验值，缺乏自适应确定专家数量的机制
2. **FOFS 等分子空间**：将特征维度均匀分给各专家，未考虑不同专家可能需要不同维度
3. **仅在 ViT-L/14 验证**：未探索不同 backbone 规模（如 ViT-B、ViT-H）的效果
4. **训练数据单一**：始终用 VisA 做辅助训练集，未探索不同训练集对泛化的影响
5. **PAA 窗口尺度固定**：$s \in \{1,3,5\}$ 是手动设定，可考虑自适应或可学习的尺度选择

## 相关工作与启发

- **ZSAD 系列方法演进**：WinCLIP → April-GAN → AnomalyCLIP → AdaCLIP → AA-CLIP → Bayes-PFL → MoECLIP，从手工prompt到学习prompt，再到适配器，最终到 MoE 动态路由
- **MoE 功能冗余解决方案**：现有方法（对比损失、正交正则化）仅作用于输出端，本文同时约束输入端和输出端，思路可推广到其他 MoE 场景
- **LoRA 冻结 A 矩阵**的启发：VeRA 等工作已验证共享/冻结下投影矩阵的可行性，本文将其与正交分离结合，值得在其他 PEFT+MoE 场景探索

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 MoE 引入 ZSAD 开创 patch-specialized 新范式，FOFS+ETF 双端约束思路独到
- 实验充分度: ⭐⭐⭐⭐⭐ 14个数据集（工业+医学）全面对比+消融+可视化+专家相似度量化，非常扎实
- 写作质量: ⭐⭐⭐⭐ 动机清晰、方法描述系统、可视化丰富
- 价值: ⭐⭐⭐⭐ 对 ZSAD 和 MoE 功能冗余两个问题同时给出了优雅解法，可推广到其他 PEFT+MoE 场景

<!-- RELATED:START -->

## 相关论文

- [VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)
- [AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP](../../CVPR2025/medical_imaging/aa-clip_enhancing_zero-shot_anomaly_detection_via_anomaly-aware_clip.md)
- [Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)
- [Dual Distillation for Few-Shot Anomaly Detection](../../ICLR2026/medical_imaging/dual_distillation_for_few-shot_anomaly_detection.md)
- [InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](invad_inversionbased_reconstructionfree_anomaly_de.md)

<!-- RELATED:END -->
