---
title: >-
  [论文解读] The Diffusion Duality
description: >-
  [ICML 2025][图像生成][离散扩散对偶性] 揭示了 Uniform-state 离散扩散过程本质上从底层高斯扩散中涌现（通过 argmax 映射），利用这一对偶性将高斯扩散的课程学习策略和一致性蒸馏迁移到离散设置，实现训练速度翻倍和采样加速两个数量级（1024步→8步），在零样本困惑度上 3/7 数据集超越自回归模型。
tags:
  - ICML 2025
  - 图像生成
  - 离散扩散对偶性
  - Uniform-state 扩散
  - 课程学习
  - 离散一致性蒸馏
  - 少步文本生成
---

# The Diffusion Duality

**会议**: ICML 2025  
**arXiv**: [2506.10892](https://arxiv.org/abs/2506.10892)  
**代码**: https://s-sahoo.com/duo (有，含 checkpoint)  
**领域**: 离散扩散模型 / 语言建模  
**关键词**: 离散扩散对偶性, Uniform-state 扩散, 课程学习, 离散一致性蒸馏, 少步文本生成

## 一句话总结
揭示了 Uniform-state 离散扩散过程本质上从底层高斯扩散中涌现（通过 argmax 映射），利用这一对偶性将高斯扩散的课程学习策略和一致性蒸馏迁移到离散设置，实现训练速度翻倍和采样加速两个数量级（1024步→8步），在零样本困惑度上 3/7 数据集超越自回归模型。

## 研究背景与动机

**领域现状**：离散扩散模型（如 MDLM、SEDD）已在文本生成上展现潜力，但性能持续落后于自回归（AR）模型。目前主要有两类离散扩散：Masked Diffusion Models（MDM，使用 [MASK] token）和 Uniform-state Diffusion Models（USDM，token 可转换为词汇表中任意 token）。

**现有痛点**：
   - **USDM 性能差**：虽然 USDM 天然具有自纠错能力（token 在任何步都可修改），但历史上一直表现不如 MDM
   - **训练方差高**：离散扩散的 ELBO 训练方差远大于高斯扩散，导致收敛慢
   - **无法少步生成**：MDM 缺少 Probability Flow ODE（因为 prior 是确定性的 mask），无法使用一致性蒸馏等技术；USDM 也没有开发过类似技术
   - **与高斯扩散的技术鸿沟**：高斯扩散有丰富的加速技术（高效参数化、快速采样器、蒸馏），但这些技术无法直接迁移到离散扩散

**核心矛盾**：离散扩散模型的设计空间仍然原始——仍在使用均值参数化和慢速祖先采样——而高斯扩散已有 15+ 年的方法论积累。

**本文目标**：(1) 建立离散扩散与高斯扩散的理论联系，(2) 借此迁移高斯扩散的高效训练和采样技术到离散设置。

**切入角度**：数学上发现 argmax 操作将高斯扩散过程变换为 Uniform-state 离散扩散过程——这不是近似，而是精确的数学关系。

**核心 idea**：离散扩散是高斯扩散的"涌现现象"（Diffusion Duality），利用这个对偶性可以自由借用高斯扩散的工具箱。

## 方法详解

### 整体框架

Duo 框架建立在以下理论发现上：

给定高斯扩散隐变量 $\mathbf{w}_t \sim \mathcal{N}(\tilde{\alpha}_t \mathbf{x}, (1-\tilde{\alpha}_t^2)\mathbf{I}_K)$，定义 $\mathbf{z}_t = \arg\max(\mathbf{w}_t)$，则 $\mathbf{z}_t$ 服从 Uniform-state 离散扩散：

$$\mathbf{z}_t \sim \text{Cat}(\cdot; \mathcal{T}(\tilde{\alpha}_t)\mathbf{x} + (1-\mathcal{T}(\tilde{\alpha}_t))\mathbf{1}/K)$$

其中 $\mathcal{T}$ 是**扩散变换算子**，将高斯扩散参数 $\tilde{\alpha}_t$ 映射到离散扩散参数 $\alpha_t$。

### 关键设计

1. **扩散对偶性的数学建立**:

    - **边际分布对应**：argmax 将高斯边际映射为分类分布，参数通过 $\mathcal{T}$ 算子关联 
    $\mathcal{T}(\tilde{\alpha}_t) = \frac{K}{K-1}\left[\int_{-\infty}^{\infty} \phi\left(\frac{z-\tilde{\alpha}_t}{\sqrt{1-\tilde{\alpha}_t^2}}\right)\Phi^{K-1}(z)dz - \frac{1}{K}\right]$
    - **转移动力学对应**：离散边际的时间演化满足 Uniform-state 扩散的两两转移矩阵 $Q_t$
    - **ELBO 关系**（Theorem 3.1）：离散扩散的 ELBO **严格紧于**底层高斯扩散的 ELBO
    - **为什么重要**：这意味着离散空间比连续空间更适合建模——在离散空间训练可以获得更紧的似然下界

2. **课程学习加速训练**:

    - 核心思想：用 tempered softmax（$\tau > 0$）替代 argmax（$\tau \to 0$）来松弛离散化
    - 训练损失：
    $\mathcal{L}_{train} = \mathbb{E}_{t, \tilde{q}_t} \sum_{\ell} f_{Duo}(\mathbf{z}_t^\ell := \arg\max(\mathbf{w}_t^\ell), \mathbf{x}_\theta([\text{softmax}(\mathbf{w}_t^{\ell'}/\tau)]^L_{\ell'=1}, t), \alpha_t; \mathbf{x}^\ell)$
    - 课程策略：$\tau = 0.001$（前 500K 步）→ $\tau = 0$（后 500K 步）
    - 同时限制训练时间窗口 $t \in [\beta, \gamma]$，避免梯度信号极弱的区域
    - **为什么有效**：argmax 对小扰动极度敏感——微小的高斯噪声导致 token 剧烈变化。Tempered softmax 保留了更多连续信号，降低去噪难度 → 降低梯度方差（实测降低一个数量级）→ 训练收敛快 2×
    - 额外优化：**Rao-Blackwellized ELBO** 避免材料化 one-hot 向量，减少内存开销并进一步降方差

3. **离散一致性蒸馏（DCD）**:

    - 挑战：离散空间没有 Probability Flow ODE，无法直接用一致性蒸馏
    - 解决方案：利用对偶性，在**高斯空间**中用最优去噪器构建确定性轨迹（Deterministic Discrete Trajectories, DDT）：
    $\mathcal{P}_{DDT}(\mathbf{x}^{1:L}, \epsilon^{1:L}) = \{[\arg\max(\tilde{\alpha}_t \mathbf{x}^\ell + \sqrt{1-\tilde{\alpha}_t^2}\epsilon^\ell)]^L_{\ell=1}\}_{t \in [0,1]}$
    - 蒸馏损失：学生模型 $\mathbf{x}_\theta$ 匹配教师模型 $\mathbf{x}_{\theta^-}$ 在 DDT 相邻点上的预测
    $\mathcal{L}_{DCD} = D_{KL}(\mathbf{x}_\theta(\mathbf{z}_t^{1:L}, t) \| \mathbf{x}_{\theta^-}(\mathbf{z}_s^{1:L}, s))$
    - 蒸馏过程：$N=5$ 轮，每轮 $M=10K$ 步，步长 $\delta$ 每轮翻倍
    - **Greedy-Tail Sampler**：在最后一步用 greedy decoding 替代采样，进一步降低 NFE（1024→8步）
    - **为什么 DDT 有效**：虽然离散空间没有确定的 ODE 轨迹，但通过高斯空间的 ODE + argmax 投影，可以构建一个"伪确定性"轨迹，其中 token 基本只翻转一次，行为类似 MDM 的 carry-over

### 损失函数 / 训练策略

- 基础模型：170M 参数改进 DiT，RoPE 位置编码，AdaLN 时间条件化
- 训练：8×H100，bfloat16，batch size 512，学习率 3e-4，1M 步
- 数据：LM1B（context 128）和 OpenWebText（context 1024）
- 蒸馏：float64 精度采样（避免低精度导致的虚假低 Gen PPL）

## 实验关键数据

### 主实验：语言建模困惑度（LM1B + OWT）

| 方法 | 类型 | LM1B PPL ↓ | OWT PPL ↓ |
|------|------|-----------|-----------|
| Transformer | 自回归 | 22.3 | 17.5 |
| MDLM | 吸收态扩散 | 27.0 | 23.2 |
| SEDD Absorb | 吸收态扩散 | 32.7 | 24.1 |
| SEDD Uniform | 均匀态扩散 | 40.3 | 29.7 |
| UDLM | 均匀态扩散 | 31.3 | 27.4 |
| **Duo (Ours)** | **均匀态扩散** | **29.9** | **25.2** |

Duo 大幅缩小了 USDM 与 MDM/AR 的差距。

### 消融实验

| 配置 | OWT PPL ↓ | 说明 |
|------|----------|------|
| Duo 完整版 | 33.7 (LM1B w/ packing) | 两项优化的合力 |
| 去掉课程学习 | 35.0 | +1.3 PPL，课程学习贡献 ~1.3 |
| 进一步去掉 Rao-Blackwell | 36.7 | +1.7 PPL，改进训练损失贡献 ~1.7 |

### 零样本困惑度（OWT 训练 → 7 个数据集）

| 方法 | PTB | Wiki | LM1B | Lambada | AG News | Pubmed | Arxiv |
|------|-----|------|------|---------|---------|--------|-------|
| AR Transformer | 82.1 | 25.8 | 51.3 | 51.3 | 52.1 | 49.0 | 41.7 |
| MDLM | 95.3 | 32.8 | 67.0 | 47.5 | 61.2 | 41.9 | 37.4 |
| **Duo** | 89.4 | 33.6 | 73.9 | **49.8** | 67.8 | **44.5** | **40.4** |

Duo 在 Lambada、Pubmed、Arxiv **三个数据集上超越 AR 模型**（加粗标注）。

### 蒸馏后采样质量（Gen PPL ↓）

| 采样步数 | MDLM+SDTT | Duo+DCD (Ancestral) | Duo+DCD (Greedy-Tail) |
|---------|-----------|--------------------|-----------------------|
| 1024 | 36.9 | 50.6 | **36.5** |
| 128 | 42.0 | 54.2 | **40.1** |
| 32 | 62.3 | 61.3 | **46.3** |
| 16 | 89.2 | 75.2 | **54.1** |
| 8 | 193.1 | 111.9 | **69.6** |

Duo 在低 NFE 区间显著优于 MDLM——因为 USDM 的自纠错特性在少步时更有价值。

### 关键发现
- **课程学习降低梯度方差一个数量级**：top-100 权重的梯度方差从 ~55 降到 ~0.86（100K 步时）
- **USDM 在少步生成上天然占优**：MDM 一旦 unmask 就不能修改，少步时错误无法纠正；USDM 每步都能更新任何 token
- **DDT 轨迹类似 MDM**：token 基本只翻转一次（先是随机 token，然后变为正确 token），但保持了 USDM 的灵活性
- **直接用去噪权重（非 EMA）做教师效果更好**：与标准一致性模型的做法不同

## 亮点与洞察
- **理论深刻**："数学中的永恒主题——离散性从底层的连续性中涌现"，这一发现优雅地统一了连续和离散扩散
- **实用价值大**：训练 2× 加速 + 采样 128× 加速（1024→8 步），同时性能不降反升
- **USDM 的潜力被重新发现**：之前被认为不如 MDM 的 USDM，在少步采样场景下反而更强
- **Rao-Blackwellized ELBO**：通过解析消除 one-hot 向量物化，既省内存又降方差，是很实用的工程优化

## 局限与展望
- 模型规模仅 170M 参数，尚未在 1B+ 级别验证
- 生成文本质量（Gen PPL ~70）与 AR 模型（~22）仍有差距
- DCD 蒸馏需要多轮训练（5 轮 × 10K 步），额外训练成本不可忽略
- Greedy-Tail sampler 降低了多样性（entropy 从 5.55 降到 5.19~5.30）
- $\mathcal{T}$ 算子需要预计算 100K 个 $(\tilde{\alpha}_t, \mathcal{T}(\tilde{\alpha}_t))$ 对，增加了实现复杂度

## 相关工作与启发
- **MDLM/SEDD/UDLM**：主要对比基线，Duo 从理论角度统一并超越
- **Consistency Models**：高斯扩散蒸馏技术，DCD 是其向离散空间的推广
- **Block Diffusion**：同期工作，在 AR 和扩散之间做插值
- **启发**：对偶性框架可能适用于其他离散生成任务（图、分子、蛋白质）；课程学习策略对任何离散扩散模型都适用

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 对偶性是深刻的理论发现，DCD 填补了离散扩散少步生成的空白
- 实验充分度: ⭐⭐⭐⭐⭐ 全面的似然评估、零样本评估、生成质量评估、消融和蒸馏轮次分析
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨完整，实验分析深入，附录详尽
- 价值: ⭐⭐⭐⭐⭐ 对离散扩散领域有重大推进，打开了从高斯扩散迁移技术的大门

<!-- RELATED:START -->

## 相关论文

- [CoLoGen: Progressive Learning of Concept-Localization Duality for Unified Image Generation](../../CVPR2026/image_generation/cologen_progressive_learning_of_concept-localization_duality_for_unified_image_g.md)
- [Progressive Tempering Sampler with Diffusion](progressive_tempering_sampler_with_diffusion.md)
- [Towards a Mechanistic Explanation of Diffusion Model Generalization](towards_a_mechanistic_explanation_of_diffusion_model_generalization.md)
- [Efficient Diffusion Models for Symmetric Manifolds](efficient_diffusion_models_for_symmetric_manifolds.md)
- [TCP-Diffusion: A Multi-modal Diffusion Model for Global Tropical Cyclone Precipitation Forecasting with Change Awareness](tcp-diffusion_a_multi-modal_diffusion_model_for_global_tropical_cyclone_precipit.md)

<!-- RELATED:END -->
