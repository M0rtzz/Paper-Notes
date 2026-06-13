---
title: >-
  [论文解读] Hierarchical Koopman Diffusion: Fast Generation with Interpretable Diffusion Trajectory
description: >-
  [NeurIPS 2025][图像生成][扩散模型加速] 基于 Koopman 算子理论，将扩散模型的非线性去噪动力学提升到线性 Koopman 空间，通过层次化分解实现一步采样，同时保留中间生成状态的可解释性和可控性。 扩散模型在高保真图像生成上取得了巨大成功，但其采样过程本质上是迭代式的，需要数十甚至上千步去噪…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "扩散模型加速"
  - "一步生成"
  - "Koopman算子"
  - "可解释生成"
  - "层次化动力学"
  - "谱分析"
---

# Hierarchical Koopman Diffusion: Fast Generation with Interpretable Diffusion Trajectory

**会议**: NeurIPS 2025  
**arXiv**: [2510.12220](https://arxiv.org/abs/2510.12220)  
**作者**: Hanru Bai (Fudan University), Weiyang Ding (Fudan University), Difan Zou (The University of Hong Kong)  
**领域**: 图像生成  
**关键词**: 扩散模型加速, 一步生成, Koopman算子, 可解释生成, 层次化动力学, 谱分析  

## 一句话总结

基于 Koopman 算子理论，将扩散模型的非线性去噪动力学提升到线性 Koopman 空间，通过层次化分解实现一步采样，同时保留中间生成状态的可解释性和可控性。

## 研究背景与动机

扩散模型在高保真图像生成上取得了巨大成功，但其采样过程本质上是迭代式的，需要数十甚至上千步去噪，这严重限制了实际应用效率。

现有的加速方案主要分为两大类：

**蒸馏方法**（Knowledge Distillation, Progressive Distillation, Rectified Flow 等）：将预训练扩散模型蒸馏为一步生成器

**一致性模型**（Consistency Models, iCT, ECM 等）：学习时间一致的映射，从噪声直接到干净数据

这些方法虽然实现了一步生成，但本质上都是学习"黑盒"的噪声→图像映射，**完全放弃了扩散模型固有的时序去噪轨迹**。这导致：

- 无法访问中间生成状态
- 丧失了对生成过程的可解释性
- 无法在推理时对特定阶段进行精细干预（如可控编辑）

本文的核心动机：**能否在保持一步采样效率的同时，保留扩散轨迹的可解释性和可控性？** 作者从 Koopman 算子理论出发，提出了一种"显式"的一步生成范式来解决这一矛盾。

## 方法详解

### 整体框架

Hierarchical Koopman Diffusion (HKD) 框架包含三个核心组件：

1. **编码器 $\mathcal{E}_\theta$**：基于 U-Net 下采样结构，将噪声图像 $\boldsymbol{x}_t$ 投影到多尺度 Koopman 空间
2. **层次化 Koopman 动力学模块**：在每个分辨率层级上通过线性算子驱动状态演化
3. **解码器 $\mathcal{D}_\phi$**：将演化后的 Koopman 表示解码回图像空间

工作流程：噪声 $\boldsymbol{x}_T$ → 编码器提取多尺度特征 $\{\boldsymbol{z}_T^{(l)}\}_{l=1}^L$ → Koopman 线性演化 → 解码器输出 $\hat{\boldsymbol{x}}_\epsilon$。

### 关键设计

#### 1. Koopman 算子理论基础

Koopman 算子将非线性动力系统提升到无穷维可观测函数空间，使系统演化在该空间中变为线性。具体地，对于扩散 ODE 的状态演化 $\boldsymbol{x}_{t+\Delta t} = \Phi(\boldsymbol{x}_t)$，存在线性算子 $\mathcal{K}$ 作用于可观测函数 $g$：

$$(\mathcal{K} \circ g)(\boldsymbol{x}) = g(\Phi(\boldsymbol{x}))$$

在连续时间下，Koopman 空间中的动力学简化为线性 ODE：$d\boldsymbol{z}_t / dt = \boldsymbol{A} \boldsymbol{z}_t$，从而拥有闭式解。

#### 2. 层次化 Koopman 子空间

图像生成具有内在的多尺度特性：全局结构先形成、局部纹理后出现。为此，HKD 在不同空间分辨率层级设计独立的 Koopman 子空间：

- 编码器提取 $L$ 个层级的特征 $\boldsymbol{z}_t^{(l)} \in \mathbb{R}^{d_l \times h_l \times w_l}$
- 每个空间位置 $(i,j)$ 处的特征向量 $\boldsymbol{z}_t^{(l)}(i,j)$ 由局部线性算子 $\boldsymbol{A}^{(l)}(i,j)$ 驱动演化
- $\boldsymbol{A}^{(l)}(i,j)$ 采用块对角结构，每个 $2 \times 2$ 块对应一对共轭复数特征值 $\alpha_k^{(l)} \pm i\beta_k^{(l)}$

这种空间自适应设计允许不同区域有不同的时频行为，细粒度建模生成动力学。

#### 3. 闭式一步映射

由于 Koopman 空间中动力学是线性的，时间 $s$ 到 $t$ 的状态映射有显式解：

$$\boldsymbol{z}_t^{(l)}(i,j) = e^{\boldsymbol{A}^{(l)}(i,j)(t-s)} \boldsymbol{z}_s^{(l)}(i,j)$$

推理时直接计算矩阵指数即可完成一步映射，无需迭代积分。

### 损失函数

总训练损失为：

$$\mathcal{L} = \mathcal{L}_{t\text{-consist}} + \mathcal{L}_{\text{recon}}$$

- **轨迹一致性损失 $\mathcal{L}_{t\text{-consist}}$**：从任意中间时间 $t$ 的状态编码并演化到 $\epsilon$，解码后应与真实 $\boldsymbol{x}_\epsilon$ 一致。这是相比隐式方法的核心优势——显式监督中间状态
- **重建损失 $\mathcal{L}_{\text{recon}}$**：直接监督从噪声 $\boldsymbol{x}_T$ 到干净图像 $\boldsymbol{x}_\epsilon$ 的一步映射
- 距离度量 $d(\cdot, \cdot) = \lambda_1 \mathcal{L}_{\text{MSE}} + \lambda_2 \mathcal{L}_{\text{LPIPS}}$，$\lambda_1$ 退火从粗匹配过渡到感知精细化

值得注意的是，作者在理论上证明：在结构化假设下，图像空间中最小化轨迹一致性损失等价于在 Koopman 潜空间中最小化。选择图像空间是因为可以利用 LPIPS 等感知度量，梯度信号更有效。

## 训练与推理

### 训练流程

1. 编码器和解码器使用 EDM 预训练的 U-Net 权重初始化
2. Koopman 矩阵 $\boldsymbol{A}^{(l)}$ 初始化为零矩阵（即初始时 Koopman 演化为恒等映射）
3. 每次迭代均匀采样 $s-1$ 个中间时间点加上终端 $T$，对所有时间点计算轨迹一致性损失
4. 使用 Adam 优化器，学习率 $1 \times 10^{-3}$，权重衰减 0.95
5. 端到端训练编码器 $\mathcal{E}_\theta$、解码器 $\mathcal{D}_\phi$ 和所有 Koopman 矩阵 $\{\boldsymbol{A}^{(l)}\}$
6. 训练数据来自预训练扩散模型生成的 ODE 轨迹 $\{\boldsymbol{x}_t\}_{t \in [0,T]}$

### 推理（一步采样）

推理极其简洁：给定噪声 $\boldsymbol{x}_T \sim \mathcal{N}(0, I)$，一步计算：

$$\hat{\boldsymbol{x}}_\epsilon = \mathcal{D}_\phi(\{e^{(\epsilon - T)\boldsymbol{A}^{(l)}} \mathcal{E}_\theta^{(l)}(\boldsymbol{x}_T)\}_{l=1}^L)$$

整个过程：编码 → 矩阵指数乘法 → 解码，无需迭代。

### 训练效率

- 8×V100 GPU 上 2-3 天完成训练
- 远快于一致性模型训练（8 GPU 一周以上）
- 训练更稳定：Koopman 空间的指数形式在谱幅度接近 1 时保证充足梯度（避免一致性模型的高方差梯度问题），多时间点监督稳定谱估计

## 实验关键数据

### 主实验：CIFAR-10 一步生成

| 方法 | 类别 | NFE | FID ↓ |
|:---|:---|:---:|:---:|
| DDPM | 多步扩散 | 1000 | 3.17 |
| EDM | 多步扩散 | 35 | 1.97 |
| DDIM | 多步扩散 | 10 | 13.36 |
| KD | 蒸馏 | 1 | 9.36 |
| PD | 蒸馏 | 1 | 8.34 |
| DMD | 蒸馏 | 1 | 3.77 |
| 2-Rectified Flow++ | 蒸馏 | 1 | 3.38 |
| CT (LPIPS) | 一致性 | 1 | 8.70 |
| CD (LPIPS) | 一致性蒸馏 | 1 | 3.55 |
| iCT-deep | 一致性 | 1 | 2.51 |
| ECM | 一致性 | 1 | 3.60 |
| **HKD (本文)** | **Koopman** | **1** | **3.30** |

### 主实验：FFHQ 64×64

| 方法 | NFE | FID ↓ |
|:---|:---:|:---:|
| EDM | 79 | 2.47 |
| EDM | 15 | 9.85 |
| ECM | 1 | 5.99 |
| **HKD (本文)** | **1** | **5.70** |

### 消融实验

| Koopman 演化 | 轨迹一致性损失 | 层次化设计 | FID ↓ |
|:---:|:---:|:---:|:---:|
| ✗ | ✗ | ✓ | 5.72 |
| ✓ | ✗ | ✓ | 5.57 |
| ✓ | ✓ | ✗ | 4.78 |
| ✓ | ✓ | ✓ | **3.30** |

### 关键发现

1. **FID 3.30** 在 CIFAR-10 一步生成上优于大部分蒸馏方法和 ECM，接近 iCT-deep (2.51) 但后者依赖大量超参调优且训练不稳
2. **FFHQ 上 FID 5.70** 优于 ECM (5.99)，验证了框架在更复杂数据集上的有效性
3. 消融实验显示三个组件均有显著贡献：Koopman 演化 (5.72→5.57)、轨迹一致性损失 (5.57→4.78)、层次化设计 (4.78→3.30)
4. 训练仅需 8×V100 GPU 2-3 天，远快于一致性模型 (8 GPU 一周)，且训练更稳定

### Koopman 谱分析实验

作者通过分析学到的 Koopman 矩阵 $\boldsymbol{A}$ 的谱结构，揭示了生成过程的语义层次：

- **谱掩码实验**：在每个分辨率层级，按特征值实部大小排序，选择性保留特定谱段（最小/中间/最大），其余置零后解码
    - 低频谱模式 → 全局结构（大致轮廓和背景）
    - 中频谱模式 → 整体形状和姿态
    - 高频谱模式 → 局部纹理和精细细节
- **累积效应 (CE) 追踪**：随时间观察各谱分量对重建的贡献，量化不同模式的演化过程

### 一步图像编辑实验

利用框架的可解释性，作者在扩散轨迹的中间状态进行频率感知干预：

- 在 Koopman 轨迹中间点，将参考图像的高频特征以不同混合比（10%-90%）注入生成图像的下半部分
- 随注入比例增大，参考图像的面部细节逐渐显现，证明频率分解建立了有意义的对应关系
- 对比实验：频率无关编辑（混合所有频段）会破坏全局结构，缺乏解耦控制
- 还在 CIFAR-10 上进行了 inpainting 和 coloring 的图像修复实验，沿轨迹迭代混合参考图像与生成图像

## 亮点与洞察

1. **理论创新**：首次将 Koopman 算子理论引入图像生成，提供了扩散模型动力学的全新数学视角。理论证明 HKD 的表示能力不弱于黑盒一步映射：$err_{\text{HKD}} \leq err_{\text{one-step}} + O(\kappa)$
2. **可解释性**：通过 Koopman 谱分析揭示了生成过程中的语义层次——低频谱模式对应全局结构，中频对应整体形状和姿态，高频对应局部纹理细节
3. **可控编辑**：基于频率感知的中间状态干预，实现了一步生成中的图像编辑（如将参考图像的高频特征注入生成图像的指定区域），这是隐式一步方法无法实现的
4. **训练稳定性**：Koopman 空间的指数形式确保谱幅度接近 1 时仍有足够梯度，避免一致性模型常见的高方差梯度问题；多时间点轨迹监督进一步稳定谱估计
5. **优雅的架构设计**：复用 EDM 的 U-Net 作为编解码器骨架，Koopman 动力学作为中间桥梁，设计简洁且易于实现

## 局限性

1. **生成质量仍有差距**：FID 3.30 与 iCT-deep (2.51) 仍有差距，未使用对抗训练或高级训练技巧
2. **数据集规模有限**：仅在 CIFAR-10 (32×32) 和 FFHQ (64×64) 上验证，未在高分辨率数据集（如 ImageNet 256×256）上测试
3. **Koopman 空间假设**：有限维 Koopman 空间近似可能在更复杂的数据分布上表现不足
4. **编辑能力初步**：频率感知编辑仅为概念验证，未扩展到文本引导编辑或属性级控制
5. **缺少条件生成的全面评估**：虽附录提供了条件生成结果，但主实验以无条件生成为主

## 未来方向（作者讨论）

- **对抗训练整合**：引入 adversarial learning 可能进一步缩小与 iCT-deep 的差距
- **高分辨率生成**：层次化设计天然适合高分辨率扩展，但尚未实验验证
- **语义编辑扩展**：显式谱分解支持可解释干预，可扩展至 text-guided editing、attribute-specific control 等更丰富的语义编辑任务
- **Koopman 动力学的通用性**：框架有望推广到视频生成（时空多尺度）和 3D 生成等领域

## My Notes

### 核心贡献评价

- **首次将 Koopman 算子理论引入图像生成**，这不是简单的"套理论"，而是提供了真正有用的框架：线性化后不仅能一步采样，还能做谱分析和中间状态干预
- **理论证明**（Thm 3.1）给出了 HKD 误差不超过黑盒一步方法 + 小量 $O(\kappa)$ 的保证，$\kappa$ 随数据集 $N$ 和 Koopman 维度 $m$ 增大而趋零，为方法的可行性提供了理论背书
- **"显式"vs"隐式"一步生成**的区分很有洞察力：蒸馏和一致性模型都是隐式的噪声→图像黑盒映射，HKD 是首个"白盒"一步方法

### 与相关工作的对比思考

- **与蒸馏方法的对比**：蒸馏方法 (KD, PD, DMD, Rectified Flow) 学习黑盒映射，HKD 提供了一种"白盒"替代方案，在保持竞争性能的同时赋予可解释性
- **与一致性模型的对比**：一致性模型训练不稳定且对超参敏感，HKD 通过 Koopman 线性结构天然更稳定
- **Koopman 在其他领域的成功**：时间序列分析 (Koopa, KoNODE)、动力系统控制等，本文是首次应用于图像生成

### 可能的局限与瓶颈

- 空间自适应的 $\boldsymbol{A}^{(l)}(i,j)$ 意味着每个空间位置独立参数化，参数量随分辨率平方增长，高分辨率场景下可能需要参数共享策略
- 有限维 Koopman 近似在数据分布极复杂时的表现值得进一步考察
- 编辑能力展示的是频率层面的干预，距离真正实用的语义编辑（如"改变发色"）仍有距离

### 启发

- Koopman 理论为"从非线性到线性"提供了原理性工具，这一思路可能推广到视频生成（时空多尺度）、3D 生成等更复杂的生成任务
- 层次化线性空间设计思路可能启发其他领域的多尺度建模（如多尺度 NeRF、层次化 flow matching）
- 轨迹一致性损失的思想——在中间状态施加监督——可以迁移到其他一步生成框架中增强稳定性

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次将 Koopman 算子引入图像生成，框架设计新颖且有理论支撑
- **实验充分度**: ⭐⭐⭐ — 数据集规模和分辨率有限（仅 CIFAR-10 和 FFHQ 64×64），但消融和分析实验较充分
- **写作质量**: ⭐⭐⭐⭐ — 理论严谨，框架描述清晰，谱分析可视化直观
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Emergence and Evolution of Interpretable Concepts in Diffusion Models](emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)
- [\[NeurIPS 2025\] State-Covering Trajectory Stitching for Diffusion Planners](state-covering_trajectory_stitching_for_diffusion_planners.md)
- [\[AAAI 2026\] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](../../AAAI2026/image_generation/hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)
- [\[NeurIPS 2025\] Fast Solvers for Discrete Diffusion Models: Theory and Applications of High-Order Algorithms](fast_solvers_for_discrete_diffusion_models_theory_and_applications_of_high-order.md)
- [\[NeurIPS 2025\] Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders](learning_interpretable_features_in_audio_latent_spaces_via_sparse_autoencoders.md)

</div>

<!-- RELATED:END -->
