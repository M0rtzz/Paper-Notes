---
title: >-
  [论文解读] Offline Preference Optimization for Rectified Flow with Noise-Tracked Pairs
description: >-
  [ICML 2026][图像生成][Rectified Flow] 本文针对 rectified flow（RF）类文生图模型，提出 PNAPO——一种把"生成时用的先验噪声"和"赢者/输者图片"一起保存为六元组的离线偏好优化框架，配合 RF 直线轨迹假设做轨迹估计和动态正则系数调度…
tags:
  - "ICML 2026"
  - "图像生成"
  - "Rectified Flow"
  - "扩散模型"
  - "偏好优化"
  - "先验噪声"
  - "动态正则"
---

# Offline Preference Optimization for Rectified Flow with Noise-Tracked Pairs

**会议**: ICML 2026  
**arXiv**: [2605.09433](https://arxiv.org/abs/2605.09433)  
**代码**: 无（论文未公开仓库链接）  
**领域**: 对齐 RLHF / 扩散模型 / 文生图  
**关键词**: Rectified Flow、Diffusion-DPO、偏好优化、先验噪声、动态正则

## 一句话总结
本文针对 rectified flow（RF）类文生图模型，提出 PNAPO——一种把"生成时用的先验噪声"和"赢者/输者图片"一起保存为六元组的离线偏好优化框架，配合 RF 直线轨迹假设做轨迹估计和动态正则系数调度，相比 Diffusion-DPO 在 SD3-M/FLUX 上同时提点又把训练算力降到 1/12。

## 研究背景与动机

**领域现状**：文生图（T2I）后训练对齐的主流做法是收集 (prompt, winner, loser) 三元组偏好数据，然后用 RL（DDPO、DPOK）或者 RL-free 的 DPO 风格目标（Diffusion-DPO、D3PO 等）让生成器更倾向赢者。RL-free 因为稳定简单更受欢迎。

**现有痛点**：现有偏好数据集（Pick-a-Pic、HPDv2、ImageReward 等）只保存最终图像，丢掉了"生成这张图所用的先验噪声"——但扩散/流模型的生成本质上是从某个特定噪声出发的轨迹过程。Diffusion-DPO 等方法只能用独立采样的前向噪声去近似反向轨迹，对真正的反向动力学是错配的，导致训练不稳定、信用分配低效。

**核心矛盾**：在标准扩散模型里反向轨迹是随机且弯曲的，给定端点采样精确反向路径是不可解的。但 RF 不一样——RF 的训练目标本来就是把数据-噪声耦合"拉直"成接近直线的轨迹，先验噪声直接决定了一条轨迹。所以"丢弃先验噪声"在 RF 上是一个比在普通扩散模型上更严重的损失。

**本文目标**：（1）让偏好数据保留先验噪声；（2）设计与 RF 几何一致的 DPO 风格目标；（3）解决 DPO 训练后期固定 $\beta$ 导致更新过弱、对所有样本一视同仁的两个老问题。

**切入角度**：作者观察到 RF 的关键性质：$\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_T$ 是端点之间的直线插值。如果数据集里同时存有 $\boldsymbol{x}_0$ 和 $\boldsymbol{x}_T$，那么中间状态可以直接由插值估计，根本不需要额外加噪。这把不可解的反向采样降级成一次线性插值，方差骤减。

**核心 idea**：把偏好三元组扩展为六元组 $(\boldsymbol{c}, \boldsymbol{x}_0^w, \boldsymbol{x}_0^l, \boldsymbol{x}_T^w, \boldsymbol{x}_T^l)$ 并加一个连续奖励差 $\delta r$，用 RF 直线插值估中间状态，加上由奖励差和训练步数共同调度的动态 $\beta$。

## 方法详解

### 整体框架
PNAPO 是离线、off-policy 的 RL-free 对齐管线，三步走：（1）**数据构造**——用 RF 基础模型对每个 prompt 采两个先验噪声 → 生成图像对 → 用 HPSv2.1 奖励模型打分 → 得到六元组 + 连续奖励差 $\delta r$；（2）**轨迹估计**——利用 RF 直线性质，从存储的 $(\boldsymbol{x}_0^*, \boldsymbol{x}_T^*)$ 端点对用 $\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_T$ 直接插值出中间态，跳过任何重采样；（3）**优化**——用 RF 一致的 PNAPO 目标 + 动态 $\beta(\delta r, n)$ 调度做 LoRA 风格更新，参考模型 $v_{\text{ref}}$ 冻结。

### 关键设计

1. **先验噪声追踪的偏好六元组**:

    - 功能：把传统三元组 $(\boldsymbol{c}, \boldsymbol{x}_0^w, \boldsymbol{x}_0^l)$ 扩展为六元组，附带 $\boldsymbol{x}_T^w, \boldsymbol{x}_T^l$ 和奖励差 $\delta r$，让 DPO 损失的轨迹估计能从端点条件出发。
    - 核心思路：用 DiffusionDB 选 20k 高质量 prompt（NSFW 过滤 → Jaccard/CLIP 去重 → 100 KNN 聚类重采样），每个 prompt 用 RF 基础模型采两次噪声 → 两张图，HPSv2.1 评分给出 $\delta r = r_\theta(\boldsymbol{x}_0^w) - r_\theta(\boldsymbol{x}_0^l)$。注意这里用模型自己采的图（off-policy 但同模型族），保证 noise 和模型策略一致。
    - 设计动机：传统数据集丢弃噪声，导致 DPO 必须从独立 $\boldsymbol{x}_T^* \sim \mathcal{N}(0, I)$ 重采样去估反向过程，引入与训练实际不匹配的方差源。保留噪声后，$p_\theta(\boldsymbol{x}_T^* | \boldsymbol{x}_0^*)$ 被显式保留，等价于把决策空间从"所有可能轨迹"缩小到"实际产生这张图的那条轨迹"。

2. **RF 一致的轨迹估计与目标函数**:

    - 功能：把不可解的 $p_\theta(\boldsymbol{x}_{1:T-1} | \boldsymbol{x}_0)$ 用 $p_\theta(\boldsymbol{x}_T | \boldsymbol{x}_0) q(\boldsymbol{x}_{1:T-1} | \boldsymbol{x}_0, \boldsymbol{x}_T)$ 近似，证明这个近似在 RF 上是更紧的 surrogate。
    - 核心思路：经过 Jensen 不等式和 KL 分解后，损失简化为 $\mathcal{L}_{\text{PNAPO}}(\theta) = -\mathbb{E}_{(\boldsymbol{c}, \boldsymbol{x}_0^w, \boldsymbol{x}_0^l, \boldsymbol{x}_T^w, \boldsymbol{x}_T^l), t} \log \sigma(-\beta(\boldsymbol{s}_\theta^t(\boldsymbol{x}_0^w, \boldsymbol{x}_T^w, \boldsymbol{c}) - \boldsymbol{s}_\theta^t(\boldsymbol{x}_0^l, \boldsymbol{x}_T^l, \boldsymbol{c})))$，其中 $\boldsymbol{s}_\theta^t(\boldsymbol{x}_0^*, \boldsymbol{x}_T^*, \boldsymbol{c}) = \|(\boldsymbol{x}_T^* - \boldsymbol{x}_0^*) - v_\theta(\boldsymbol{x}_t^*, t, \boldsymbol{c})\|^2_2 - \|(\boldsymbol{x}_T^* - \boldsymbol{x}_0^*) - v_{\text{ref}}(\boldsymbol{x}_t^*, t, \boldsymbol{c})\|^2_2$，其中 $\boldsymbol{x}_t^* = (1-t)\boldsymbol{x}_0^* + t\boldsymbol{x}_T^*$。目标本质是让 $v_\theta$ 在赢者轨迹上比 ref 更准、在输者轨迹上比 ref 更差。
    - 设计动机：作者形式化证明 $D_{KL}(p_\theta(\boldsymbol{x}_T|\boldsymbol{x}_0) q(\boldsymbol{x}_{1:T-1}|\boldsymbol{x}_0, \boldsymbol{x}_T) \| p_\theta(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)) \leq D_{KL}(q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0) \| p_\theta(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0))$，说明 PNAPO 的轨迹近似严格优于 Diffusion-DPO 的前向加噪近似。类比 RL 里的稀疏奖励问题，决策空间缩小直接降低梯度方差、加速训练。

3. **基于奖励差和训练进度的动态 $\beta$ 调度**:

    - 功能：让正则强度 $\beta$ 自动响应"样本难度"（赢/输奖励差）和"训练阶段"，缓解固定 $\beta$ 后期把模型拉回参考模型的问题。
    - 核心思路：$\beta(\delta r, n) = \beta \cdot f(\delta r) \cdot g(n)$，其中 $f(\delta r) = 2\sigma(\delta r) - 1$ 是单调递增到 1 的样本控制器；$g(n)$ 是退火因子——前 $n_1$ 步保持 1，$n_1$ 到 $n_2$ 间按余弦下降到 $1/2$，之后保持 $1/2$。当 margin 为负时增大 $\delta r$ 抬高 $\beta$ 加速对齐；margin 转正后效果反转给出更柔和的更新。
    - 设计动机：通过对 $\nabla_\theta \mathcal{L}_{\text{PNAPO}}$ 的梯度分解，作者发现固定 $\beta$ 有两个问题——对所有图对无差别加权（忽略难度），训练后期强正则把模型拉回 ref。动态 $\beta$ 让奖励差大的对子（"明显更好"）权重更大，又让训练后期允许更多偏离 ref。

### 损失函数 / 训练策略
核心损失就是 PNAPO 目标函数。优化器 AdamW，学习率 $1\mathrm{e}{-6}$；FLUX 的 $\beta=2000$，SD3-M 的 $\beta=5000$。20k prompt × 每 prompt 2 图，Euler 离散调度器、50 步、guidance scale=1。8× NVIDIA H800 GPU。

## 实验关键数据

### 主实验
基线包括原模型、SFT、Diffusion-DPO、IPO、CaPO，全部用相同超参数和模型配置复现以保证公平。在 HPDv2（3200 prompt）和 OPDv1（7459 prompt）上评 PickScore、HPSv2.1、ImageReward、LAION Aesthetic、CLIP；在 GenEval 评对象生成对齐。

| 测试集 / 模型 | 指标 | 原模型 | DPO | PNAPO | 提升 |
|--------------|------|--------|-----|-------|------|
| OPDv1 SD3-M | HPSv2.1 | 31.96 | 32.39 | 33.09 | +1.13 (vs base) |
| OPDv1 FLUX | HPSv2.1 | 30.74 | 30.79 | 32.10 | +1.36 (vs base) |
| OPDv1 FLUX | ImageReward | 1.202 | 1.209 | 1.238 | +0.036 |
| OPDv1 FLUX | Aesthetic | 6.550 | 6.548 | 6.692 | +0.142 |
| GenEval SD3-M | Overall | 0.68 | — | 0.73 | +7.4% 相对 |
| GenEval FLUX | Overall | 0.65 | 0.66 | 0.69 | +6.2% 相对 |
| HPSv2.1 胜率 FLUX | PNAPO vs DPO | — | — | 84.6% | — |

### 消融实验
训练算力对比（NVIDIA H800 GPU-Hours）：

| 模型 | Diffusion-DPO | PNAPO | 节省 |
|------|--------------|-------|------|
| SD3-M | ~249.6 | ~20.8 | 12× |
| FLUX | ~422.4 | ~35.2 | 12× |

用户研究（10 名参与者，20 对图，PNAPO-FLUX vs baselines）：

| 评估维度 | PNAPO 偏好率 |
|---------|------------|
| 整体偏好 | 56% |
| 视觉吸引力 | 72% |
| 文本-图像对齐 | 52% |

### 关键发现
- **质量与算力双赢**：在所有指标都超过 Diffusion-DPO 的同时把 GPU 时间砍到 1/12，验证了"轨迹估计变紧"直接带来的训练效率提升。
- **背景模糊问题被缓解**：FLUX 标志性的背景模糊瑕疵在 PNAPO 下显著减少，质化结果显示文字渲染和构图也变好。
- **跨架构泛化**：在 RF 家族两个不同骨架（SD3-M / FLUX）上一致提升，说明方法依赖的是 RF 几何性质而不是特定模型。
- **CLIP 文本-图像对齐**：FLUX 上从 35.97 提到 36.89，证明动态 $\beta$ 没有牺牲文本对齐去换美学。

## 亮点与洞察
- **从"丢弃噪声"到"追踪噪声"的范式翻转**：以前的偏好数据集都只存图，本文指出对 RF 而言噪声是轨迹身份的一部分——这是一个被长期忽视的"免费午餐"，只要在数据构造阶段额外存一份噪声 tensor 就能拿到 12× 算力节省，性价比极高。
- **几何一致的近似带来理论保证**：作者用 KL 链式不等式严格证明 PNAPO 的轨迹近似比 Diffusion-DPO 更紧，把"更好"从经验观察提升到理论结果，少见的扎实。
- **动态 $\beta$ 的两个相互独立的因子设计**：$f(\delta r)$ 管样本难度，$g(n)$ 管训练进度，解耦得很干净，可以独立组合到其他 DPO 变体上（D3PO、IPO、Diffusion-KTO 都能套）。
- **离线 RL-free 的工程友好性**：相比 GRPO 类在线 RL 方法，PNAPO 只需要采一次数据然后稳定离线训练，对生产环境算力/调度限制更友好。

## 局限与展望
- **依赖奖励模型**：用 HPSv2.1 当伪人类标注，奖励模型本身的偏见和盲点会被放大；论文没讨论 reward hacking 风险。
- **只覆盖 RF 类模型**：核心机制（直线插值）严格依赖 RF 的轨迹直线性，对纯 DDPM/DDIM 不能直接迁移；作者也明确把适用范围限定在 RF。
- **数据规模较小**：20k prompt 在 T2I 偏好数据集里偏少，规模化到 100k+ 时动态 $\beta$ 调度是否仍然稳定还需要验证。
- **没有与在线 RL 比对**：论文定位为 RL-free 的补充方案，但缺少和 GRPO 系列方法的同算力公平对比，离线/在线收益的真实差距没量化。
- **超参 $n_1, n_2$ 需手调**：余弦退火的两个阈值靠经验设置，不同模型/数据集需要重新调，没有自适应方案。

## 相关工作与启发
- **vs Diffusion-DPO (Wallace 2024)**：核心思路类似但用前向加噪近似反向轨迹；PNAPO 在 RF 上证明轨迹近似更紧、训练快 12×，且明确利用 RF 几何。
- **vs D3PO (Yang 2024)**：D3PO 用迭代反向过程估计每步偏好，计算昂贵；PNAPO 用插值跳过反向过程，效率更高。
- **vs SPO / InPO / SmPO**：这些方法在去噪全程对齐偏好，需要 DDIM Inversion；PNAPO 端到端直接用存储的噪声，工程更简单。
- **vs Diffusion-NPO / Self-NPO**：从 CFG 角度训"负样本模型"做引导；PNAPO 是正面更新，思路互补，可以结合。
- **vs GRPO 系列（在线 RL）**：高对齐但需大量在线采样和细调；PNAPO 走"采一次离线训练"路线，在算力/工程约束下更实用。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "存噪声"这个数据结构改动看起来简单，但配合 RF 几何分析直击 Diffusion-DPO 的方差源头，思路漂亮。
- 实验充分度: ⭐⭐⭐⭐ 双模型两数据集多指标 + 用户研究 + GPU-Hours 对比，唯一缺失是与在线 RL 方法的对比和大规模数据验证。
- 写作质量: ⭐⭐⭐⭐ 推导清楚，从动机到目标函数到动态 $\beta$ 一气呵成，公式记号略密。
- 价值: ⭐⭐⭐⭐⭐ 对 RF 类 T2I 后训练是即插即用且节省一个数量级算力的方法，工程价值大。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Straighten Viscous Rectified Flow via Noise Optimization](../../ICCV2025/image_generation/straighten_viscous_rectified_flow_via_noise_optimization.md)
- [\[AAAI 2026\] Rectified Noise: A Generative Model Using Positive-incentive Noise](../../AAAI2026/image_generation/rectified_noise_a_generative_model_using_positive-incentive_noise.md)
- [\[ICLR 2026\] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](../../ICLR2026/image_generation/flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)
- [\[NeurIPS 2025\] GuideFlow3D: Optimization-Guided Rectified Flow For Appearance Transfer](../../NeurIPS2025/image_generation/guideflow3d_optimization-guided_rectified_flow_for_appearance_transfer.md)
- [\[NeurIPS 2025\] Efficient Rectified Flow for Image Fusion](../../NeurIPS2025/image_generation/efficient_rectified_flow_for_image_fusion.md)

</div>

<!-- RELATED:END -->
