---
title: >-
  [论文解读] ReAlign: Text-to-Motion Generation via Step-Aware Reward-Guided Alignment
description: >-
  [AAAI 2026][图像生成][文本生成动作] 提出 ReAlign（Reward-guided sampling Alignment），通过步感知（step-aware）奖励模型和奖励引导采样策略，在扩散推理过程中动态引导采样轨迹朝向文本-动作高对齐的分布，无需微调任何扩散模型即可显著提升多种动作生成方法的质量。以 MLD 为例，R@1 提升 17.9%，FID 改善 58.8%。
tags:
  - AAAI 2026
  - 图像生成
  - 文本生成动作
  - 扩散模型
  - 奖励引导采样
  - 文本-动作对齐
  - 即插即用
---

# ReAlign: Text-to-Motion Generation via Step-Aware Reward-Guided Alignment

**会议**: AAAI 2026  
**arXiv**: [2511.19217](https://arxiv.org/abs/2511.19217)  
**代码**: [https://wengwanjiang.github.io/ReAlign-page](https://wengwanjiang.github.io/ReAlign-page)  
**领域**: 图像生成  
**关键词**: 文本生成动作, 扩散模型, 奖励引导采样, 文本-动作对齐, 即插即用

## 一句话总结

提出 ReAlign（Reward-guided sampling Alignment），通过步感知（step-aware）奖励模型和奖励引导采样策略，在扩散推理过程中动态引导采样轨迹朝向文本-动作高对齐的分布，无需微调任何扩散模型即可显著提升多种动作生成方法的质量。以 MLD 为例，R@1 提升 17.9%，FID 改善 58.8%。

## 研究背景与动机

### 领域现状

文本-动作生成（Text-to-Motion Generation）旨在根据自然语言描述合成 3D 人体动作，在游戏、影视、机器人等领域有广泛应用。扩散模型（如 MDM、MLD、MotionDiffuse 等）已成为主流方法，能生成多样且逼真的动作。

### 现有痛点

**文本-动作对齐不足**：扩散模型依赖 CLIP 编码文本嵌入，但 CLIP 是在文本-图像对上训练的，天然不适合捕捉文本与动作之间的语义对齐。结果是生成的动作经常与输入描述不一致（如描述"向右走"却生成"向左走"）。

**高概率密度 ≠ 语义一致性**：扩散采样过程中，分数函数 $\nabla \log p_t(\mathbf{x})$ 引导采样点朝向高密度区域，但这些区域不一定与文本语义一致。模型倾向于生成"概率高但语义偏"的动作。

**先前对齐方法的局限**：基于强化学习的方法（ReinDiffuse、MotionRL、SoPo）聚焦于微调生成模型来对齐偏好或提升质量，但它们无法处理噪声动作输入，且对齐问题应在去噪过程中解决而非事后校正。

**动作生成与检索割裂**：两个相关任务往往独立研究，缺乏统一模型或互相增强的探索。

### 核心矛盾

扩散采样分布 $p_t(\mathbf{x})$ 优先追求高概率密度而忽视语义保真度。如何在不改变扩散模型本身的情况下，让采样轨迹同时兼顾概率密度和语义对齐？

### 本文切入角度

**核心 idea**：构造一个理想采样分布 $p_t^I(\mathbf{x}|c) = p_t(\mathbf{x}|c) \cdot p_t^r(\mathbf{x}|c) / Z(c)$，将原始采样分布与基于奖励的对齐分布相乘。通过训练一个步感知奖励模型来估计奖励分布，并在推理时将奖励梯度注入反向 SDE，引导采样朝向语义和动作质量的双重最优。方法完全即插即用，无需微调任何扩散模型。

## 方法详解

### 整体框架

ReAlign 由两个核心组件构成：

1. **步感知奖励模型（Step-Aware Reward Model）**：适应不同噪声水平的文本-动作对齐评估
2. **奖励引导采样策略（Reward-Guided Sampling）**：将奖励梯度注入扩散去噪过程

### 关键设计

#### 1. **步感知奖励模型**

**功能**：在不同去噪步骤（不同噪声水平）下准确评估噪声动作与文本的对齐程度。

**问题**：现有对齐模型（如 TMR、LaMP）假设输入是干净无噪声的动作，无法处理去噪过程中的噪声动作。而去噪过程正是需要引导的关键时刻。

**核心思路**：引入时间步 token $[e_t]$，将其与动作帧序列 $[x_t^1, x_t^2, \ldots, x_t^N]$ 拼接为 $[e_t, x_t^1, x_t^2, \ldots, x_t^N]$，送入 Transformer 编码器。这使模型能在处理动作动态的同时适应不同噪声水平。

训练时对动作添加不同步骤 $t$ 的噪声，使用两个互补损失优化：
$$\mathcal{L}_{RM}(\varphi; \mathbf{x}_t, c) = \mathcal{L}_C(\varphi; \mathbf{x}_t, c) + \mathcal{L}_R(\varphi; \mathbf{x}_t, c)$$

- $\mathcal{L}_C$：对比损失，确保文本-动作检索的准确性
- $\mathcal{L}_R$：表示损失，学习有意义的动作嵌入

训练后奖励为余弦相似度：$R_\varphi(\mathbf{x}, c) = \cos(\mathbf{z_x}, \mathbf{z}_c)$

**设计动机**：去噪过程中从纯噪声到清晰动作，噪声水平不断变化。传统对齐模型在高噪声步骤完全失效。时间步 token 让模型"知道"当前噪声程度，从而在每个步骤都给出准确的对齐评估。

#### 2. **动作-动作奖励（Motion-to-Motion Reward）**

**功能**：确保生成动作与真实世界动作模式一致,弥补文本描述的歧义性。

**核心思路**：用步感知奖励模型从训练集中检索与文本条件最匹配的参考动作：
$$\mathbf{x}^c = \arg\max_{\mathbf{x} \in \mathcal{D}_{tr}} R_\varphi(\mathbf{x}, c)$$

然后计算生成动作与参考动作的嵌入余弦相似度：
$$R_m(\mathbf{x}_t, c) = \cos(\mathbf{z_x}, \mathbf{z_{x^c}})$$

参考动作 $\mathbf{x}^c$ 作为动态锚点，确保生成动作忠实于文本隐含的真实动作模式。

**双重对齐奖励**：
$$R(\mathbf{x}_t, c) = \mu R_\varphi(\mathbf{x}_t, c) + \eta R_m(\mathbf{x}_t, c)$$

**奖励分布**：$p_t^r(\mathbf{x}_t|c) = \exp(R(\mathbf{x}_t, c)) / Z^r(c)$

#### 3. **奖励引导采样**

**功能**：将奖励分布整合进扩散反向过程，修改采样轨迹朝向理想分布。

**理论推导**（Theorem 1-3）：

定义理想分布 $p_t^I(\mathbf{x}|c) = p_t(\mathbf{x}|c) \cdot p_t^r(\mathbf{x}|c) / Z(c)$，代入反向 SDE 得：

$$\mathbf{dx} = [\mathbf{f}(\mathbf{x},t) - g(t)^2 \nabla(\log p_t(\mathbf{x}|c) + \log p_t^r(\mathbf{x}|c))] dt + g(t) d\mathbf{w}$$

离散化到 DDPM 框架下（Theorem 3）：
$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\bar{\mathbf{x}}_{t-1} + \sqrt{\beta_t}\epsilon\right) + \frac{\beta_t}{\sqrt{\alpha_t}} \nabla R(\mathbf{x}_t, c)$$

为了采样稳定性，移除权重 $\frac{\beta_t}{\sqrt{\alpha_t}}$，得到最终实用公式：
$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\bar{\mathbf{x}}_{t-1} + \sqrt{\beta_t}\epsilon\right) + \nabla R(\mathbf{x}_t, c)$$

奖励梯度在每个去噪步骤中引导采样点同时朝向高密度和高对齐方向。

### 损失函数 / 训练策略

**奖励模型训练**：
- 架构：SkipTransformer（9 层，4 头，潜在维度 256）
- 最大时间步 1000，噪声动作概率 0.5
- AdamW，lr $10^{-4}$，batch size 512
- 训练遵循 TMR 框架

**推理**：奖励模型即插即用，与 CFG 结合使用，零额外训练。

## 实验关键数据

### 主实验

**HumanML3D 数据集上的文本-动作生成**（ReAlign 应用于 MLD 和 MLD++）：

| 方法 | R@1 ↑ | R@3 ↑ | FID ↓ | MM Dist ↓ | Diversity → |
|------|-------|-------|------|-----------|-------------|
| Real | 0.511 | 0.797 | 0.002 | 2.974 | 9.503 |
| MLD | 0.481 | 0.772 | 0.473 | 3.196 | 9.724 |
| **MLD + ReAlign** | **0.567 (+17.9%)** | **0.848 (+9.8%)** | **0.195 (+58.8%)** | **2.704 (+15.4%)** | 9.474 |
| MLD++ | 0.548 | 0.829 | 0.073 | 2.810 | 9.658 |
| **MLD++ + ReAlign** | **0.572 (+4.4%)** | **0.852 (+2.8%)** | **0.055 (+24.7%)** | **2.648 (+5.8%)** | 9.478 |

**KIT-ML 数据集**（ReAlign 应用于 MDM）：

| 方法 | R@1 ↑ | R@3 ↑ | FID ↓ | MM Dist ↓ |
|------|-------|-------|------|-----------|
| MDM | 0.403 | 0.731 | 0.497 | 3.096 |
| **MDM + ReAlign** | **0.451 (+11.9%)** | **0.784 (+7.3%)** | **0.276 (+44.5%)** | **2.775 (+10.4%)** |

### 消融实验

**各组件在 HumanML3D（MLD 基线）上的消融**：

| T2M奖励 | M2M奖励 | 步感知 | R@1 ↑ | FID ↓ | MM Dist ↓ |
|---------|---------|--------|-------|------|-----------|
| ✗ | ✗ | ✗ | 0.481 | 0.473 | 3.196 |
| ✓ | ✗ | ✗ | 0.556 | 0.213 | 2.761 |
| ✗ | ✓ | ✗ | 0.517 | 0.205 | 2.932 |
| ✓ | ✓ | ✗ | 0.556 | 0.199 | 2.750 |
| ✓ | ✗ | ✓ | 0.568 | 0.212 | 2.714 |
| **✓** | **✓** | **✓** | **0.567** | **0.195** | **2.704** |

**即插即用能力验证**（应用于 5 种不同扩散模型）：

| 基线方法 | R@1 提升 | FID 提升 | MM Dist 提升 |
|---------|---------|---------|-------------|
| Mo.Diffuse | +8.8% | +41.3% | +9.9% |
| MDM | +3.3% | +33.5% | +6.0% |
| MLD | +17.9% | +58.8% | +15.4% |
| MotionLCM | +7.6% | +10.2% | +7.1% |
| MLD++ | +4.4% | +24.7% | +5.8% |

### 关键发现

1. **T2M 奖励是主力**：仅使用 T2M 奖励即将 R@1 从 0.481 提升至 0.556（+15.6%），FID 从 0.473 降至 0.213（+54.9%）。
2. **步感知训练不可或缺**：图 3 的对比显示，无步感知的奖励模型在高噪声步骤失效，步感知版本在所有去噪步骤上均显著更优。
3. **M2M 奖励提供额外真实感增益**：单独效果有限（受文本-动作检索精度制约），但与 T2M 和步感知结合后进一步降低 FID。
4. **通用性验证**：在 5 种不同基线上一致有效，FID 改善幅度从 10.2% 到 58.8%，证明了即插即用的普适性。
5. 动作检索任务上也有提升：HumanML3D 上 R@1 达 67.59%（T2M retrieval）和 68.94%（M2T retrieval），超越 TMR 和 LaMP。

## 亮点与洞察

1. **理论优美**：从理想分布 $p_t^I$ 的定义出发，经 SDE 推导到 DDPM 离散化，数学链条完整（Theorem 1-3），奖励梯度自然融入去噪过程。
2. **即插即用的核心价值**：不修改任何扩散模型，仅在推理阶段添加奖励引导。所有基线方法均获得大幅提升，验证了方法的通用性。
3. **生成与检索的统一**：步感知奖励模型同时服务于生成引导和动作检索两个任务，构建了生成-检索的桥梁。
4. **噪声增强策略的妙用**：在训练奖励模型时加入噪声动作，不仅使其适应去噪过程的噪声，还相当于数据增强，提升了检索模型对细微动作差异的区分能力。
5. **步感知是首次在动作生成中提出**：将时间步信息作为 token 注入对齐模型，思路简单但效果显著。

## 局限与展望

1. **推理开销增加**：每步去噪都需额外的奖励模型前向传播 + 梯度计算，推理时间数倍增加。
2. **检索动作的质量瓶颈**：M2M 奖励依赖从训练集检索的参考动作，训练集动作不丰富时效果受限。
3. **Diversity 下降**：在部分设置下 diversity 略降，反映了奖励引导对多样性的约束。虽然作者论证这是"更好的对齐排斥了错误动作"，但过强的引导可能导致模式坍塌。
4. **奖励权重 $\mu$、$\eta$ 的选择**：消融未详细探讨最优超参数搜索，手动调节可能不具备普遍性。
5. **仅验证了 DDPM 框架**：DDIM、ODE 采样器下的表现未验证，理论推导的适配性有待扩展。

## 相关工作与启发

- **TMR/LaMP**：文本-动作对齐模型的基石，但均假设干净输入，ReAlign 的步感知设计扩展了其适用范围。
- **Classifier Guidance** (Dhariwal et al.)：图像扩散中用分类器梯度引导生成，ReAlign 将此思想推广到动作领域的对齐奖励。
- **ReinDiffuse/MotionRL/SoPo**：通过 RL 微调扩散模型对齐偏好，但需要训练。ReAlign 的免训练方案更灵活。
- **EnergyMoGen**：同期工作，也关注动作生成对齐，但 ReAlign 在 R@1 上大幅领先（0.567 vs. 0.526）。
- 奖励引导采样思路可推广到其他文本条件生成任务（3D 生成、音乐生成等）。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 步感知奖励 + 推理时引导的组合有新意，理论推导完整
- 实验充分度: ⭐⭐⭐⭐⭐ — 两个数据集、五个基线验证、详细消融、检索任务验证
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，动机阐述到位
- 价值: ⭐⭐⭐⭐⭐ — 即插即用方案实用性极强，改善幅度显著

<!-- RELATED:START -->

## 相关论文

- [BiMotion: B-spline Motion for Text-guided Dynamic 3D Character Generation](../../CVPR2026/image_generation/bimotion_b-spline_motion_for_text-guided_dynamic_3d_character_generation.md)
- [Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](../../ECCV2024/image_generation/local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [MACS: Multi-source Audio-to-Image Generation with Contextual Significance and Semantic Alignment](macs_multi-source_audio-to-image_generation_with_contextual_significance_and_sem.md)
- [Step-Aware Residual-Guided Diffusion for EEG Spatial Super-Resolution](../../ICLR2026/image_generation/step-aware_residual-guided_diffusion_for_eeg_spatial_super-resolution.md)
- [Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution](mixture_of_ranks_with_degradation-aware_routing_for_one-step_real-world_image_su.md)

<!-- RELATED:END -->
