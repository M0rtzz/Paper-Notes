---
title: >-
  [论文解读] Threshold-Guided Optimization for Visual Generative Models
description: >-
  [ICML 2026][图像生成][阈值引导] 作者把 DPO 的成对偏好假设拆掉，证明 KL 正则化最优策略本质上是把每个样本的 reward 与一个无法计算的实例相关基线 $\tau^*(x)=\beta\log Z(x)$ 比较，于是用从分数分位数估出的全局阈值 $\tau$ 替代它…
tags:
  - "ICML 2026"
  - "图像生成"
  - "阈值引导"
  - "无配对偏好优化"
  - "标量反馈"
  - "扩散模型对齐"
  - "MaskGIT"
---

# Threshold-Guided Optimization for Visual Generative Models

**会议**: ICML 2026  
**arXiv**: [2605.04653](https://arxiv.org/abs/2605.04653)  
**代码**: 无  
**领域**: 图像生成 / 偏好对齐  
**关键词**: 阈值引导, 无配对偏好优化, 标量反馈, 扩散模型对齐, MaskGIT

## 一句话总结
作者把 DPO 的成对偏好假设拆掉，证明 KL 正则化最优策略本质上是把每个样本的 reward 与一个无法计算的实例相关基线 $\tau^*(x)=\beta\log Z(x)$ 比较，于是用从分数分位数估出的全局阈值 $\tau$ 替代它，再加一个与 $|s-\tau|$ 成正比的置信度权重，让扩散模型和 MaskGIT 在仅有标量打分（无成对偏好）时也能稳定对齐，并在五个 reward model 三个测试集上一致优于 Diffusion-DPO / KTO / DSPO。

## 研究背景与动机
**领域现状**：视觉生成模型对齐的主流是把 LLM 的 RLHF / DPO 套过来：先收集成对偏好 $(y_w, y_l)$，再用 Bradley-Terry 模型让 $\pi_\theta$ 给 $y_w$ 更高的概率比。Diffusion-DPO、AlignProp、DSPO 都是这条路。

**现有痛点**：实际场景里反馈往往不是成对，而是 1–5 星打分、reward model 的连续分，或者人工对单张图的标量分。硬把这些分凑成对（同 batch 内比较）会丢掉绝对数值信息，且分数聚集时人为对会被噪声放大。Diffusion-KTO 用 desirable/undesirable 集合绕开成对，但需要把分数硬切两堆。

**核心矛盾**：DPO 系列方法之所以能避开 KL 最优解里的难解配分函数 $Z(x)$，靠的是成对差值里 $\log Z(x)$ 自然抵消。一旦只有单样本标量分，这个抵消机制不再成立，必须直接面对 $\tau^*(x)=\beta\log Z(x)$ 这个实例相关的基线。

**本文目标**：(i) 给标量反馈下的 KL-正则化对齐推出一个**可计算**的代理决策规则；(ii) 让规则同时适用于扩散模型 (MSE 似然代理) 和 MaskGIT (token 级 cross-entropy 似然)；(iii) 不引入额外的成对采样开销，纯离线、单遍打分即可训练。

**切入角度**：作者从 KL 最优解出发，发现最优策略的更新方向其实是一个二元决策——只有当样本 reward 高过 $\tau^*(x)$ 时才该提升其概率。既然 $\tau^*(x)$ 难算，那能否用**整个数据集**上 reward 的某个分位数（如中位数）作为统一阈值 $\tau$ 来近似？分数距离阈值远的样本天然提供更强的监督信号，这就启发了"置信度加权"。

**核心 idea**：用经验分数分布的分位阈值 $\tau$ 作为全局代理替换难解的实例级基线 $\tau^*(x)$，把对齐变成一个带置信度权重的二元分类任务，从而在无配对的标量反馈上做直接策略拟合。

## 方法详解

### 整体框架
TGO（Threshold-Guided Optimization）要解决的是"只有标量打分、没有成对偏好时怎么对齐视觉生成模型"。它的核心转换是把对齐从"比较两个样本谁更好"重写成"判断单个样本好不好"：先用 reward model 给离线数据集打分，再从这堆分数的经验分布里取一个全局分位阈值，把每个样本按是否过阈值打成伪正/伪负标签，最后用一个带置信度权重的二元交叉熵把策略往伪正方向推、伪负方向拉。整个流程纯离线，不需要在线 rollout，也不需要微调 reward model。

### 关键设计

**1. 从 KL 最优解推出的阈值决策规则：把"该不该提升概率"化简成与全局阈值比大小**

DPO 系列方法之所以能避开难解的配分函数 $Z(x)$，靠的是成对差值里 $\log Z(x)$ 自然抵消；一旦只有单样本标量分，这个抵消机制失效，就得正面对付 $Z(x)$。作者从 KL 正则化目标 $\max \mathbb E[\mathcal R(x,y)] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ 的闭式最优解入手，发现它等价于一个二元决策：$\log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} > 0 \iff \mathcal R(x,y) > \tau^*(x)$，其中实例相关的基线 $\tau^*(x) = \beta \log Z(x)$ 难以计算。于是用两个假设把它做成可算——假设标量分 $s$ 是 reward 的单调变换，再用整个数据集分数分布的全局分位 $\tau = \text{Percentile}(\{s_i\}, p)$（默认 $p=0.5$ 中位数）替代 $\tau^*(x)$，决策规则就落成"当 $s \ge \tau$ 时应让 $\pi_\theta(y|x) \gtrsim \pi_{\text{ref}}(y|x)$"。这个全局阈值是最简单且统计上有保证的代理：附录定理证明替换后的估计量在 $n \to \infty$ 时一致、误差 $O(1/n)$，并校准回原 KL 最优规则。

**2. 置信度加权的二元分类损失：让分数离阈值越远的样本梯度越大**

中位数附近的样本本身处在"好坏难分"的灰色地带，若与极端样本同权处理就会把噪声放进梯度。为此作者定义 implicit policy score $\hat s_{\theta,\text{ref}}(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$，伪标签 $l = \mathbb 1[s \ge \tau]$，损失写成加权二元交叉熵 $\mathcal L_{\text{TG}} = -\mathbb E[w(s,\tau)(l\log\sigma(\hat s) + (1-l)\log(1-\sigma(\hat s)))]$，其中置信度权重 $w(s,\tau) = 1 + c|s-\tau|$、超参 $c \ge 0$。分数离阈值越远（越确定是好或坏）权重越大，越靠近阈值（越模糊）权重越接近 1。这样既保留了全数据集的利用率，又自然增强信号噪声比，相当于在分类问题里加了一个软 margin，且实测对超参不敏感（$c=5$ 跨任务稳定）。

**3. 针对两类视觉生成模型的似然代理：让框架同时吃下连续扩散和离散 token 范式**

损失里需要 $\log \pi_\theta(y|x)$，而它在两类模型上算法不同。扩散模型的精确似然不可解，沿用 Diffusion-DPO 的高斯观测近似 $\log \pi_\theta(y|x) \approx -\frac{1}{T}\text{MSE}(y, \hat y_\theta(x))$，温度 $T$ 控制尺度（默认 $T=0.001$），避免重新发明轮子；MaskGIT 是离散 token 模型，VQ-GAN tokenize 后掩码位置的对数似然 $\log \pi_\theta(y|x) = \frac{1}{|M|}\sum_{i\in M}\log p_\theta(t_i | y_{\setminus M}, x)$ 天然可算，反而是更"干净"的实验场景——它能验证 TGO 不依赖扩散特有的 MSE 近似，对生成范式无偏。

### 损失函数 / 训练策略
最终损失即上面的 $\mathcal L_{\text{TG}}$。训练超参：$\beta = 1$，扩散温度 $T=0.001$，置信度尺度 $c=5$，batch 128，78 个更新步（10K 提示集合），学习率 $1\text{e}{-5}$。阈值 $\tau$ 在大数据时可在更小的 proxy 集合（由 $\pi_{\text{ref}}$ 生成 + reward 打分）上估出后复用，估计误差按定理也是 $O(1/n)$ 衰减。SFT baseline 用相同优化超参但只在伪正样本上训。

## 实验关键数据

### 主实验
在 SD v1.5 上用 Pick-a-Pic v2（成对转标量）训练，对比 7 种 baseline，三个测试集 × 五个 reward model：

| 测试集 | 指标 | SD v1.5 | Diffusion-DPO | Diffusion-KTO | TGO (本文) |
|---|---|---|---|---|---|
| Pick-a-Pic | HPSv2.1 | 0.2469 | 0.2594 | 0.2814 | **0.2860** |
| Pick-a-Pic | ImageReward | 0.1131 | 0.3433 | 0.6381 | **0.6703** |
| PartiPrompts | PickScore | 21.15 | 21.41 | 21.50 | **21.55** |
| HPSv2 | ImageReward | 0.1384 | 0.3672 | 0.7365 | **0.7595** |
| HPSv2 | Aesthetic | 5.29 | 5.39 | 5.50 | **5.53** |

在 10K 标量反馈集合上跨范式比较：

| 范式 | 模型 | HPSv2.1 | ImageReward | Aesthetic |
|---|---|---|---|---|
| Diffusion | SD v1.4 | 0.2454 | 0.1406 | 5.4277 |
| Diffusion | + SFT | 0.2506 | 0.2348 | 5.4927 |
| Diffusion | + TGO | **0.2618** | **0.3523** | **5.6036** |
| MaskGIT | Meissonic | 0.2810 | 0.8230 | 5.7692 |
| MaskGIT | + SFT | 0.2912 | 0.9215 | 5.8013 |
| MaskGIT | + TGO | **0.2915** | **0.9369** | **5.8270** |

### 消融实验

| 配置 | 关键变化 | 影响 |
|---|---|---|
| Full TGO | $\tau$=中位数, $c=5$ | 全维度最优 |
| 无置信度加权 ($c=0$) | 退化为均权 BCE | 在 ImageReward 等高方差指标上掉幅明显，验证加权对样本效率贡献 |
| 提高/降低 $\tau$ 分位 | 改变正负样本比例 | 偏向极端分位时正样本太少，监督信号稀疏；中位数最稳 |
| 单 reward 训练 → 多 reward 评估 | 跨 reward 泛化 | 在未训练过的 reward 上也提升，说明 TGO 不是 reward hacking |

### 关键发现
- 在所有 reward 维度上一致打败 Diffusion-DPO（成对对照），说明"成对偏好假设"本身并非必须，标量打分加阈值就够。
- TGO 在 MaskGIT（精确似然）和扩散（近似似然）上都有效，证明方法对生成范式无偏。
- 阈值 $\tau$ 可以用 proxy 集合便宜估计，理论上误差 $O(1/n)$，工程上对大规模训练很友好。

## 亮点与洞察
- **理论上把 DPO 拆穿**：作者点破 DPO 能避开 $Z(x)$ 不是因为成对偏好"更对"，而是因为成对差值数学上让 $\log Z(x)$ 抵消。一旦换成单样本，"成对"就不再有特权——这是把 DPO 系列方法的护城河重新审视了一遍。
- **置信度加权 = 软margin**：把 $w = 1 + c|s-\tau|$ 看作分类问题里的样本权重，等价于让模型在"信号 margin"上更激进，是非常简洁的 trick，可以直接迁移到任何带分数的标签场景（如 LLM 的 reward score-based fine-tune）。
- **跨范式统一**：扩散 + MaskGIT 共用一个框架本身是个工程亮点，说明 TGO 不绑定扩散的 MSE 假设，对未来的 token-based 视频/3D 生成模型也可即插即用。

## 局限与展望
- 全局阈值 $\tau$ 隐含假设所有 prompt 的最优基线"差不多"，但 $\tau^*(x)$ 本质是实例相关的——困难 prompt 上可能本应有更高的基线，简单 prompt 应更低。文中未对 prompt-conditional 阈值做对比。
- 离线训练前提下，$\pi_{\text{ref}}$ 与训练时的 $\pi_\theta$ 越走越远，伪标签可能过时；论文虽给出可选的 $\pi_{\text{ref}} \leftarrow \pi_\theta$ 滚动更新但没系统验证。
- reward model 自身的偏置会被直接放大（TGO 没有任何"去 reward hacking"机制），跨 reward 评估虽改进但远小于训练 reward 上的提升，仍存在过拟合 scorer 的风险。
- 改进方向：把 $\tau$ 做成 prompt embedding 的函数；引入在线 rollout 让 $\tau$ 跟随策略更新；把 TGO 与 GRPO 结合用作 actor-critic 中的 critic 估计。

## 相关工作与启发
- **vs Diffusion-DPO**：DPO 必须成对，TGO 只需标量；DPO 通过差值抵消 $Z(x)$，TGO 用全局阈值近似 $\tau^*(x)$。在所有实验中 TGO 一致更优。
- **vs Diffusion-KTO**：KTO 也用 desirable/undesirable 集合，但基于 Kahneman-Tversky 价值函数；TGO 直接从 KL 最优解推出阈值规则，理论更干净，超参少（KTO 需要两个 desirable/undesirable 权重）。
- **vs QRPO**：QRPO 把 reward 做分位变换让 $Z$ 解析；TGO 不变换 reward，而是用分位"切"为正负，逻辑上更接近 DPO 的分类框架，工程上更轻。
- **vs DSPO**：DSPO 在 SD 上常常退化回基线（多个指标和原始 SD v1.5 完全相同），TGO 一致改进，证明对 score-based 监督的开发更彻底。

## 评分
- 新颖性: ⭐⭐⭐⭐ 理论上把 DPO 拆解为"全局阈值的近似"是个干净的新视角，但工程上和 KTO/QRPO 同属 unpaired 路线。
- 实验充分度: ⭐⭐⭐⭐ 三测试集 × 五 reward × 两生成范式 × 多 baseline，coverage 良好，但缺少在线策略下的对比和阈值条件化的消融。
- 写作质量: ⭐⭐⭐⭐ 从 KL 公式一路推到算法，逻辑链条非常清晰，附录定理给出一致性、偏差、校准的完整保证。
- 价值: ⭐⭐⭐⭐ 对实际工程意义大，因为绝大多数 reward 数据天然是标量而非成对，TGO 直接降低了数据采集成本。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Generative Visual Code Mobile World Models](generative_visual_code_mobile_world_models.md)
- [\[ICML 2026\] EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization](evogm_learning_to_merge_llms_via_evolutionary_generative_optimization.md)
- [\[ICML 2026\] E²PO: Embedding-perturbed Exploration Preference Optimization for Flow Models](embedding-perturbed_exploration_preference_optimization_for_flow_models.md)
- [\[CVPR 2026\] MapReduce LoRA: Advancing the Pareto Front in Multi-Preference Optimization for Generative Models](../../CVPR2026/image_generation/mapreduce_lora_advancing_the_pareto_front_in_multi-preference_optimization_for_g.md)
- [\[ICML 2025\] Direct Discriminative Optimization: Your Likelihood-Based Visual Generative Model is also a GAN Discriminator](../../ICML2025/image_generation/direct_discriminative_optimization_your_likelihood-based_visual_generative_model.md)

</div>

<!-- RELATED:END -->
