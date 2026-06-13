---
title: >-
  [论文解读] Classifier-Free Guidance inside the Attraction Basin May Cause Memorization
description: >-
  [CVPR 2025][图像生成][扩散模型记忆化] 从动力系统视角提出"吸引盆地"概念解释扩散模型记忆化现象——CFG 在吸引盆地内施加会导致轨迹收敛到记忆化训练图像，通过检测转折点延迟 CFG 启动（配合反向引导 OG）可零额外开销地缓解记忆化。 领域现状：扩散模型（如 Stable Diffusion）会精确复制训练数…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "扩散模型记忆化"
  - "吸引盆地"
  - "分类器无关引导"
  - "隐私泄露"
  - "缓解策略"
---

# Classifier-Free Guidance inside the Attraction Basin May Cause Memorization

**会议**: CVPR 2025  
**arXiv**: [2411.16738](https://arxiv.org/abs/2411.16738)  
**代码**: [https://github.com/SonyResearch/mitigating_memorization](https://github.com/SonyResearch/mitigating_memorization)  
**领域**: 扩散模型 / AI安全  
**关键词**: 扩散模型记忆化, 吸引盆地, 分类器无关引导, 隐私泄露, 缓解策略

## 一句话总结
从动力系统视角提出"吸引盆地"概念解释扩散模型记忆化现象——CFG 在吸引盆地内施加会导致轨迹收敛到记忆化训练图像，通过检测转折点延迟 CFG 启动（配合反向引导 OG）可零额外开销地缓解记忆化。

## 研究背景与动机

**领域现状**：扩散模型（如 Stable Diffusion）会精确复制训练数据中的图像（verbatim memorization），引发版权侵犯和隐私泄露担忧。已知因素包括训练数据重复、过度特定的 prompt 和小数据集微调，但即使去重后问题仍存在。

**现有痛点**：现有缓解方法要么在训练时修改（需重训，昂贵）、要么在推理时扰动 prompt/embedding（损害文本对齐）或降低 trigger token 的注意力权重（仅对特定类型的记忆化有效）。关键问题是这些方法针对特定记忆化场景设计，无法跨场景泛化。例如 Wen et al. 的方法在数据重复场景有效但在微调场景失效，Ren et al. 的方法需要 trigger token 存在。

**核心矛盾**：记忆化发生在去噪过程的高噪声阶段，此时 CFG 的条件引导力量异常强大，将不同初始化的轨迹都"吸引"到同一个记忆化图像。但完全不用 CFG 又会导致图像质量和文本对齐度下降。

**本文目标** 在推理时零额外开销地缓解扩散模型的记忆化，不依赖于 prompt 修改，且跨多种记忆化场景泛化。

**切入角度**：观察到 CFG 在去噪后期某个时间步之后启用不会产生记忆化输出。这意味着存在一个"转折点"$\tau^*$——在此之前去噪轨迹处于吸引盆地内（CFG 会导致记忆化），之后轨迹逃离盆地（CFG 正常工作）。转折点可通过条件-无条件噪声预测差的幅度突降来检测。

**核心 idea**：不在吸引盆地内施加 CFG（延迟到转折点后），或用反向引导（OG）加速逃离吸引盆地，以零额外计算开销缓解记忆化。

## 方法详解

### 整体框架
在标准扩散推理过程中，监控条件引导的 L2 幅度 $\|\epsilon_\theta(x_t, e_p) - \epsilon_\theta(x_t, e_\emptyset)\|^2$。当检测到第一个局部最小值时标记为转折点，此后开始施加正常 CFG。转折点之前可选择不施加引导（零 CFG）或施加反向引导（负 CFG）。整个过程复用 CFG 已有的条件和无条件预测计算，零额外开销。

### 关键设计

1. **吸引盆地理论（Attraction Basin）**

    - 功能：提供记忆化现象的动力系统解释
    - 核心思路：将去噪过程视为动力系统，记忆化训练图像 $x^a$ 是吸引子。吸引盆地定义为状态空间中所有在 CFG 下会收敛到 $x^a$ 附近的点集 $X^b(x^a, \epsilon) = \{(x,t) | \mathbb{P}(\varphi(x,t,e) \in B_D(x^a, \epsilon)) > 1-\delta\}$。盆地在 $t=T$ 时最宽（几乎覆盖整个空间），随去噪推进逐渐收窄。关键发现：不施加 CFG 时，轨迹会自然逃离盆地；施加 CFG 等于给轨迹施加了指向吸引子的力，使其留在盆地内
    - 设计动机：已有解释只关注 trigger token 或数据重复等表面因素，吸引盆地提供了统一的机制解释

2. **转折点检测与延迟 CFG（STP/DTP）**

    - 功能：找到从记忆化到非记忆化的临界时间步，精确确定 CFG 的最佳启动时间
    - 核心思路：**静态转折点（STP）**：某些模型（如 SDv2.1 微调版）所有样本共享同一转折点（如 t=500），直接硬编码即可。**动态转折点（DTP）**：某些模型（如预训练 SDv1.4）每个 prompt/初始化组合有不同转折点。通过实时追踪 $d_t = \|\epsilon_\theta(x_t, e_p) - \epsilon_\theta(x_t, e_\emptyset)\|_2^2$，检测第一个局部最小值（$d_{t+2} > d_{t+1} < d_t$），此后切换为正常 CFG。关键：条件和无条件预测在 CFG 中本就会计算，无需额外前向传播
    - 设计动机：从 Figure 2 的经验观察中总结出规律——幅度高且稳定时在盆地内，急剧下降时逃离盆地

3. **反向引导（Opposite Guidance, OG）**

    - 功能：加速去噪轨迹逃离吸引盆地，使转折点更早出现
    - 核心思路：在转折点之前施加负 CFG：$\hat{\epsilon} = \epsilon_\theta(x_t, e_\emptyset) - s(\epsilon_\theta(x_t, e_p) - \epsilon_\theta(x_t, e_\emptyset))$，将轨迹推向与条件引导相反的方向。实验发现 OG 使转折点从约 t=779 提前到约 t=839，从而为正常 CFG 留出更多步数，改善图像质量和文本对齐
    - 设计动机：当转折点很晚才出现（$t \leq 500$），零 CFG 阶段太长导致正常 CFG 步数不足，图像质量下降。OG 通过主动推离盆地解决这一问题

### 损失函数 / 训练策略
完全是推理时方法，无需训练或修改模型权重。所有计算（条件/无条件预测）在标准 CFG 推理中已包含。SDv1.4 推理时间 1.26s vs Wen et al. 的 2.86s（A100 GPU）。

## 实验关键数据

### 主实验

| 场景 | 方法 | 相似度(95%)↓ | CLIP↑ | FID↓ |
|------|------|-------------|-------|------|
| S1: SDv2.1微调 LAION-10k | 无缓解 | 0.6504 | 0.3027 | 16.84 |
| S1 | Wen et al. | 0.3853 | 0.2895 | 16.72 |
| S1 | **OG+STP** | **0.3811** | **0.3020** | **15.67** |
| S3: SDv1.4数据重复 | 无缓解 | 0.7977 | 0.3105 | 106.49 |
| S3 | Wen et al. (lt=1) | 0.6038 | 0.3050 | 136.34 |
| S3 | **DTP** | **0.5885** | 0.3020 | 138.92 |

### 消融实验

| 配置 | 相似度(95%)↓ | CLIP↑ | 说明 |
|------|-------------|-------|------|
| 标准 CFG (全程) | 0.6504 | 0.3027 | 记忆化 |
| 零 CFG [1000,500] + CFG [500,0] (STP) | 0.2857 | 0.2976 | 记忆化消除，CLIP 略降 |
| OG [1000,STP] + CFG [STP,0] | 0.3811 | 0.3020 | OG 改善 CLIP（更多 CFG 步数） |

### 关键发现
- 吸引盆地在所有四种记忆化场景中都存在，验证了理论的普适性
- 其他方法只对特定场景有效：Ren et al. 在 Scenario 1 几乎无效（相似度仅从 0.6504 降到 0.6028），Wen et al. 在 Scenario 1 虽降低相似度但图像质量差
- 本方法是唯一在所有场景中一致有效的方法
- OG 使 FID 从 19.85（STP only）改善到 15.67，说明更早的转折点为 CFG 留出更多生成空间
- 零额外计算开销（1.26s vs Wen et al. 2.86s），因为复用了 CFG 中已有的计算

## 亮点与洞察
- **吸引盆地视角**为扩散模型记忆化提供了优雅的动力系统解释，统一了多种记忆化成因（数据重复、微调过拟合、trigger token），本质都是 CFG 在盆地内的过强引导
- **零额外开销的缓解**极其实用：不修改模型、不修改 prompt、不增加计算，仅改变 CFG 的时间调度。可直接部署到任何使用 CFG 的扩散模型
- **反向引导（OG）**概念新颖：负 CFG 不是负提示词，而是在吸引盆地内主动推离记忆化轨迹，在图像质量和缓解效果之间取得更好平衡

## 局限与展望
- 需要先检测是否存在记忆化（检测本身需要额外计算或先验知识）
- 非记忆化样本没有转折点，对它们施加方法无害但也无益
- 转折点很晚时（$t \leq 500$），即使用 OG，CFG 步数仍可能不足影响质量
- 仅在 SD v1.4/v2.1 上验证，对 SDXL、Flux 等新架构的泛化性未知

## 相关工作与启发
- **vs Wen et al.**: 通过优化 prompt embedding 降低条件噪声差异，但增加推理时间且不跨场景泛化。本文方法零额外开销且在 Wen et al. 失效的场景中仍有效
- **vs Ren et al.**: 降低 trigger token 注意力权重，仅在 trigger token 存在时有效。无 trigger token 时从 0.6504 仅降到 0.6028
- **vs Chen et al.**: 设计不同 CFG 权重调度器，但需要预知记忆化类型。本方法自动检测转折点

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 吸引盆地的理论框架和反向引导概念高度原创，为理解和缓解记忆化提供了新范式
- 实验充分度: ⭐⭐⭐⭐ 四种场景全面验证+与多种基线对比，但缺乏新架构测试
- 写作质量: ⭐⭐⭐⭐⭐ 理论定义严谨，可视化极其出色（Figure 2 是全文精华），行文流畅
- 价值: ⭐⭐⭐⭐⭐ 对扩散模型安全和隐私保护有直接价值，零开销特性使其高度可部署

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] TCFG: Tangential Damping Classifier-Free Guidance](tcfg_tangential_damping_classifier-free_guidance.md)
- [\[AAAI 2026\] Studying Classifier(-Free) Guidance From A Classifier-Centric Perspective](../../AAAI2026/image_generation/studying_classifier-free_guidance_from_a_classifier-centric_perspective.md)
- [\[ICCV 2025\] TeEFusion: Blending Text Embeddings to Distill Classifier-Free Guidance](../../ICCV2025/image_generation/teefusion_blending_text_embeddings_to_distill_classifier-free_guidance.md)
- [\[CVPR 2025\] Enhancing Privacy-Utility Trade-offs to Mitigate Memorization in Diffusion Models](enhancing_privacy-utility_trade-offs_to_mitigate_memorization_in_diffusion_model.md)
- [\[NeurIPS 2025\] Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations](../../NeurIPS2025/image_generation/towards_a_golden_classifier-free_guidance_path_via_foresight_fixed_point_iterati.md)

</div>

<!-- RELATED:END -->
