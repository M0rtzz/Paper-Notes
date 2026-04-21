---
title: >-
  [论文解读] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation
description: >-
  [CVPR2026][多模态][持续学习] 提出 SeGP-CL，通过对抗锚点探测旧-新语义边界的脆弱区域，结合锚点引导的跨模态几何蒸馏（ACGD）和文本语义几何正则化（TSGR），在无样本回放条件下有效保持 VLM 的跨模态语义几何结构，显著缓解灾难性遗忘。
tags:
  - CVPR2026
  - 多模态
  - 持续学习
  - 视觉语言模型
  - 语义几何保持
  - 对抗锚点
  - 跨模态蒸馏
  - CLIP
  - 无样本回放
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Continual Learning with Vision-Language Models via Semantic-Geometry Preservation

**会议**: CVPR2026  
**arXiv**: [2603.12055](https://arxiv.org/abs/2603.12055)  
**代码**: 待确认  
**领域**: multimodal_vlm  
**关键词**: 持续学习, 视觉语言模型, 语义几何保持, 对抗锚点, 跨模态蒸馏, CLIP, 无样本回放

## 一句话总结

提出 SeGP-CL，通过对抗锚点探测旧-新语义边界的脆弱区域，结合锚点引导的跨模态几何蒸馏（ACGD）和文本语义几何正则化（TSGR），在无样本回放条件下有效保持 VLM 的跨模态语义几何结构，显著缓解灾难性遗忘。

## 研究背景与动机

**VLM 持续学习的核心挑战**：预训练视觉语言模型（如 CLIP）在持续学习中面临灾难性遗忘，现有方法在适配新任务时没有显式保持跨模态语义几何结构，导致新任务监督信号引发几何畸变。

**语义边界处的脆弱性**：作者关键观察——有害的表征漂移并非均匀分布在嵌入空间中，而是集中在旧-新语义交界处。在这些区域，新样本与旧类共享视觉模式，容易被新文本语义"重新解释"，从而破坏已建立的视觉-文本对齐。

**现有方法的不足**：冻结骨架+任务特异组件的保守策略（L2P、DualPrompt、PROOF 等）过度隔离知识、限制正向迁移；参数高效适配方法（LoRA/Adapter）缺乏对跨模态稳定性的针对性建模；利用文本先验的方法（DesCLIP、CLG-CBM）仍未充分关注无样本条件下的跨模态几何保持。

**参考数据方案的局限**：一些方法（ZSCL、DualTeacher）使用额外参考数据集来稳定几何结构，但引入非平凡的数据开销，且约束不够精准——无法集中约束最容易发生畸变的边界区域。

**模态间隙问题**：VLM 中视觉和文本嵌入空间并非完美对应（modality gap），仅依赖文本语义无法完全表征视觉空间，需要结合原始视觉线索进行互补推理。

**对抗攻击的建设性利用**：VLM 对微小扰动敏感，可被建设性地利用——通过对抗扰动暴露和覆盖旧几何结构中最脆弱的邻域，为无样本条件下的几何保持提供高效探测手段。

## 方法详解

### 整体框架（SeGP-CL）

SeGP-CL 是一个分三阶段的无样本持续学习框架：

- **训练前**：冻结教师快照 $(F^T, G^T)$，通过双目标投影梯度下降（DPGD）从新任务数据中构建一组对抗锚点 $\mathcal{A}_t$，探测旧-新语义边界的脆弱区域
- **训练中**：用新任务数据优化交叉熵损失，同时在锚点上执行 ACGD 蒸馏保持跨模态结构，并用 TSGR 稳定文本语义参考框架
- **训练后**：利用锚点估计原始视觉空间中的漂移，迁移旧类视觉原型，并通过双路径推理融合跨模态与视觉线索

### 关键设计 1：双目标投影梯度下降（DPGD）构建对抗锚点

核心思想是从新任务数据中选取与旧类语义亲和度最高的种子样本，通过对抗扰动将其推向旧类语义区域：

1. **种子选择**：对每个旧类 $c$，按教师模型的跨模态相似度 $Q(x, c) = \bar{v}^T(x)^\top u_c^T$ 排序，取 top-$K_{\text{seed}}$ 个新任务样本作为种子
2. **双目标优化**：文本目标将扰动样本推向旧类文本嵌入（$\mathcal{L}_{\text{adv}}$），视觉目标将其拉向旧类原始视觉原型（$\mathcal{L}_{\text{v-adv}}$），修正模态间隙带来的不稳定性
3. **PGD 迭代**：在 $\ell_\infty$ 约束下运行 $K_{\text{adv}}=10$ 次符号梯度迭代，步长 $\gamma = 1.5 \times 10^{-3}$

$$\delta^{(k+1)} = \Pi_{\|\delta\|_\infty \leq \epsilon}\big(\delta^{(k)} - \gamma \cdot \text{sign}(\nabla_\delta \mathcal{L}'_{\text{adv}})\big)$$

### 关键设计 2：锚点引导的跨模态几何蒸馏（ACGD）

在对抗锚点上对齐教师与学生的旧类概率分布，约束脆弱边界区域的跨模态结构：

$$\mathcal{L}_{\text{ACGD}} = \tau_A^2 \cdot \mathbb{E}_{x^{adv} \sim \mathcal{A}_t}\left[\text{KL}(\pi_T^{\tau_A}(\cdot | x^{adv}) \| \pi_S^{\tau_A}(\cdot | x^{adv}))\right]$$

其中 $\tau_A = 20$ 为蒸馏温度，教师/学生分布均在旧类集合 $\mathcal{C}_{<t}$ 上计算。

### 关键设计 3：文本语义几何正则化（TSGR）

文本概念间的相对几何结构若在任务间漂移，会隐式重参数化旧类语义。TSGR 通过 $k$-NN 子图匹配约束新类的文本邻域结构：

- 用 LoRA 重置后的预训练文本编码器 $G^0$ 构建参考子图
- 对每个新类 $c \in \mathcal{C}_t$，找其 $k=10$ 个最近邻，匹配教师与学生的子图邻
- 仅约束新类根的子图，复杂度 $\mathcal{O}(|\mathcal{C}_t| \cdot k)$，远低于全局约束

### 关键设计 4：锚点驱动的原型迁移与双路径推理

- **原型迁移**：利用锚点在训练前后的原始视觉特征位移 $d_t(x^{adv})$，加权估计每个旧类的漂移方向 $\Delta_{t,c}$，并按锚点-原型接近度调制幅度，迁移旧类原型
- **双路径推理**：融合 CLIP 跨模态分数与视觉原型分数：$\ell_t(x, c) = s_t^{\text{clip}}(x, c) + \beta \cdot s_t^v(x, c)$，$\beta=0.5$

### 总损失函数

$$\mathcal{L}_{\text{CL}}^t = \mathcal{L}_{\text{cls}} + \lambda_{\text{ACGD}} \cdot \mathcal{L}_{\text{ACGD}} + \lambda_{\text{GR}} \cdot \mathcal{L}_{\text{GR}}$$

其中 $\lambda_{\text{ACGD}}=5$, $\lambda_{\text{GR}}=1$，仅更新 LoRA 的上投影矩阵 B。

## 实验

### 主实验：五大基准 SOTA 对比（CLIP ViT-B/16）

| 方法 | CIFAR100 Avg/Last | ImageNet-R Avg/Last | ImageNet-Sub Avg/Last | CUB-200 Avg/Last | UCF Avg/Last |
|---|---|---|---|---|---|
| MG-CLIP (ICCV'25) | 87.0/80.6 | 87.6/82.7 | 87.3/78.4 | 80.6/72.0 | – |
| RAPF (ECCV'24) | 86.2/79.0 | 85.6/80.3 | 87.5/80.2 | 82.7/76.2 | 92.5/87.5 |
| ENGINE (ICCV'25) | 82.1/73.1 | 84.4/77.0 | – | 83.9/76.2 | 95.0/90.1 |
| **SeGP-CL (Ours)** | **89.8/84.6** | **88.9/84.8** | **89.9/80.5** | **85.4/80.1** | **95.9/92.8** |

SeGP-CL 在全部五个基准上取得 SOTA，CIFAR100 Last 较 MG-CLIP 提升 +4.0，CUB-200 Last 较 RAPF 提升 +3.9。

### 迁移与遗忘指标（仅 CLIP 分支，CIFAR100）

| 方法 | FWT ↑ | BWT ↑ | Forgetting ↓ |
|---|---|---|---|
| MG-CLIP | 70.2 | -3.9 | 4.9 |
| DesCLIP | 68.7 | -2.1 | 6.5 |
| **SeGP-CL** | **72.3** | **-0.43** | **0.9** |

SeGP-CL 的 Forgetting 仅 0.9，远低于 MG-CLIP 的 4.9，BWT 接近零（-0.43），表明几乎无后向遗忘。

### 消融实验

| ACGD | TSGR | 原型迁移 | 视觉分支 | CIFAR100 Last | Forgetting ↓ |
|---|---|---|---|---|---|
| ✗ | ✗ | ✗ | ✗ | 77.0 | 10.9 |
| ✓ | ✗ | ✗ | ✗ | 81.7 | 5.8 |
| ✓ | ✓ | ✗ | ✗ | 82.8 | 4.7 |
| ✓ | ✓ | ✓ | ✗ | 83.2 | 4.3 |
| ✓ | ✓ | ✓ | ✓ | 84.6 | 4.5 |

ACGD 贡献最大（Last +4.7, Forgetting -5.1），TSGR、原型迁移、视觉分支逐步提升。

### 关键发现

- **对抗锚点 vs 其他蒸馏数据源**：锚点蒸馏（+5.8 Last）远超参考数据（ZSCL +1.9）、合成数据（GIFT +2.7）和新任务数据（-0.5）
- **跨场景泛化**：在 CIFAR100 上训练后，仍在 Food101/Oxford-Pets/ImageNet-1K 上保持接近 zero-shot 的泛化能力（TSGR 功不可没）
- **参数效率**：LoRA rank=32 下仅 3.44M 可训练参数（vs MoE-Adapter 13.35M），每迭代额外开销仅 ~79ms
- **DPGD 迭代次数**：10 次迭代即可稳定收敛，文本目标收敛慢于视觉目标（印证模态间隙）

## 亮点

- **问题定位精准**：首次系统揭示 VLM 持续学习中跨模态几何畸变集中于旧-新语义边界的现象，并以 JSD 度量提供实证
- **对抗攻击的建设性利用**：巧妙将 VLM 的对抗脆弱性转化为定位脆弱区域的手段，无需存储旧数据即可探测边界邻域
- **双目标设计解决模态间隙**：DPGD 的视觉锚定项补偿模态间隙，避免纯文本目标产生不稳定锚点
- **轻量且高效**：TSGR 仅约束新类的 $k$-NN 子图，参数开销小，每迭代额外时间可控
- **理论与实验统一**：从对抗优化的一阶最优性到实验中五个基准全面 SOTA，论证逻辑完整

## 局限性

- 对抗锚点的质量依赖 $\ell_\infty$ 预算和迭代次数等超参数，不同数据集可能需要调优
- TSGR 仅约束新类的文本邻域子图，若旧类间的文本关系发生漂移则无法检测
- 原型迁移假设锚点的特征漂移可作为旧类漂移的可靠代理，当新旧类语义差异过大时假设可能不成立
- 仅在 CLIP ViT-B/16 上验证，未测试更大规模 backbone（如 ViT-L）或其他 VLM（如 SigLIP、EVA-CLIP）
- 双路径推理的融合系数 $\beta$ 固定，未探索自适应融合策略

## 相关工作

- **VLM 持续学习**：与 MG-CLIP（保持模态间隙）、ZSCL/DualTeacher（参考数据蒸馏）、ENGINE/RAPF（任务特异组件）形成对比，SeGP-CL 无需额外数据且精准约束脆弱区域
- **跨模态蒸馏**：SGCL 在新任务数据上蒸馏语义伪标签参考分布，但不如对抗锚点精准
- **合成数据**：GIFT 用 Stable Diffusion 合成旧类图像进行蒸馏，但域差距限制效果
- **对抗鲁棒性**：利用 PGD 攻击框架，但目标从"攻击"转为"探测脆弱邻域"

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 对抗锚点探测语义边界的思路非常新颖，将攻击转化为防御工具
- 实验充分度: ⭐⭐⭐⭐⭐ — 五个基准全面 SOTA，包含详尽的蒸馏方案对比、消融、泛化分析
- 写作质量: ⭐⭐⭐⭐ — 公式推导严谨，但符号较多，阅读门槛偏高
- 价值: ⭐⭐⭐⭐⭐ — 为 VLM 持续学习提供了新的几何保持范式，无样本回放条件下大幅超越前作

<!-- RELATED:START -->

## 相关论文

- [Enhanced Continual Learning of Vision-Language Models with Model Fusion](../../ICLR2026/multimodal_vlm/enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)
- [Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning](../../AAAI2026/multimodal_vlm/harnessing_textual_semantic_priors_for_knowledge_transfer_and_refinement_in_clip.md)
- [Branch, or Layer? Zeroth-Order Optimization for Continual Learning of Vision-Language Models](../../AAAI2026/multimodal_vlm/branch_or_layer_zeroth-order_optimization_for_continual_lear.md)
- [BriMA: Bridged Modality Adaptation for Multi-Modal Continual Action Quality Assessment](brima_bridged_modality_adaptation_for_multi-modal_continual_action_quality_asses.md)
- [On Token's Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models](on_tokens_dilemma_dynamic_moe_with_drift-aware_token_assignment_for_continual_le.md)

<!-- RELATED:END -->
