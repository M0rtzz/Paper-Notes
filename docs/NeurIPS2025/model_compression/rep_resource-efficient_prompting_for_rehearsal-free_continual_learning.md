---
title: >-
  [论文解读] REP: Resource-Efficient Prompting for Rehearsal-Free Continual Learning
description: >-
  [NeurIPS 2025][模型压缩][持续学习] REP 通过轻量代理模型的快速提示选择、自适应 Token 合并（AToM）和自适应层丢弃（ALD）三种互补技术，将基于提示的无排练持续学习方法的训练时间减少最高 51%、内存降低最高 41%，精度损失微乎其微。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "持续学习"
  - "资源效率"
  - "提示学习"
  - "Token合并"
  - "层丢弃"
---

# REP: Resource-Efficient Prompting for Rehearsal-Free Continual Learning

**会议**: NeurIPS 2025  
**arXiv**: [2406.04772](https://arxiv.org/abs/2406.04772)  
**代码**: 无  
**领域**: 模型压缩 / 持续学习  
**关键词**: 持续学习, 资源效率, 提示学习, Token合并, 层丢弃

## 一句话总结

REP 通过轻量代理模型的快速提示选择、自适应 Token 合并（AToM）和自适应层丢弃（ALD）三种互补技术，将基于提示的无排练持续学习方法的训练时间减少最高 51%、内存降低最高 41%，精度损失微乎其微。

## 研究背景与动机

持续学习（Continual Learning）在多个顺序任务上训练模型，核心挑战是灾难性遗忘。基于提示（prompt）的无排练方法（如 L2P、DualPrompt、CODA-Prompt）在冻结的预训练 ViT 上通过学习少量提示参数来适配新任务，避免存储旧数据，适合边缘设备部署。

**现有痛点**：

**提示选择开销大**：通常需要用主干网络（如 ViT-L）的前向传播来计算查询特征进行提示检索，增加高达 28% 的计算时间

**提示更新代价高**：尽管主干网络冻结，但每个 mini-batch 仍需完整的前向和反向传播来优化提示和分类头，需要存储所有中间激活

**边缘设备限制严格**：设备内存通常 1-8GB，计算效率直接影响能耗和设备寿命

**核心洞察**：
- 提示选择阶段容许较大近似误差——无需使用全尺寸主干网络
- 冻结主干网络的各层对新任务的贡献不均等——浅层更重要（注意力距离更多样化），深层趋于全局且同质化

## 方法详解

### 整体框架

REP 框架包含三个互补技术：
1. 用轻量代理模型替代主干网络进行提示选择
2. AToM：在提示更新中自适应合并冗余 token
3. ALD：在提示更新中自适应丢弃深层层

三者分别优化不同瓶颈，可独立或组合使用。

### 关键设计

1. **轻量代理模型的提示选择**：用紧凑的 ViT-Ti（5.8M 参数）替代 ViT-L（307M 参数）计算查询特征。由于 ViT-Ti 的特征维度 d < D，应用固定随机投影 ϕ 将低维特征映射回原始 D 维空间进行提示匹配：
    $p^*_{\text{efficient}} = \underset{p_k \in P}{\text{argmax}} \frac{\langle \phi(q_{\text{efficient}}(x_i^j)), p_k \rangle}{\|\phi(q_{\text{efficient}}(x_i^j))\| \|p_k\|}$
   经验证该策略保留了约 97% 的表征相似性（CKA 度量）。

2. **自适应Token合并（AToM）**：与传统 ToMe（Token Merging）的两个关键区别：

    - **保护提示Token**：ToMe 不区分提示和非提示 token，导致提示的任务特定信息被稀释，甚至引起梯度爆炸。AToM 在合并时排除提示 token。
    - **渐进式调度器**：传统 ToMe 每层均匀合并 n 个 token。AToM 使用渐进调度：
    $r'(l) = \min(\delta \times (l-1), r_{\max})$
      其中 $\delta = r_{\max}/(L-1)$。浅层少合并（保留重要的局部/任务特定信息），深层多合并（信息已全局化、冗余度高）。默认 $r_{\max} = 2n$。

3. **自适应层丢弃（ALD）**：不同于均匀随机丢弃（Progressive Layer Dropping, PLD），ALD 同时考虑时间和空间维度，利用 AToM 的反馈来指导层丢弃。层保留概率：
    $\theta_{t,l} = \alpha(l) \times ((1-\bar{\theta})\exp(-\gamma \cdot t) + \bar{\theta})$
   其中 $\alpha(l)$ 根据该层的 token 合并量调整：当已合并 token 数超过阈值 τ 时（通常在深层），$\alpha(l) = 0.9$（更容易丢弃），否则 $\alpha(l) = 1$（保留）。这确保浅层优先保留，深层更积极丢弃。

### 损失函数 / 训练策略

REP 不改变基础 CL 方法的损失函数。标准框架为：
$$L = L_{\text{class}}(f_{\text{update}}(x_i^j), y_i^j) + \epsilon_1 L_{\text{prompt}}(p^*, q(x_i^j)) + \epsilon_2 L_{\text{aux}}$$

REP 仅优化计算路径（哪些 token 参与计算、哪些层执行前向/反向传播），不修改损失或学习目标。

## 实验关键数据

### 主实验（7种提示方法 × 3种ViT × 3个数据集）

| 模型 | 方法 | 数据集 | w/o REP Acc | w/ REP Acc | 时间加速 | 内存节省 |
|------|------|--------|------------|-----------|---------|---------|
| ViT-L | L2P | Split ImageNet-R | 75.6 | 75.3 | 1.9× | 1.4× |
| ViT-L | DualPrompt | Split ImageNet-R | 71.2 | 70.6 | 2.0× | 1.4× |
| ViT-L | HiDe-Prompt | Split ImageNet-R | 78.7 | 78.0 | 1.8× | 1.2× |
| ViT-L | ConvPrompt | Split ImageNet-R | 79.1 | 78.5 | 1.3× | 1.3× |
| ViT-B | HiDe-Prompt | Split ImageNet-R | 64.5 | 64.4 | 1.7× | 1.7× |

精度损失范围：0.0-1.2%（Split CIFAR-100），0.1-1.1%（Split ImageNet-R），0.0-0.8%（Split PlantDisease）。部分情况下 REP 甚至提升精度（如 L2P+ViT-L 在 PlantDisease 从 75.9% 提升到 81.1%）。

### 消融实验

| 消融配置 | 精度 | 迭代时间(ms) | 内存(GB) | 说明 |
|---------|------|-------------|---------|------|
| 完整 REP-L2P | 75.3 | 240 | 4.5 | 最佳平衡 |
| w/o AToM+ALD | 74.8 | 349 | 5.5 | 两模块联合贡献大 |
| w/ ToMe（替代AToM） | 70.2 | 275 | 3.7 | 传统方法精度暴跌5.1% |
| w/ PLD（替代ALD） | 73.3 | 259 | 4.5 | PLD 精度差2% |
| Random Drop-25% | 70.6 | 398 | 6.5 | 均匀丢弃不可行 |
| ALD | 75.8 | 401 | 6.5 | 自适应丢弃保持精度 |

### 超参数敏感性

| Token合并数 n | 精度 | 时间(ms) | 内存(GB) |
|--------------|------|---------|---------|
| 4 | 75.3 | 256 | 5.2 |
| 8（默认） | 75.3 | 240 | 4.5 |
| 10 | 73.6 | 228 | 4.1 |

| 保留概率 θ̄ | 精度 | 时间(ms) |
|------------|------|---------|
| 0.1 | 72.9 | 217 |
| 0.5（默认） | 75.3 | 240 |
| 0.9 | 74.3 | 282 |

### 关键发现

- AToM 的核心在于保护提示 token——传统 ToMe 导致梯度爆炸，AToM 排除提示后梯度稳定
- ALD 利用 AToM 的合并量信息进行联动决策，空间-时间两维度协同优化
- 浅层比深层更重要的洞察在多种提示方法和主干架构上一致成立
- REP 可扩展到非提示方法（SLCA、RanPAC），训练时间减少 37-48%，内存降低最高 48%
- 更大的模型（ViT-L）从 REP 中获益更多，因为有更多可优化的空间

## 亮点与洞察

- **成本-精度分析驱动设计**：不是盲目压缩，而是基于对注意力距离的深入分析，识别出提示选择和更新阶段的不同优化机会
- **AToM vs ToMe 的关键区别**：保护提示 token + 渐进调度，两个简单修改带来 5% 的精度差异，揭示了 CL 场景下 token 合并的特殊需求
- **AToM-ALD 联动**：ALD 不独立设计调度参数，而是利用 AToM 的合并量作为反馈，实现两个技术的自然耦合
- **广泛适用性**：在 7 种提示方法 + 3 种主干 + 3 个数据集上验证，又扩展到 2 种非提示方法，可复制性强

## 局限与展望

- 代理模型需要额外维护一个 ViT-Ti，增加部署复杂度
- AToM 和 ALD 的超参数（n, θ̄, τ, α）需要针对不同模型规模调整
- 仅在图像分类任务上验证，视觉-语言、检测等任务的适用性未知
- 随机投影是否在所有情况下都能保持 97% 的 CKA 相似性有待验证
- 层丢弃的理论保证不够强——依赖经验观察而非正式分析

## 相关工作与启发

REP 填补了提示学习方法在资源效率方面的空白。与 BudgetCL 和 CarM 等基于 CNN 的资源感知 CL 方法不同，REP 专注于 ViT 架构。其核心洞察——ViT 浅层比深层对新任务更敏感——与 ViT 预训练研究中的发现一致，但首次被应用于持续学习的计算优化。AToM 中保护提示 token 的设计可以推广到任何在 Transformer 中使用可学习 token 的方法。

## 评分

- 新颖性: ⭐⭐⭐⭐ AToM保护提示token和ALD联动设计有新意，但各组件技术相对简单
- 实验充分度: ⭐⭐⭐⭐⭐ 7种方法×3种模型×3个数据集+详尽消融+非提示方法扩展
- 写作质量: ⭐⭐⭐⭐ 动机分析清晰，实验组织系统化
- 价值: ⭐⭐⭐⭐ 解决了提示学习在边缘设备部署的实际瓶颈，工程价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning](../../CVPR2025/model_compression/cl-lora_continual_low-rank_adaptation_for_rehearsal-free_class-incremental_learn.md)
- [\[CVPR 2026\] Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge](../../CVPR2026/model_compression/critical_patch-aware_sparse_prompting_with_decoupled_training_for_continual_lear.md)
- [\[ICCV 2025\] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning](../../ICCV2025/model_compression/achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [\[CVPR 2025\] LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning](../../CVPR2025/model_compression/lora_subtraction_for_drift-resistant_space_in_exemplar-free_continual_learning.md)

</div>

<!-- RELATED:END -->
