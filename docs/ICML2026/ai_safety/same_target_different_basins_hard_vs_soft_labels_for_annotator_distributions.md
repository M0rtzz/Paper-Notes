---
title: >-
  [论文解读] Same Target, Different Basins: Hard vs. Soft Labels for Annotator Distributions
description: >-
  [ICML 2026 (EIML Workshop)][AI安全][标注者分布] 在 CIFAR-10H 上把"标注者分布"以硬标签方式投喂给模型（multipass 按票循环 / SLS 每个 epoch 重采样），证明它和软标签交叉熵期望目标等价…
tags:
  - "ICML 2026 (EIML Workshop)"
  - "AI安全"
  - "标注者分布"
  - "软硬标签"
  - "损失景观"
  - "OOD 检测"
  - "校准"
---

# Same Target, Different Basins: Hard vs. Soft Labels for Annotator Distributions

**会议**: ICML 2026 (EIML Workshop)  
**arXiv**: [2605.20642](https://arxiv.org/abs/2605.20642)  
**代码**: 无  
**领域**: AI 安全 / 不确定性估计 / 训练方法论  
**关键词**: 标注者分布, 软硬标签, 损失景观, OOD 检测, 校准

## 一句话总结
在 CIFAR-10H 上把"标注者分布"以硬标签方式投喂给模型（multipass 按票循环 / SLS 每个 epoch 重采样），证明它和软标签交叉熵期望目标等价，但收敛到更平坦的 basin、在稀疏标注下更优、且 OOD 检测略胜。

## 研究背景与动机
**领域现状**：多标注者数据集（CIFAR-10H、ChaosNLI、LeWiDi）的主流做法是要么投票塌成单标签、要么直接在每例的经验软标签分布上做交叉熵。Peterson et al. (2019) 已证明软标签训练能保留"歧义性 / 主观性"信息，明显优于硬投票。

**现有痛点**：软标签训练把"分布留在 loss 里"是一种假设——它默认经验分布 $p_i$ 就是好的训练目标。但 (a) 当每例只有 5–10 个标注时，$p_i$ 是稀疏估计，离真分布有距离；(b) 它把"目标是什么"和"目标怎么投喂"两件事耦合在了一起，无法解耦研究优化路径的影响。

**核心矛盾**：要保留标注者信息，分布是否一定要每一步都进 loss？如果换成"硬标签按分布采样投喂"，能否在同一个期望目标下走出不同的优化路径，进而得到不同几何性质的解？

**本文目标**：在固定每例标注者目标的前提下，只改变"投喂格式"（hard vs soft），系统比较两种 delivery 在稀疏 / 充分两种标注预算下的端点性能、basin 几何和 OOD 行为。

**切入角度**：作者把 SLS（每 epoch 从 $p_i$ 采一个硬标签）和 soft cross-entropy 做形式化对照，证明二者**期望梯度相同**、差异只在采样方差，而采样方差和标注分歧高度相关——也就是说硬标签 delivery 等价于"在分歧大的样本上注入结构化噪声"。

**核心 idea**：固定标注者分布作为目标，把投喂格式当作独立变量，用 multipass（按票多遍循环）和 SLS（按分布重采样）替代软标签 CE；理论上期望目标等价，实验上得到更平坦 basin + 稀疏标注下更好的 soft NLL。

## 方法详解

### 整体框架
设数据集 $\mathcal{D}=\{(x_i,p_i)\}_{i=1}^N$，$p_i\in\Delta^{K-1}$ 是第 $i$ 个样本的经验标注者分布，$q_\theta(x)=\mathrm{softmax}(z_\theta(x))$ 是模型预测。对照组是 soft-label CE $\mathcal{L}_{\mathrm{soft}}=\sum_i H(p_i,q_\theta(x_i))$。本文设计两种 hard-label delivery（multipass、SLS）+ 两种控制（deterministic control、shuffled SLS），每个 epoch 都用普通硬标签 CE，但每例的硬标签随 epoch / 采样不同而变。所有方法都在 CIFAR-adapted ResNet-18 上跑 200 epochs，cosine annealing。

### 关键设计

1. **Multipass（按票循环的 count-preserving 硬标签）**:

    - 功能：当原始投票计数 $\{c_{ik}\}_k$ 可得时，把每例的票展开成"标签多重集"，固定种子打乱一次，然后跨 epoch 按 $\texttt{epoch}\bmod m_i$ 循环投喂（$m_i$ 是该例总票数）。每个 epoch 仍然只有 $N$ 个样本，数据集基数不变。
    - 核心思路：和 Sheng et al. (2008) 的"重复标注展开成多样本"不同，multipass 保留数据集大小但改变 epoch 内的"标签身份"。一例若有 50 票则 50 个 epoch 走完一轮，每个观察到的票出现且只出现一次。
    - 设计动机：用确定性的、可复现的硬标签序列承载分布信息——避免 SLS 的采样方差，又能验证"是不是只要 epoch 间标签变化就够了"。它是有原始计数时的首选默认。

2. **Stochastic Label Sampling (SLS) + 期望目标等价命题**:

    - 功能：每个 epoch 开始时为每例独立采样 $y_i^{(t)}\sim\mathrm{Categorical}(p_i)$，然后按普通硬标签 CE 训练当前 epoch。只需要 $p_i$ 不需要原始票数，是 multipass 的轻量替代。
    - 核心思路：作者用 Proposition 1 证明 $\mathbb{E}_{y\sim p}[-\log q_y]=H(p,q)$、$\mathbb{E}_{y\sim p}[\nabla_z\ell]=q-p$、$\mathrm{Cov}_{y\sim p}[\nabla_z\ell]=\mathrm{Diag}(p)-pp^\top$，因此 SLS 和软标签 CE 期望目标完全等价，差别只是额外梯度方差 $\mathbb{E}\|q-e_y\|^2-\|q-p\|^2=1-\|p\|_2^2$，分歧越大方差越大。
    - 设计动机：把"目标"和"优化路径"严格解耦——两种方法目标相同，端点性能差异必然来自优化路径和噪声协方差（$\mathrm{Diag}(p)-pp^\top$ 的结构），从而可以归因到 basin 几何。

3. **Deterministic / Shuffled 双重控制**:

    - 功能：Deterministic control 是 multipass 的 traversal-order 消融，换一个固定打乱种子；shuffled SLS 是把每例的 $p_i$ 在样本间整体置换一次后再做 SLS，破坏"样本-分布"配对但保留全局分歧统计。
    - 核心思路：两个控制各回答一个 confound——前者验证"循环顺序"无关，后者验证"每例匹配自己的分布"才是关键；如果 shuffled SLS 仍能 work，那 SLS 的好处就只是"噪声正则"。
    - 设计动机：把"分布作为标签结构信息"和"分布作为通用噪声源"两种解释分开。实验中 shuffled SLS 退化到 12% 准确率，确认配对是 first-order 因子。

### 损失函数 / 训练策略
- Soft label：$\mathcal{L}_{\mathrm{soft}}=-\sum_i\sum_k p_{ik}\log q_{\theta,k}(x_i)$；hard 方法都用普通 hard-label CE。
- 优化器：SGD，lr=0.1，momentum=0.9，weight_decay=$5\times 10^{-4}$，cosine annealing 200 epochs，batch=128，仅随机裁剪+水平翻转。
- 评估遵循 proper-scoring-rule 视角：soft NLL 为主，KL-to-annotator 与 soft Brier 为辅，hard_acc / equal-mass ECE / Spearman 熵相关 为次。Hessian 用 power iteration 估 $\lambda_{\max}$、Hutchinson 估 trace，都在 best-soft-NLL checkpoint 的 eval 模式 + BN frozen 下计算。

## 实验关键数据

### 主实验
CIFAR-10H 10000 张图，80/20 stratified split，10 seeds 主对照 / 5 paired seeds 用于 hard-delivery 家族和稀疏扫描。

| 方法 | Soft NLL ↓ | Hard Acc ↑ | ECE_eqmass ↓ | EntCorr ↑ |
|------|-----------|-----------|-------------|-----------|
| Majority vote | 0.7284 | 0.8570 | 0.0704 | 0.2902 |
| Label smoothing | 0.6263 | 0.8590 | 0.0598 | 0.2117 |
| Mixup | 0.5526 | **0.8824** | 0.0977 | 0.2499 |
| Soft labels | 0.5096 | 0.8687 | 0.0185 | 0.3909 |
| **SLS** | **0.5052** | 0.8695 | 0.0186 | **0.3946** |

全分布时 SLS 与 soft 在 4 项指标上均无显著差异（$p\in[0.38,0.92]$），mixup 准确率最高但 ECE 几乎差 5 倍。

稀疏标注扫描（K∈{5,10,25,50}，每格 5 paired seeds 的 soft NLL）：

| 方法 | K=5 | K=10 | K=25 | K=50 |
|------|-----|------|------|------|
| Soft labels | 0.5860 | 0.5785 | 0.5388 | 0.5628 |
| SLS | 0.5599 (p=.031) | 0.5485 (.031) | 0.5169 (.063) | 0.5291 (.094) |
| Multipass | 0.5649 (.063) | 0.5371 (.031) | 0.5117 (.031) | 0.5241 (.031) |
| Det. control | **0.5555** (.031) | **0.5388** (.031) | **0.5077** (.031) | **0.5231** (.031) |

12/12 cell 均方向上优于 soft，其中 9/12 全 5 seed 一致；K 越小提升越大；改进与"经验 $p_i$ 到真分布 JS 距离"Spearman 相关 0.05–0.16 正相关。

### 消融实验

| 配置 | Soft NLL | Hard Acc | EntCorr | $\lambda_{\max}$ (full) | Trace (full) |
|------|----------|----------|---------|-------------------------|--------------|
| Soft labels | 0.5096 | 0.8687 | 0.3909 | 242.2 | 4946.0 |
| SLS | 0.5052 | 0.8695 | 0.3946 | **104.9** | **1633.9** |
| Multipass | 0.4942 | 0.8714 | 0.4000 | 103.8 | 1571.4 |
| Det. control | 0.4921 | 0.8724 | 0.3963 | 101.6 | 1581.6 |
| Shuffled SLS | 2.2973 | 0.1199 | -0.006 | 18.4 | 33.9 |

Multipass / SLS / Det. control 几何上几乎相同（$\lambda_{\max}$ ~100，比 soft 小 2.4×），shuffled SLS 形式上最"平"但接近随机分类——说明"平坦"必须在保持配对的前提下解读。Stale-target 探针：把采样标签固定 1/5/10/50 epoch，soft NLL 从 0.5027 单调恶化到 0.6689。

### 关键发现
- **配对是 first-order**：shuffled SLS 几何最平却 12% 准确率，证明 hard delivery 的好处不是"通用噪声正则"，而是"每例匹配自己的分布"+ 噪声协方差被分歧重塑。
- **结构化梯度噪声**：最后一层梯度方差与标注者熵的 Spearman 相关跨 seed 均值 0.939——SLS 注入的额外方差恰好集中在分歧大的样本上，和 Proposition 1 的 $\mathrm{Diag}(p)-pp^\top$ 一致。
- **同目标不同 basin**：SLS 和 soft checkpoint 间 mean loss barrier=2.05 远大于 0，CKA 0.920 vs 0.887、Grad-CAM 跨 seed 稳定性 0.901 vs 0.804——端点 loss 几乎相同但占据不同 basin，hard delivery 的表征更可复现。
- **OOD 端点收益**：SVHN 6 个 detector 中 5 个 hard 优于 soft；CIFAR-100 上 SLS 在所有 score 上 AUROC 优于 soft，Energy/ODIN paired $p=0.0186$。

## 亮点与洞察
- **把 delivery 当独立变量是聪明的研究范式**：以往关于"用标注者分布训练"的工作总是把 target 和 delivery 一起换，本文用 Proposition 1 把二者数学解耦，再用 deterministic control 把"采样随机性"也解耦——三层 ablation 设计可直接复用到任何 stochastic vs deterministic 优化对比。
- **Multipass 是被忽视的实用基线**：当你有原始投票计数时，按多重集循环既是确定性的、可复现的，又免去了 SLS 的方差且不像 Sheng 2008 那样改变数据集大小，是"最便宜的硬标签 delivery"。
- **"flat basin via structured label noise"的新证据**：以往 flat-minima 文献多用 SGD 噪声或对称 label noise 论证（Keskar、Smith、Damian），本文证明"按真实分布采样的硬标签"也能驱使模型进入 ~2.4× 更平坦的 basin，且方差结构与 $\mathrm{Diag}(p)-pp^\top$ 严格对应，给 HaoChen/Wu "covariance-Hessian alignment" 提供了具体可解释的例证。

## 局限与展望
- 作者承认：单数据集 (CIFAR-10H) + 单架构 (ResNet-18 改 CIFAR stem)，未在 ChaosNLI / LeWiDi / 更大视觉模型上验证；稀疏扫描 12 个 cell 都方向上优但 Holm 校正后无单 cell 显著（5 paired seeds 的最小 raw p=0.03125）；OOD 比较跨了 10-seed 主表和 5-seed 消融表，作者明确按描述性证据处理。
- 本笔记认为：稀疏 K 是通过对 dense CIFAR-10H 重采样模拟的，真实稀疏数据集还会叠加"标注者选择偏倚"和"任务相关分歧结构"，hard delivery 的相对收益是否保留未知；命题 1 只给了 per-step 方差，basin 几何到端点指标的因果链仍是观察性的。
- 改进方向：把 multipass 推到 ChaosNLI / SBIC 这类高分歧 NLP 数据；用 SAM 类显式 flatness 优化器对比，看"结构化标签噪声"是否能替代显式 sharpness 正则；研究 multipass 的循环长度 $m_i$ 与 LR schedule 的相互作用。

## 相关工作与启发
- **vs Peterson et al. (2019)**：他们提出 CIFAR-10H 并证明 soft-label CE > majority vote；本文保留同一目标分布只换 delivery，是更细的因果切片。
- **vs Sheng et al. (2008) 重复标注展开**：Sheng 让数据集变 $N\to\sum m_i$；multipass 保持 $N$ 不变，循环投喂——同一思想的 cardinality-preserving 版本。
- **vs DisturbLabel (Xie et al., 2016)**：DisturbLabel 用均匀噪声扰动标签；SLS 从真实 $p_i$ 采样，噪声结构对齐目标分布，因而是"信息保持型噪声"而非破坏型。
- **vs Label smoothing / Mixup**：LS 是均匀软目标、mixup 是输入与目标同插值，二者都不保留每例的 epistemic 结构；表 2 显示二者 ECE 远差于 SLS/soft，印证"必须保留 example-to-distribution 配对"。
- **vs flat-minima 文献 (Keskar 2017, Smith 2021, HaoChen 2021)**：本文给"hard-label delivery 也能让 basin 变平"提供新机制，且方差协方差有精确解析形式。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把 delivery 当独立变量并用期望等价命题严格解耦，是 annotator-disagreement 领域少见的因果切片；但 SLS 的采样思想在 Peterson 2019 已有雏形。
- 实验充分度: ⭐⭐⭐⭐ 三种主方法 + 两种控制 + 稀疏扫描 + Hessian + barrier + CKA + Grad-CAM + OOD 全套，但单数据集单架构限制泛化结论。
- 写作质量: ⭐⭐⭐⭐ 章节组织清晰，"regime split"叙事贯穿全文，命题与控制的角色都解释到位；表格略多需细读。
- 价值: ⭐⭐⭐⭐ 给"有标注者分布的训练任务"提供了 multipass 这个实用默认，并把 flat-minima 与 structured label noise 之间建立了具体可验证的桥梁。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Retraining with Predicted Hard Labels Provably Increases Model Accuracy](../../ICML2025/ai_safety/retraining_with_predicted_hard_labels_provably_increases_model_accuracy.md)
- [\[ICML 2026\] How Hard Can It Be? Hardness-Aware Multi-Objective Unlearning](how_hard_can_it_be_hardness-aware_multi-objective_unlearning.md)
- [\[ICML 2026\] Exposing Vulnerabilities in Explanation for Time Series Classifiers via Dual-Target Adversarial Attack](exposing_vulnerabilities_in_explanation_for_time_series_classifiers_via_dual-tar.md)
- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](../../AAAI2026/ai_safety/rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[AAAI 2026\] Easy to Learn, Yet Hard to Forget: Towards Robust Unlearning Under Bias](../../AAAI2026/ai_safety/easy_to_learn_yet_hard_to_forget_towards_robust_unlearning_under_bias.md)

</div>

<!-- RELATED:END -->
