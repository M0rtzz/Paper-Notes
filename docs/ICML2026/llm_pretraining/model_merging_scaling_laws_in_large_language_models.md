---
title: >-
  [论文解读] Model Merging Scaling Laws in Large Language Models
description: >-
  [ICML 2026][预训练][模型合并] 作者用 10,866 个合并模型实测出一条形如 $L=L_*+BN^{-\beta}+A_0 N^{-\gamma}/(k+b)$ 的双轴幂律：基座规模 $N$ 决定 floor，专家数 $k$ 决定 tail…
tags:
  - "ICML 2026"
  - "预训练"
  - "模型合并"
  - "scaling law"
  - "power law"
  - "task arithmetic"
  - "TIES/DARE"
---

# Model Merging Scaling Laws in Large Language Models

**会议**: ICML 2026  
**arXiv**: [2509.24244](https://arxiv.org/abs/2509.24244)  
**代码**: https://github.com/InfiXAI/Merging-Scaling-Law (有)  
**领域**: LLM 预训练 / 模型合并 / Scaling Law  
**关键词**: 模型合并、scaling law、power law、task arithmetic、TIES/DARE

## 一句话总结
作者用 10,866 个合并模型实测出一条形如 $L=L_*+BN^{-\beta}+A_0 N^{-\gamma}/(k+b)$ 的双轴幂律：基座规模 $N$ 决定 floor，专家数 $k$ 决定 tail，且四种主流合并方法（Average、TA、TIES、DARE）都共用同一条曲线，从而把"合多少个专家、合到哪一步停"变成一个可预测、可预算的工程问题。

## 研究背景与动机
**领域现状**：模型合并（Model Merging）已经成为继多任务 SFT 之后的低成本"专家整合"范式。线性加权（Model Soups、Task Arithmetic）和带预处理的版本（TIES、DARE）在 LLM、LoRA 适配器等场景被广泛使用。

**现有痛点**：合并本质上还是"凭手感"——试不同子集、不同顺序、不同归一化系数，开销很大却没有像预训练那样的 scaling law 指导。给定一个目标 loss，没人能预先回答"我到底需要几个专家"或者"基座放大一倍 vs 多融一个专家哪个更划算"。

**核心矛盾**：合并的收益曲线明显不是线性，但又确实存在某种规律性（早期收益陡、后期饱和）。如果没有解析形式去描述这条曲线，工程实践就只能用穷举搜索，浪费 GPU。

**本文目标**：（1）找到一条同时刻画 $N$（基座参数量）和 $k$（合并专家数）影响的紧凑公式；（2）证明它对不同合并算法、不同骨干、in-domain 与 cross-domain 都成立；（3）给出"只测三个点就能外推整条曲线"的实操方法。

**切入角度**：把合并看作"对若干 task vector 做等权重平均"。在二阶 Taylor 展开下，等权平均的方差会以 $1/k$ 速率收缩，而方差通过 Hessian 进入 loss 就是 $A(N)/k$ 那一项。作者由此预期"floor + 1/k tail"的结构，并大规模实证验证。

**核心 idea**：用一条"floor + 1/(k+b) tail"的幂律统一描述所有合并方法的 CE 曲线，把基座规模和专家数这两个尺度统一进同一公式，使合并变成 budget-aware 的可预测过程。

## 方法详解

### 整体框架
作者在 Qwen2.5 系列（0.5B/1.5B/3B/7B/14B/32B/72B）上从同一个基座微调九个领域专家（algebra、analysis、geometry、discrete、number_theory、code、chemistry、physics、biology），覆盖 in-domain 与 cross-domain 两种评估。对每个 $(N,k)$ 组合，遍历或均匀采样所有 $\binom{9}{k}$ 的专家子集，跑四种合并算法（Average、TA、TIES、DARE）合成模型并测 token-level CE，最终得到 10,866 个合并模型的网格数据。然后用加权非线性最小二乘拟合一条形如 $\mathbb{E}[L\mid N,k]=L_\infty(N)+A(N)/(k+b)$ 的曲线，其中 $L_\infty(N)=L_*+BN^{-\beta}$、$A(N)=A_0 N^{-\gamma}$，再用 R² 与残差结构验证。

### 关键设计

1. **统一 floor+tail 幂律**:

    - 功能：用一个公式同时刻画基座规模和专家数对合并 loss 的影响。
    - 核心思路：$\mathbb{E}[L\mid N,k]=L_*+BN^{-\beta}+\frac{A_0 N^{-\gamma}}{k+b}$，其中 floor 项 $L_*+BN^{-\beta}$ 随 $N$ 单调下降，tail 项 $A_0 N^{-\gamma}/(k+b)$ 随 $k$ 以倒数速率衰减；拟合时用权重 $\propto k$ 来稳定早期 $k$ 噪声，所有方法在所有切片上 $R^2>0.98$。
    - 设计动机：把"更大基座更好合"和"专家越多收益递减"两个观察合并到同一表达式，使预算决策（"再加一个专家 vs 把基座放大一档"）可以直接通过两项的相对量级来比较。

2. **从二阶 Taylor 导出 1/k tail 的理论**:

    - 功能：解释为什么所有合并算法在等权归一化下都呈现 $1/k$ 的尾部。
    - 核心思路：把每个 task vector 写成 $v_i$，等权合并后扰动均值是 $c\mu$、协方差是 $\Sigma/k$；对 loss 做二阶 Taylor 展开得到 $\mathbb{E}[L]=L(\theta_0)+cg^\top\mu+\frac{1}{2}c^2\mu^\top H\mu+\frac{c^2}{2k}\mathrm{Tr}(H\Sigma)+\mathcal{O}(k^{-3/2})$，前面三项凝聚成 $L_\infty(N)$，最后一项就是 $A(N)/k$；Corollary 进一步说明 subset 间的 std 以 $1/\sqrt{k}$ 收缩。TIES/DARE 这类预处理算法被吸收成对 $\Psi(v)$ 的修改，不改变 leading-order 形式。
    - 设计动机：从理论上把"为什么 1/k"讲清楚，而不是仅给出经验拟合；同时解释为什么 TIES、DARE 这些差异巨大的实现最终也都落在同一曲线上。

3. **三点拟合 + 推荐 $k^*$ 的预算算法**:

    - 功能：只用 $k\in\{1,2,4\}$ 三个点就能外推整条 $k$-曲线，并给出"性价比最高的专家数" $k^*$。
    - 核心思路：因为公式只有 $L_\infty$、$A$、$b$ 三个自由度，理论上三点定型；论文经验显示三点拟合可以恢复完整 9 点曲线，并把 $k^*$ 稳定地估计在 $5\sim 6$，对应 elbow 位置 $\Delta_k\approx A/[(k+b)(k+1+b)]\sim k^{-2}$。
    - 设计动机：在真实场景下做完整 $k$-grid 的开销很大；三点法把"先测一小批再决定预算"做成可行流程，让合并从"试错"变成"测量+外推"。

### 损失函数 / 训练策略
论文不引入新的训练损失，所有数据点来自冻结的基座+独立微调的 9 个领域专家，用 token-level cross-entropy 在 30M held-out token 上评估；合并系数采用等权归一化 $\alpha_{i,k}=c/k$。拟合采用加权非线性最小二乘，权重 $\propto k$ 以抑制小 $k$ 时的高方差。

## 实验关键数据

### 主实验

| 设置 | 模型规模 $N$ | $k=9$ 时域均 CE | 相比 0.5B 降幅 |
|------|------------|-----------------|---------------|
| In-domain | 0.5B | 0.739 | — |
| In-domain | 7B | ~0.52 | ~30% |
| In-domain | 32B | 0.430 | 41.9% |
| Cross-domain | 0.5B→32B | 同步下移 | floor 与 tail 都缩小 |
| 拟合质量 | 全部点 | $R^2>0.98$ | floor/tail 均匀残差 |

### 消融实验

| 配置 | 关键观察 | 说明 |
|------|---------|------|
| Average / TA / TIES / DARE | 同一公式 $R^2>0.98$ | 方法差异被吸收进 $L_\infty$、$A$、$b$ 三个常数 |
| 候选池 $M=9\to 8\to 7$ | floor 几乎不变，tail 减小幅度变小 | 多样性主要拉低 tail 而非 floor |
| 三点 $k\in\{1,2,4\}$ 拟合 | 推断 9 点曲线误差 < 全拟合的几倍 | 三点法足够支撑预算决策 |
| 不同 donor 顺序（DARE） | $k=8$ 时 whisker 长度缩 ~83% | 顺序敏感性以 $1/(k+b)$ 收缩 |
| 跨骨干（LLaMA-3.2 3B / LLaMA-3 8B） | 同样的 1/k tail | 公式形态可迁移 |

### 关键发现
- "更大基座更好合"被定量化：32B 相比 0.5B 在 $k=9$ 时 CE 直降 41.9%，floor 和 tail 同时缩小，相当于既给了更低的渐近性能又减少了所需的专家数。
- elbow 普遍出现在 $k\approx 5\sim 6$：达到 85% 收益只需 5 个专家、90% 只需 6 个；超过这个数，新增专家几乎只是"刷数据"。
- 方法差异在大尺度下被压平：$N=32B$、$k\approx 8$ 时 Avg/TA/TIES/DARE 的 mean CE 差距 $\lesssim 2\%$，merge-to-merge 方差按 $\sim 1/k$ 收缩到共同 floor。
- order sensitivity 同样以 $1/(k+b)$ 衰减，$k\geq 6$ 之后精挑顺序基本没有意义。

## 亮点与洞察
- 用 10,866 个真实合并模型把"folk wisdom"拍成 $R^2>0.98$ 的硬曲线，规模和系统性远超此前任何 merging 论文，是这条领域目前最权威的实证依据。
- floor 与 tail 解耦的视角非常实用：用 $A/L$ 的相对量级就能秒判"再融一个专家 vs 把基座升一档"哪个 ROI 更高，这是对工业界算力分配的直接价值。
- 三点拟合法把 scaling law 从"事后总结"升级为"提前预测"工具，不需要跑完所有 $k$ 就能锁定 elbow，这种"测量-外推"思路可以迁移到其它合成性研究（如 RAG 检索源数量、ensemble 模型数）。

## 局限与展望
- 公式只覆盖等权归一化合并，对非等权或学得的权重（如基于路由/优化的 merge）只能解释 leading order，差异要靠 finite-$k$ 偏差吸收。
- 专家容量被当作隐变量塞进了 $A(N)$，没有显式建模 LoRA rank、微调 token 数等"专家强度"维度，论文也承认这是自然扩展。
- 评测只用 cross-entropy，与下游 task accuracy 之间还有距离，对"代码/数学"这种长尾任务的 elbow 是否一致仍需验证。
- 9 个领域虽多样但都是 Mixture-of-Thoughts/OpenScience 这一系列数据，对真正异质（如多语言、多模态、安全对齐）的合并场景外推性待考。

## 相关工作与启发
- **vs Kaplan/Chinchilla 等预训练 scaling law**: 它们刻画 $(N, D, C)$ 与 loss 的关系，本文新增了"专家数 $k$"这一组合维度，并显示它和 $N$ 是可解耦的两条坐标轴。
- **vs Yadav et al. (2024) 经验研究**: 后者经验上指出"方法差异随专家数变小"，本文用统一公式把这一观察解释为"共同 $L_\infty(N)$ 主导大 $k$、$A(N)/(k+b)$ tail 主导小 $k$"。
- **vs TIES/DARE 等具体合并算法**: 本文不与之竞争而是把它们"放进同一框架"，说明这些预处理只是把任务向量的均值/协方差稍作修改，不改变幂律骨架。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次给出 $(N,k)$ 双轴 merging scaling law，并配上一阶可证明的理论；公式本身简洁，但思路在 scaling law 谱系里属于自然延伸。
- 实验充分度: ⭐⭐⭐⭐⭐ 10,866 个合并模型、9 个领域、7 个规模、4 种方法、跨骨干验证，规模在 merging 文献里几乎独一档。
- 写作质量: ⭐⭐⭐⭐ 公式与图配合清晰，把 floor/tail 物理意义讲透；只是 in-domain/cross-domain 章节略有重复叙述。
- 价值: ⭐⭐⭐⭐⭐ 直接给出"三点拟合→预算决策"的可落地流程，对工业界合并、LoRA 仓库管理、专家路由都有立刻可用的工程意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)
- [\[CVPR 2026\] Model Merging in the Essential Subspace](../../CVPR2026/llm_pretraining/model_merging_in_the_essential_subspace.md)
- [\[ICML 2026\] Predicting Large Model Test Losses with a Noisy Quadratic System](predicting_large_model_test_losses_with_a_noisy_quadratic_system.md)
- [\[NeurIPS 2025\] Gemstones: A Model Suite for Multi-Faceted Scaling Laws](../../NeurIPS2025/llm_pretraining/gemstones_a_model_suite_for_multi-faceted_scaling_laws.md)
- [\[ICML 2026\] Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models](softplus_attention_with_re-weighting_boosts_length_extrapolation_in_large_langua.md)

</div>

<!-- RELATED:END -->
