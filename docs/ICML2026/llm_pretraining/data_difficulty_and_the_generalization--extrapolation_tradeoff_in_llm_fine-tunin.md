---
title: >-
  [论文解读] Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning
description: >-
  [ICML 2026][预训练][数据难度] 本文系统研究 SFT 中数据难度的作用，发现并不存在"普适最优难度"，而是存在一个**随数据规模增大而向更难方向漂移**的最优难度，并用"in-distribution 泛化 gap"与"extrapolation gap"两个 gap 的 trade-off 给…
tags:
  - "ICML 2026"
  - "预训练"
  - "数据难度"
  - "监督微调"
  - "泛化-外推权衡"
  - "PAC-Bayes"
  - "数据规模"
---

# Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning

**会议**: ICML 2026  
**arXiv**: [2605.12906](https://arxiv.org/abs/2605.12906)  
**代码**: 无  
**领域**: LLM 预训练 / SFT 数据选择  
**关键词**: 数据难度, 监督微调, 泛化-外推权衡, PAC-Bayes, 数据规模

## 一句话总结
本文系统研究 SFT 中数据难度的作用，发现并不存在"普适最优难度"，而是存在一个**随数据规模增大而向更难方向漂移**的最优难度，并用"in-distribution 泛化 gap"与"extrapolation gap"两个 gap 的 trade-off 给出 PAC-Bayes 解释。

## 研究背景与动机

**领域现状**：SFT 时挑数据的启发式五花八门——有人说要剔除"太简单"的（LIMO、s1、Marion et al.），有人说要保留"和 base model 分布接近的简单"数据（BERTIN、DFT、Anchored-SFT），还有人主张"中等难度最好"。每个论文都拿出漂亮的对比表，但相互之间矛盾重重。

**现有痛点**：上述结论缺乏统一框架解释，导致工程上"该挑难数据还是简单数据"成了玄学。Table 1 中作者在 OpenR1-Math-94k 上 medium 最优、在 OpenMath 上 easy 最优、在 OpenScience 上 easy/medium 接近 hard 暴跌，同一模型同一评测在不同数据集结论就翻转。

**核心矛盾**：先前工作几乎都在"固定数据规模"下比较难度，但**难度与数据规模并非独立变量**——它们共同决定了 SFT 后的模型性能。Figure 2 给出关键观察：剔除"难"样本在小数据时有益、大数据时有害；剔除"易"样本反之。

**本文目标**：(1) 建立 (数据规模 $n$, 数据难度) 二维实验图谱；(2) 用一个 mechanism 同时解释"难度非单调"和"最优难度随 $n$ 漂移"两件事；(3) 给出可解释的理论上界。

**切入角度**：把测试风险分解成**in-distribution 泛化 gap** $G_{\mathrm{gen}}$ 与 **extrapolation gap** $G_{\mathrm{ext}}$——前者随难度升高（更难拟合）、随 $n$ 减小，后者随难度升高反而下降（更难的训练分布覆盖更难的测试分布）。两个 gap 反向运动产生**单峰**的"最优难度"。

**核心 idea**：用"训练-测试两个分布之间的 TV/KL gap 与 posterior-prior KL gap 的 trade-off"替代"难/易的二分逻辑"，并指出 $n$ 增大主要压缩 $G_{\mathrm{gen}}$，因此最优难度随 $n$ 单调右移。

## 方法详解

本文几乎不是"提方法"的论文，而是"建机制+理论上界+大量受控实验"。

### 整体框架
全文分三层：(a) 真实数据上的 SFT 二维扫描（Qwen2.5-Math-1.5B/7B × OpenMath 各 difficulty bucket × 各 size）；(b) 合成数据 iGSM 上的精确难度控制实验，以 op 数量代表难度，并按测试集 difficulty slice 分别评估，分离出"in-distribution 拟合崩了"vs"extrapolation 不动"两种失败模式；(c) PAC-Bayes 理论框架给出可解释的两-gap 分解上界（Proposition 4.1）。

### 关键设计

1. **CoT-length 作为难度度量**:

    - 功能：用 ground-truth Chain-of-Thought 长度作为问题难度的代理指标。
    - 核心思路：Figure 1 验证 CoT 长度与外部 LLM 通过率呈强负相关——CoT 越长，pass rate 越低，说明问题越难；这避免了"用本模型困惑度度量自身要学的难度"带来的循环依赖。
    - 设计动机：perplexity 难度依赖被评估模型本身，会随 SFT 漂移；CoT 长度是 task-side 属性，跨模型可比，方便构造"同样难度、不同模型"的实验。

2. **二维实验图谱 (size × difficulty) + decomposed evaluation**:

    - 功能：取消"固定 $n$ 只扫难度"或"固定难度只扫 $n$"的局部视角，绘制完整热图，并把测试集按 op 数切片，看 SFT 模型在每个测试难度上的提升量。
    - 核心思路：Figure 6 展示两种典型失败模式——easy 训练时 in-domain 测试涨、hard 测试掉（extrapolation 失败）；hard 训练且 $n$ 小时全 slice 一起掉（generalization 失败）。这是把"总分变化"拆成"哪里崩了"的关键诊断。
    - 设计动机：单一总分会掩盖机制；切片才能看出 trade-off 的两端在哪里出问题，进而支持后续 PAC-Bayes 上界的物理解释。

3. **两-gap PAC-Bayes 分解 (Proposition 4.1)**:

    - 功能：把测试 risk 上界写成 $\mathbb{E}_{\theta\sim\pi_\mathrm{train}}[R_{\mathcal D_\mathrm{test}}(\theta)]\le \mathbb E[\hat R_S(\theta)] + G_\mathrm{gen}+G_\mathrm{ext}+\epsilon$，其中 $G_\mathrm{gen}=\mathcal O(\sqrt{\mathrm{KL}(\pi_\mathrm{train}\|\pi_\mathrm{pre})/n})$ ，$G_\mathrm{ext}=\mathcal O(\mathrm{TV}(\mathcal D_\mathrm{test},\mathcal D_\mathrm{train}))$。
    - 核心思路：把预训练当作 prior $\pi_\mathrm{pre}$、SFT 得到的参数分布当作 posterior $\pi_\mathrm{train}$，PAC-Bayes 给出 posterior-prior KL 的复杂度项；TV 项捕捉训练分布到测试分布的偏移。难度上升 → posterior 离 prior 更远 → $G_\mathrm{gen}$ 升；但训练分布更靠近难测试集 → $G_\mathrm{ext}$ 降；两者反向运动产生单峰。
    - 设计动机：先前 SFT 数据选择缺乏理论锚点；这个上界既能解释 4 大观察（数据规模、难度非单调、最优难度漂移、模型相对难度），又给出"调难度等价于在 KL-TV 之间做正则"的几何直觉。

### 损失函数 / 训练策略
SFT 使用标准 CE，难度切片由 CoT 长度三等分（easy/medium/hard）或 op 数等距划分（iGSM 实验）；DFT 案例研究中 token 权重为 $\mathrm{sg}(p_\theta)\cdot \nabla\log p_\theta$，将"token 概率"作为隐式难度信号——这部分作为延伸案例验证理论对"token 级数据选择"同样成立。

## 实验关键数据

### 主实验

| 数据集 | base 模型 | Easy | Medium | Hard | 最优难度 |
|---|---|---|---|---|---|
| OpenR1-Math-94k (Math500) | Qwen2.5-Math-1.5B | 61.1 | **68.3** | 61.7 | medium |
| OpenMath 200k subset (Math500) | Qwen2.5-Math-1.5B | **71.7** | 70.1 | 69.0 | easy |
| OpenScience 200k subset (MMLU) | Qwen2.5-Math-1.5B | **53.4** | 53.0 | 41.2 | easy |

二维扫描结论（Figure 3-4，OpenMath/Qwen2.5-Math-7B）：固定 $n$ 时性能-难度曲线呈倒 U 型；固定难度时性能-规模呈对数饱和；最优难度随 $n$ 增大向更难漂移。

### 消融实验（合成 iGSM 控制实验，Section 4-5）

| 配置 | 现象 | 解释 |
|---|---|---|
| Base Ops[2–8]2k, hard 训练 + 小 $n$ | 全 slice 测试一起掉 | $G_\mathrm{gen}$ 主导（拟合不上） |
| Base Ops[2–8]2k, easy 训练 + 任意 $n$ | 易测试涨、难测试掉 | $G_\mathrm{ext}$ 主导（覆盖不到） |
| Base Ops[2–8]2k, medium 训练 | 整体提升最高 | $G_\mathrm{gen}+G_\mathrm{ext}$ 之和最小 |
| 强 base (Ops[2–8]2k vs Ops[2–6]2k) | 强 base 最优难度右移 | prior 更强 → $G_\mathrm{gen}$ 项更小 |
| DFT vs SFT，小 $n$ + 难数据 | DFT 优于 SFT | DFT 偏向高概率 token，等价于隐式降难度 |
| DFT vs SFT，大 $n$ | SFT 反超 | DFT 高概率 token 偏置压低了 $G_\mathrm{ext}$ 改善 |

### 关键发现
- "最优难度"是 $n$ 的递增函数：小数据偏好简单样本（降 $G_\mathrm{gen}$），大数据偏好困难样本（降 $G_\mathrm{ext}$），这一论断在真实数学/科学数据与合成 iGSM 上同时复现。
- 难度是**相对**的：同一份"hard"对强 base 是 medium、对弱 base 是 ultra-hard；因此数据选择必须考虑 base 模型能力，而非绝对 token 长度。
- DFT 的非普适增益被理论自然解释——它本质上是一个隐式 easy-shift，因此在"训练难度过高且 $n$ 不足"时受益、在"数据充足"时反被 $G_\mathrm{ext}$ 拖后腿。

## 亮点与洞察
- 把先前论文里看似矛盾的"easy/medium/hard 谁更好"全部统一到一个 $G_\mathrm{gen}$-$G_\mathrm{ext}$ 反向运动的图像里——这是少见的"用理论收口实验混乱"的好例子。
- iGSM 上的 decomposed evaluation 是真正能"看见 mechanism"的诊断利器——一旦你按测试难度切片画图，"模型为什么掉点"立刻揭穿，应迁移到所有 SFT 数据消融工作。
- 把 SFT 看作"posterior 偏离 prior + 训练-测试分布偏移"的双源风险，是把传统 PAC-Bayes 直接落到 LLM SFT 的最自然形式，给"按数据规模调难度"提供了清晰的物理解释。

## 局限与展望
- 理论上界仍是 worst-case 形式，TV 与 KL 的具体取值在真实文本上几乎不可估计，只能定性指引；如何把"最优难度"做成可计算的可执行准则，作者自己也承认是 open。
- 实验主要在 Qwen2.5-Math 与 Llama 数学家族，扩展到代码、agent、通用对话等多领域 SFT 时，"CoT 长度"作为难度度量未必稳定。
- DFT 案例只是延伸验证，没有给出"在 token 级如何按本理论自适应调节"的具体算法；未来可设计 size-dependent token weighting 把理论直接落地。

## 相关工作与启发
- **vs LIMO / s1 (Ye et al. 2025, Muennighoff et al. 2025)**: 它们一律剔除"base 已会的简单题"，对应"无脑选最难"。本文证明这只在 $n$ 大时合理，小数据下反而灾难。
- **vs BERTIN / Zhang et al. 2025**: 提倡"贴近 base 分布的简单样本"，对应"无脑选最简单"。本文证明这只在小数据合理。
- **vs DFT (Wu et al. 2025)**: token 级隐式偏向 easy，被本文吸收为 token-level 案例，统一解释为何 DFT 在不同 setting 表现忽好忽坏。
- **vs Curriculum Learning**: 本文给出 curriculum 之所以"经常有效但不总是"的根因——curriculum 等价于在训练过程中沿着"最优难度随 $n$ 增长"的曲线移动，但若 schedule 错位反而走偏。

## 评分
- 新颖性: ⭐⭐⭐⭐ 不是新方法，但用统一框架收口一堆矛盾结论，本身是稀缺贡献。
- 实验充分度: ⭐⭐⭐⭐⭐ 真实数据 + iGSM 合成数据 + 多 base 模型 + 多评测，二维热图+ slice 分析覆盖很全。
- 写作质量: ⭐⭐⭐⭐ 逻辑层次清晰，PAC-Bayes 解释紧扣 4 大观察；公式排版略密但可读。
- 价值: ⭐⭐⭐⭐ 对正在做 SFT 数据筛选的团队是直接 actionable 的指导——不要再固定挑 easy/hard，而要按 base 能力与数据预算联合决定。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Token-level Data Selection for Safe LLM Fine-tuning](../../ICLR2026/llm_pretraining/token-level_data_selection_for_safe_llm_fine-tuning.md)
- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](../../ICLR2026/llm_pretraining/pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)
- [\[ICML 2026\] Tuning the Implicit Regularizer of Masked Diffusion Language Models: Enhancing Generalization via Insights from k-Parity](tuning_the_implicit_regularizer_of_masked_diffusion_language_models_enhancing_ge.md)
- [\[ICML 2025\] DipLLM: Fine-Tuning LLM for Strategic Decision-Making in Diplomacy](../../ICML2025/llm_pretraining/dipllm_fine-tuning_llm_for_strategic_decision-making_in_diplomacy.md)
- [\[ACL 2025\] Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](../../ACL2025/llm_pretraining/data_whisperer_data_selection.md)

</div>

<!-- RELATED:END -->
