---
title: >-
  [论文解读] On the Convergence Rate of LoRA Gradient Descent
description: >-
  [ICML 2026][优化/理论][LoRA] 本文首次在不假设 adapter 矩阵有界、不要求重参数化损失 Lipschitz 平滑的前提下，证明了原始 LoRA 梯度下降的最小梯度范数以 $O(1/\log T)$ 速率收敛（若参数范数有界则恢复经典 $O(1/T)$）…
tags:
  - "ICML 2026"
  - "优化/理论"
  - "LoRA"
  - "收敛分析"
  - "非 Lipschitz 平滑"
  - "Burer-Monteiro"
  - "自适应学习率"
---

# On the Convergence Rate of LoRA Gradient Descent

**会议**: ICML 2026  
**arXiv**: [2512.18248](https://arxiv.org/abs/2512.18248)  
**代码**: https://github.com/siqiaomu/lora  
**领域**: 优化理论 / LLM 高效微调 / LoRA  
**关键词**: LoRA、收敛分析、非 Lipschitz 平滑、Burer-Monteiro、自适应学习率

## 一句话总结
本文首次在不假设 adapter 矩阵有界、不要求重参数化损失 Lipschitz 平滑的前提下，证明了原始 LoRA 梯度下降的最小梯度范数以 $O(1/\log T)$ 速率收敛（若参数范数有界则恢复经典 $O(1/T)$），并据此设计了与理论严格对应的自适应/归一化学习率，在 logistic regression、ResNet-18、TinyLlama 上验证了训练加速与稳定性提升。

## 研究背景与动机

**领域现状**：LoRA（Low-Rank Adaptation）已成为 LLM 微调最流行的方案——冻结预训练权重 $W_0$，只训两个小矩阵 $A, B$ 使得新权重为 $W_0 + BA$。算法本身极简：每步同时对 $A, B$ 做梯度下降。

**现有痛点**：尽管 LoRA 简单且实证有效，它的收敛理论一直是个悖论——即使原始损失 $\mathcal{L}(W)$ 是 Lipschitz 平滑的，重参数化后的 $\mathcal{L}(BA)$ 关于 $A, B$ 不再 Lipschitz 平滑（因为 $\nabla_B \mathcal{L}(BA)$ 含 $A$ 的乘法因子），让经典的"用 descent lemma 推 $O(1/T)$"分析直接失效。

**核心矛盾**：现有 LoRA 理论分析分三类，都回避了这个核心难题——（1）**无穷正则极限**类（Kim 2025、Jang 2024、NTK 分析等）：只给渐近收敛或无限宽神经网络结论，不给有限模型的非渐近速率；（2）**类 LoRA 的变体**（GaLore、RAC-LoRA 等）：只更新单个 adapter 或加投影来保持 Lipschitz 平滑性，但不对应实际部署的 LoRA；（3）**强假设下的收敛**（Jiang 2024、Ghiasvand 2025）：假设 $A, B$ 的范数被某常数一致上界，相当于人为强加 Lipschitz 平滑，并把常数塞进收敛界——证明过程没有实质新意。

**本文目标**：在最弱可能的假设（仅原损失 Lipschitz 平滑 + 下有界）下，给出原始 LoRA 同步梯度下降的**非渐近收敛速率**，不假设 $A, B$ 范数有界。

**切入角度**：把 $A, B$ 堆叠成单个矩阵 $V$，则 $BA$ 出现在 $VV^T$ 的特定块中——这就是经典的 Burer-Monteiro 对称参数化。在 $V$ 视角下，LoRA 梯度下降变成对 $\mathcal{J}(V) = \mathcal{L}(E_1 V V^T E_2)$ 的标准梯度下降，问题被还原到"$VV^T$ 形式的非平滑优化"，可以套用更精细的修正 descent lemma。

**核心 idea**：堆叠重参数化 → 推导含高阶项的"Lipschitz-like" descent lemma → 通过限定学习率与 $\|V_t\|^2$ 和当前梯度成反比，保证每步下降；分析 $\|V_t\|^2 = O(t)$ 增长，让 $\sum \eta_t = \Theta(\log T)$，得 $O(1/\log T)$ 收敛率。

## 方法详解

### 整体框架
证明分三步走：（1）**问题重构**——把 LoRA 的 $A, B$ 堆成 $V = [B; A^T] \in \mathbb{R}^{(m+n) \times r}$，原损失变成 $\mathcal{J}(V) = \mathcal{L}(E_1 V V^T E_2)$，其中 $E_1, E_2$ 是抽取矩阵；LoRA 同步梯度下降等价于在 $V$ 上的标准梯度下降。（2）**修正 descent lemma**——证明 $\mathcal{J}$ 满足含 $\|V_2 - V_1\|^k$ ($k=2,3,4$) 高阶项的"Lipschitz-like"不等式（Lemma 3.3）。（3）**学习率控制 + 收敛**——选 $\eta_t = \min\{1/(4\sqrt{2}L(\|V_t\|^2 + \|\nabla\mathcal{L}(E_1 V_t V_t^T E_2)\|)), 1\}$ 保证每步下降至少 $\eta_t \|\nabla\mathcal{J}(V_t)\|^2 / 4$（Lemma 3.4），结合 $\|V_t\|^2 = O(t)$ 的最坏情况估计推出 $\sum_t \eta_t = \Theta(\log T)$，最后由 $\min_t \|\nabla\mathcal{J}(V_t)\|^2 \leq 4(\mathcal{J}(V_0) - \mathcal{L}^*) / \sum_t \eta_t$ 得 $O(1/\log T)$。

### 关键设计

1. **Burer-Monteiro 堆叠重参数化**:

    - 功能：把 LoRA 关于两个矩阵 $A, B$ 的非凸非平滑问题转化为关于单一矩阵 $V$ 的问题，让 $BA$ 出现在 $VV^T$ 的右上角块，从而可以套用 $VV^T$ 形式优化的成熟工具。
    - 核心思路：定义 $V = \begin{bmatrix} B \\ A^T \end{bmatrix} \in \mathbb{R}^{(m+n) \times r}$，则 $VV^T = \begin{bmatrix} BB^T & BA \\ A^T B^T & A^T A \end{bmatrix}$，用抽取矩阵 $E_1 = [I_m, 0]$、$E_2 = [0, I_n]^T$ 取出 $BA = E_1 V V^T E_2$。定义 $\mathcal{J}(V) = \mathcal{L}(E_1 V V^T E_2)$，由链式法则 $\nabla \mathcal{J}(V) = 2\,\mathrm{Sym}(E_1^T \nabla\mathcal{L}(E_1 V V^T E_2) E_2^T) V$，其乘法因子 $V$ 正是非平滑性的根源。LoRA 同步梯度更新等价于在 $V$ 上做标准梯度下降。
    - 设计动机：在 $V$ 视角下，多个看似零散的现象有统一解释——$V = 0$ 自动成为驻点（无论原损失结构如何），梯度的 $V$ 因子让小 $V$ 时梯度小（"原点附近的扁平区域"），范数大时学习率必须减小，这些都能在新坐标系下精确量化。Burer-Monteiro 形式还自动让结论可推广到一般 $VV^T$ 型参数化。

2. **修正的"Lipschitz-like" descent lemma**:

    - 功能：给出非 Lipschitz 平滑函数 $\mathcal{J}$ 的一步下降不等式，含一阶项 + 三个高阶项 + 一个梯度依赖项，对应"如果学习率足够小，仍能保证单步下降"。
    - 核心思路：Lemma 3.3 证明 $\mathcal{J}(V_2) \leq \mathcal{J}(V_1) + \langle \nabla\mathcal{J}(V_1), V_2 - V_1 \rangle_F + \sqrt{2}L\|V_2 - V_1\|^2 \|V_1\|^2 + \sqrt{2}L\|V_2 - V_1\|^3 \|V_1\| + \frac{\sqrt{2}L}{4}\|V_2 - V_1\|^4 + \|\nabla\mathcal{L}(E_1 V_1 V_1^T E_2)\|\|V_2 - V_1\|^2$。和经典 descent lemma 对比，多了三个高阶项（$\|V_1\|^2$、$\|V_1\|$、单独 $\|V_2 - V_1\|^4$）和一个梯度依赖项，反映 $V$ 范数和原梯度对下降步长的约束。
    - 设计动机：直接在 $V$ 上证 Lipschitz 平滑是不可能的（梯度有乘法因子 $V$），但通过细心展开 $\mathcal{J}(V_2) - \mathcal{J}(V_1)$ 的 Taylor 形式并用原损失的 Lipschitz 平滑性界出高阶项系数，得到一个"含高阶修正"的弱版下降条件。这是把非 Lipschitz 问题归约到可控形式的关键技巧。

3. **位置依赖的自适应学习率与 $O(1/\log T)$ 速率**:

    - 功能：根据当前迭代点的范数和梯度自动调节学习率，保证每步下降，并通过控制 $\|V_t\|$ 的增长速度推出非渐近收敛率。
    - 核心思路：Lemma 3.4 选 $\eta_t = \min\{1/(4\sqrt{2}L(\|V_t\|^2 + \|\nabla\mathcal{L}(E_1 V_t V_t^T E_2)\|)), 1\}$ 让高阶项被一阶项主导，得到一步下降 $\mathcal{J}(V_{t+1}) \leq \mathcal{J}(V_t) - \frac{\eta_t}{4}\|\nabla\mathcal{J}(V_t)\|^2$。对 $t$ 求和并由 $\mathcal{J} \geq \mathcal{L}^*$ 得 $\min_t \|\nabla\mathcal{J}(V_t)\|^2 \leq \frac{4(\mathcal{J}(V_0) - \mathcal{L}^*)}{\sum_t \eta_t}$。关键在估计 $\sum_t \eta_t$——最坏情况下 $\|V_t\|^2 = O(t)$，让 $\eta_t = \Omega(1/t)$，调和级数给出 $\sum_t \eta_t = \Theta(\log T)$，得 $O(1/\log T)$ 速率（Theorem 3.5）。如果额外假设 $\|V_t\| \leq C$，则 $\sum_t \eta_t = \Theta(T)$，恢复经典 $O(1/T)$。
    - 设计动机：理论上这条速率体现了 LoRA 的"位置依赖性"——迭代点远离原点时（$\|V\|$ 增大）必须降学习率，让收敛减速；靠近原点时可以激进。$V = 0$ 是人为造出的驻点，所以 LoRA 完全可能收敛到原点（即使原全连接最优在远处）——这是 LoRA 和全秩训练给出不同解的理论根源。实验中作者从这个公式出发设计了 $\eta^{adapt}$、$\eta^{adapt2}$、$\eta^{norm}$ 三种实用调度，把理论直接翻译成可部署的学习率策略。

### 损失函数 / 训练策略
证明只用两条假设：原损失 $\mathcal{L}$ 是 $L$-Lipschitz 平滑且下有界。算法是标准 LoRA 同步梯度下降：$A_{t+1} = A_t - \eta_t \nabla_A \mathcal{L}(B_t A_t)$，$B_{t+1} = B_t - \eta_t \nabla_B \mathcal{L}(B_t A_t)$。理论结果可自然扩展到多权重矩阵情形（Lemma 3.6 证明分块构造的 $\tilde{\mathcal{L}}$ 是 $2L$-Lipschitz 平滑）。

## 实验关键数据

### 主实验
实验目的不是 SOTA 而是验证理论。任务：CIFAR-10 分类，模型分三档——logistic regression on ResNet-18 embeddings（loss 已知 Lipschitz 平滑）、ResNet-18 直接训（卷积层加 LoRA，关闭 BatchNorm）、TinyLlama-1.1B 在 Alpaca 上 LoRA 微调。三种学习率方案：

| 学习率方案 | 公式 |
|----------|------|
| 自适应 $\eta^{adapt}$ | $\alpha / (\|V_t\|^2 + \|\nabla\mathcal{L}(E_1 V_t V_t^T E_2)\|)$ |
| 自适应 $\eta^{adapt2}$ | $\alpha / (\|V_t\|^2 + \sqrt{\mathcal{L}(E_1 V_t V_t^T E_2)})$ |
| 归一化 $\eta^{norm}$ | $\alpha / \|\nabla\mathcal{L}(V_t)\|^{1/2}$ |

| 实验 | 关键观察 |
|------|---------|
| Logistic regression（rank 4 / 20） | 三种非常数学习率都比同量级常数 lr 收敛更快且更稳；常数 lr 大就发散、小就慢；$\eta^{adapt}$ 和 $\eta^{adapt2}$ 初期高度相关 |
| ResNet-18 + LoRA on CIFAR-10 | $\eta^{adapt2}$ 和 $\eta^{norm}$ 显著稳定训练，$\eta^{norm}$ 表现最好；$\|V_t\|$ 在有限步后停止增长 |
| TinyLlama LoRA on Alpaca（$\sigma = 10^{-3}$） | 小初始化下自适应学习率比同量级常数 lr 收敛更快更稳 |
| TinyLlama LoRA on Alpaca（$\sigma = 1/r$） | 大初始化下自适应学习率接近常数 lr（因 $\|V_t\|$ 极大且增长极慢），优势减弱 |

### 消融实验
Logistic regression 训 1000 epoch 的长期行为观察：

| 现象 | 解释 |
|------|------|
| 损失早期看似收敛 | 但 $\|V_t\|$ 在所有 $t$ 上单调增长，无界 |
| 渐近收敛速度确实比 $O(1/T)$ 慢 | 验证理论中"$\|V_t\| \to \infty$ 时收敛减速到 $O(1/\log T)$" |
| ResNet-18 上 $\|V_t\|$ 有限步后停止增长 | 落入有界情形，速率为 $O(1/T)$ |

### 关键发现
- **理论与实验的精确对应**：实验观察到的"$\|V_t\|$ 增长 ↔ 学习率下降 ↔ 收敛减速"链条直接来自证明，三种学习率方案是公式 (8) 的实用近似。
- **LoRA 在初始化附近是平坦区域**：因为 $V=0$ 是驻点，标准初始化（$B=0$）让模型卡在低梯度区，需要大初始学习率"逃离"，这解释了为什么 $\eta^{adapt}/\eta^{adapt2}/\eta^{norm}$ 早期取高值后递减能稳定训练。
- **位置依赖性是 LoRA 独有的**：标准 GD 收敛速率不依赖参数范数，但 LoRA 因 $V$ 因子让远离原点时减速、靠近时加速——这是低秩重参数化对 loss landscape 几何结构改造的直接体现。
- **高维下自适应优势减弱**：当 $\|V_t\|$ 极大且变化慢（高维 LLM）时，$\eta^{adapt}$ 退化为接近常数 lr，看上去像 $O(1/T)$，但渐近行为仍是 $O(1/\log T)$。
- **收敛率与秩 $r$ 无关**：除了通过 $\|V_0\|^2$ 间接出现，$r$ 不直接进入速率——因为梯度下降本身就是 dimension-free 的。

## 亮点与洞察
- **第一个完全去除人为有界假设的非渐近 LoRA 收敛证明**：之前所有非渐近结果都靠"假设 $\|A\|, \|B\|$ 一致有界"绕开 Lipschitz 平滑性失败，本文用 $V$ 重参数化 + 修正 descent lemma 真正解决了这个核心难题。
- **$V = 0$ 永远是驻点的几何洞察**：揭示了"LoRA 可能收敛到原点（即使原最优在远处）"这一反直觉现象，从理论上解释了 LoRA 和全秩训练给出不同解的根本原因。
- **理论直接产出实用算法**：三个学习率公式 $\eta^{adapt}/\eta^{adapt2}/\eta^{norm}$ 不是事后凑出来的，而是直接从 Lemma 3.4 的 $\eta_t$ 选择公式推出，每一项都有明确理论含义；这种"理论 → 算法"的紧密对应在优化论文里少见。
- **Burer-Monteiro 视角的统一性**：作者顺便证明了对一般 Lipschitz 平滑函数，Burer-Monteiro 形式 $\min_V f(VV^T)$ 的梯度下降也收敛到驻点，把 LoRA 收敛理论嵌入更广的低秩参数化优化框架。
- **$O(1/\log T)$ 的"位置依赖减速"是 LoRA 内在性质**：不是证明松了，而是反映 LoRA 几何结构的实际行为，实验也观察到了对应的减速现象。

## 局限与展望
- **只覆盖确定性梯度下降**：实际 LoRA 训练用 SGD（随机批），随机噪声需要四阶矩界等额外工具才能推广，作者明确把这留为未来工作。
- **没分析凸/强凸情形**：经典 GD 在凸/强凸下有 $O(1/T)$ 或 $O((1-\mu/L)^T)$ 速率，LoRA 重参数化让原本的凸性也丧失，是否有更快收敛尚不清楚。
- **学习率公式依赖原梯度 $\nabla\mathcal{L}(E_1 V_t V_t^T E_2)$**：实际 LoRA 实现为省内存通常算 $xA^T B^T$ 而不显式构造 $BA$，所以 $\eta^{adapt}$ 实际开销大；$\eta^{adapt2}$ 用 loss 代替梯度缓解了这点但仍需额外评估。
- **$O(1/\log T)$ 的下界缺失**：作者没证这个速率是紧的——实际可能更快，理论只是上界。
- **没考虑常用变体（QLoRA、ReLoRA、LoRA+）**：这些变体的理论分析没覆盖，期待后续工作。
- **多矩阵扩展的常数有点放松**：Lemma 3.6 给出 $\tilde{\mathcal{L}}$ 是 $2L$-Lipschitz 平滑而不是 $L$，规模上去后常数累积可能不优。

## 相关工作与启发
- **vs Jiang 2024 / Ghiasvand 2025**：这两者都假设 $A, B$ 范数一致有界，本质上是把问题强行 Lipschitz 化，证明没新技巧；本文完全去掉这个假设，给出真正面对非 Lipschitz 困难的证明。
- **vs NTK 类分析（Jang 2024、Hayou 2024）**：只给无限宽极限的渐近性质，本文给有限模型非渐近速率。
- **vs RAC-LoRA / Bernoulli-LoRA**：这些变体只更新一个 adapter 来保持 Lipschitz 平滑性，不对应实际 LoRA；本文直接分析原始同步 LoRA。
- **vs GaLore / RSO / LDAdam**：这些是 LoRA-like 的低秩内存优化方法，但更新公式不同；本文证明的是 LoRA 本身，不直接迁移但提供了类似分析范式。
- **vs Burer-Monteiro 一般理论**：本文证明顺便覆盖了 BM 参数化梯度下降的收敛，与 BM 优化和半定规划文献接轨。
- **vs LoRA+（Hayou 2024）**：LoRA+ 提出对 $A, B$ 用不同学习率，本文的位置依赖学习率是另一个独立思路，两者可结合。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 真正解决了 LoRA 收敛理论的核心难题，证明技巧（$V$ 重参数 + 修正 descent lemma）是新的。
- 实验充分度: ⭐⭐⭐⭐ Logistic regression + ResNet-18 + TinyLlama 三档验证，主要目的不是 SOTA，覆盖足以支撑理论。
- 写作质量: ⭐⭐⭐⭐⭐ 证明三步走结构清晰，每一步动机解释到位，理论到实验的过渡自然。
- 价值: ⭐⭐⭐⭐ 理论价值高（LoRA 收敛终于有干净答案），实用价值中等（三种学习率方案值得在实际微调里试）。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Limits of Convergence-Rate Control for Open-Weight Safety](limits_of_convergence-rate_control_for_open-weight_safety.md)
- [\[ICML 2026\] Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence](balanced_lora_removing_parameter_invariance_to_accelerate_convergence.md)
- [\[ICML 2026\] Pseudospectral Bounds for Transient Amplification in Coupled Gradient Descent](pseudospectral_bounds_for_transient_amplification_in_coupled_gradient_descent.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](../../NeurIPS2025/optimization/learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[ICLR 2026\] Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks](../../ICLR2026/optimization/directional_convergence_benign_overfitting_of_gradient_descent_in_leaky_relu_two.md)

</div>

<!-- RELATED:END -->
