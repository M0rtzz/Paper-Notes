---
title: >-
  [论文解读] Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models
description: >-
  [ICML 2026][LLM推理][Looped LM] 本文从动力系统视角诊断 Looped Language Model (LoopLM) 在 test-time 扩展深度时"先涨后崩"的根因——归一化位置导致的"稳定—有效"二元困境…
tags:
  - "ICML 2026"
  - "LLM推理"
  - "Looped LM"
  - "Test-time Scaling"
  - "Jacobian 谱半径"
  - "动力系统稳定性"
  - "随机循环采样"
---

# Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models

**会议**: ICML 2026  
**arXiv**: [2605.26733](https://arxiv.org/abs/2605.26733)  
**代码**: https://github.com/njuyxw/STARS (有)  
**领域**: LLM 推理 / 潜在推理 / 循环 Transformer  
**关键词**: Looped LM、Test-time Scaling、Jacobian 谱半径、动力系统稳定性、随机循环采样

## 一句话总结
本文从动力系统视角诊断 Looped Language Model (LoopLM) 在 test-time 扩展深度时"先涨后崩"的根因——归一化位置导致的"稳定—有效"二元困境，并提出 STARS：用 Jacobian 谱半径正则化 (JSRR) + 随机循环采样把潜在轨迹拉向"渐近稳定的有效不动点"，在 GSM8K 上把 8 步循环的性能跌幅从 20.47% 压到 8.26%，同时峰值提升 4.01%。

## 研究背景与动机

**领域现状**：LLM test-time scaling 的主流路径是显式扩展输出长度（CoT、多采样投票、ToT、MCTS），但这些都受限于自然语言序列的带宽和效率。近年兴起的 Looped Language Models (LoopLM, 例如 Huginn、Ouro) 走另一条路：通过对同一组共享参数的 Transformer block 进行深度递归，把"思考"放进连续潜空间，理论上迭代越多次表征越精炼，且不需扩长 context。

**现有痛点**：作者发现这种"想得越久越准"的假设并不成立。在 GSM8K 上 Ouro-1.4B 的准确率在某个迭代深度达到峰值后会"急剧崩塌"——直接 SFT 后峰值更高，但 8 步时从 70.46% 跌到 52.97%。这意味着 LoopLM 并没有真正学到"可扩展的潜在推理能力"，只是过拟合了训练时使用的固定迭代深度。

**核心矛盾**：作者从动力系统视角做诊断实验，把循环 block 看成一个离散时间映射 $\mathbf{h}^{(t+1)}=\Phi_\theta(\mathbf{h}^{(t)})$，发现 LoopLM 存在一个被忽略的根本性二元困境——**有效性与稳定性由 LayerNorm 的位置决定，且二者不可兼得**：
- **内部归一化**（Pre-Norm / Pre-Sandwich）：残差跳过归一化，信息高速公路畅通（有效），但更新向量直接累加到 backbone 上，隐状态范数随迭代指数级膨胀，轨迹偏离数据流形 → 性能崩塌。
- **外部归一化**（Post-Norm / Post-Sandwich）：归一化包住残差，隐状态有界（稳定），但推理浅，训练时性能就上不去。

作者还系统验证了常见补救手段——加 Prelude/Coda 非循环层、L2 正则、随机循环采样——**都无法同时破解这个 deadlock**。

**本文目标**：让 LoopLM 真正具备 test-time scalable 的潜在推理能力，即：迭代越深，潜在状态越收敛、性能越稳健。

**切入角度**：作者把"推理"概念化为"不确定性的迭代消减过程"。从动力学语言讲，这意味着隐状态应该收敛到一个既"稳定"（不发散、不振荡）又"有效"（停在能解出题的位置）的不动点。光稳定不行（思想浅），光有效不行（思维混乱）。

**核心 idea**：用 Lyapunov 线性化定理——不动点的稳定性由 Jacobian 谱半径决定——把"渐近稳定"显式写成训练时的正则项 $\rho(J) < 1$，再用随机循环采样把约束推广到整条轨迹，从而让模型自己学会收敛到"既稳又准"的不动点。

## 方法详解

### 整体框架

STARS (STAbility-driven Recurrent Scaling) 是一个训练框架，作用在任意已有的 LoopLM 上（论文用 Ouro-1.4B 做 fine-tune）。整体 pipeline 与标准 LoopLM 训练几乎相同：输入是文本 token 序列，prelude 嵌入后进入循环 block $\Phi_\theta = \mathcal{M}^L$，迭代 $t$ 次后经 coda + lmhead 输出。改动只在**损失函数**上：每个 batch 先从分布 $\mathcal{P}$（log-normal）随机采样循环深度 $t$，然后同时计算两项 loss——标准 SFT 交叉熵 $\mathcal{L}_{SFT}^{(t)}$ 和 Jacobian 谱半径正则 $\mathcal{L}_{JSRR}^{(t)}$，加权求和反向传播。推理阶段无任何额外开销，照常按所需迭代数前向即可。

### 关键设计

1. **动力系统诊断 + 后置 sandwich 归一化结构选择**:

    - 功能：在做训练改进之前，先把"哪个架构有救"挑出来。
    - 核心思路：在 4 位加法的可控小实验上，把 12 种归一化结构（LayerNorm/RMSNorm/SimpleNorm × Pre/Post/Pre-Sandwich/Post-Sandwich）穷举训练，用 PCA 投影观察潜在轨迹的"尺度"和准确率随 $T_{test}$ 的演化，得到关键结论：归一化的"位置"决定动力学，归一化的"类型"几乎无影响。基于此，STARS 选定 **Post-Sandwich LayerNorm**——天然有界、容易收敛到吸引子，只要再施加引导，就有机会同时拿到稳定性和有效性。
    - 设计动机：内部归一化系列在 4 位加法上轨迹尺度 PCA 图 explode，外部归一化系列轨迹紧凑但准确率撑不到测试期；Prelude/Coda、L2、随机循环单独用都无法破解 deadlock。这一节本身就是论文最有价值的"负面结果"，把后续方法约束在"以外部归一化为底座，主动注入稳定性约束"这条路上。

2. **Jacobian 谱半径正则化 (JSRR)**:

    - 功能：在训练时显式压制循环映射 $\Phi_\theta$ 在当前隐状态处的 Jacobian 谱半径 $\rho(J)$，把不动点拉到"渐近稳定"侧。
    - 核心思路：根据 Lyapunov 线性化定理，离散时间系统 $\mathbf{h}^{(t+1)}=\Phi_\theta(\mathbf{h}^{(t)})$ 在不动点 $\mathbf{h}^\star$ 的局部稳定性由 $\rho(J(\mathbf{h}^\star)) = \max_i |\lambda_i|$ 决定，$\rho<1$ 即可保证小扰动指数衰减、迭代收敛。但 $J\in\mathbb{R}^{D\times D}$ 中 $D=M\cdot d$ 太大，直接求特征值不可行；作者改用 **单步幂迭代 + Jacobian-vector product (JVP)**：随机初始化 $\mathbf{v}$ 后用 PyTorch 的 JVP 算 $J\mathbf{v}$，无需显式构造 $J$，谱半径估计为 $\rho(J)\approx \|J\mathbf{v}\|_2$。损失定义为 $\mathcal{L}_{JSRR}^{(t)} = \frac{1}{N}\sum_i \|J^{(t,i)} \mathbf{v}^{(t,i)}\|_2^2$。另外由于训练中精确不动点 $\mathbf{h}^\star$ 未知，作者采用"代理"做法——直接在当前迭代 $t$ 的隐状态 $\mathbf{h}^{(t)}$ 上算谱半径。
    - 设计动机：DEQ 系列 (Bai 2019, 2021) 用 Frobenius 范数 $\|J\|_F$ 正则，但 $\rho(J) \le \|J\|$ 是宽松上界，约束 $\|J\|$ 会过度压缩模型表达能力；直接管谱半径既数学精确又只挤压"最不稳定"那条主方向。选择单步幂迭代而非多步，是因为多步会引入复杂的二阶梯度依赖容易爆炸，单步虽然单样本噪声大、但 batch 平均后优化方向统计正确，且显存/计算几乎无额外开销。

3. **随机循环采样 × JSRR 的轨迹级正则化**:

    - 功能：把"某一个 $t$ 处的稳定性"升级成"整条轨迹的全局稳定性"，同时也是让模型不绑死单一训练深度的关键。
    - 核心思路：每个 batch 从分布 $\mathcal{P}$（论文用 log-normal $\mu=1.7, \sigma=0.4$, range $[1,16]$）采样一个循环步数 $t$，然后按 $\mathcal{L}_{STARS} = \mathbb{E}_{t\sim\mathcal{P}}[(1-\lambda)\cdot\mathcal{L}_{SFT}^{(t)} + \lambda\cdot\mathcal{L}_{JSRR}^{(t)}]$ 联合优化（$\lambda=0.1$）。这样既让 SFT 项覆盖各种深度（不再过拟合固定 $T_{train}$），又让 JSRR 项把谱半径约束施加到轨迹上 $\mathcal{P}$ 支撑集的每一个点上。
    - 设计动机：单点 JSRR 只能保证某一深度收敛，对更深的迭代不一定有效；单纯随机循环采样的诊断实验已经显示——它能让性能在训练范围外保持更久，但仍然无法阻止 internal norm 的状态漂移，也不能保证 external norm 的 attractor 是"有益"的。两者组合后才真正做到"全程有界 + 全程有效"的统一约束。

### 损失函数 / 训练策略

最终训练目标（公式 4）：

$\mathcal{L}_{STARS} = \mathbb{E}_{t\sim\mathcal{P}}\left[(1-\lambda)\cdot\mathcal{L}_{SFT}^{(t)} + \lambda\cdot\mathcal{L}_{JSRR}^{(t)}\right]$

数学推理实验中：基于 Ouro-1.4B 在 NuminaMath-1.5 的 400K 子集上 fine-tune 1 epoch，4×A800 + AdamW + cosine schedule + 起始 lr $1\times10^{-6}$；随机循环 log-normal $\mu=1.7, \sigma=0.4$, range $[1,16]$，$\lambda=0.1$。加法实验则用 log-normal $\mu=2, \sigma=0.7$, range $[1,100]$，lr $1\times10^{-4}$。

## 实验关键数据

### 主实验（数学推理，Ouro-1.4B fine-tune）

| 模型 | 循环步数 | GSM8K | MATH500 | ASDiv | SVAMP | AMC23 | 平均 |
|------|---------|-------|---------|-------|-------|-------|------|
| Ouro-1.4B (base) | 4 | 75.21 | 59.60 | 76.57 | 75.67 | 50.00 | 67.41 |
| Ouro-1.4B (base) | 8 | 58.23 | 40.80 | 70.07 | 66.33 | 40.00 | 55.09 |
| Ouro-1.4B-SFT | 4 | 80.06 | 64.60 | 83.47 | 76.67 | 47.50 | 70.46 |
| Ouro-1.4B-SFT | 8 | 60.05 | 39.20 | 75.10 | 68.00 | 22.50 | 52.97 |
| **Ouro-1.4B-STARS** | 4 | **81.96** | **67.40** | **84.73** | **84.33** | **52.50** | **74.18** |
| **Ouro-1.4B-STARS** | 8 | 74.45 | 54.80 | 82.52 | 81.00 | 35.00 | 65.55 |

关键对比：GSM8K 上从峰值（4 步）到 8 步的相对跌幅 Ouro 是 20.47%，SFT 是 25.0%，STARS 只有 8.26%；同时 STARS 的 4 步峰值 81.96% 比 SFT 还高 1.90%。多位加法任务上 STARS 在 4–100 步范围内准确率稳定 100%。

### 消融实验（Figure 4 右图，4 个数学 benchmark 平均）

| 配置 | 趋势特征 |
|------|---------|
| Ouro-1.4B (base) | 4 步后急剧下降 |
| + Random Loop only | 下降变缓，但仍有明显衰退 |
| + JSRR only | 下降变缓，与 Random Loop 互补 |
| **Full STARS (RL+JSRR)** | 下降最慢、峰值最高，两者都不可少 |

### 关键发现
- **归一化位置才是 LoopLM 命门**：12 种结构穷举显示归一化类型几乎无影响，但 Pre vs Post 决定潜空间是发散还是收敛；这是论文最值得迁移的洞察。
- **常见补救方案全数失败**：Prelude/Coda 层、L2 正则、纯随机循环采样都无法同时实现稳定与有效，论文用这一连串"负面消融"把读者推向"必须有显式稳定性约束"的结论。
- **JSRR 与 Random Loop 互为补充**：JSRR 给"局部稳定性"，Random Loop 把约束推广到全局轨迹，两者缺一不可，单独使用都不够。

## 亮点与洞察
- **从动力系统视角重审 LoopLM**：把 LoopLM 的"想得越久越差"翻译成"轨迹不收敛到不动点"，并用 PCA 投影把潜在轨迹真的画出来——这种"先做诊断再开药"的范式比直接 propose loss 要可靠得多，整篇论文的说服力很大程度来自第 4 节的可视化。
- **JSRR 用 Lyapunov 借力打力**：把控制论里 70 年代就成熟的稳定性条件直接搬到 Transformer 训练里，再用 JVP 把不可计算的全 Jacobian 谱半径降到 $O(D)$ 的单步幂迭代，工程上几乎零成本。这一招对所有用循环/平衡结构的工作（DEQ、Neural ODE、Universal Transformer）都可以复用。
- **"既稳又准的不动点"是个可推广的设计哲学**：把"推理"形式化为不动点收敛过程，那么所有 latent reasoning / continuous CoT 方法（Coconut、SIM-CoT、CODI 等）都可以用同样的稳定性—有效性二元框架重新审视，谁的潜在状态轨迹更接近收敛，谁就更"想得清楚"。

## 局限与展望
- **代理点而非真不动点**：JSRR 实际约束的是当前迭代 $\mathbf{h}^{(t)}$ 处的谱半径，而非真正不动点 $\mathbf{h}^\star$ 处，理论保证不严格——只能说期望意义下统计正确。
- **只在 1.4B 规模验证**：实验止步于 Ouro-1.4B + 400K NuminaMath 子集 + 1 epoch，没在更大基座或更长循环深度（>16）上验证；8 步以上的衰退也只是变缓，并未根除。
- **8 步处与峰值仍有 8.62 个点的差距**：方法把崩塌延后了但没有完全消除，说明"深度无穷扩展即性能单调上升"这一终极目标尚未达成。
- **改进方向**：将 JSRR 与显式 fixed-point solver（DEQ 式 implicit differentiation）结合、引入二阶幂迭代多步估计、把 $\lambda$ 改为可学习/自适应、把 STARS 推广到 Coconut 等 sequential 形 latent reasoning。

## 相关工作与启发
- **vs Geiping et al. (Huginn) / Zhu et al. (Ouro)**：他们直接训练 LoopLM 并依靠 Prelude/Coda 缓解；本文证明这些非循环层只能"略微减缓"内部归一化系统的漂移、并把外部归一化系统的吸引子变紧但非良性。STARS 在他们的基础上加 JSRR 才真正解决可扩展性。
- **vs DEQ (Bai et al. 2019, 2021)**：DEQ 用 Frobenius 范数 $\|J\|_F$ 正则化平衡模型，但 $\|J\|$ 是 $\rho(J)$ 的宽松上界，会过度压缩表达力；本文直接管谱半径，更精确、对模型干扰更小，且首次把这种思想用到大规模 LoopLLM 上。
- **vs Coconut / SIM-CoT / CODI 等 continuous CoT**：这些方法沿"序列维"扩展潜在表征，本质上仍受 token 生成带宽约束；LoopLM 走"深度维"，STARS 让这条路真正可扩展，二者未来可能结合（在 Coconut 内部把 latent thought block 套上 JSRR）。
- **vs Universal Transformer / Looped Transformer (Giannou et al.)**：早期循环 Transformer 工作多在小规模/理论层面，缺乏稳定性分析；本文用动力系统语言把这一系工作和 DEQ 联通起来，是后续"recurrent depth for reasoning"路线的方法论坐标。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次用动力系统视角系统诊断 LoopLM scaling 失败，并把 Lyapunov 谱半径以可微方式注入 LLM 训练。
- 实验充分度: ⭐⭐⭐⭐ 4 位加法可控实验 + 5 个数学 benchmark + 12 种归一化结构穷举 + 完整消融，缺规模上推。
- 写作质量: ⭐⭐⭐⭐⭐ 故事线"诊断 → 矛盾 → 哲学 → 方法 → 实验"环环相扣，PCA 可视化和负面结果都用得极好。
- 价值: ⭐⭐⭐⭐⭐ 对所有走"深度循环"路线的 latent reasoning 工作（DEQ、Huginn、Ouro、Coconut）都直接可借鉴，是 LoopLM 走向真正 test-time scalable 的关键一步。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models](prioritize_the_process_not_just_the_outcome_rewarding_latent_thought_trajectorie.md)
- [\[ICML 2026\] Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models](prism_efficient_test-time_scaling_via_hierarchical_search_and_self-verification_.md)
- [\[ICML 2026\] Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models](lookahead_sample_reward_guidance_for_test-time_scaling_of_diffusion_models.md)
- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](../../ACL2026/llm_reasoning/parallel_test-time_scaling_for_latent_reasoning_models.md)
- [\[ICML 2026\] Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](dynamics_within_latent_chain-of-thought_an_empirical_study_of_causal_structure.md)

</div>

<!-- RELATED:END -->
