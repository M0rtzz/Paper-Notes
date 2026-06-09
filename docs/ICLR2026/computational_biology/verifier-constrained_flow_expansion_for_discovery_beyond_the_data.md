---
title: >-
  [论文解读] Verifier-Constrained Flow Expansion for Discovery Beyond the Data
description: >-
  [ICLR 2026][计算生物][Flow Expansion] 提出Flow Expander (FE)，通过验证器约束的熵最大化在概率空间中扩展预训练流模型的覆盖范围，使其生成超越训练数据分布但保持有效性的设计样本，在分子构象设计中增加多样性同时保持化学有效性。
tags:
  - "ICLR 2026"
  - "计算生物"
  - "Flow Expansion"
  - "验证器约束"
  - "熵最大化"
  - "Mirror Descent"
  - "分子设计"
---

# Verifier-Constrained Flow Expansion for Discovery Beyond the Data

**会议**: ICLR 2026  
**arXiv**: [2602.15984](https://arxiv.org/abs/2602.15984)  
**代码**: 无  
**领域**: 计算生物
**关键词**: Flow Expansion, 验证器约束, 熵最大化, Mirror Descent, 分子设计

## 一句话总结
提出Flow Expander (FE)，通过验证器约束的熵最大化在概率空间中扩展预训练流模型的覆盖范围，使其生成超越训练数据分布但保持有效性的设计样本，在分子构象设计中增加多样性同时保持化学有效性。

## 研究背景与动机

**领域现状**：流模型和扩散模型通过散度最小化训练，仅覆盖训练数据分布对应的设计空间的极小子集。在科学发现任务中（如药物设计、材料设计），需要探索超越数据分布的有效设计。

**现有痛点**：(1) 预训练流模型集中在高数据密度区域，低概率区域可能是无效设计；(2) 流形探索方法（如密度重平衡）在数据稀疏区域失去有效性信号；(3) 缺乏利用外部验证器（如原子键检查器）来引导探索的原则性方法。

**核心矛盾**：探索超越数据分布的设计空间需要增加覆盖范围（熵最大化），但无约束扩展会生成无效设计。需要在扩展和有效性之间取得平衡。

**本文目标**：如何利用给定验证器调整预训练流模型，使其密度扩展到高数据可用性区域之外，同时保持样本有效性？

**切入角度**：形式化强/弱验证器概念，针对两种情况分别提出全局和局部流扩展的数学框架。

**核心 idea**：通过验证器约束的熵最大化和噪声空间上的Mirror Descent优化，实现预训练流模型的有原则扩展。

## 方法详解

### 整体框架
FE先按验证器能力把问题分成两类——强验证器（$\Omega_v = \Omega$，能完全刻画有效空间）走全局扩展、目标是把密度铺成有效空间上的均匀分布；弱验证器（$\Omega_v \supset \Omega$，只能当过滤器）走局部扩展、在预训练分布附近做受约束的扩散。两种情形都被统一成噪声空间上的Mirror Descent优化，再由ExpandThenProject算法把"扩展"和"投影回有效区"交替迭代来求解。

### 关键设计

**1. 全局流扩展：强验证器下直接逼近均匀分布。** 当验证器足够强、能精确判定一个设计是否有效时，问题被写成熵最大化 $\pi^* = \arg\max_{\pi} \mathcal{H}(p_1^\pi)$，约束为期望有效性 $\mathbb{E}_{x \sim p_1}[v(x)] = 1$ 且起点固定 $p_0^\pi = p_0^{\text{pre}}$。这个凸问题的最优解非常干净：终端分布就是有效设计空间上的均匀分布 $p_1^{\pi^*} = \mathcal{U}(\Omega)$。关键在于此时**根本不需要依赖预训练模型**——既然验证器已经完整刻画了有效空间，最大熵的答案完全由 $\Omega$ 决定，于是探索可以毫无包袱地铺满整个有效区域。

**2. 局部流扩展：弱验证器下用KL锚住预训练分布。** 现实中的验证器（如原子键检查器）往往只是过滤器，会漏掉一些它检测不到的无效设计。如果照搬全局扩展，模型就会把密度分配到这些"验证器看不见的无效区"。FE的做法是在熵最大化里加一项KL正则 $\max_\pi \mathcal{H}(p_1^\pi) - \alpha D_{\text{KL}}(p_1^\pi \| p_1^{\text{pre}})$，仍保留有效性约束 $\mathbb{E}[v(x)] = 1$。KL项把扩展后的分布拉回预训练分布附近，等于借预训练模型的先验来兜住弱验证器的盲区；超参 $\alpha$ 就是保守度旋钮——$\alpha$ 越大越贴近预训练分布、越不敢往外探。

**3. ExpandThenProject：把扩展与投影拆成两步交替。** 直接求解上面的约束优化很难，FE把每一步Mirror Descent拆成两个易实现的子步并迭代K次。**扩展步**在噪声空间上做无约束优化（Eq. 15），用running cost $f_t = \lambda_t \delta\mathcal{G}_t$ 驱动密度往外摊开；**投影步**则是一次reward-guided fine-tuning（Eq. 16），把验证器对数 $\log v$ 当奖励，把刚才扩出去的密度拉回有效区。先尽情扩、再约束投影，这种"先放后收"的交替正好对应Mirror Descent的一步更新，既保留了探索力度又不会失控。

**4. 闭式梯度：用速度场线性变换近似score，避免显式估计。** 扩展步的核心是running cost的梯度 $\delta\mathcal{G}_t$。FE给出了闭式表达：全局情形 $\nabla_x \delta\mathcal{G}_t = -s_t^\pi$，即score function取负；局部情形多一项把当前与预训练score的差也算进来，$\nabla_x \delta\mathcal{G}_t = -s_t^\pi - \alpha_t(s_t^\pi - s_t^{\text{pre}})$，这正是KL正则在梯度上的体现。而score本身不必单独训练，可以从流的速度场 $\pi(x,t)$ 经线性变换得到 $s_t^\pi(x) = \frac{1}{\kappa_t(\frac{\dot{\omega}_t}{\omega_t}\kappa_t - \dot{\kappa}_t)}(\pi(x,t) - \frac{\dot{\omega}_t}{\omega_t}x)$，让整套优化直接复用预训练流模型的输出。

**5. 噪声空间探索（NSE）：用整段轨迹的score稳住高维探索。** NSE是FE去掉投影步后的副产品，却解决了一个独立的痛点：现有流探索方法只用终端 $t=1$ 处的score $s_1^\pi$，而它在数据稀疏处会发散。FE的扩展步沿用整个流过程 $t \in [0,1]$ 的score信息，相当于把探索信号在时间上均摊，避免被终端的奇异点带偏，因此在高维分子设置中比只看终端的方法稳定得多。

**6. 收敛保证：从精确到近似的两级理论支撑。** 整套交替迭代有理论兜底：Proposition 1证明ExpandThenProject恰好精确求解一步Mirror Descent的解；Theorem 5.1在理想化的精确更新下给出有限时间收敛率 $D_{\text{KL}}(\mathbf{Q}^* \| \mathbf{Q}^K) \leq \frac{C}{K}$；Theorem 5.2进一步放宽到实际的近似更新，在温和的噪声/偏差假设下保证渐近收敛，把"扩展步和投影步都只是近似"的现实情形也纳入保证。

## 实验关键数据

### 合成实验（可视化验证）
- FE成功将预训练分布扩展到整个有效区域
- NSE在高维设置中稳定性显著优于现有方法

### 分子设计实验
- FE增加构象多样性的同时比现有流探索方法更好地保持有效性
- 弱验证器（原子键检查器）有效过滤无效构象
- 多弱验证器组合 $\Omega_v = \bigcap_i \Omega_{v_i}$ 进一步收紧有效区域

### 关键发现
- 噪声空间探索（使用整个过程而非终端score）在高维中显著提升稳定性
- 验证器约束的投影步至关重要——无约束扩展会产生大量无效样本
- $\alpha$ 的选择应反映验证器质量：强验证器→小 $\alpha$，弱验证器→大 $\alpha$

## 亮点与洞察
- **问题形式化优雅**：强/弱验证器的区分及对应的全局/局部扩展框架，概念清晰且实用
- **理论严谨**：从连续时间RL到Mirror Descent的理论链条完整，convergence guarantees扎实
- **噪声空间优化是关键创新**：解决了终端score发散的实际问题，且NSE作为副产品本身就有价值
- **通用框架**：适用于任何有验证器的科学发现任务

## 局限与展望
- score function的近似精度影响实际性能，需要高质量预训练模型
- $\alpha_t$ 和 $\lambda_t$ 的选择缺乏自动调节机制
- 分子设计实验规模相对较小，大规模评估有待进一步开展
- 可以探索学习型验证器（如GNN）替代手工规则

## 相关工作与启发
- **vs De Santi et al. 2025**：仅使用终端score $s_1^\pi$ 进行探索，存在发散问题；FE利用整个过程稳定
- **vs reward-guided fine-tuning**：FE额外提供验证器约束，防止扩展到无效区域
- **连续时间RL视角**：将流模型微调统一为最优控制问题的创新

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 验证器约束的流扩展是全新问题，理论框架完整
- 实验充分度: ⭐⭐⭐⭐ 合成+分子设计实验，但大规模实验有待进一步验证
- 写作质量: ⭐⭐⭐⭐ 理论密集但逻辑清晰，图示有效
- 价值: ⭐⭐⭐⭐⭐ 对科学发现中的生成模型应用有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Constrained Flow Optimization via Sequential Fine-Tuning for Molecular Design](../../ICML2026/computational_biology/constrained_flow_optimization_via_sequential_fine_tuning_for_molecular_design.md)
- [\[NeurIPS 2025\] Flow Density Control: Generative Optimization Beyond Entropy-Regularized Fine-Tuning](../../NeurIPS2025/computational_biology/flow_density_control_generative_optimization_beyond_entropy-regularized_fine-tun.md)
- [\[NeurIPS 2025\] Constrained Discrete Diffusion](../../NeurIPS2025/computational_biology/constrained_discrete_diffusion.md)
- [\[AAAI 2026\] Constrained Best Arm Identification with Tests for Feasibility](../../AAAI2026/computational_biology/constrained_best_arm_identification_with_tests_for_feasibility.md)
- [\[ICLR 2026\] Zatom-1: A Multimodal Flow Foundation Model for 3D Molecules and Materials](zatom-1_a_multimodal_flow_foundation_model_for_3d_molecules_and_materials.md)

</div>

<!-- RELATED:END -->
