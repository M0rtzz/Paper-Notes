---
title: >-
  [论文解读] Verifier-Constrained Flow Expansion for Discovery Beyond the Data
description: >-
  [ICLR 2026][图像生成][Flow Expansion] 提出Flow Expander (FE)，通过验证器约束的熵最大化在概率空间中扩展预训练流模型的覆盖范围，使其生成超越训练数据分布但保持有效性的设计样本，在分子构象设计中增加多样性同时保持化学有效性。
tags:
  - ICLR 2026
  - 图像生成
  - Flow Expansion
  - 验证器约束
  - 熵最大化
  - Mirror Descent
  - 分子设计
---

# Verifier-Constrained Flow Expansion for Discovery Beyond the Data

**会议**: ICLR 2026  
**arXiv**: [2602.15984](https://arxiv.org/abs/2602.15984)  
**代码**: 无  
**领域**: 流模型/科学发现  
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
定义强验证器（$\Omega_v = \Omega$，完全刻画有效空间）和弱验证器（$\Omega_v \supset \Omega$，仅作为过滤器）。全局扩展适用于强验证器（目标：有效空间上的均匀分布），局部扩展适用于弱验证器（目标：受约束的局部扩展）。

### 关键设计

1. **全局流扩展（Problem 5）**:

    - 强验证器下，求解 $\pi^* = \arg\max_{\pi} \mathcal{H}(p_1^\pi)$ s.t. $\mathbb{E}_{x \sim p_1}[v(x)] = 1$，$p_0^\pi = p_0^{\text{pre}}$
    - 最优解为有效设计空间上的均匀分布 $p_1^{\pi^*} = \mathcal{U}(\Omega)$
    - 不依赖预训练模型，因为强验证器完全刻画了有效空间

2. **局部流扩展（Problem 7）**:

    - 弱验证器下，加入KL正则化：$\max_\pi \mathcal{H}(p_1^\pi) - \alpha D_{\text{KL}}(p_1^\pi \| p_1^{\text{pre}})$ s.t. $\mathbb{E}[v(x)] = 1$
    - KL项防止模型在弱验证器无法检测的无效区域分配密度
    - $\alpha$ 控制保守程度：大 $\alpha$ → 接近预训练分布

3. **Flow Expander算法（ExpandThenProject）**:

    - **扩展步**：通过噪声空间优化（Eq. 15），利用running cost $f_t = \lambda_t \delta\mathcal{G}_t$ 进行无约束扩展
    - **投影步**：通过reward-guided fine-tuning（Eq. 16）利用验证器 $\log v$ 约束扩展结果
    - 交替执行K次迭代

4. **闭式梯度表达式**:

    - 全局FE：$\nabla_x \delta\mathcal{G}_t = -s_t^\pi$（score function的负值）
    - 局部FE：$\nabla_x \delta\mathcal{G}_t = -s_t^\pi - \alpha_t(s_t^\pi - s_t^{\text{pre}})$
    - 利用流的速度场通过线性变换近似score：$s_t^\pi(x) = \frac{1}{\kappa_t(\frac{\dot{\omega}_t}{\omega_t}\kappa_t - \dot{\kappa}_t)}(\pi(x,t) - \frac{\dot{\omega}_t}{\omega_t}x)$

5. **噪声空间探索（NSE）**:

    - FE去掉投影步的副产品
    - 利用整个流过程的score信息（而非仅终端 $t=1$），解决 $s_1^\pi$ 发散问题
    - 在高维设置中优于现有流探索方法

### 理论保证
- Proposition 1：ExpandThenProject精确求解MD步的解
- Theorem 5.1（理想化）：精确更新下有限时间收敛 $D_{\text{KL}}(\mathbf{Q}^* \| \mathbf{Q}^K) \leq \frac{C}{K}$
- Theorem 5.2（一般设定）：近似更新下在温和噪声/偏差假设下渐近收敛

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

## 相关论文

- [Pseudo-Nonlinear Data Augmentation: A Constrained Energy Minimization Viewpoint](pseudo-nonlinear_data_augmentation_a_constrained_energy_minimization_viewpoint.md)
- [VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model](vfscale_intrinsic_reasoning_through_verifier-free_test-time_scalable_diffusion_m.md)
- [Beyond Confidence: The Rhythms of Reasoning in Generative Models](beyond_confidence_the_rhythms_of_reasoning_in_generative_models.md)
- [Infinity and Beyond: Compositional Alignment in VAR and Diffusion T2I Models](infinity_and_beyond_compositional_alignment_in_var_and_diffusion_t2i_models.md)
- [Beyond the Golden Data: Resolving the Motion-Vision Quality Dilemma via Timestep Selective Training](../../CVPR2026/image_generation/beyond_the_golden_data_resolving_the_motion-vision_quality_dilemma_via_timestep_.md)

<!-- RELATED:END -->
