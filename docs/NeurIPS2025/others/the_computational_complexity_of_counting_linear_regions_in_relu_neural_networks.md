---
title: >-
  [论文解读] The Computational Complexity of Counting Linear Regions in ReLU Neural Networks
description: >-
  [NeurIPS 2025][ReLU网络] 系统梳理了ReLU网络"线性区域"的六种非等价定义，证明对所有定义计数线性区域都是#P-hard的（一层隐藏层即如此），并在多层网络中证明了强不可近似结果和多项式空间上界。
tags:
  - NeurIPS 2025
  - ReLU网络
  - 线性区域
  - 计算复杂性
  - "#P-hard"
  - NP-hard
---

# The Computational Complexity of Counting Linear Regions in ReLU Neural Networks

**会议**: NeurIPS 2025  
**arXiv**: [2505.16716](https://arxiv.org/abs/2505.16716)  
**代码**: 暂无  
**领域**: 理论深度学习 / 计算复杂性  
**关键词**: ReLU网络, 线性区域, 计算复杂性, #P-hard, NP-hard

## 一句话总结

系统梳理了ReLU网络"线性区域"的六种非等价定义，证明对所有定义计数线性区域都是#P-hard的（一层隐藏层即如此），并在多层网络中证明了强不可近似结果和多项式空间上界。

## 研究背景与动机

ReLU网络计算的函数是连续分段线性（CPWL）的——输入空间可以被划分为有限个"线性区域"，每个区域上函数是仿射的。自Pascanu等人（2014）和Montúfar等人（2014）的开创性工作以来，线性区域的数量一直被视为衡量ReLU网络表达能力的标准度量。大量后续工作致力于推导线性区域数量的上下界、开发计数算法等。

然而，一个最基本的计算问题却被忽视了：**给定一个网络，到底需要多少时间和空间才能计算出它有多少个线性区域？** 更令人意外的是，在讨论计数之前，连"什么是线性区域"这个看似简单的问题都存在混淆——文献中至少有六种不同的、非等价的定义。这些不一致性已经导致了先前工作中的混乱甚至错误。

本文的动机正是从计算复杂性视角系统地解答这些问题：首先统一和区分各种定义，然后为每种定义建立严格的复杂性分类。

## 方法详解

### 整体框架

本文的贡献分为三个层次：（1）定义梳理——识别并比较六种非等价定义；（2）下界证明——建立各种#P-hard和NP-hard结果；（3）上界算法——证明某些定义下可在多项式空间内计数。

### 关键设计

1. **六种线性区域定义的系统分类**:

    - **定义1（Activation Region）**: 具有相同激活模式 $a \in \{0,1\}^{s(N)}$ 的输入集合 $S_a$（可低维、可非凸）
    - **定义2（Proper Activation Region）**: 全维的Activation Region
    - **定义3（Convex Region）**: 与 $f_N$ 兼容的多面体复形中的全维多面体（取最小数量）
    - **定义4（Open Connected Region）**: $f_N$ 为仿射的开连通极大子集
    - **定义5（Closed Connected Region）**: $f_N$ 为仿射的闭连通极大子集
    - **定义6（Affine Region）**: $f_N$ 为仿射的极大子集（无需连通）
   
   关系链：$R_6 \leq R_5 \leq R_4 \leq R_3 \leq R_2 \leq R_1$，且每个不等号都可以严格成立。设计动机：本文不推荐"最佳"定义，但通过明确指出各定义的区别和关系来消除混淆，并系统标注了20篇代表性论文各自使用的定义。

2. **浅层网络（一个隐藏层）的复杂性分析**: 

    - **正面结果**：判定网络是否只有一个线性区域（1-region-decision）对所有六种定义都在P中。核心思路是检测每个神经元对应的超平面是否被"抵消"（cancelled），即该超平面上 $f_N$ 的梯度是否连续。
    - **负面结果**：精确计数对定义1-4是#P-完全的，对定义5-6是#P-hard的。证明通过从计数超平面排列的cell数（已知#P-完全）规约。构造一个神经网络使得其线性区域恰好对应超平面排列的cell。设计动机：说明即使是"最简单"的一层网络，精确计数也从根本上不可行（除非#P = FP）。

3. **深层网络（多个隐藏层）的复杂性分析**:

    - **决策版本的NP-hard**（定理5.2）：对任意固定常数 $K,L \geq 2$，判定L层网络是否有超过K个线性区域是NP-hard的。关键引理（引理5.3）：从SAT规约——给定SAT公式 $\phi$，构造两层网络 $N_\phi$ 使得：(a) $\phi$ 不可满足时 $N_\phi$ 计算常数零函数（1个区域），(b) $\phi$ 可满足时有至少2个区域（满足assignment附近的非零区域）。改进了Wang(2022)需要 $\Theta(\log n)$ 深度的结果。
    - **近似不可行**（定理5.7）：对 $L \geq 3$，除非NP=RP，不存在多项式时间算法能以大于 $(2^n+1)^{-1}$ 的近似比近似线性区域数。直觉：不可满足公式→1个区域，可满足→至少 $2^n+1$ 个区域，任何有效近似都能区分这两种情况，从而解决SAT。
    - **两层网络的近似也困难**（定理5.8）：对两层网络，近似比以指数衰减 $2^{-O(n^{1-\epsilon})}$。

4. **多项式空间上界**（正面结果）: 对定义1、2、6，线性区域计数在FPSPACE中。算法思路：

    - 定义1-2：遍历所有 $2^{s(N)}$ 个可能的激活模式，逐一检验是否构成有效区域
    - 定义6：遍历所有编码长度在多项式上界内的仿射函数系数组合，检查是否被某个proper activation region实现
   
   设计动机：虽然不期望多项式时间算法存在，但证明问题至少不需要指数空间。

### 损失函数 / 训练策略

本文为纯理论工作，不涉及训练。所有结果均为最坏情况复杂性分析。

## 实验关键数据

### 主实验（理论结果汇总）

| 问题 | 网络深度 | 定义1-2 | 定义3 | 定义4-5 | 定义6 |
|------|---------|---------|-------|---------|-------|
| 1-region-decision | 1层 | P | P | P | P |
| 精确计数 | 1层 | #P-完全 | #P-完全 | #P-hard | #P-hard |
| K-region-decision | ≥2层 | ? | NP-hard | NP-hard | NP-hard |
| 近似计数 | ≥3层 | ? | NP-hard | NP-hard | NP-hard |
| 多项式空间 | 任意 | ✓ | ? | ? | ✓ |

### 消融实验（各定义间区域数比较）

| 网络函数示例 | $R_1$ | $R_2$ | $R_3$ | $R_4$ | $R_5$ | $R_6$ |
|-------------|-------|-------|-------|-------|-------|-------|
| $f(x,y)=\max(-y,\min(0,-x))$ | 7 | 5 | 4 | 3 | 3 | 3 |
| 图2示例函数 | - | - | - | 9 | 8 | 7 |

### 关键发现

- 即使只有一个隐藏层，精确计数线性区域就已经是#P-hard的，这意味着不太可能存在高效的精确计数算法
- 两个隐藏层就足以使得判定"是否有超过1个区域"变成NP-hard
- 三层及以上时，不仅精确计数困难，甚至在指数因子范围内的近似也不可行
- 开连通区域和闭连通区域之间有微妙但重要的区别：闭连通区域可以在另一个闭连通区域的低维切片上"延伸"

## 亮点与洞察

- **对领域的清理功能**：系统梳理了文献中的六种定义及其非等价关系，发现并指出了先前工作中的混淆（如Wang 2022的算法实际适用于定义4而非其声称的定义5），对后续研究有重要参考价值
- **规约构造的精巧性**：从SAT到两层网络的规约特别优雅——构造的网络在满足assignment附近产生非零值的有界区域，在其他地方恒为零，使得可满足性与区域数产生严格对应
- **启发性**：虽然是最坏情况结果，但暗示了一个有趣的研究方向——经过梯度下降训练的网络可能有更少的线性区域，是否允许更快的计数算法？

## 局限与展望

- 所有困难性结果都是最坏情况的——实际训练得到的网络可能有特殊结构使问题变得可行
- 不是所有结果都适用于全部六种定义（如近似不可行对定义1-2的开放性、计数是否在#P中对定义5-6的开放性）
- 本文是纯理论工作，没有将复杂性结果与实际网络分析工具的性能做联系
- 固定参数可处理性（FPT）在不同参数化和定义下的状态仍不清楚

## 相关工作与启发

- **与Wang(2022)的对比**：Wang的NP-hard结果需要 $\Theta(\log n)$ 深度，本文改进到任意常数 $L \geq 2$；Wang声称的EXPTIME上界实际适用于定义4而非定义5
- **与Serra等人的实际算法对比**：混合整数规划方法可以枚举具体网络的区域，但没有理论复杂性保证
- **启发**：计算困难性暗示需要替代度量——或许应该研究训练后网络的"有效"区域数量或更可计算的表达力度量

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次系统分类六种定义并建立完整复杂性图景，多个技术创新（SAT→两层网络规约、近似不可行、多项式空间上界）
- **实验充分度**: ⭐⭐⭐ — 纯理论工作，通过精确的数学例子验证定义间的区别，但无实验
- **写作质量**: ⭐⭐⭐⭐⭐ — 数学表述严谨，组织结构清晰，定义表是极好的参考工具
- **价值**: ⭐⭐⭐⭐ — 理论意义重大，澄清了文献中的混淆并建立了问题的基本复杂性边界，对理论深度学习研究有持久影响

<!-- RELATED:START -->

## 相关论文

- [The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets](the_cost_of_robustness_tighter_bounds_on_parameter_complexity_for_robust_memoriz.md)
- [Modeling Neural Activity with Conditionally Linear Dynamical Systems](modeling_neural_activity_with_conditionally_linear_dynamical_systems.md)
- [Sheaf Cohomology of Linear Predictive Coding Networks](sheaf_cohomology_of_linear_predictive_coding_networks.md)
- [Depth-Bounds for Neural Networks via the Braid Arrangement](depth-bounds_for_neural_networks_via_the_braid_arrangement.md)
- [SAD Neural Networks: Divergent Gradient Flows and Asymptotic Optimality via o-minimal Structures](sad_neural_networks_divergent_gradient_flows_and_asymptotic_optimality_via_o-min.md)

<!-- RELATED:END -->
