---
title: >-
  [论文解读] Towards Understanding Adam Convergence on Highly Degenerate Polynomials
description: >-
  [ICML2026][优化/理论][Adam] 本文挑出一类高阶退化多项式 $L(x)=\tfrac{1}{k}x^k$（$k\ge 4$ 偶数）作为最小问题模型，证明在常数学习率下 Adam 通过 $v_t$ 与 $g_t^2$ 的"解耦"机制把有效学习率指数放大，从而实现局部线性收敛…
tags:
  - "ICML2026"
  - "优化/理论"
  - "Adam"
  - "退化最小"
  - "线性收敛"
  - "相图"
  - "自适应步长"
---

# Towards Understanding Adam Convergence on Highly Degenerate Polynomials

**会议**: ICML2026  
**arXiv**: [2603.09581](https://arxiv.org/abs/2603.09581)  
**代码**: 论文未给出公开链接  
**领域**: optimization  
**关键词**: Adam, 退化最小, 线性收敛, 相图, 自适应步长  

## 一句话总结
本文挑出一类高阶退化多项式 $L(x)=\tfrac{1}{k}x^k$（$k\ge 4$ 偶数）作为最小问题模型，证明在常数学习率下 Adam 通过 $v_t$ 与 $g_t^2$ 的"解耦"机制把有效学习率指数放大，从而实现局部线性收敛，而 GD 与动量在同一问题上只能拿到 $\Theta(t^{-1/(k-2)})$ 的次线性速率，并完整刻画了 Adam 在 $(\beta_1,\beta_2)$ 平面上"稳定收敛 / spike / SignGD 振荡"三个相区。

## 研究背景与动机
**领域现状**：Adam 几乎是深度学习默认优化器，但理论分析长期局限于两类设定：要么需要外加 $\eta/\sqrt T$ 衰减调度并要求 $\beta_2$ 接近 1（如 Zhang et al. 2022a 的 $O(\log T/\sqrt T)$ 速率），要么直接构造反例说明 Adam 在简单凸问题上发散（Reddi et al. 2018）。

**现有痛点**：实践里 Adam 用恒定学习率、$\beta_2\in[0.9,0.999]$ 都好用，可是现有理论既不解释这种"裸 Adam 也收敛"的常态，也没说清楚 Adam 在什么类型的目标上比 GD/动量真的有优势——已有解释多分散在 Hessian 异质性、坐标 $\ell_\infty$ 几何、heavy-tailed 噪声等假设各异的机制里。

**核心矛盾**：理论分析普遍盯着强凸或一般凸目标，而深度网络损失景观早被实证为高度退化（Hessian 谱集中在 0 附近）；强凸理论恰好是 Adam 最容易出 spike 的场景，与实际跑 Transformer 时 Adam 远胜 SGD 的现象错位。

**本文目标**：找到一个干净的目标函数类，使 Adam 的"内在自适应"优势能被纯粹隔离出来；在这个类上完成局部稳定性、收敛速率、与 GD/Momentum 的复杂度分离三件事；并把超参数相图与实验现象（spike、SignGD 振荡）对齐。

**切入角度**：从 Fig. 1 的对比观察出发——在 $L=\tfrac{1}{2}x^2$ 上常数步长 Adam 最终 spike，但在 $L=\tfrac{1}{4}x^4$ 上反而稳定指数下降。这说明退化阶 $k\ge 4$ 才是 Adam 真正"舒服"的曲率结构，应当被当成模型问题来分析。

**核心 idea**：把"$v_t$ 是否跟踪 $g_t^2$"作为相变锚点——当 $g_t$ 衰减足够快时 $v_t\to\beta_2 v_{t-1}$ 进入几何衰减，等价于给 GD 装上了 $\beta_2^{-t/2}$ 的指数学习率调度，正好把退化多项式的次线性诅咒拉回线性速率。

## 方法详解

### 整体框架
本文不引入新算法，而是对原版 Adam（$\varepsilon=0$、忽略 bias correction）做了一套"模型问题 → 状态空间 → 相图"的理论分析流程：

1. **模型问题层**：固定 $L(x)=\tfrac{1}{k}x^k$，$k\ge 4$ 偶数；这是深网损失景观中"退化方向"的最小代表。
2. **状态空间层**：用 $\omega_t:=m_t/x_t^{k-1}$ 和 $\lambda_t:=x_t^{k-2}/\sqrt{v_t}$ 把 Adam 的 3 个时序变量重新归一化，把动力学从 $(x_t,m_t,v_t)$ 写成 $(\omega_t,\lambda_t,x_t)$，迭代变成 $x_{t+1}=(1-\eta\omega_t\lambda_t)x_t$。
3. **稳定性/相图层**：解出非平凡不动点，分析其 Jacobian 谱半径，得到稳定条件 $\beta_1<\beta_2^{k/(2(k-2))}$；与不动点是否存在/稳定对应 Adam 三种实证行为。
4. **加速机制层**：在 RMSProp（$\beta_1=0$）上隔离自适应效应，证明 $v_t$ 几何衰减 → 有效学习率指数增长 → 退化目标上由次线性升级到线性。
5. **离散化稳定层**：把"指数增长学习率"压回离散 GD，引入有效 sharpness $u_t=\eta_t x_t^{k-2}$ 并研究 1D 映射 $u_{t+1}=\gamma u_t(1-u_t)^{k-2}$ 的分岔行为，得到全局收敛门槛 $\gamma<(\tfrac{k}{k-2})^{k-2}$。

### 关键设计

**1. 归一化状态空间 $(\omega_t,\lambda_t,x_t)$：把含 $x_t$ 高次幂的耦合系统压成尺度无关的低维动力学**

Adam 在 $L=\tfrac{1}{k}x^k$ 上的原始变量 $(x_t,m_t,v_t)$ 里，$x_t$ 在多个幂次出现，直接算谱半径会被尺度污染。本文代入 $g_t=x_t^{k-1}$ 后引入两个归一化量——$\omega_t=m_t/x_t^{k-1}$ 表示"归一化一阶矩"、$\lambda_t=x_t^{k-2}/\sqrt{v_t}$ 表示"有效曲率"，把 $x_t$ 的衰减彻底剥离，迭代化简成 $x_{t+1}=(1-\eta\omega_t\lambda_t)x_t$，损失单调下降等价于 $0\le\omega_t\lambda_t\le 2/\eta$。这样非平凡不动点出现在 $x^\star=0$ 而 $(\omega^\star,\lambda^\star)$ 取与 $x_t$ 无关的有限值，稳定性就只剩一个 $2\times2$ 子 Jacobian 的问题——这是整套相图分析能解析推到底的前提。

**2. $v_t$–$g_t^2$ 解耦机制：指数放大有效学习率，把次线性拔成线性**

退化目标上 GD 只能拿 $\Theta(t^{-1/(k-2)})$ 的次线性速率，Adam 凭什么更快？关键在 $v_t$ 是否还跟得上 $g_t^2$。在 RMSProp 设定下，引理 5.4 证明若 $x_t\to 0$ 则 $g_t/\sqrt{v_t}\to 0$，进而 $v_t/v_{t-1}\to\beta_2$，即 $v_t\sim\beta_2^t$ 几何衰减，于是有效学习率 $\eta_{\mathrm{eff},t}=\eta/\sqrt{v_t}\propto\beta_2^{-t/2}$ 指数膨胀。引理 5.5 把这个指数 schedule 喂给连续 GD 流 $\dot x=-\eta(t)x^{k-1}$，解得 $x(t)\sim\exp(-\tfrac{\alpha}{k-2}t)$——幂律衰减被拔成指数衰减。值得强调的是，这个加速来源跟 SignGD 视角是机制对立的：SignGD 是 $\beta_2=0$ 极限、$v_t$ 紧跟 $g_t^2$，反而不收敛到 0；真正的加速来自"$v_t$ 落后于 $g_t^2$ 的几何记忆"，而这个机制只在 $g_t$ 足够快衰减时才打开，正好对应 $k\ge 4$ 的退化结构。

**3. 三相相图与 Jacobian 谱条件（定理 4.1 + 定理 6.1）：用 $(\beta_1,\beta_2)$ 把全部稳态行为分三类**

把以前各文章零散观察到的 Adam 失败模式（限环、loss spike、振荡）一次性纳入同一相图。非平凡不动点的存在条件是 $\beta_1<\beta_2^{(k-1)/(2(k-2))}$，稳定条件是 $\beta_1<\beta_2^{k/(2(k-2))}$，把两条曲线和"无不动点"区放进 $(\beta_1,\beta_2)$ 平面得到三个区：(I) 双条件都满足 → 稳定线性收敛，速率 $x_{t+1}/x_t\to\beta_2^{1/(2(k-2))}$；(II) 不动点存在但失稳 → 早期被吸引产生指数收敛、后期 $\omega_t\lambda_t>2/\eta$ 触发 spike；(III) 不动点不存在 → $v_t$ 紧跟 $g_t^2$、等价 SignGD，损失在 $L(\eta/2)$ 附近振荡。这套用同一组不等式区分所有行为的刻画，预测的理论速率 $k\ln\beta_2/(2(k-2))$ 与 Fig. 2(a) 实证斜率完全对齐。

### 损失函数 / 训练策略
没有训练损失（理论性论文），但给出三个量级关键预测可被代入实践：(a) 线性收敛速率 $\beta_2^{1/(2(k-2))}$；(b) GD 在退化目标上的复杂度 $T_\varepsilon\sim\varepsilon^{-(k-2)}$ 而 Adam 是 $T_\varepsilon\sim(k-2)\ln(1/\varepsilon)$；(c) 备 1D 映射的全局稳定门槛 $\gamma_{\mathrm{crit}}=(\tfrac{k}{k-2})^{k-2}$，对 $k=4$ 等价 $\beta_2>0.0625$。

### 损失函数 / 训练策略
没有训练损失（理论性论文），但给出三个量级关键预测可被代入实践：(a) 线性收敛速率 $\beta_2^{1/(2(k-2))}$；(b) GD 在退化目标上的复杂度 $T_\varepsilon\sim\varepsilon^{-(k-2)}$ 而 Adam 是 $T_\varepsilon\sim(k-2)\ln(1/\varepsilon)$；(c) 备 1D 映射的全局稳定门槛 $\gamma_{\mathrm{crit}}=(\tfrac{k}{k-2})^{k-2}$，对 $k=4$ 等价 $\beta_2>0.0625$。

## 实验关键数据

### 主实验
所有实验均在解析最小问题 $L(x)=\tfrac{1}{k}x^k$ 上做，目的是逐一验证理论预测。

| 实验 | 设置 | 关键观察 | 对应理论 |
|------|------|---------|---------|
| 强凸 vs 退化收敛 | $L=\tfrac{1}{2}x^2$ vs $L=\tfrac{1}{4}x^4$，$\beta_1=0.9,\beta_2=0.99$ | 强凸上 Adam 最终 spike；$k=4$ 上稳定指数下降，斜率 $\approx-0.0726$ | 印证 Adam 的"退化偏好" |
| 收敛速率验证 | $k=4,6$ Adam loss 曲线斜率 | $k=4$ 实测斜率 $\approx-0.0726$，理论 $k\ln\beta_2/(2(k-2))=-0.0726$；$k=6$ 实测 $\approx-0.0544$，理论 $-0.0544$ | 完全对齐 |
| $\lambda_t$ 演化 | 对比 $k=2$ vs $k=4$ | $k=2$ 时 $\lambda_t=1/\sqrt{v_t}$ 无界增长越过 $2/\eta$ 阈值（红区）；$k>2$ 时 $\lambda_t$ 收敛到常数 | 直观展示稳定机制 |
| 理论 vs 实证相图 | $k=4$，$x_0=1, \eta=0.001$，100k 步 | 稳定区 final loss $\approx 10^{-300}$（机器精度），不稳定区显著更高 | 印证定理 4.1 的稳定不等式 |
| 三相典型轨迹 | $\beta_1=0.9$, $\beta_2\in\{0.91,0.895,0.8\}$ | 分别得到稳定指数、指数+spike、SignGD 振荡 | 对应三相 |
| 相图扫描 | $(\beta_1,\beta_2)$ 网格记录 $\min_t L(x_t)$ 与 $L(x_T)$ | Regime II 表现为 $\min L$ 低但 $L_T$ 高（典型 spike 特征） | 用两张图区分相 I/II/III |
| 耦合比 $R_t^{(v)}=v_t/g_t^2$ | 同上 | 相 I/II 中 $\max R_t^{(v)}$ 很大（解耦发生），相 III 中 $\approx 1$（紧耦合） | 直接验证解耦机制是加速根源 |
| 退化与非退化混合 | $\tfrac{1}{4}(x-y)^2+\tfrac{1}{16}(x+y)^4$ 等 4 种耦合损失 | Adam 在含退化方向时显著快于 GD/Momentum，但 quadratic 分量会引入 spike | 解释为何实际任务里 Adam 既快又偶尔 spike |
| 架构相关性 | MLP 中 ReLU vs softmax；Transformer vs CNN | softmax 与 Transformer 的 Hessian 谱更集中在 0（退化更重），Adam 优势更显著 | 把理论关联到真实模型 |

### 消融实验

| 配置 | 关键变化 | 说明 |
|------|---------|------|
| 完整 Adam | $\beta_1,\beta_2$ 都生效 | 三相相图均出现 |
| $\beta_1=0$（退化为 RMSProp） | 去掉一阶动量 | 状态空间从 3D 降到 2D，定理 5.7 给出**全局**线性收敛，证明加速来自 $v_t$ 一支 |
| $\beta_2=0$ 且 $\varepsilon\to 0$（SignGD） | 去掉二阶记忆 | 常数步长下不收敛，停在 $O(L(\eta))$，反证"几何记忆"是加速关键 |
| 把单项 $\tfrac{1}{k}x^k$ 推广到 $\tfrac{1}{k}x^k(1+h(x))$ | $h$ 解析且 $h(0)=0$ | 局部稳定条件与速率完全一致——理论对一般退化最小普适（Remark 4.4） |

### 关键发现
- 退化阶 $k$ 越大，稳定区在 $(\beta_1,\beta_2)$ 平面上**单调扩张**（Remark 4.2，$\beta_1<\beta_2^{k/(2(k-2))}$ 的指数随 $k$ 递减），即更难的退化反而让 Adam 超参更宽松。
- Spike 的物理来源被精确锁定：相 II 中不动点存在但失稳，$v_t$ 先解耦驱动加速，等 $x_t$ 突然反弹后 $v_t$ 需要数步才能重新跟上 $g_t^2$——这段"响应延迟"就是 loss spike。
- SignGD 视角（如 Kunstner et al. 2023, 2024）只能解释相 III；相 I/II 的真正加速来自与 SignGD 相反的 $v_t$–$g_t^2$ 解耦。

## 亮点与洞察
- **找对了"最小问题"**：把 $L=\tfrac{1}{k}x^k$ 作为研究对象，既保留了深网损失景观最关键的退化结构，又使分析能解析推到底；这种"用最朴素一维 model problem 隔离机制"的范式对后续做自适应方法理论很有借鉴价值。
- **复杂度分离严谨**：明确写出 GD 在退化目标上的 $T_\varepsilon\sim\varepsilon^{-(k-2)}$ 指数爆炸 vs Adam 的 $(k-2)\ln(1/\varepsilon)$ 线性复杂度，把"Adam 比 GD 快"从经验上升为复杂度类分离结论，且不需要随机性假设。
- **机制对立的洞察**：解耦机制（$v_t$ 落后于 $g_t^2$）与 SignGD 机制（$v_t$ 紧跟 $g_t^2$）是相反的物理过程，可同时存在于训练的不同阶段或不同坐标——这一区分能直接指导诊断"实际网络训练里 Adam 的优势到底来自哪一类机制"。

## 局限与展望
- 全部分析在 1D 退化多项式上，多维深网损失景观虽与之有 Hessian 谱相似性，但耦合更复杂；Fig. 8 与第 7 节也只在 2D 玩具损失上演示。
- 假设 $\varepsilon=0$、忽略 bias correction、纯确定性梯度（无 mini-batch 噪声）——这三条简化与实际 Adam 跑大模型时差距明显，作者把随机 batch 设置明确列为未来工作。
- Adam 全局基底吸引域无封闭解，定理 4.1 只给局部稳定；Fig. 3(b) 的"经验吸引域宽广"是观察而非证明。
- 改进思路：(a) 把当前定理扩到含 mini-batch 噪声的 SDE 设定；(b) 在真实 Transformer 的 Hessian 谱上分块匹配 $k$ 估计，预测每块的稳定区；(c) 用解耦比 $R_t^{(v)}$ 作训练健康监控信号，触发自适应 $\beta_2$ 调度。

## 相关工作与启发
- **vs Zhang et al. (2022a)**：他们要求 $\beta_2$ 接近 1 + 学习率衰减才证 $O(\log T/\sqrt T)$；本文在 $\beta_1,\beta_2\in[0,1)$ 全域常数步长下给出局部线性收敛，证明 Zhang et al. 的条件本质是为了避开本文相 III。
- **vs Davis et al. (2025)**：他们证 GD + Polyak 自适应步长在四阶增长下能线性收敛；本文证 Adam 不需要外接 Polyak 规则，**内置**的 $v_t$ 几何记忆即可达到同等效果。
- **vs SignGD 加速理论 (Kunstner et al. 2023, 2024)**：本文相 III 明确对应 SignGD 行为且不收敛；线性加速来自相反的解耦机制——补全了 Adam 优势中此前缺失的一块。
- **vs Cohen et al. (2023) "edge of stability"**：他们观察到自适应方法在 EoS 附近的 spike 现象；本文用相 II 的不动点失稳给出精确的代数刻画。
- **vs Zhang et al. (2024) Hessian block heterogeneity**：他们解释 Transformer 中 Adam 优势来自块对角异质性；本文给出另一互补维度——退化阶差异；两者可共存。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 解耦机制 vs SignGD 视角的对立，以及在常数步长下给出 Adam 全相图，都是文献里没出现过的清晰刻画。
- 实验充分度: ⭐⭐⭐⭐ 在解析模型问题上做到了"理论与实证斜率完全对齐"，但缺少真实网络上的端到端复现；架构相关性章节只是 preliminary。
- 写作质量: ⭐⭐⭐⭐⭐ 状态空间归一化的引入、三相相图的对照表（Table 1）、机制图（Fig. 6）层次清楚，公式与图协同非常好读。
- 价值: ⭐⭐⭐⭐ 对理论方向是重要补全，对实践指导（动态 $\beta_2$、自适应调度设计）有启发，但不直接改进现有训练管线。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](../../NeurIPS2025/optimization/understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [\[NeurIPS 2025\] Understanding Adam Requires Better Rotation Dependent Assumptions](../../NeurIPS2025/optimization/understanding_adam_requires_better_rotation_dependent_assumptions.md)
- [\[ICML 2026\] Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence](balanced_lora_removing_parameter_invariance_to_accelerate_convergence.md)
- [\[NeurIPS 2025\] In Search of Adam's Secret Sauce](../../NeurIPS2025/optimization/in_search_of_adams_secret_sauce.md)
- [\[ICML 2026\] Towards Understanding Continual Factual Knowledge Acquisition of Language Models: From Theory to Algorithm](towards_understanding_continual_factual_knowledge_acquisition_of_language_models.md)

</div>

<!-- RELATED:END -->
