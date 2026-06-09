---
title: >-
  [论文解读] TPV: Parameter Perturbations Through the Lens of Test Prediction Variance
description: >-
  [ICML2026][优化/理论][预测方差] 作者把"训好模型对参数扰动的局部预测敏感度"形式化为 Test Prediction Variance（TPV），证明其在一阶近似下化为 $\mathrm{Tr}(H_{\mathrm{eff}}C)$ 的迹形式…
tags:
  - "ICML2026"
  - "优化/理论"
  - "预测方差"
  - "参数扰动"
  - "良性过拟合"
  - "宽极小"
  - "训练集模型选择"
---

# TPV: Parameter Perturbations Through the Lens of Test Prediction Variance

**会议**: ICML2026  
**arXiv**: [2512.11089](https://arxiv.org/abs/2512.11089)  
**代码**: 论文文末标注 "Code Available Here"（具体仓库需查 arXiv 页面）  
**领域**: optimization  
**关键词**: 预测方差, 参数扰动, 良性过拟合, 宽极小, 训练集模型选择  

## 一句话总结
作者把"训好模型对参数扰动的局部预测敏感度"形式化为 Test Prediction Variance（TPV），证明其在一阶近似下化为 $\mathrm{Tr}(H_{\mathrm{eff}}C)$ 的迹形式，从而把 SGD 噪声、标签噪声、量化、剪枝放进同一个曲率–协方差框架，并给出一个完全用训练集就能估计 TPV 的稳定性定理，落地为 label-free 剪枝准则 JBR 和无需测试标签的模型选择信号。

## 研究背景与动机
**领域现状**：理解训练好的网络对"训练后噪声"（SGD 收敛噪声、有限精度、微调时的标签噪声、剪枝掩码）的鲁棒性，目前各有理论分支：宽极小/平坦极小理论（Keskar 等）、隐式优化偏置（Soudry 等）、benign overfitting（Bartlett 等）、NTK 理论（Jacot 等）。

**现有痛点**：这些理论都用各自的工具，回答的几乎是同一类问题——"优化器最终落在哪个 $w^\star$"。但实际部署中我们关心的是给定一个已经训完的 $w^\star$，它在面对真实扰动时输出会变多少；这是个局部、固定模型的问题，前述理论没有一个统一量直接表达。

**核心矛盾**：现有视角把变化建模为"重新训练得到不同 $w$"，是全局变量；而真实噪声（小步长 SGD 抖动、量化、按掩码剪枝）都作用在固定 $w^\star$ 的邻域，本质是局部参数扰动。两种视角统计意义完全不同，混着用会出错。

**本文目标**：定义一个直接刻画"固定 $w^\star$ 下输出对参数扰动的敏感度"的量；证明它能被训练集估计；用它把各类噪声机制统一起来；并落到可用的应用（剪枝、模型选择）上。

**切入角度**：在 $w^\star$ 附近做一阶 Taylor 展开 $f_{w^\star+\delta w}(x)\approx f_{w^\star}(x)+J(x)\delta w$，那么预测方差自然分解成"模型雅可比的几何"与"扰动协方差"两部分。这一拆分恰好对应大多数训练后噪声源的物理结构——噪声机制只决定 $C$，而 $H_{\mathrm{eff}}$ 是与噪声无关的纯几何对象。

**核心 idea**：用一阶 TPV 的迹形式 $\mathrm{Tr}(H_{\mathrm{eff}}C)$ 作为唯一统一量，把各种训练后扰动的鲁棒性问题压缩成"扰动协方差 $C$ 怎么与 Jacobian 几何 $H_{\mathrm{eff}}$ 耦合"。

## 方法详解

### 整体框架
TPV 框架是一个分析工具而非新训练算法，主线由三部分组成：

1. **定义层**：把 TPV 定义为 $\mathrm{TPV}:=\mathbb{E}_{x,\delta w}\bigl[\|f_{w^\star+\delta w}(x)-f_{w^\star}(x)\|^2\bigr]$；一阶近似下化为 $\mathrm{Tr}(H_{\mathrm{eff}}C)$，其中 $H_{\mathrm{eff}}:=\mathbb{E}_x[J(x)^\top J(x)]$ 是模型雅可比的二阶矩，$C:=\mathbb{E}_R[\delta w\delta w^\top]$ 是扰动协方差。
2. **稳定性层**：证明在过参数化、各向同性扰动下，$\mathrm{TPV}(w^\star;X_{\mathrm{tr}})$ 与 $\mathrm{TPV}(w^\star;X_{\mathrm{te}})$ 的差被一个跟泛化无关的小量上界控制，从而训练集就能估计测试集 TPV。
3. **机制层**：对每种噪声源（标签噪声、SGD 稳态噪声、量化）显式算出 $C$ 的形式，把 TPV 写成可解释的函数，并恢复现有理论结论（benign overfitting、wide minima）。

下游再接 JBR 剪枝准则与训练集模型选择两个应用。

### 关键设计

**1. TPV 的迹分解 $\mathrm{Tr}(H_{\mathrm{eff}}C)$ 作为统一镜片：把异构扰动映到同一标量。**

旧分析往往要为 SGD 噪声、标签噪声、量化、剪枝各搞一套数学工具。TPV 的出发点是一阶 Taylor 展开 $f_{w^\star+\delta w}(x)\approx f_{w^\star}(x)+J(x)\delta w$：先对每个 $x$ 写 $(J(x)\delta w)^2=\mathrm{Tr}(J(x)^\top J(x)\delta w\delta w^\top)$，再对 $x$ 和 $\delta w$ 双重取期望（两者独立），就得到

$$\mathrm{TPV}\approx\mathrm{Tr}(H_{\mathrm{eff}}C),\quad H_{\mathrm{eff}}:=\mathbb{E}_x[J(x)^\top J(x)],\ C:=\mathbb{E}_R[\delta w\delta w^\top]$$

$H_{\mathrm{eff}}$ 是 label-free 的纯几何量（只跟测试输入与训好的 Jacobian 有关），$C$ 完全由噪声机制决定。于是分析每种噪声只剩"算 $C$"这一步：SGD 稳态噪声 $C\propto(\eta/b)\nabla^2 L(w^\star)$，量化 $C=\tfrac{\delta^2}{12}I$，标签噪声 $C$ 与 $J^\top J$ 的伪逆相关。这个"形式不变、只换 $C$"的模板也顺带解释了为什么宽极小假设对 SGD 与量化都成立（两者 $C$ 都正比 Hessian），却对标签噪声不成立（它的 $C$ 走雅可比方向、与 Hessian 谱无关）。

**2. TPV Trace Stability 定理：训练集就能估计测试集 TPV。**

sharpness 文献一直默默用"训练集 sharpness 近似测试 sharpness"做无标签模型选择，却从未严格证明。定理 3.1 第一次把它写成上界

$$\big|\mathrm{TPV}(w^\star;X_{\mathrm{tr}})-\mathrm{TPV}(w^\star;X_{\mathrm{te}})\big|\le c_1\mathrm{Tr}(C),\quad c_1=\tfrac{n_{\mathrm{tr}}+n_{\mathrm{te}}}{p}\varepsilon_{\mathrm{NTK}}+o(1)$$

证明靠两点：NTK 在训练全程基本保持稳定（Jacot/Allen-Zhu）；$H_{\mathrm{eff}}(w_0;X)$ 在初始化时对训练/测试集都集中到总体量（大数律）。两件事拼起来说明 $H_{\mathrm{eff}}$ 在训练前后、训练集与测试集上差异都极小，于是 $\mathrm{Tr}(H_{\mathrm{eff}}C)$ 也差异极小。最关键的是这个上界与泛化间隙 $L_{\mathrm{test}}-L_{\mathrm{train}}$ 无关——传统直觉以为只有泛化好的模型才能用训练集统计代替测试集统计，TPV 稳定性说不是这样。

**3. 标签噪声下的非线性版 benign overfitting（定理 4.2）：把线性回归结论搬到深网。**

经典 benign overfitting 只给线性回归的 $\sigma_\varepsilon^2\mathrm{Tr}((XX^\top)^{-1})$。这里把"训练标签被 $\varepsilon_i$ 污染、选最小范数解"在一阶线性化下重做，得到

$$\mathrm{TPV}_{\mathrm{label}}\approx\sigma_\varepsilon^2\sum_{i=1}^r B_{ii}/s_i^2$$

其中 $s_i$ 是训练集 Jacobian 的非零奇异值，$B_{ii}=(V^\top H_{\mathrm{eff}}V)_{ii}$ 是测试分布 Jacobian 在训练 Jacobian 右奇异向量上的投影能量；线性情况是它把 Jacobian 换成数据矩阵 $X$ 的特例。这个式子直接说明非线性深网的标签噪声敏感度由"小奇异值方向上是否恰好有测试 Jacobian 的能量"主导——而 NTK 理论保证过参数化网络的最小奇异值有下界，从而 $\sum B_{ii}/s_i^2$ 受抑制，这正是过参数化为何降低标签噪声敏感度的几何解释。

### 损失函数 / 训练策略
TPV 本身不是训练目标，文中给出的训练相关结论只有定理 4.3：squared loss 下 SGD 稳态噪声的 TPV 近似为 $\tfrac{\eta\sigma_\varepsilon^2}{2b}\mathrm{Tr}(\nabla_w^2 L(w^\star))$，即"学习率/批量比 × 平方残差 × Hessian 迹"，恢复宽极小直觉；量化噪声给出 $\tfrac{\delta^2}{12}\mathrm{Tr}(\nabla_w^2 L(w^\star))$。

### 损失函数 / 训练策略
TPV 本身不是训练目标，文中给出的训练相关结论只有定理 4.3：squared loss 下 SGD 稳态噪声的 TPV 近似为 $\tfrac{\eta\sigma_\varepsilon^2}{2b}\mathrm{Tr}(\nabla_w^2 L(w^\star))$，即"学习率/批量比 × 平方残差 × Hessian 迹"，恢复宽极小直觉；量化噪声给出 $\tfrac{\delta^2}{12}\mathrm{Tr}(\nabla_w^2 L(w^\star))$。

## 实验关键数据

### 主实验
作者把实验拆成"TPV 稳定性验证"和"TPV–test loss 相关性验证"两个层级。

| 实验场景 | 配置规模 | 关键观察 | 含义 |
|---------|---------|---------|------|
| 合成数据 TPV 稳定性 | 324 配置 × 2 噪声源（标签 / SGD），$n_{\mathrm{train}}=1000$ | TPV 跨 5 个数量级，所有点紧贴 $y=x$ 对角线，宽度甚至 = 1 时也成立 | 稳定性远超定理要求 |
| 合成数据小样本断点 | 同上但 width = 256，$n_{\mathrm{train}}\in\{10,1000\}$ | $n=10$ 时严重偏离对角线，$n=1000$ 时贴合 | 稳定性只在样本太少时崩 |
| CIFAR-10 MobileNetV2 | 多宽度乘子 | 稳定性跨架构保持 | 真实数据上成立 |
| CIFAR-100 logit 噪声 + 微调 | 宽度递增的 MobileNetV2 | 宽度↑ → 训练/测试 TPV 都↓ → 测试 CE 也↓ | TPV 与泛化在低训练损失区正相关 |

### 消融实验

| 消融维度 | 设置 | 结论 |
|---------|------|------|
| 噪声方差 $\sigma_\varepsilon$ | 合成数据 $\sigma=0.01$ vs $0.1$ | $\sigma=0.1$ 时训练集 TPV 饱和到 $\sigma^2$，测试集 TPV 仍随宽度下降——稳定性在大扰动下崩，与定理上界变松一致 |
| 正则化扫描 | CIFAR-10 上扫 weight decay / dropout / label smoothing | TPV 与 test loss 呈 U 形：低训练损失区共降，高训练损失区反向（欠拟合区 TPV 小但 test loss 大） |
| 训练轨迹 | ResNet-18 / CIFAR-100 / 30% 标签噪声 | TPV 最大值时刻恰好分隔欠拟合阶段与 epoch-wise double descent 阶段 |
| 剪枝准则对比 | JBR vs 7 个基线（Jacobian / L1 / BN Scale / FPGM / WHC / Taylor / Random），CIFAR-10/100 + ImageNet | JBR 在 4 个设置上全部匹配或超过 SOTA |
| 训练配方稳健性诊断 | 同 MLP 扫 7 个 weight decay，对比 sharpness vs label-noise TPV | sharpness 与标签噪声敏感度不正相关，TPV 能正确挑出最稳健配方 |

### 关键发现
- TPV 稳定性与模型是否泛化得好**解耦**：泛化间隙大的模型同样贴合对角线，颠覆了"用训练集统计预测测试集统计需要先泛化好"的默认直觉。
- TPV 与 test loss 的 U 形关系跨架构、跨正则化方式都成立；通过 "argmax TPV epoch" 这一可观察地标，能在单条训练曲线上把 underfit 与 double descent 阶段切开。
- 把宽极小假说与 benign overfitting **统一**：两个长期被认为来自不同物理机制的结论，在 TPV 框架下分别对应 $C\propto$Hessian 与 $C\propto J^\dagger J^{\dagger\top}$ 两种扰动协方差形式。

## 亮点与洞察
- **统一性**：一个 $\mathrm{Tr}(H_{\mathrm{eff}}C)$ 式子同时回收宽极小、benign overfitting、量化敏感度、剪枝重要性四件事，且只靠"换 $C$"。这种"形式不变，仅噪声协方差变"的思想可迁移到任何"小参数扰动"问题。
- **训练集就够**：TPV 稳定性把"无标签模型选择"这个工程难题从经验法则上升到理论结论，并且不依赖泛化间隙——意味着可以在标签很贵的场景（医学、隐私）下用纯训练数据挑模型。
- **U 形机制解释**：把"模型选择曲线为什么有时单调、有时 U 形"归结为 bias 与 variance 在 $L_{\mathrm{train}}$ 上的相对主导关系，给后续做 hyperparameter 自动调优一个清晰的相位划分。

## 局限与展望
- 定理 3.1 只对各向同性 $C$ 和过参数化极限严格成立，对各向异性扰动（如标签噪声 $C$ 与 Jacobian 几何强耦合时）需要经验稳定性补位。
- 一阶 Taylor 近似在大扰动或 $\sigma_\varepsilon$ 大、解离开线性化邻域时会失效；附录 D.2/D.3 自己也承认非线性标签噪声 TPV 在实践中难以精确估计。
- TPV 是均方/MSE 框架内的量；分类任务里实际用的是 logit 扰动或微调过程，论文用 CE 与 logit 噪声做了实证桥接但缺正式定义；推广到结构化输出（检测、生成）仍是开放问题。
- 改进方向：(i) 把稳定性证明从 NTK 区域扩展到 feature learning 区域；(ii) 把输入分布漂移也纳入 $C$；(iii) 给标签噪声 TPV 提供数值稳定的实践估计算法。

## 相关工作与启发
- **vs 宽极小/平坦极小理论 (Keskar et al., 2017; Foret et al., 2020 SAM)**：传统理论说"$\mathrm{Tr}(\nabla^2 L)$ 小则泛化好"，TPV 重新表述为"$H_{\mathrm{eff}}$ 与 SGD/量化产生的 $C$ 共同决定鲁棒性"，并解释 Dinh et al. (2017) 的 reparameterization 反例为什么没真正推翻宽极小直觉——固定噪声物理后 $C$ 也跟着重参数变。
- **vs Benign Overfitting (Bartlett et al., 2020)**：他们在线性回归下证 $\sigma^2\mathrm{Tr}((XX^\top)^{-1})$；定理 4.2 把 $X$ 换成 Jacobian、把数据条件数换成 Jacobian 谱条件数，直接推广到任意非线性深网。
- **vs Bordelon & Pehlevan (2023) 有限宽预测涨落**：他们走 DMFT 描述训练动力学的全局方差；TPV 是 post-training 的局部量，互补而非竞争。
- **vs Jacobian Criterion (Chen et al., 2025) 剪枝准则**：JBR 在 TPV 几何下重新推导，得到的 score 形式与 JC 同源但加入了 $\delta_u$ 这一"对预测类负对数概率的输出方向"，从而保留训练集已分类正确样本的预测——附录 H.1 给了具体差异分析。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 单个 $\mathrm{Tr}(H_{\mathrm{eff}}C)$ 同时回收四类长期独立的鲁棒性理论，且 TPV 稳定性是文献里第一份。
- 实验充分度: ⭐⭐⭐⭐ 324 配置的合成实验 + CIFAR/ImageNet/NLU 多任务覆盖，但剪枝与模型选择的具体数字主要放在附录。
- 写作质量: ⭐⭐⭐⭐ 主线清晰，定理与机制对应工整；公式密度大，部分推导（如定理 4.2 的最小范数解假设）需要附录配合才完全清楚。
- 价值: ⭐⭐⭐⭐⭐ "训练集 TPV 替代测试集模型选择"如果落到实际管线里会显著降低标注成本，理论统一性也对后续做训练后鲁棒性研究是显然的脚手架。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Test-Time Augmentation Improves Efficiency in Conformal Prediction](../../CVPR2025/optimization/test-time_augmentation_improves_efficiency_in_conformal_prediction.md)
- [\[ICML 2026\] Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence](balanced_lora_removing_parameter_invariance_to_accelerate_convergence.md)
- [\[ICLR 2026\] Personalized Collaborative Learning with Affinity-Based Variance Reduction](../../ICLR2026/optimization/personalized_collaborative_learning_with_affinity-based_variance_reduction.md)
- [\[ICML 2026\] Test time training enhances in-context learning of nonlinear functions](test_time_training_enhances_in-context_learning_of_nonlinear_functions.md)
- [\[ICML 2026\] Colorful Pinball: Density-Weighted Quantile Regression for Conditional Guarantee of Conformal Prediction](colorful_pinball_density-weighted_quantile_regression_for_conditional_guarantee_.md)

</div>

<!-- RELATED:END -->
