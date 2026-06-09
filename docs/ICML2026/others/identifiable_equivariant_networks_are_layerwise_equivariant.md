---
title: >-
  [论文解读] Identifiable Equivariant Networks are Layerwise Equivariant
description: >-
  [ICML2026][等变性] 本文在一个架构无关的抽象框架下证明：只要参数满足"弱可辨识性"，端到端 $G$-等变的深度网络一定存在等价参数化，使得每一层都对某个潜在群作用等变；这从理论上解释了"端到端等变会自动塌缩为逐层等变"这一长期被实验观察到的现象。
tags:
  - "ICML2026"
  - "等变性"
  - "可辨识性"
  - "子模型"
  - "隐空间对称"
  - "MLP"
  - "多头注意力"
---

# Identifiable Equivariant Networks are Layerwise Equivariant

**会议**: ICML2026  
**arXiv**: [2601.21645](https://arxiv.org/abs/2601.21645)  
**代码**: 无  
**领域**: 等变神经网络 / 几何深度学习理论  
**关键词**: 等变性, 可辨识性, 子模型, 隐空间对称, MLP, 多头注意力

## 一句话总结
本文在一个架构无关的抽象框架下证明：只要参数满足"弱可辨识性"，端到端 $G$-等变的深度网络一定存在等价参数化，使得每一层都对某个潜在群作用等变；这从理论上解释了"端到端等变会自动塌缩为逐层等变"这一长期被实验观察到的现象。

## 研究背景与动机

**领域现状**：几何深度学习的主流范式是"逐层等变"——先给定一个作用在输入/输出上的对称群 $G$，再去构造每一层都 $G$-等变的线性算子（CNN 的平移等变、GNN 的置换等变、equivariant transformer 的旋转等变都属于这一类）。这条路线已经在视觉、图、分子建模中被反复验证有效。

**现有痛点**：但"逐层等变"是不是构造端到端等变网络的*唯一*办法？实验中人们观察到一个反直觉现象：即使用一个**没有任何等变约束**的普通 MLP 去拟合带对称性的数据（或者加 symmetry-based data augmentation），训练完之后中间层的权重也会自发呈现等变结构（Lenc & Vedaldi 2015, Gruver et al. 2023, Bökman & Kahl 2023）。这种"自发等变"什么时候出现、为什么出现，缺乏统一的理论解释。

**核心矛盾**：端到端等变这个"全局约束"和逐层等变这个"局部结构"之间存在一道鸿沟——后者显然蕴含前者，但反过来是否成立、在什么条件下成立，过去只有针对浅层 ReLU MLP 的零散结果（Agrawal & Ostrowski 2022, Marchetti et al. 2024）。

**本文目标**：在一个尽可能抽象的层面上回答"端到端等变 $\Rightarrow$ 逐层等变？"这个反问句，并把所需的充分条件梳理清楚。

**切入角度**：作者借鉴范畴论里的"参数化函数"语言，把每一层抽象成 $f_i: V_{i-1} \times \Theta_i \to V_i$，把对称性抽象成任意群在隐空间 $V_i$ 上的作用 $K_i$，再借助"参数可辨识性"（同一函数对应的参数只能差一个对称变换）这把万能钥匙撬开问题。

**核心 idea**：用"弱可辨识性"作为唯一假设，证明 $G$-等变性会自动沿层向内传播——输入端的群作用通过隐空间的辨识对称群 $K_i$ 一路传递到输出端，每一层都因此被迫等变。

## 方法详解

本文不是算法论文而是理论论文，"方法"指的是其证明框架与抽象形式化。

### 整体框架

作者把深度模型重新定义为四元组 $(V_i, \Theta_i, f_i, K_i)$：$V_i$ 是隐空间集合、$\Theta_i$ 是参数空间、$f_i: V_{i-1}\times\Theta_i \to V_i$ 是层映射、$K_i$ 是作用在 $V_i$ 上的对称群。整个端到端函数定义为 $f(\bullet; \theta) = f_L(\bullet; \theta_L)\circ\cdots\circ f_1(\bullet; \theta_1)$。在这个抽象层面上，MLP、注意力网络、卷积网络都只是同一定义的不同具体实例。证明思路是：先给出"子模型 (submodel)"的形式化定义，再用子模型刻画"退化参数"，最后通过"弱可辨识性 + 伴随性质 (adjunction property)"完成逐层等变性的归纳推导。

### 关键设计

**1. 子模型（Submodel）：把冗余/失活神经元这类退化情形提升到架构无关的统一形式**

谈"参数唯一性"绕不开一个麻烦：MLP 里某些神经元可能根本不参与前向计算，或两个神经元功能完全相同，于是同一函数对应无穷多个等价参数，可辨识性直接被破坏。本文的对策是把所有退化都解释成"来自一个更小的子模型"。子模型由 $(\widetilde V_i,\widetilde\Theta_i,\widetilde f_i)$ 加一组联系映射 $\alpha_i:\widetilde V_i\to V_i$、$\alpha_i^*:V_i\to\widetilde V_i$（满足 $\alpha_i^*\circ\alpha_i=\mathrm{Id}$）和 $\beta_i:\widetilde\Theta_i\to\Theta_i$ 构成，并要求一个交换图成立；对 MLP 取线性 $\alpha_i$ 时，子模型恰好就是"删掉失活神经元、合并冗余神经元"后得到的小网络。这样一来，退化的部分被子模型吃干净，剩下的非退化部分就可以单独要求可辨识性，从而绕开 ReLU 等病态激活带来的麻烦——这是整个证明能架构无关的地基。

**2. 弱可辨识性（Weak Identifiability）：把各家针对具体激活的辨识结果统一成一个最小假设**

Sussmann、Fefferman、Vlačić & Bölcskei 等人对 Tanh、polynomial、sigmoidal 等具体激活各自证过可辨识性，本文要的是一把能直接复用这些结论的万能钥匙。定义是：参数 $\theta$ 弱可辨识，指存在某个子模型上的可辨识参数 $\widetilde\theta$ 使 $\theta_i=\beta_i(\widetilde\theta_i)$；而"可辨识"指若 $f(\bullet;\theta)=f(\bullet;\theta')$，则存在唯一的 $k_i\in K_i$ 序列满足 $f_i(x;\theta_i')=k_i\cdot f_i(k_{i-1}^{-1}\cdot x;\theta_i)$。这个假设的妙处在于它把"证明"和"具体激活"解耦：很多激活在去掉退化后已被证可辨识，本框架直接拿来当输入；ReLU 虽仍是开放问题，但作者明确指出一旦将来证明了 ReLU 的弱可辨识性，本文结论立刻自动适用。

**3. 伴随性质（Adjunction Property）：把端到端等变的全局约束翻译成可逐层推导的局部条件**

主定理要做的是把"整张网络端到端等变"这个全局事实，一层层地传导成"每层各自等变"。关键支点是伴随性质：要求 $G$ 也作用在首末两层的参数上，并满足 $f_1(g\cdot x_0;\theta_1)=f_1(x_0;g^{-1}\cdot\theta_1)$ 与 $g\cdot f_L(x_{L-1};\theta_L)=f_L(x_{L-1};g\cdot\theta_L)$。有了它，就能把输入端的 $g$ 作用"搬"到第一层参数上，再借弱可辨识性把它强行转成某个 $k_1\in K_1$ 在 $V_1$ 上的作用，然后归纳推进到下一层。这个条件并非凭空假设——MLP（$G$ 线性作用于输入/输出时）、加位置编码的注意力网络都天然满足；对于无位置编码的注意力或会"吸收"群作用的 CNN，作者给出广义伴随性质 $g^{-1}\cdot f_i(g\cdot x,\theta)=f_i(x,g^{-1}\cdot\theta)$ 使证明照样成立；而 Deep Sets、equivariant GNN 这类"层本身就是等变线性算子"的架构会破坏伴随条件，作者诚实地指出定理在此不适用。

### 主定理与证明梗概

**定理 4.1**：若 $\theta$ 弱可辨识，且模型在 $\theta$ 处端到端 $G$-等变，则存在 $G$ 在每个 $V_i$（$i=1,\dots,L-1$）上的作用，使得每个 $f_i$ 在 $\theta_i$ 处都 $G$-等变。

证明的归纳骨架是：对任意 $g\in G$，利用伴随性质把第一层的"输入侧 $g$ 作用"转化为"参数侧 $g^{-1}$ 作用"，从而得到两个端到端相等的参数；再由弱可辨识性得到唯一的隐空间对称 $k_i(g)\in K_i$；最后验证 $g\mapsto k_i(g)$ 是群同态，并把它定义为 $G$ 在 $V_i$ 上的作用。Remark 4.2 指出，该隐空间作用实际上由群同态 $G\to K_i$ 给出，即潜在群作用一定"分解"经过隐空间自带的辨识对称群。

## 实验关键数据

理论结果用 CIFAR-10 上的小型 MLP（深度 4，第一层非线性为 Tanh 或 GELU，后续均为 GELU）和单层多头注意力网络做定性 + 定量验证。训练阶段后半段加 mirror-equivariance 软损失（而非硬约束），观察隐空间是否自发形成等变结构。

### 主实验：第一/第二层的相对等变误差

通过最小二乘估出线性变换 $A_i$ 使 $A_i f_i(x)\approx f_i(\mathrm{mirror}[x])$，在 10 万独立噪声样本上计算 $|f_i(\mathrm{mirror}[y]) - A_i f_i(y)|/|f_i(y)|$ 的中位数：

| 任务 | 层 | 镜像方向 | Tanh | GELU |
|------|-----|---------|------|------|
| Autoencoder | $f_1$ | 左右 | **0.029** | 0.40 |
| Autoencoder | $f_2$ | 左右 | **0.022** | 0.11 |
| Autoencoder | $f_1$ | 上下（参考） | 0.15 | 0.49 |
| Classifier | $f_1$ | 左右 | **0.077** | 0.19 |
| Classifier | $f_2$ | 左右 | **0.054** | 0.064 |
| Classifier | $f_1$ | 上下（参考） | 0.48 | 0.56 |

结论非常清晰：训练目标只要求*端到端*近似镜像等变，Tanh 网络第一层就自发逼近线性等变（误差 0.029-0.077，对应符号置换矩阵，正是 Tanh 的 intertwiner 群 $\{\pm 1\}^{d_i}\rtimes S_{d_i}$）；GELU 网络则不然，因为 GELU 满足 $\sigma(x)-\sigma(-x)=x$，可以"绕过"非线性，需要两层组合 $f_2$ 才能恢复线性等变性。

### 消融对比：激活函数 vs 隐空间作用结构

| 激活函数 | 隐空间作用 $K_i$ | 弱可辨识性 | $f_1$ 是否线性等变 |
|---------|---------------------|------------|-------------------|
| Tanh | $\{\pm 1\}^{d_i}\rtimes S_{d_i}$（符号置换） | 已证明 | 是（误差 ≈ 0.03） |
| 大次幂 $t^m$ | $(\mathbb R^\times)^{d_i}\rtimes S_{d_i}$（monomial） | 已证明 | 理论支持 |
| GELU | 仅置换 + 满足 $\sigma(x)-\sigma(-x)=x$ | 难（含 ReLU 类病态） | 否（误差 0.4） |
| ReLU | 正缩放 + 置换 | 开放问题 | 仅浅层 + 重参数化情形成立 |

### 关键发现
- **理论与现象一一对应**：Tanh 网络第一层学到的 64 个 filter 在视觉上呈现"自身/镜像/取反"三类成对结构（Pink/Light Blue/Gold 三色），恰好就是符号置换群的几何表现；GELU 出现大量"取反副本"则源于 $\sigma(x)-\sigma(-x)=x$ 的 bypass 效应，恰恰说明此时 $V_1$ 上的作用不是简单的置换。
- **注意力网络的等变结构是 head permutation**：在 CIFAR-10 自编码器的单层多头注意力中，输入图像左右翻转后大部分 head 的 attention map 同样翻转，但其中第 1 / 第 5 两个 head 之间发生互换——这正是 $K_i = \mathrm{GL}(d_i)^{h_i}\rtimes S_{h_i}$ 里 $S_{h_i}$ 部分的直观体现。
- **加位置编码后注意力满足标准伴随性质**：因此即使是 $G\subseteq S_n$（token 置换）类对称，可辨识性也能蕴含位置编码本身的等变性，给"为什么位置编码会自发学到有结构的几何模式"提供了一种解释。

## 亮点与洞察
- **首次给出架构无关的"端到端等变 $\Rightarrow$ 逐层等变"定理**：过去结果只覆盖浅层 ReLU MLP 或带特殊缩放对称的层，本文用范畴论 + 可辨识性把整个证明"一次性"做完，MLP/attention/polynomial CNN 都是 corollary。
- **"弱可辨识性"的引入是关键技巧**：直接要求可辨识会被退化参数破坏，而用"子模型"吃掉所有退化情形之后，问题就退化成对每一类已知架构核查现成的可辨识性结果即可，把抽象证明和具体架构干净解耦。
- **给"为什么没有等变约束也会学到等变结构"提供解释**：以前只能说"经验上观察到了"，现在能说"只要数据/augmentation 强迫网络端到端等变，且参数辨识性成立，那么逐层等变是数学上必然的"。
- **指明 ReLU 仍是开放问题**：定理的形式让 ReLU 网络的"层等变性研究"和"可辨识性研究"完全等价，把一个深度学习理论里的硬骨头变成了纯代数几何问题（因为 ReLU MLP 与 polynomial 模型有平行结构）。

## 局限与展望
- **理论假设是"精确"等变**，但实际网络只是近似等变；作者承认补救只能靠实验定性展示（CIFAR-10 mirror 实验），缺乏定量的"近似等变 $\Rightarrow$ 近似逐层等变"稳定性界。
- **跳跃连接 (skip connection) 未覆盖**：残差块会增加层间依赖、降低可辨识性（可以把残差通道置零等价改变有效深度），现代主流架构（ResNet/Transformer）严格说不能直接套用定理。
- **不告诉你隐空间作用具体是什么**：定理只保证"存在某个 $K_i$ 上的作用使每层等变"，但没有给出最优作用应如何选择、训练过程中实际会收敛到哪一个。
- **不涉及训练动力学**：是存在性结果而非构造性结果——它告诉你"等价参数化存在"，但不告诉你随机初始化的无约束 MLP 是否一定会在 SGD 下收敛到这种参数化。
- **Deep Sets / 等变 GNN 不适用**：这些架构的层本身会"吸收"特定群作用（而非作用于参数空间），破坏伴随性质，定理在此失效，作者明确列为未来方向。

## 相关工作与启发
- **vs Agrawal & Ostrowski (2022)**：他们针对浅层 ReLU MLP 用特殊重参数化恢复可辨识性，本文则把"可辨识性 $\Rightarrow$ 逐层等变"的归约推广到任意架构和任意群，浅层 ReLU 结果成为本文的 Corollary。
- **vs Marchetti et al. (2024)**：他们只对带 neuron-wise scaling 对称的模型的第一层证明了等变结构，本文证明的是任意深度任意层。
- **vs Xie & Smidt (2025) / Brehmer et al. (2024)**：他们经验性地辩论"硬等变约束 vs data augmentation"哪个更好，本文给出一个理论侧的结论：在可辨识性成立的前提下，两条路线给出的*函数类*是相同的（augmentation 训练出的网络存在等价的逐层等变参数化），把争论转移到"训练动力学 / 优化效率"层面。
- **vs Categorical Deep Learning (Gavranović et al. 2024)**：本文部分采用了范畴论的形式化（Remark 3.3），是该路线在等变性问题上的一次实质性应用，可作为后续把范畴论用于深度学习理论证明的样板。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 给出架构无关的"端到端 $\Rightarrow$ 逐层"等变性首个统一定理
- 实验充分度: ⭐⭐⭐ 理论论文 + 定性 + 一张定量表，CIFAR-10 小网络验证够用但谈不上充分
- 写作质量: ⭐⭐⭐⭐⭐ 抽象形式化与具体例子（MLP / attention）来回切换，节奏清晰
- 价值: ⭐⭐⭐⭐⭐ 为"自发等变"现象提供理论根基，并把 ReLU 层等变性归约为已知开放问题

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] How the Optimizer Shapes Learned Solutions in Equivariant Neural Networks](how_the_optimizer_shapes_learned_solutions_in_equivariant_neural_networks.md)
- [\[NeurIPS 2025\] Equivariance by Contrast: Identifiable Equivariant Embeddings from Unlabeled Finite Group Actions](../../NeurIPS2025/others/equivariance_by_contrast_identifiable_equivariant_embeddings_from_unlabeled_fini.md)
- [\[NeurIPS 2025\] Learning (Approximately) Equivariant Networks via Constrained Optimization](../../NeurIPS2025/others/learning_approximately_equivariant_networks_via_constrained_optimization.md)
- [\[NeurIPS 2025\] On Universality Classes of Equivariant Networks](../../NeurIPS2025/others/on_universality_classes_of_equivariant_networks.md)
- [\[ICML 2025\] Permutation Equivariant Neural Networks for Symmetric Tensors](../../ICML2025/others/permutation_equivariant_neural_networks_for_symmetric_tensors.md)

</div>

<!-- RELATED:END -->
