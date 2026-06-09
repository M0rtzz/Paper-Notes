---
title: >-
  [论文解读] Turning Stale Gradients into Stable Gradients: Coherent Coordinate Descent with Implicit Landscape Smoothing for Lightweight Zeroth-Order Optimization
description: >-
  [ICML 2026][模型压缩][零阶优化] 本文把"陈旧的"块循环坐标下降梯度估计存进 FIFO buffer，配上 momentum 衰减重用，并证明这等价于带 warm-start 的 BCCD…
tags:
  - "ICML 2026"
  - "模型压缩"
  - "零阶优化"
  - "坐标下降"
  - "陈旧梯度"
  - "隐式平滑"
  - "设备端学习"
---

# Turning Stale Gradients into Stable Gradients: Coherent Coordinate Descent with Implicit Landscape Smoothing for Lightweight Zeroth-Order Optimization

**会议**: ICML 2026  
**arXiv**: [2605.14373](https://arxiv.org/abs/2605.14373)  
**代码**: https://github.com/chen-dylan-liang/CoCD (有)  
**领域**: 模型压缩 / 零阶优化 / 边缘训练  
**关键词**: 零阶优化, 坐标下降, 陈旧梯度, 隐式平滑, 设备端学习

## 一句话总结
本文把"陈旧的"块循环坐标下降梯度估计存进 FIFO buffer，配上 momentum 衰减重用，并证明这等价于带 warm-start 的 BCCD；同时给出反直觉结论——更大的有限差分步长 $\epsilon$ 会隐式平滑 loss landscape、降低有效 Lipschitz 常数，从而让 stale gradient 反而能换来稳定下降。

## 研究背景与动机

**领域现状**：当前零阶（ZO）优化主要走两个流派——经典的逐坐标有限差分（FD）方差极低但每步要 $O(d)$ 次函数评估，scalability 几乎没有；现代随机派（Evolution Strategies / SPSA / DeepZero 的随机子空间）每步只要 $O(1)$ 次评估，但梯度估计方差高，必须用很小学习率或很大 batch 才能压噪声。

**现有痛点**：两派各有死穴。FD 派"准但慢"，根本上不了大模型；随机派"快但不稳"，在非凸 landscape 上很容易发散，对一些回归任务（比如 SARCOS）甚至会跑飞。而且大家有两个隐含共识——"陈旧梯度是 lag、是噪声，应该丢"和"有限差分步 $\epsilon$ 越小越接近真梯度、越好"，本文要正面挑战这两个共识。

**核心矛盾**：单次 ZO 估计的方差 vs 多次估计的样本效率之间是硬 trade-off；同时，随机化引入的噪声其实掩盖了 landscape 本身的几何结构信息。

**本文目标**：(1) 设计一个 deterministic、O(1)-budget、低方差的 ZO 优化器；(2) 证明 stale 梯度可以"复用而不是丢弃"；(3) 证明大的 $\epsilon$ 不是 bug 而是 feature。

**切入角度**：作者观察到优化轨迹有"temporal coherence"——相邻 step 的梯度被 Lipschitz 平滑约束着只能小幅变化。既然如此，上一步算出来的 partial derivative 在下一步几乎仍然有效，扔掉就是浪费；同时，大 $\epsilon$ 等价于在一个 coordinate 邻域内做平均，相当于给目标函数做了隐式 Gaussian smoothing。

**核心 idea**：维护一个 dense gradient buffer，每步只刷新 $B$ 个坐标的 FD，其余坐标用衰减后的旧值，组成"hybrid gradient"做下降——就是 Block Cyclic Coordinate Descent 加上 momentum-style warm start。

## 方法详解

### 整体框架
CoCD（Coherent Coordinate Descent）想要一个确定性、每步只花 $O(1)$ 函数评估、又低方差的零阶优化器，做法是把"上一步算出来的偏导"当成资产留着复用而不是丢掉。它维护一份与参数等大的 dense 梯度 buffer $\hat{\mathbf{g}} \in \mathbb{R}^n$，全局参数 $\mathbf{x}_t$ 按 cyclic 顺序轮流选下一组坐标。每一步先把整个 buffer 衰减一下 $\hat{\mathbf{g}}_{t-1}^{\text{decay}} = \gamma \cdot \hat{\mathbf{g}}_{t-1}$，再只对当前被选中的 $B$ 个坐标用中心有限差分算新值 $\tilde{\nabla}_i f(\mathbf{x}_t) = \frac{f(\mathbf{x}_t+\epsilon\mathbf{e}_i)-f(\mathbf{x}_t-\epsilon\mathbf{e}_i)}{\epsilon}$ 覆盖进 buffer 对应位置，最后拿整份 buffer 做一次全维度更新 $\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha \hat{\mathbf{g}}_t$。于是每步真正算 FD 的只有 $B$ 个坐标（典型 $B \ll n$），但下降方向始终是满维的，绝大多数分量来自衰减后的 stale 值。

### 关键设计

**1. 用一个衰减因子 $\gamma$ 把"丢弃 stale"和"完美记忆"连成一族算法**

零阶优化的两难是：单次估计方差大，多次估计又烧不起预算，而分布式 SGD 的经验又告诉大家陈旧梯度是延迟噪声、该丢。CoCD 反着用——既然连续 step 间梯度变化被 Lipschitz 平滑约束着只能小幅变动，那上一步的偏导就是一个几乎免费的"warm start"，比每步从零做随机估计稳得多。一个标量 $\gamma \in [0,1]$ 就把整个谱系串起来：$\gamma=0$ 时 buffer 每步清零，退化成经典的 Block Cyclic Coordinate Descent（只有当前 $B$ 个坐标有效）；$\gamma=1$ 时 stale 梯度一直保留到被循环刷新，等价于带 time-lagged gradient 的 full-gradient descent；$0<\gamma<1$ 则形成"fading memory"，旧梯度被指数级压低，给高度非凸的 landscape 留出 robustness。效果在 SARCOS 的 $B=1$ 极端压力测试里很直观：$\gamma=0.95$ 的 CoCD 收敛到 loss 68.6，而 BCCD（$\gamma=0$）即使把 $B$ 提到 64 仍卡在 188。

**2. 把有限差分步长 $\epsilon$ 从误差源变成 landscape 平滑器**

传统 FD 文献把 $\epsilon \to 0$ 当黄金法则，可在非凸 landscape 上这反而把高频噪声塞进梯度估计。作者换个视角：中心差分其实等价于在邻域内对真梯度做平均 $\tilde{\nabla}_i^\epsilon f(\mathbf{x}) = \frac{1}{2\epsilon}\int_{-\epsilon}^{\epsilon}\nabla_i f(\mathbf{x}+u\mathbf{e}_i)\,du$，所以 $\epsilon$ 就是一个隐式的 Gaussian smoothing 半径。定义随之平滑后的有效 Lipschitz 常数 $L_\epsilon$，可证 $L_\epsilon \le L$ 且单调随 $\epsilon$ 增大而变小。这一点直接进了 CoCD 的近似误差界

$$\|\hat{\mathbf{g}}_t - \tilde{\nabla}^\epsilon f(\mathbf{x}_t)\| \le \frac{L_\epsilon \delta}{2}\big(BK(K-1)+2rK\big),\quad K=\lfloor n/B\rfloor.$$

$L_\epsilon$ 越小，同样的 stale 程度就能容忍越大的 step size $\delta$——也就是说放大 $\epsilon$ 反过来让"复用 stale 梯度"更安全。SARCOS 上 $\epsilon=1$ 让 BCCD 直接退化、却让 CoCD 拿到全场最好结果，正是这个反直觉论断的正面验证。

**3. 用扁平 FIFO buffer + 虚拟化索引把内存压到"只占一份参数"**

要在边缘设备上跑，内存是硬约束，而神经网络参数是一堆形状各异的 tensor，朴素实现每步都要 reshape/concat 成 flat 再 reshape 回去，复制成本高到离谱。CoCD 的工程 trick 是预分配一段连续 1D 内存当 FIFO buffer（长度 $m$，典型 $m=n$，也可调小做 memory-constrained 部署），用三个整数指针 `cur_param_idx` / `cur_weight_idx` / `cur_grad_idx` 维护 flat index 到各 tensor view 的映射；下降阶段直接在 tensor 的临时 view 上（不复制）做 in-place 减法 $\mathbf{x} \leftarrow \mathbf{x} - \alpha \hat{\mathbf{g}}$。这样整个优化器的内存严格等于一份参数副本，比 Adam（要存 2 份 moment）甚至比 SPSA（要存投影矩阵）都省，正好契合 on-device 训练的诉求。

### 损失函数 / 训练策略
不引入新损失，只是替换优化器。理论侧在 L-smooth + PŁ 条件下证明 CoCD 线性收敛：$f(\mathbf{x}_t)-f(\mathbf{x}^*) \le (1-\frac{2\mu C_1}{C_2})^t (f(\mathbf{x}_0)-f(\mathbf{x}^*))$，其中 staleness factor $\tau = n/B - 1$，要求学习率 $\alpha < \frac{2}{L(1+n\tau)}$。$B$ 越大 $\tau$ 越小、可用学习率越大；$\gamma$ 与 $m$ 则通过定义 effective LR 进一步影响稳定阈值。

## 实验关键数据

### 主实验
模型规模 13k–270k 参数（MLP/CNN/ResNet-20），数据集 SARCOS（回归）、MNIST、CIFAR-10。

| 数据集 | 方法 | 最终指标 | Time (s/epoch) |
|--------|------|----------|---------------|
| SARCOS | SGD (Oracle, 一阶) | Loss 5.38 | – |
| SARCOS | BCCD ($\gamma=0$) | 188.73 | 6.23 |
| SARCOS | **CoCD (本文)** | **31.18** | 6.12 |
| MNIST | BCCD | 27.03% | ~44.4 |
| MNIST | **CoCD** | **95.48%** | ~44.6 |
| CIFAR-10 | BCCD | 10.13%（瞎猜） | ~77.0 |
| CIFAR-10 | **CoCD** | **45.08%** | ~77.0 |

### 消融实验

| 配置 | 关键发现 | 说明 |
|------|---------|------|
| $\gamma$ 扫描 | $\gamma=0$ 最慢、$\gamma \to 1$ 单调更快 | momentum 是收敛速度的主导因子 |
| $\epsilon$ 扫描 (SARCOS) | $\epsilon$ 越大 loss 越低 | 隐式平滑在高非凸场景压倒 FD 误差 |
| $\epsilon$ 扫描 (MNIST) | 大 $\epsilon$ 训练曲线更平滑、稳定性更好 | 与理论的 $L_\epsilon$ 解释一致 |
| 内存预算 $M=0.25n$ | 仍能收敛 | 验证 "只存一份参数" 设计的健壮性 |
| CoCD vs ZO-SGD | 收敛 + 8.1s vs 15.7s 每 episode | 没有 Gaussian 采样开销，wall-clock 2x 快 |
| CoCD vs SPSA | CoCD 稳定收敛、SPSA 在 SARCOS 中途发散 | 确定性更新比随机更新更稳 |

### 关键发现
- **momentum $\gamma$ 是最关键的超参**：直接决定 CoCD 与经典 BCCD 之间的鸿沟（CIFAR-10 上 10% vs 45%）。
- **大 $\epsilon$ 的反直觉收益在非凸回归任务上最显著**，分类任务上则更多体现在训练曲线的平滑度而非最终精度。
- **wall-clock 上明显优于随机 ZO**：因为 deterministic 更新省掉了 Gaussian 采样和大 batch 的开销。

## 亮点与洞察
- **"陈旧 = 资源"这一视角的反转**：分布式 SGD 文献里 stale gradient 一直被当成不可控延迟噪声，本文证明在 cyclic 结构 + temporal coherence 下，stale 其实就是 warm start，几乎免费的几何信息。
- **理论与实证的双向印证**：误差界 $\frac{L_\epsilon \delta}{2}(\cdot)$ 直接预测了"$L_\epsilon$ 小 → 容忍更大 $\delta$"，实验中 $\epsilon=1$ 在 SARCOS 上反败为胜正好验证。
- **$\gamma$/$\epsilon$ 双调节是个可迁移的 trick**：任何"用陈旧近似 + 估计噪声"的场景（联邦学习、异步 SGD、低精度训练）都可以借鉴"显式 decay 历史 + 主动放大平滑半径"这两个手柄。

## 局限与展望
- **scalability 卡在小到中等规模**：参数量超过 270k 后，要让坐标级 FD 估计够准，$B$ 必须涨到不实用的程度，目前还到不了 LLM 级别。
- **超参敏感**：$\gamma$、$\epsilon$、$B$、$m$ 四个旋钮交互复杂，论文给出的最优值是逐任务手调，没有自适应方案。
- **理论假设较强**：linear convergence 依赖 PŁ 条件，深度网络通常不满足；论文也承认这只是 best-case ceiling。
- **未与 MeZO 直接比较**：LLM 微调最热的 ZO 方法是 MeZO（ZO-SGD 变种），作者把跟 MeZO 的对比放到 future work，说明 LLM 场景的有效性还没验证。

## 相关工作与启发
- **vs DeepZero (Chen 2024)**: DeepZero 为了上百万参数模型，先用 pruning at initialization 选随机子空间，再在子空间内做 CGE，相当于把空间外参数永久冻结；CoCD 在 memory-constrained 中等规模上做真正全空间 cyclic，避免了这种结构性偏差，但代价是不能直接上百亿参数。
- **vs SPSA / Evolution Strategies**: 那些方法靠随机扰动估计梯度，方差大、需要大 batch；CoCD 完全 deterministic、low-variance、wall-clock 更快，且 implicit smoothing 把随机派的"显式 Gaussian smoothing"替代掉了。
- **vs WASP (Rakita 2025)**: WASP 也明确假设 temporal coherence、把过去梯度当几何约束，但靠 affine subspace 矩阵分解，适合几百维的机器人控制；CoCD 只靠向量级 buffer 操作，扩展到上万参数仍然轻量。

## 评分
- 新颖性: ⭐⭐⭐⭐ "stale 当 warm start + 大 $\epsilon$ 隐式平滑" 两个反直觉观点都很硬核
- 实验充分度: ⭐⭐⭐ 模型规模偏小（≤270k）、没跟 MeZO/LLM 微调对比，但 SARCOS/MNIST/CIFAR + 多 baseline + 多 ablation 已够支撑论点
- 写作质量: ⭐⭐⭐⭐⭐ 三段式 introduction / 理论铺陈 / 实证验证一气呵成，公式与表格交替推进
- 价值: ⭐⭐⭐⭐ 对边缘训练 / 黑盒优化 / 异步 SGD 三个方向都有可迁移的设计思路

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Fine-tuning Quantized Neural Networks with Zeroth-order Optimization](../../ICLR2026/model_compression/fine-tuning_quantized_neural_networks_with_zeroth-order_optimization.md)
- [\[ICML 2026\] GradPower: Powering Gradients for Faster Language Model Pre-Training](gradpower_powering_gradients_for_faster_language_model_pre-training.md)
- [\[CVPR 2026\] FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation](../../CVPR2026/model_compression/fozo_forward-only_zeroth-order_prompt_optimization_for_test-time_adaptation.md)
- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](../../ACL2025/model_compression/wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[ICML 2026\] Bounded Hyperbolic Tangent: A Stable and Efficient Alternative to Pre-Layer Normalization in Large Language Models](bounded_hyperbolic_tangent_a_stable_and_efficient_alternative_to_pre-layer_norma.md)

</div>

<!-- RELATED:END -->
