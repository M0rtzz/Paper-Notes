---
title: >-
  [论文解读] TITAN: A Trajectory-Informed Technique for Adaptive Parameter Freezing in Large-Scale VQE
description: >-
  [NeurIPS 2025][物理/科学计算][VQE] 提出TITAN框架，用深度学习模型预测VQE中的"冻结参数"（训练过程中始终不活跃的参数），在初始化阶段即冻结40-60%参数，实现最高3倍收敛加速和40-60%电路评估量减少，在30量子比特的分子系统上匹配或超越基线精度。
tags:
  - "NeurIPS 2025"
  - "物理/科学计算"
  - "VQE"
  - "参数冻结"
  - "barren plateau"
  - "量子化学"
  - "深度学习辅助量子计算"
---

# TITAN: A Trajectory-Informed Technique for Adaptive Parameter Freezing in Large-Scale VQE

**会议**: NeurIPS 2025  
**arXiv**: [2509.15193](https://arxiv.org/abs/2509.15193)  
**代码**: GitHub (开源)  
**领域**: 物理学  
**关键词**: VQE, 参数冻结, barren plateau, 量子化学, 深度学习辅助量子计算

## 一句话总结
提出TITAN框架，用深度学习模型预测VQE中的"冻结参数"（训练过程中始终不活跃的参数），在初始化阶段即冻结40-60%参数，实现最高3倍收敛加速和40-60%电路评估量减少，在30量子比特的分子系统上匹配或超越基线精度。

## 研究背景与动机

**领域现状**：变分量子本征求解器（VQE）是利用量子计算机进行电子结构计算的旗舰协议，通过迭代调整参数化量子线路来最小化分子哈密顿量的期望值。

**现有痛点**：(a) no-cloning定理要求每个参数每步梯度需要两次电路评估（parameter-shift rule），测量开销随参数数线性增长；(b) 更深的线路面临barren plateaus（BP），梯度指数级衰减；(c) 苯分子（C₆H₆）单个VQE需10⁶-10⁸次电路评估（离子阱量子计算机约55小时）。

**核心矛盾**：需要足够深的线路保证表达能力，但深度增加导致参数数量爆炸和BP问题。

**本文目标** 从参数空间压缩的角度降低VQE的测量开销——发现并冻结那些在训练中始终不活跃的参数。

**切入角度**：实证发现VQE存在"冻结参数"现象——初始化决定了参数的长期活跃性，一部分参数在整个训练过程中始终不重要。

**核心 idea**：用深度学习模型（ResNet-18+自注意力）从哈密顿量和ansatz结构预测哪些参数可以被冻结，在训练前即裁剪参数空间。

## 方法详解

### 整体框架
三阶段：(1) APFA算法收集VQE训练轨迹中的参数冻结模式作为训练标签；(2) 用CFCSA编码将任意(H, A(L,N,D))对映射为统一的输入张量，训练ResNet-18+MHSA预测冻结强度；(3) 推断时对新的VQE实例预测冻结参数，直接裁剪参数空间后再优化。

### 关键设计

1. **APFA（Adaptive Parameter Freezing and Activation）算法**：

    - 功能：记录VQE训练中每个参数的时序冻结/激活状态
    - 核心思路：维护每个参数的梯度EMA $\hat{g}_t^{(i)}$；自适应冻结/激活阈值 $\tau_f^{(t)}, \tau_a^{(t)}$ 由梯度衰减比 $r_t$ 动态调节；连续低于冻结阈值$N_f$次即冻结，连续高于激活阈值$N_a$次即重激活
    - 设计动机：首次量化VQE的参数冻结现象，提供时序mask轨迹作为深度学习训练标签

2. **增强高斯初始化（Theorem 1）**：

    - 功能：保证数据集中的VQE轨迹避免barren plateaus
    - 核心思路：证明当使用HEA ansatz且参数从 $\mathcal{N}(0, c/L)$ 采样时，梯度方差下界为 $\Theta(1/L)$（仅随深度多项式衰减而非指数衰减）
    - 设计动机：BP下APFA无法提取有意义的冻结模式；该定理保证训练数据的信息质量

3. **CFCSA编码（Coordinate-aware FC Self-Attention）**：

    - 功能：将任意规模的VQE实例编码为统一输入格式
    - 核心思路：每个参数关联3D坐标 $[\ell/(L{-}1), d/(D{-}1), q/(N{-}1)]$（层/门/量子比特）+ K维描述符（耦合常数等），形成 $\mathbf{X} \in \mathbb{R}^{(3+K) \times L \times (DN)}$ 张量；全卷积+MHSA架构对维度不敏感
    - 设计动机：一个训练好的模型可处理任意(L,N,D)组合，无需重训练

### 损失函数 / 训练策略
- Frobenius范数最小化：$\min_\omega \frac{1}{N}\sum_i \|\hat{\mathbf{C}}_i - \mathbf{C}_i\|_F^2$
- ResNet-18 + 通道维MHSA，Adam优化器

## 实验关键数据

### 主实验 — Heisenberg模型 (HEA, 5-15量子比特, 5-10层)

| 指标 | 结果 |
|------|------|
| $\Delta E \leq 0$ 比例 | >90% 的(层,量子比特)配置 |
| 冻结参数比例 (τ=80) | 约40-60% |
| 收敛加速 | 最高3× |
| 电路评估减少 | 40-60% |

### 消融实验 — 不同ansatz泛化

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| TITAN on HEA | $\Delta E \leq 0$ | 大多数配置匹配或超越基线 |
| TITAN on SU2 | $\Delta E \leq 0$ | 泛化到未见过的ansatz |
| TITAN on SEL | $\Delta E \leq 0$ | 泛化到未见过的ansatz |
| Random Freeze | 显著能量损失 (0.3-1.2 Ha) | 随机冻结远不如TITAN |

### 关键发现
- **冻结参数现象普遍存在**：从5量子比特到15量子比特、不同ansatz和哈密顿量中均观察到，初始化决定了长期冗余模式
- **冻结模式有规律**：小电路(5×5)中前两层几乎全部冻结；大电路(5×10)中活跃区域集中在输出端
- **随机冻结完全无效**：证明TITAN的收益来自有针对性的裁剪而非简单减少参数
- **跨ansatz泛化**：在HEA上训练的TITAN成功泛化到SU2和SEL ansatz

## 亮点与洞察
- **深度学习辅助量子计算的新范式**：用经典神经网络预测量子线路的冗余模式，将VQE的参数空间压缩问题转化为监督学习问题
- **CFCSA的通用编码设计**：坐标+描述符编码使单一模型适用于任意规模的VQE实例，极具工程价值
- **互补性强**：TITAN可与现有的测量分组、架构搜索、优化器改进叠加使用

## 局限与展望
- 理论保证（Theorem 1）仅适用于HEA类ansatz，对UCC等化学启发ansatz未证明
- 数据生成本身需要运行大量VQE轨迹，初始投资不可忽略
- 30量子比特仍远低于实际化学应用（需100+量子比特）
- 真实量子硬件的噪声对冻结模式的影响未验证

## 相关工作与启发
- **vs QuACK**: QuACK优化VQE的优化器层面，TITAN优化参数空间层面，两者互补
- **vs 量子架构搜索 (QAS)**: QAS搜索最优线路结构，TITAN在给定结构内裁剪参数，互补方向
- **vs 经典剪枝**: 类似神经网络剪枝思想但应用于量子线路，是量子-经典交叉的有趣类比

## 评分
- 新颖性: ⭐⭐⭐⭐ 冻结参数现象的发现和DL预测框架新颖
- 实验充分度: ⭐⭐⭐⭐ 多哈密顿量+多ansatz+消融+分子系统，覆盖广
- 写作质量: ⭐⭐⭐⭐ 理论和实验组织清晰
- 价值: ⭐⭐⭐⭐ 为VQE规模化提供了实用的参数压缩工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning](f-adapter_frequency-adaptive_parameter-efficient_fine-tuning_in_scientific_machi.md)
- [\[ICML 2025\] Rethink the Role of Deep Learning towards Large-scale Quantum Systems](../../ICML2025/physics/rethink_the_role_of_deep_learning_towards_large-scale_quantum_systems.md)
- [\[NeurIPS 2025\] Simulation-Based Inference for Neutrino Interaction Model Parameter Tuning](simulation-based_inference_for_neutrino_interaction_model_parameter_tuning.md)
- [\[ICML 2025\] Erwin: A Tree-based Hierarchical Transformer for Large-scale Physical Systems](../../ICML2025/physics/erwin_a_tree-based_hierarchical_transformer_for_large-scale_physical_systems.md)
- [\[NeurIPS 2025\] FEAT: Free Energy Estimators with Adaptive Transport](feat_free_energy_estimators_with_adaptive_transport.md)

</div>

<!-- RELATED:END -->
