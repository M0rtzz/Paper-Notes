---
title: >-
  [论文解读] Mechanistic Interpretability of RNNs Emulating Hidden Markov Models
description: >-
  [NeurIPS 2025][语义分割][RNN可解释性] 训练vanilla RNN复现隐马尔可夫模型（HMM）的发射统计量，然后通过逆向工程揭示RNN利用噪声维持的轨道动力学、"kick neuron"电路和自致随机共振机制来实现离散随机状态转换的计算原理。
tags:
  - "NeurIPS 2025"
  - "语义分割"
  - "RNN可解释性"
  - "隐马尔可夫模型"
  - "噪声驱动动力学"
  - "随机共振"
  - "组合动力学基元"
---

# Mechanistic Interpretability of RNNs Emulating Hidden Markov Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.25674](https://arxiv.org/abs/2510.25674)  
**代码**: [GitHub](https://github.com/EliaTorre/hmmrnn)  
**领域**: 人体理解  
**关键词**: RNN可解释性, 隐马尔可夫模型, 噪声驱动动力学, 随机共振, 组合动力学基元

## 一句话总结

训练vanilla RNN复现隐马尔可夫模型（HMM）的发射统计量，然后通过逆向工程揭示RNN利用噪声维持的轨道动力学、"kick neuron"电路和自致随机共振机制来实现离散随机状态转换的计算原理。

## 研究背景与动机

**领域现状**：RNN在神经科学中被广泛用于建模神经群体动力学、生成关于神经计算的假说。过去的工作主要聚焦于确定性的、输入驱动的简单任务（如运动判别、达到运动等）。

**现有痛点**：自然行为通常表现为离散隐状态之间的随机转换（如HMM所描述），而RNN依赖连续状态空间——两者看似不兼容。对于RNN如何产生自发的、随机的离散行为知之甚少。

**核心矛盾**：HMM用离散状态+随机转换建模行为，RNN用连续轨迹建模动力学——如何桥接？

**本文目标**：RNN能否用连续动力学实现离散随机状态转换？如果能，其内部机制是什么？

**切入角度**：直接训练RNN拟合HMM的输出分布，然后多层次逆向工程。

**核心 idea**：RNN通过噪声维持的闭合轨道动力学实现离散状态，慢速噪声积累+快速kick neuron触发组成"动力学基元"，多个基元组合可模拟复杂HMM结构。

## 方法详解

### 整体框架

1. **训练管线**：RNN接收i.i.d.高斯噪声输入 $x_t \sim \mathcal{N}(0, I_d)$，经ReLU循环更新隐状态，线性投影到3维输出，通过Gumbel-Softmax转换为类别样本。用Sinkhorn divergence（最优传输距离）作为损失函数训练。
2. **逆向工程**：从全局动力学→局部动力学→单神经元/连接→计算原理四个层次分析。

### 关键设计

1. **训练范式（噪声驱动RNN + Sinkhorn Loss）**：

    - **功能**：训练RNN学习HMM的概率性输出行为
    - **为什么**：目标序列是概率性的，传统的逐步监督损失不适用；需要比较分布而非单个样本
    - **怎么做**：使用Sinkhorn divergence——一种正则化的最优传输距离，通过软化的耦合矩阵使优化可微：
    $h_t = \text{ReLU}(h_{t-1}W_{hh}^T + x_t W_{ih}^T), \quad y_t = h_t A^T$
      参数 $\Theta = \{W_{hh}, W_{ih}, A\}$ 通过最小化预测输出与HMM目标序列之间的Sinkhorn divergence优化
    - **区别**：标准RNN训练用确定性损失，本文首次引入最优传输损失训练RNN复现概率行为

2. **HMM架构族**：

    - **线性链HMM**：2-5个状态，带宽对角转移矩阵，系统考察从最大离散到准连续的频谱
    - **全连接HMM**：3状态，每个状态可转移到任意其他状态
    - **循环HMM**：4状态，双向闭环转移结构

3. **全局动力学发现——噪声维持的轨道动力学**：

    - 无输入时，RNN活动收敛到单一不动点（无法实现离散状态切换）
    - 有随机输入时，轨迹沿闭合轨道运动，轨道上出现慢速区域（对应不同输出类）
    - 轨道半径与输入方差线性缩放
    - 训练过程中经历明显的相变：从稳定不动点→不稳定→轨道动力学出现

4. **局部动力学发现——三个功能区域**：

    - **Clusters**（驻留时间>8步）：稳定的慢速区域，对应不同输出概率分布，对噪声敏感
    - **Kick-zones**（驻留时间2-8步）：位于cluster下游，有少量不稳定方向，是状态转换的触发区
    - **Transitions**（驻留时间<2步）：短暂的快速通道，近乎确定性地导向下一个cluster，对噪声不敏感

5. **"Kick Neuron"电路**：

    - 发现两组三联体kick neurons，其预激活值在cluster中强负、在kick-zone近零、在transition中变正
    - 连接结构：组内互相兴奋，组间互相抑制
    - 更大的"噪声积累群体"（~70个神经元）通过结构化连接调制kick neuron的门控开启
    - 因果干预验证：消除kick neuron（$\mu=0$）→轨迹被困在cluster中；增强（$\mu=2$）→轨迹过冲

### 损失函数 / 训练策略

- 损失函数：Sinkhorn divergence（最优传输距离的正则化版本）
- 网络：vanilla RNN，隐状态维度 $|h| \in \{50, 150, 200\}$，输入维度 $d \in \{1, 10, 100, 200\}$
- Gumbel-Softmax温度 $\tau = 1$
- 主要分析聚焦于 $|h|=150, d=100$ 配置

## 实验关键数据

### 主实验

**RNN复现HMM发射统计量**：

| 指标 | 描述 | 结果 |
|------|------|------|
| Sinkhorn距离 | 全局重建误差 | 收敛到接近0 |
| 转移矩阵误差 | 3×3发射转移矩阵 | 与目标HMM高度吻合 |
| 边缘频率 | 稳态分布 | 精确匹配 |
| 输出易变性 | 输出变化比例 | 与目标一致 |

所有四种指标在所有HMM架构（线性链/全连接/循环）上均验证成功。

### 消融实验

**因果干预实验（Kick Neuron验证）**：

| 干预类型 | 调制因子$\mu$ | 效果 | 关键特征指标变化 |
|----------|-------------|------|-----------------|
| 消融kick neuron | $\mu=0$ | 轨迹困在当前cluster，无法转换 | 临界特征值对消失 |
| 消融噪声积累群体 | $\mu=0$ | 同上，验证因果一致性 | 同上 |
| 增强kick neuron | $\mu=2$ | 轨迹过冲到目标cluster之外 | 临界特征值对保持 |
| 控制组（非积累群体） | $\mu=0$ | cluster切换正常，保持kick neuron噪声驱动 | 无显著变化 |

### 关键发现

- RNN通过单一不动点+噪声驱动的闭合轨道实现离散状态表示，而非预期的多稳态吸引子
- 训练过程中发生明显的相变（二次下降），对应不稳定特征值出现和轨道动力学涌现
- **组合动力学基元**：相同的"慢噪声积累+快kick触发"基元可组合产生复杂的离散隐结构（全连接/循环HMM可分解为多个线性链基元的组合）
- 该机制构成自致随机共振（SISR）：无需外部周期信号，噪声自身驱动准周期振荡

## 亮点与洞察

- **桥接离散与连续**：优雅地展示了连续动力系统如何实现离散随机转换，弥合了RNN和HMM之间的概念鸿沟
- **多层次逆向工程**：从群体动力学到单神经元连接的完整因果链条，机制可解释性做到了标杆水平
- **组合基元思想**：一个"动力学基元"被模块化复用来构建复杂结构，类似于编程中的函数复用
- **生物学启示**：大脑皮层本身具有内在噪声（随机离子通道、概率性突触传递），本文机制与随机共振的生物学现象高度呼应
- **训练方法创新**：Sinkhorn divergence用于训练概率性RNN行为的方法可推广到其他随机过程建模

## 局限与展望

- 仅使用vanilla RNN（ReLU激活），未验证GRU/LSTM等更复杂架构是否发展出相同机制
- HMM规模有限（2-5个状态），扩展到数十/数百状态的HMM时机制是否仍然成立？
- 未在真实神经数据或行为数据上验证——当前仅在合成HMM输出上进行
- Gumbel-Softmax的温度参数固定为1，不同温度对学习结果的影响未探索
- 高维输入更利于收敛，但其物理意义（对应神经回路中的什么？）尚不清楚

## 相关工作与启发

- **RNN固定点分析**（Sussillo, Barak等）：通过线性化分析固定点拓扑结构理解RNN计算，本文扩展到噪声驱动的非平衡态分析
- **低秩RNN**（Mastrogiuseppe, Barak等）：低秩连接可揭示连接与动力学的关系，本文发现kick neuron天然形成低秩结构化连接
- **多任务RNN**（Driscoll等）：RNN在多任务中复用共享动力学基元，与本文的组合基元发现相呼应
- **启发**：该方法框架（训练模拟→逆向工程→因果干预）可推广到其他计算模型的机制解释

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 开创性地训练RNN模拟HMM并发现噪声驱动轨道动力学和组合基元
- 实验充分度: ⭐⭐⭐⭐⭐ 多层次分析（全局/局部/单神经元/计算原理）+因果干预验证
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑层层递进，图示精美直观，叙事一气呵成
- 价值: ⭐⭐⭐⭐⭐ 对计算神经科学和可解释AI均有深远启发意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Attention (as Discrete-Time Markov) Chains](attention_as_discrete-time_markov_chains.md)
- [\[NeurIPS 2025\] SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation](sansa_unleashing_the_hidden_semantics_in_sam2_for_few-shot_segmentation.md)
- [\[NeurIPS 2025\] Instant Video Models: Universal Adapters for Stabilizing Image-Based Networks](instant_video_models_universal_adapters_for_stabilizing_image-based_networks.md)
- [\[NeurIPS 2025\] RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing](roma_scaling_up_mamba-based_foundation_models_for_remote_sensing.md)
- [\[NeurIPS 2025\] Mars-Bench: A Benchmark for Evaluating Foundation Models for Mars Science Tasks](mars-bench_a_benchmark_for_evaluating_foundation_models_for_mars_science_tasks.md)

</div>

<!-- RELATED:END -->
