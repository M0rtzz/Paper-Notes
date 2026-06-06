---
title: >-
  [论文解读] Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting
description: >-
  [NeurIPS 2025 (Spotlight)][时间序列][时空预测] 提出 ST-TTC，一种轻量级测试时计算范式，通过频域相位-幅值校准器和闪电梯度更新机制，在推理阶段实时修正时空预测中的周期性偏差，无需修改骨干网络即可一致性提升多种模型性能。
tags:
  - "NeurIPS 2025 (Spotlight)"
  - "时间序列"
  - "时空预测"
  - "测试时计算"
  - "频域校准"
  - "非平稳性"
  - "在线适应"
---

# Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting

**会议**: NeurIPS 2025 (Spotlight)  
**arXiv**: [2506.00635](https://arxiv.org/abs/2506.00635)  
**代码**: [https://github.com/Onedean/ST-TTC](https://github.com/Onedean/ST-TTC) (开源)  
**领域**: 时间序列  
**关键词**: 时空预测, 测试时计算, 频域校准, 非平稳性, 在线适应

## 一句话总结

提出 ST-TTC，一种轻量级测试时计算范式，通过频域相位-幅值校准器和闪电梯度更新机制，在推理阶段实时修正时空预测中的周期性偏差，无需修改骨干网络即可一致性提升多种模型性能。

## 研究背景与动机

**领域现状**: 时空预测（交通流、气象、能源等）已有大量深度网络方案（Transformer、GNN、MLP 等），在标准基准上取得了不错效果。

**现有痛点**: 实际部署中，传感器数据常受噪声、异常值和**非平稳分布漂移**（如季节性变化、传感器老化）影响，导致训练好的模型在测试时性能下降。

**核心矛盾**: 现有解决方案主要分三类：(a) OOD 学习——在训练阶段增强鲁棒性，但假设训练数据覆盖所有未来分布不变性，实际很难成立；(b) 持续微调——将目标域分为多个周期反复训练，数据稀缺场景失效；(c) 测试时训练（TTT）——需要额外自监督任务和架构修改，计算开销大。

**本文目标**: 如何在推理阶段以极低开销利用测试时信息纠正预测偏差？时空预测有一个独特优势——**标签自相关性**：滑动窗口内的历史样本在推窗后自然获得真实标签，可以用来做实时校准。

**切入角度**: 从频域（spectral domain）入手，将非平稳导致的周期性偏差分解为振幅和相位的变化，用极少参数做节点级别的频谱校准。

**核心 idea**: 冻结骨干网络，在测试时仅训练一个频域幅值-相位校准器，利用流式历史样本做单步梯度更新，实现轻量、即插即用的测试时计算。

## 方法详解

### 整体框架

ST-TTC 包含两个协同组件：
- **SD-Calibrator**（频域校准器）：回答"计算什么"——对骨干网络的时域预测做频域校准
- **Flash Gradient Update + Streaming Memory Queue**：回答"怎么计算"——高效利用历史测试信息更新校准器参数

### 关键设计

#### 1. 频域校准器（SD-Calibrator）

**功能**: 对骨干网络预测 $\hat{y} \in \mathbb{R}^{B \times N \times T}$ 进行频域幅值和相位修正。

**核心思路**:

- **空间感知分解**: 对每个空间节点独立做 rFFT，得到频谱 $Y_f = \text{rFFT}(\hat{y}) \in \mathbb{C}^{B \times N \times M}$，其中 $M = T/2 + 1$，分解为振幅 $A = |Y_f|$ 和相位 $P = \angle Y_f$
- **分组调制**: 将 $M$ 个频率bin 分为 $G$ 组（默认 $G=4$），学习逐组逐节点的偏移参数 $\lambda^\alpha \in \mathbb{R}^{G \times N \times 1}$, $\lambda^\phi \in \mathbb{R}^{G \times N \times 1}$（均初始化为0）
- **校准公式**: $A'_g = A_g \odot (1 + \lambda_g^\alpha)$, $P'_g = P_g + \lambda_g^\phi$
- **重建**: $Y'_f = \bigcup_{g=1}^{G} A'_g \odot e^{jP'_g}$, 再 irFFT 回时域得到 $\hat{y}_{cal}$

**设计动机**: 时域校准需要大量参数且容易过拟合噪声；频域中周期性变化表现为特定频率分量的幅值和相位变化，更直接、更鲁棒。分组设计将参数量从 $2NM$（全谱）降为 $2NG$（$G \ll M$），大幅降低内存和梯度更新开销。

#### 2. 闪电梯度更新 + 流式记忆队列

**功能**: 高效地在推理时更新 SD-Calibrator 参数。

**核心思路**:

- **流式记忆队列**: 维护 FIFO 队列 $\mathcal{Q}$，最大容量为预测时域长度 $T_f$。每收到新测试样本 $(X_t, Y_t)$ 入队，队满时弹出最老样本 $(X_o, Y_o)$ 用于梯度更新——**避免信息泄露**
- **闪电梯度更新**: 用弹出的 $(X_o, Y_o)$ 做单步梯度下降：$\hat{Y}_o^b = f_\theta(X_o)$（骨干冻结），$\hat{Y}_o^{cal} = g_\theta(\hat{Y}_o^b)$，计算 $\mathcal{L}(\hat{Y}_o^{cal}, Y_o)$，然后 $\lambda \leftarrow \lambda - \eta \nabla_\lambda \mathcal{L}$

**设计动机**: 时空预测要求**时效性**——推理额外延迟必须小于滑动窗口步长。单样本单步梯度下降确保闪电般快速更新。队列延迟 $T_f$ 步使用标签，避免了信息泄露问题。

### 损失函数/训练策略

- 骨干网络权重完全冻结，仅更新 SD-Calibrator 参数（$\lambda^\alpha, \lambda^\phi$）
- 学习率 $lr = 1 \times 10^{-4}$，记忆队列更新样本数 $n=1$，频率组数 $G=4$
- 所有实验共享相同超参数，每个实验重复5次，报告均值±标准差

## 实验关键数据

### 主实验

| 数据集 | 模型 | MAE (无TTC) | MAE (有TTC) | 相对提升 |
|--------|------|------------|------------|---------|
| PEMS-03 | GWNet | 16.73 | 16.42 | ↓1.85% |
| PEMS-04 | STAEformer | 19.48 | 19.33 | ↓0.77% |
| PEMS-07 | STGCN | 24.26 | 23.83 | ↓1.77% |
| PEMS-08 | STAEformer | 14.84 | 14.73 | ↓0.74% |
| KnowAir | GWNet | 17.03 | 16.94 | ↓0.53% |
| UrbanEV | 各模型 | — | — | 提升更显著 |

跨6种骨干（Transformer/Graph/MLP-based）× 6个数据集，ST-TTC **全面一致性提升**。

### 消融实验

| 组件 | PEMS-04 MAE | PEMS-08 MAE |
|------|------------|------------|
| 完整 ST-TTC | 最优 | 最优 |
| 去掉相位校准 | 性能下降 | 性能下降 |
| 去掉幅值校准 | 性能下降 | 性能下降 |
| 时域校准替代 | 不如频域 | 不如频域 |
| 全谱替代分组 | 过拟合 | 过拟合 |

### 关键发现

1. ST-TTC 在**所有**骨干架构和数据集组合上都带来性能提升，即使是当前最优模型也能进一步增强
2. 分布漂移越大的数据集（如 UrbanEV）相对提升更显著
3. 与 DOST、TTT-MAE、TENT 等已有测试时方法相比，ST-TTC 精度更高、推理更快
4. 在大规模基准 LargeST（8600+ 节点）上也有效，参数量仅增加 ~68K（0.02%）
5. 可与OOD学习（STONE）和持续学习（EAC）方法叠加使用，进一步提升

## 亮点与洞察

- **范式创新**: 首次形式化时空预测的测试时计算范式，定义清晰（Table 1 对比极其精彩）
- **频域校准是关键洞察**: 非平稳性主要体现为周期模式的幅值/相位漂移，频域校准比时域更直接更鲁棒
- **极致轻量**: 校准器参数仅 $2NG$（例如 $G=4, N=307$ 时仅 2456 个参数），远小于骨干网络
- **即插即用**: 无需修改骨干网络架构或训练过程，任何预训练模型都可直接使用
- **理论保证**: 提供了 SD-Calibrator 输出扰动的近似上界（Theorem 1），确保校准不会过拟合

## 局限与展望

1. 当前假设周期性偏差是主要性能退化来源，对于完全随机的分布变化可能效果有限
2. 频率组数 $G$ 是固定的超参数，未探索自适应分组策略
3. 单步梯度更新虽然快但更新幅度受限，对于剧烈分布变化可能需要多步
4. 未探索不同节点使用不同学习率的可能性

## 相关工作与启发

- **测试时计算**在 NLP（o1, r1）和 CV（TTT, TENT）中已获成功，本文将其迁移到时空预测领域
- 频域处理思路可推广到其他有周期性模式的预测任务（如周期性设备监控、季节性负载预测）
- FIFO 队列+单步更新的在线学习框架设计简洁有效，值得在其他流式学习场景借鉴

## 评分

⭐⭐⭐⭐⭐

NeurIPS 2025 Spotlight，思路清晰、动机充分、设计简洁、实验全面。频域校准 + 闪电更新的组合非常优雅，解决了一个实际且重要的问题。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Exploring Neural Granger Causality with xLSTMs: Unveiling Temporal Dependencies in Complex Data](exploring_neural_granger_causality_with_xlstms_unveiling_temporal_dependencies_i.md)
- [\[NeurIPS 2025\] StRap: Spatio-Temporal Pattern Retrieval for Out-of-Distribution Generalization](strap_spatio-temporal_pattern_retrieval_for_out-of-distribution_generalization.md)
- [\[NeurIPS 2025\] Probability Calibration for Precipitation Nowcasting](probability_calibration_for_precipitation_nowcasting.md)
- [\[ACL 2026\] Test of Time: Rethinking Temporal Signal of Benchmark Contamination](../../ACL2026/time_series/test_of_time_rethinking_temporal_signal_of_benchmark_contamination.md)
- [\[NeurIPS 2025\] SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series](syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)

</div>

<!-- RELATED:END -->
