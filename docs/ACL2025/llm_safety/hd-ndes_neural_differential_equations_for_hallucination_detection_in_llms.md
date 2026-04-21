---
title: >-
  [论文解读] HD-NDEs: Neural Differential Equations for Hallucination Detection in LLMs
description: >-
  [ACL 2025][幻觉检测] 本文首次将神经微分方程（Neural DEs）应用于LLM幻觉检测，通过对隐空间中token激活的连续轨迹建模来系统评估陈述的真实性，在True-False数据集上AUC-ROC超过SOTA 14%以上。
tags:
  - ACL 2025
  - 幻觉检测
  - Neural ODE
  - Neural CDE
  - Neural SDE
  - 隐状态轨迹
  - 分类器
---

# HD-NDEs: Neural Differential Equations for Hallucination Detection in LLMs

**会议**: ACL 2025  
**arXiv**: [2506.00088](https://arxiv.org/abs/2506.00088)  
**领域**: LLM 幻觉检测 / 神经微分方程  
**关键词**: 幻觉检测, Neural ODE, Neural CDE, Neural SDE, 隐状态轨迹, 分类器  

## 一句话总结

本文首次将神经微分方程（Neural DEs）应用于LLM幻觉检测，通过对隐空间中token激活的连续轨迹建模来系统评估陈述的真实性，在True-False数据集上AUC-ROC超过SOTA 14%以上。

## 研究背景与动机

**幻觉是LLM部署的核心挑战**：LLM生成不准确或非事实陈述的问题一直是实际应用中的重大障碍，可能导致客户流失或法律风险。

**现有检测方法的不足**：
   - **证据-based方法**（检索外部知识验证）：计算密集、耗时，不适合高吞吐场景
   - **Logit-based方法**（如AvgProb、AvgEnt）：通过token级不确定性估计句子级不确定性，但粒度粗
   - **一致性-based方法**（如SelfCheckGPT）：多次生成判断一致性，效率低
   - **分类-based方法**（如SAPLMA）：效率高但仅利用最后一个token的隐状态，当非事实信息出现在序列中前或中间时性能下降

**最后一个token不够用**：通过PCA分析发现，对于同一问题的正确和错误答案，最后几个token的隐状态激活几乎相同（因为结尾token相同），而差异主要体现在序列中间部分。这说明需要利用**整个序列**的隐状态信息。

**Neural DEs的理论契合**：
   - Transformer数学上可解释为微分方程的数值求解器（Lu et al., 2019）
   - Neural DEs在时间序列建模中表现优异，天然适合建模token级隐状态的动态演化
   - 可以将token生成过程视为隐空间中的连续轨迹

## 方法详解

### 整体框架

HD-NDEs的工作流程：

1. **特征提取**：将陈述输入LLM，提取每个token在指定隐藏层的嵌入 $\boldsymbol{x} = (x_0, x_1, ..., x_n) \in \mathbb{R}^{d_x}$
2. **降维投影**：使用PCA将高维嵌入投影到低维空间 $\boldsymbol{y} = (y_0, y_1, ..., y_n) \in \mathbb{R}^{d_y}$
3. **Neural DE求解**：用Neural ODE/CDE/SDE建模隐空间轨迹 $\boldsymbol{z} = (z_0, z_1, ..., z_n)$
4. **分类判断**：从隐状态中提取 $z^*$，通过线性分类器输出幻觉概率 $P(\xi=1|\boldsymbol{x})$

### 三种Neural DE变体

**Neural ODEs**：通过确定性微分方程建模平滑、连续时间的动态：

$$z(t) = z(0) + \int_0^t f(s, z(s); \theta_f) ds$$

初始条件 $z(0) = h(\boldsymbol{y}; \theta_h)$，其中 $f$ 和 $h$ 是可学习的神经网络。使用四阶Runge-Kutta（RK4）求解。

**Neural CDEs**：引入控制信号引导系统演化，解决Neural ODE只由初始条件决定的局限：

$$z(t) = z(0) + \int_0^t f(s, z(s); \theta_f) dY(s)$$

控制路径 $Y(t)$ 采用自然三次样条或Hermite样条对时间序列数据进行插值构建。

**Neural SDEs**：加入随机噪声项来捕获系统中的不确定性：

$$z(t) = z(0) + \int_0^t f(s, z(s); \theta_f) ds + \int_0^t g(s, z(s); \theta_g) dW(s)$$

其中 $\{W_t\}_{t \geq 0}$ 是布朗运动，使用Euler-Maruyama方法求解。

### 分类器设计

从隐状态序列 $\boldsymbol{z}$ 通过函数 $k(\theta_k)$ 提取特征 $z^*$，再经过简单的线性层+sigmoid函数输出幻觉概率。整个分类器参数量极小。

### 反向传播

使用adjoint方法进行梯度计算，以常数内存代价实现从最终状态到初始状态的参数更新。

## 实验关键数据

### 实验设置

- **5个数据集**：Company*, Fact*, City*, Invention*, True-False
- **6个LLM**：LLama-2-7B, LLama-2-13B, Alpaca-13B, Vicuna-13B, Mistral-7B-v0.3, Gemma-2-9B
- **基线方法**：P(True), AvgProb, AvgEnt, EUBHD, SAPLMA, MIND, Probe@Exact
- **评估指标**：AUC-ROC

### 主要结果（AUC-ROC）

**Company数据集**：

| 方法 | LLama-2-7B | LLama-2-13B | Vicuna-13B | Gemma-2-9B |
|------|------------|-------------|------------|------------|
| SAPLMA | 54.0 | 58.2 | 68.2 | 64.8 |
| MIND | 56.4 | 60.3 | 69.8 | 65.9 |
| Neural CDEs | **65.9** | **72.8** | **79.8** | **73.6** |
| Neural SDEs | 73.8 | 78.4 | 72.3 | 72.8 |

**City数据集**：

| 方法 | LLama-2-7B | LLama-2-13B | Vicuna-13B | Gemma-2-9B |
|------|------------|-------------|------------|------------|
| SAPLMA | 60.0 | 69.3 | 64.5 | 64.7 |
| Neural ODEs | 73.0 | 82.3 | 73.2 | 72.4 |
| Neural CDEs | **75.7** | 80.6 | **80.1** | **77.2** |

### 关键数据亮点

- **True-False数据集**：HD-NDEs（Neural CDEs变体）在AUC-ROC上超过SAPLMA等SOTA方法**14%以上**
- Neural CDEs通常表现最好，因为控制信号机制能更好地利用序列中的时间信息
- Neural SDEs在部分数据集上优于Neural CDEs，因为随机项有助于捕获生成过程中的内在不确定性
- 即使是最简单的Neural ODEs也普遍优于所有分类-based基线

### 跨模型一致性

- HD-NDEs在所有6个LLM上均优于基线方法，展现了优秀的跨模型泛化能力
- 模型规模越大（7B→13B），HD-NDEs的提升幅度通常更显著

## 亮点与洞察

1. **理论动机清晰**：通过PCA可视化直接展示了仅用最后token检测幻觉的失败案例，动机令人信服
2. **Neural DEs与Transformer的深层联系**：利用Transformer可类比为ODE求解器的理论，为将Neural DEs用于LLM分析提供了坚实基础
3. **方法简洁高效**：分类器仅是简单线性层，主要学习在Neural DE建模的隐空间中做判断，无需训练大型模型
4. **三种DE变体的互补性**：ODE捕获确定性动态、CDE引入外部控制、SDE建模随机性，覆盖了不同场景需求

## 局限性

1. **需要白盒访问**：必须获取LLM的中间层隐状态，无法用于黑盒API模型（如GPT-4、Claude等）
2. **PCA降维的信息损失**：高维嵌入到低维空间的PCA投影可能丢失重要信息
3. **句子级检测粒度**：仅能判断整个陈述是否为幻觉，无法精确定位哪些token不准确
4. **训练数据需求**：需要为每个LLM分别收集标注数据训练Neural DE参数，跨模型迁移性未验证
5. **计算开销**：Neural DE求解器（尤其RK4和adjoint方法）的计算开销比简单分类器要大

## 相关工作

- **幻觉检测**：SAPLMA (Azaria and Mitchell, 2023) 用最后token隐状态训练分类器；MIND和Probe@Exact改进特征提取
- **Neural DEs**：Chen et al. (2018) 提出Neural ODE用于连续深度网络；Kidger et al. (2020) 提出Neural CDE处理时序数据
- **LLM与动态系统**：Lu et al. (2019) 首次将Transformer与ODE类比

## 评分

⭐⭐⭐⭐ — 新颖性突出，首次将Neural DEs应用于幻觉检测，理论动机清晰、实验改进显著（14%+）。白盒访问的限制是主要瓶颈，但对于开源LLM场景非常有价值。

<!-- RELATED:START -->

## 相关论文

- [Learning Auxiliary Tasks Improves Reference-Free Hallucination Detection in Open-Domain Long-Form Generation](learning_auxiliary_tasks_improves_reference-free_hallucination_detection_in_open.md)
- [Enhancing Hallucination Detection through Noise Injection](../../ICLR2026/llm_safety/enhancing_hallucination_detection_through_noise_injection.md)
- [VeriTrail: Closed-Domain Hallucination Detection with Traceability](../../ICLR2026/llm_safety/veritrail_closed-domain_hallucination_detection_with_traceable_evidence_synthes.md)
- [AGrail: A Lifelong Agent Guardrail with Effective and Adaptive Safety Detection](agrail_a_lifelong_agent_guardrail_with_effective_and_adaptive_safety_detection.md)
- [SEUF: Is Unlearning One Expert Enough for Mixture-of-Experts LLMs?](seuf_is_unlearning_one_expert_enough_for_mixture-of-experts_llms.md)

<!-- RELATED:END -->
