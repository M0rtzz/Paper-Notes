---
title: >-
  [论文解读] TimeBill: Time-Budgeted Inference for Large Language Models
description: >-
  [AAAI 2026][自动驾驶][时间预算推理] 提出TimeBill框架，通过细粒度响应长度预测器（RLP）和工作负载引导的执行时间估计器（ETE），在给定时间预算下自适应调整KV Cache驱逐比例，在保证推理完成率的同时最大化LLM响应质量。
tags:
  - AAAI 2026
  - 自动驾驶
  - 时间预算推理
  - KV Cache驱逐
  - 响应长度预测
  - 执行时间估计
  - 实时系统
---

# TimeBill: Time-Budgeted Inference for Large Language Models

**会议**: AAAI 2026  
**arXiv**: [2512.21859](https://arxiv.org/abs/2512.21859)  
**代码**: 无  
**领域**: 自动驾驶 / LLM推理优化  
**关键词**: 时间预算推理, KV Cache驱逐, 响应长度预测, 执行时间估计, 实时系统

## 一句话总结

提出TimeBill框架，通过细粒度响应长度预测器（RLP）和工作负载引导的执行时间估计器（ETE），在给定时间预算下自适应调整KV Cache驱逐比例，在保证推理完成率的同时最大化LLM响应质量。

## 研究背景与动机

### 问题背景

大语言模型（LLM）正越来越多地部署在时间关键系统中，如机器人、自动驾驶、具身智能和工业自动化。在这些场景下，LLM需要在**硬实时截止时间**内生成准确的响应，否则将被视为系统故障。例如：
- Autoware.Flex 利用LLM将自然语言指令转换为自动驾驶系统可理解的格式
- DriveGPT4 使用LLM感知驾驶环境并生成驾驶决策

### 核心挑战

**执行时间不确定性**：与CNN不同，LLM的自回归生成过程导致端到端执行时间高度不确定，取决于响应长度

**响应长度预测粗糙**：现有预测器（如ProxyModel的5类分类、S3的10类分类）粒度太粗，且BERT基础架构难以处理长输入

**固定KV Cache驱逐比例不灵活**：不同任务有不同的时间预算，固定比例要么导致超时（比例太低），要么严重降低响应质量（比例太高）

### 现有方法的不足

- **离线方法**（量化、剪枝）：部署前压缩模型，无法在运行时根据时间预算调整
- **在线方法**（KV Cache驱逐/量化）：StreamingLLM、SnapKV等使用固定驱逐比例，忽略时间预算约束
- **现有预测器**：BERT基础的预测器受限于上下文长度，无法处理长输入；粗粒度分类无法提供精确的响应时间估计

## 方法详解

### 整体框架

TimeBill框架包含三个核心组件：

1. **细粒度响应长度预测器（RLP）**：基于小语言模型（SLM），预测目标LLM的响应长度
2. **工作负载引导的执行时间估计器（ETE）**：结合FLOPs分析和性能画像，估计端到端执行时间
3. **时间预算高效推理机制**：根据执行时间预测和时间预算，自适应调整KV Cache驱逐比例 $\alpha$

### 关键设计

#### 1. 问题形式化

时间预算LLM推理被建模为约束优化问题：

$$\max_{\theta} \mathcal{M}(\hat{\mathbf{y}}(\theta), \mathbf{y})$$
$$\text{s.t.} \quad t_{\text{e2e}}(x, \theta) \leq T, \quad N \leq N_{\max}$$

其中 $\mathcal{M}(\cdot)$ 是响应性能指标，$T$ 是时间预算，$N_{\max}$ 是最大生成长度。目标是在时间约束内最大化响应质量。

#### 2. 细粒度响应长度预测器（RLP）

**核心思路**：将响应长度预测定义为细粒度分类任务，使用SLM（Qwen2.5-0.5B-Instruct）替代BERT，支持长输入处理。

- **架构**：Embedding层 + $L$个Decoder层（RMSNorm-CausalAttention-RMSNorm-FFN/SwiGLU）+ 分类头
- **桶设计**：将响应长度按固定大小 $B$ 划分为桶，默认512个桶（$B=16$）
- **知识蒸馏对齐**：收集目标LLM的实际响应长度 $N_j$，构建训练数据集 $(x_j, \lceil N_j/B \rceil)$，使RLP与目标LLM对齐

预测后进行后处理，限制最大预测长度：

$$\hat{N} = \min(N_{\max}, \text{Predict}(x) \cdot B)$$

**设计动机**：SLM相比BERT有更长的上下文窗口，能处理长输入；细粒度分类（512类）比粗粒度（5-10类）提供更精确的预测；知识蒸馏确保预测器与目标LLM一致。

#### 3. 工作负载引导的执行时间估计器（ETE）

**核心思路**：结合FLOPs理论建模和性能画像拟合，准确估计执行时间。

**FLOPs建模分析**：
- Prefill阶段：执行时间关于输入长度 $N_x$ 是二次的（CausalAttention的 $Q K^T$ 计算）
- Decoding步骤：执行时间关于KV Cache长度 $N_{kv}$ 是线性的

$$\hat{t}_{\text{prefill}}(x) = aN_x^2 + bN_x + c$$
$$\hat{t}_{\text{decoding}}^i(N_{kv}^i) = pN_{kv}^i + q$$

**性能画像拟合**：通过实际测量不同 $N_x$ 和 $N_{kv}$ 下的执行时间，使用最小二乘法拟合系数 $a, b, c, p, q$。

**KV Cache驱逐对执行时间的影响**：驱逐比例 $\alpha$ 后，第 $i$ 个解码步的KV Cache长度为：

$$N_{kv}^i(x, \alpha) = (1-\alpha)N_x + i - 1$$

引入**悲观因子** $k$（$k \geq 1$）估计最坏情况执行时间（WCET），确保满足硬实时约束。

#### 4. 时间预算高效推理机制

**核心思路**：将原优化问题转化为最小化KV Cache驱逐比例 $\alpha$（因为驱逐比例越大，响应质量越差）。

最优驱逐比例的解析解：

$$\alpha^* = \min\left(\alpha_{\max}, 1 - \frac{T - \hat{t}_{\text{prefill}}(x) - t_{\text{Predict}}(x)}{pN_x(\hat{N}_W - 1)} + \frac{\hat{N}_W - 2}{2pN_x} + \frac{q}{pN_x}\right)$$

**系统部署**：RLP预测可与LLM的Prefill阶段并行执行（在CPU或其他GPU上），如果预测器执行时间小于Prefill时间，则预测开销为零。

### 损失函数 / 训练策略

- RLP使用交叉熵损失训练分类任务
- 使用Arena-Human-Preference-100k数据集构建训练数据，避免在测试集上训练
- ETE通过性能画像数据和最小二乘法拟合，无需训练神经网络
- KV Cache驱逐使用SnapKV实现，$\alpha_{\max}$ 设为95%

## 实验关键数据

### 主实验

实验在Qwen2.5-7B-Instruct上进行，测试数据集为LongBench，硬件为NVIDIA A40 GPU。

| 方法 | 时间预算 | 平均分数（Kill） | 完成率（Kill） | 说明 |
|------|---------|----------------|--------------|------|
| Vanilla | 5-10s | 最低 | 最低 | 经常超时 |
| α=25% | 5-10s | 中等偏低 | 中等 | 驱逐比例不足 |
| α=50% | 5-10s | 中等 | 中等偏高 | 先升后降 |
| α=95% | 5-10s | 中等偏低 | 最高之一 | 驱逐过多，质量差 |
| AWQ | 5-10s | 略优于Vanilla | 略优于Vanilla | 可与TimeBill正交结合 |
| **TimeBill** | 5-10s | **最高** | **与α=95%相当** | 自适应平衡 |

### 消融实验

**响应长度预测器对比**：

| 方法 | 桶数 | MAE↓ | RMSE↓ | R²↑ |
|------|------|------|-------|-----|
| Ours (回归) | - | 64.21 | 103.30 | 0.516 |
| Ours (128桶) | 128 | 48.95 | 87.57 | 0.652 |
| Ours (256桶) | 256 | 44.15 | 78.63 | 0.719 |
| **Ours (512桶)** | **512** | **42.71** | **78.13** | **0.723** |
| ProxyModel | 5 | 105.72 | 136.79 | 0.152 |
| S3 | 10 | 108.96 | 148.91 | -0.004 |

**执行时间估计精度**：
- Prefill阶段MAPE：1.22%
- Decoding步骤MAPE：1.69%

**悲观因子 $k$ 的影响**（T=5s, Kill策略）：
- $k=1\text{-}5$：增大 $k$ → 完成率和平均分数均提升
- $k=6\text{-}8$：$k$ 过大 → $\alpha$ 过大 → 响应质量严重下降，平均分数开始降低

### 关键发现

1. 细粒度分类（512桶）比粗粒度（5/10桶）的预测精度高2.5倍以上
2. 基于SLM的预测器相比BERT基础预测器，MAE降低60%
3. TimeBill在所有时间预算（5-10s）下均获得最高平均响应得分
4. 悲观因子 $k=5$ 是最优选择，符合硬实时系统的常见做法

## 亮点与洞察

1. **问题定义新颖**：首次将LLM推理形式化为时间预算约束优化问题，提供了理论框架
2. **解析解优雅**：通过FLOPs建模和性能画像，得到最优KV Cache驱逐比例的闭合解析解，无需在线搜索
3. **系统设计巧妙**：RLP与Prefill并行执行的设计消除了额外预测开销
4. **实用性强**：框架支持不同推理任务的不同时间预算，且与量化等离线方法正交互补

## 局限与展望

1. 仅在单GPU单请求场景验证，未考虑批量推理和多请求调度
2. RLP需要针对每个目标LLM重新训练，迁移性有限
3. 悲观因子 $k$ 需要手动选择，缺乏自适应调整机制
4. KV Cache驱逐策略固定为SnapKV，未探索与其他驱逐策略的结合
5. 未在真实自动驾驶系统中验证端到端效果

## 相关工作与启发

- 与SnapKV等KV Cache驱逐方法形成互补：TimeBill提供了动态调整驱逐比例的机制
- 为LLM部署到实时系统提供了理论和实践框架
- 启发：可将类似思路扩展到多模型协作场景的时间预算分配

## 评分

- 新颖性: ⭐⭐⭐⭐ — 问题定义新颖，但方法上主要是已有组件的组合
- 实验充分度: ⭐⭐⭐⭐ — 多种基线、多种策略、多种时间预算，较为充分
- 写作质量: ⭐⭐⭐⭐⭐ — 数学推导清晰，系统设计图详尽
- 价值: ⭐⭐⭐⭐ — 对实时LLM部署有较强参考价值

<!-- RELATED:START -->

## 相关论文

- [Embracing Large Language Models in Traffic Flow Forecasting](../../ACL2025/autonomous_driving/embracing_large_language_models_in_traffic_flow_forecasting.md)
- [TawPipe: Topology-Aware Weight Pipeline Parallelism for Accelerating Long-Context Large Models Training](tawpipe_topology-aware_weight_pipeline_parallelism_for_accelerating_long-context.md)
- [Distilling Multi-modal Large Language Models for Autonomous Driving](../../CVPR2025/autonomous_driving/distilling_multi-modal_large_language_models_for_autonomous_driving.md)
- [Navigation Instruction Generation with BEV Perception and Large Language Models](../../ECCV2024/autonomous_driving/navigation_instruction_generation_with_bev_perception_and_large_language_models.md)
- [ST4VLA: Spatially Guided Training for Vision-Language-Action Models](../../ICLR2026/autonomous_driving/st4vla_spatially_guided_training_for_vision-language-action_models.md)

<!-- RELATED:END -->
