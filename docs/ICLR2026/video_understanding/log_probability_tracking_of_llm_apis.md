---
title: >-
  [论文解读] Log Probability Tracking of LLM APIs
description: >-
  [ICLR 2026][视频理解][LLM API监控] 提出 Logprob Tracking (LT) 方法，仅用单token输入和单token输出的log概率即可检测LLM API的微小变更（如单步微调），灵敏度比现有方法高2-3个数量级，成本低1000倍。
tags:
  - "ICLR 2026"
  - "视频理解"
  - "LLM API监控"
  - "log概率"
  - "模型变更检测"
  - "假设检验"
  - "非确定性"
---

# Log Probability Tracking of LLM APIs

**会议**: ICLR 2026  
**arXiv**: [2512.03816](https://arxiv.org/abs/2512.03816)  
**代码**: [有](https://github.com/timothee-chauvin/track-llm-apis)  
**领域**: 视频理解  
**关键词**: LLM API监控, log概率, 模型变更检测, 假设检验, 非确定性

## 一句话总结

提出 Logprob Tracking (LT) 方法，仅用单token输入和单token输出的log概率即可检测LLM API的微小变更（如单步微调），灵敏度比现有方法高2-3个数量级，成本低1000倍。

## 研究背景与动机

LLM API提供商通常提供版本固定的端点，暗示模型保持一致。用户（开发者、研究者、监管者）依赖这种一致性来保证应用可靠性和研究可复现性。然而用户缺乏实际验证一致性的手段。

实际中，提供商可能因多种原因改变模型：
- **性能优化**：更新推理软件/硬件基础设施
- **安全响应**：应对新的越狱攻击或修改模型行为
- **成本节省**：悄悄部署量化版本
- **流量管理**：高峰期切换到更轻量的模型
- **安全事件**：如Grok在2025年经历的三次被篡改系统提示事件

现有变更检测方法（如MET、MMLU基准测试）代价昂贵，需要大量查询和token生成，导致LLM API在实践中几乎不受第三方监控。

核心洞察：虽然log概率在实践中是非确定性的，但通过简单的统计检验，单token的logprob仍然包含足够丰富的分布信息来检测极微小的变更。

## 方法详解

### 整体框架

LT方法的流程极其简洁：
1. 向两个LLM API（同一API不同时间点）发送相同的短提示（单字母"x"）
2. 请求仅返回1个token的输出及其top-k logprobs
3. 重复 $N$ 次采样
4. 对每个token的logprob均值进行排列检验(permutation test)

### 关键设计

#### 1. 处理非确定性

LLM logprobs的非确定性来源有二：
- **有意的非确定性**：温度采样（但LT直接操作logprobs，不受此影响）
- **无意的非确定性**：批处理中其他请求的影响、不同GPU路由导致的数值差异

LT将每个logprob视为从某个分布中采样，用标准假设检验判断两个分布是否相同。

#### 2. 两样本检验

设 $\mathcal{V} = \{t_1, \dots, t_{n_{\text{tok}}}\}$ 为所有观察到的token集合。对每个token计算平均logprob：

$$\bar{a}_i^{(1)} = \frac{1}{N}\sum_{j=1}^{N} T_{j,i}^{(1)}, \quad \bar{a}_i^{(2)} = \frac{1}{N}\sum_{j=1}^{N} T_{j,i}^{(2)}$$

检验统计量为各token均值的绝对距离的平均：

$$S = \frac{1}{n_{\text{tok}}} \sum_{i=1}^{n_{\text{tok}}} |\bar{a}_i^{(1)} - \bar{a}_i^{(2)}|$$

使用排列检验获得p值：将 $2N$ 个样本随机分为两组，计算 $B$ 次排列统计量 $S^{(b)}$，p值为 $\hat{p} = \frac{1}{B}\sum_{b=1}^{B} \mathbf{1}\{S^{(b)} \geq S\}$。

#### 3. 缺失logprob处理

top-k截断导致不同采样中token集合不完全相同。对于某次采样中缺失的token，用该次采样的最小logprob进行保守填充（因为其真实logprob不大于该值）。

#### 4. TinyChange Benchmark

创建包含58种模型变体的基准，涵盖5个修改强度级别：
- **常规微调和LoRA微调**：1到512步单样本微调
- **非结构化权重剪枝**：按幅度或随机，移除比例 $2^{-10}$ 到 $1$
- **参数噪声**：高斯噪声标准差 $\sigma$ 从 $2^{-15}$ 到 $1$

应用于5个开源模型（0.5B-8B参数），共290个变体。

### 损失函数 / 训练策略

LT是纯统计推断方法，无训练过程。核心参数：
- 采样数 $N=10$
- 排列检验次数 $B$ 
- 显著性水平 $\alpha$
- 仅需1-2个token的提示

## 实验关键数据

### 主实验

| 方法 | Overall AUC (95% CI) | 输入token/测试 | 输出token/测试 | 年度成本(GPT-4.1价格) |
|------|:-:|:-:|:-:|:-:|
| MMLU-ALG | 0.878 | $2.1 \times 10^5$ | $9.9 \times 10^3$ | $332 |
| MET | 0.670 | $2.9 \times 10^4$ | $2.0 \times 10^4$ | $146 |
| **LT (Ours)** | **0.915** | **28** | **20** | **$0.14** |

LT不仅AUC最高(0.915)，token成本仅需48个(28输入+20输出)，比MET便宜约1000倍，比MMLU-ALG便宜约2400倍。

| 修改类型 | LT达到AUC>0.9的最高难度 | MET | MMLU-ALG |
|---------|:-:|:-:|:-:|
| 权重剪枝 | $\leq 2^{-10}$ | $2^{-1}$ | $2^{-4}$ |

LT在权重剪枝灵敏度上比MET高 $2^9=512$ 倍，比MMLU-ALG高 $2^6=64$ 倍。

### 消融实验

**提示长度影响**：最短提示(1.5 tokens)和最长提示(33 tokens)之间的AUC差异仅约1%，表明极短提示即可有效检测。

**真实世界部署**：监控189个端点超过4个月，收集170万+响应，检测到37次疑似变更，涉及29个端点和7个提供商。几乎所有检测到的变更(34/37)影响开源权重模型。

### 关键发现

- 短至单字母"x"的提示就足以可靠检测变更
- LoRA微调对所有方法都最难检测
- 开源权重模型同样频繁遭受未公开变更
- 部分提供商（如OpenAI）开始限制最小输出token数（≥16），可能是为了阻碍监控

## 亮点与洞察

1. **极致简约**：1-token输入 + 1-token输出 + 简单统计检验 = 超越复杂方法
2. **信息密度观点**：logprobs比生成的tokens包含更丰富的分布信息，是被严重低估的信号源
3. **实用性极强**：每小时监控一次、一年仅需$0.14的成本，使大规模持续监控成为可能
4. **透明度呼吁**：34/37变更涉及开源模型，揭示开源权重并不等于部署透明

## 局限与展望

- 需要API支持返回logprobs（目前仅约23%的端点支持）
- 无法区分基础设施变更和模型更新的具体类型
- 提供商可能通过缓存logprobs或识别监控查询来规避
- 某些修改（如调整end-of-sequence偏置）可能不影响首token
- 方法侧重于变更检测，不提供变更性质的详细信息

## 相关工作与启发

- 与模型指纹识别(LLM fingerprinting)高度相关但目标不同：LT追求对微小变化的敏感性
- 零知识证明(zkLLM, TOPLOC)提供更强保证但计算开销大得多
- 可与现有审计管线互补：LT作为低成本高灵敏度的第一道防线
- 对AI安全和可复现性研究有直接意义

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — logprob作为监控信号的洞察极具创新
- 技术深度：⭐⭐⭐⭐ — 统计方法简单但有效，理论分析清晰
- 实验充分度：⭐⭐⭐⭐⭐ — TinyChange benchmark + 大规模真实部署验证
- 实用价值：⭐⭐⭐⭐⭐ — 直接可部署，成本极低

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks](nerve_nonlinear_eigenspectrum_dynamics_in_llm_feed-forward_networks.md)
- [\[ICLR 2026\] Stabilizing Policy Gradients for Sample-Efficient Reinforcement Learning in LLM Reasoning](stabilizing_policy_gradients_for_sample-efficient_reinforcement_learning_in_llm_.md)
- [\[ICLR 2026\] Towards Reliable Benchmarking: A Contamination Free, Controllable Evaluation Framework for Multi-step LLM Function Calling](towards_reliable_benchmarking_a_contamination_free_controllable_evaluation_frame.md)
- [\[ACL 2026\] ViLL-E: Video LLM Embeddings for Retrieval](../../ACL2026/video_understanding/vill-e_video_llm_embeddings_for_retrieval.md)
- [\[NeurIPS 2025\] A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers](../../NeurIPS2025/video_understanding/a_little_depth_goes_a_long_way_the_expressive_power_of_logde.md)

</div>

<!-- RELATED:END -->
