---
title: >-
  [论文解读] STAR: Similarity-guided Teacher-Assisted Refinement for Super-Tiny Function Calling Models
description: >-
  [ICLR 2026][模型压缩][知识蒸馏] 提出 STAR 框架，通过约束知识蒸馏（CKD）和相似度引导的强化学习（Sim-RL）协同工作，将大模型的 function calling 能力有效迁移到 0.6B 级别的超小模型，在 BFCL 和 ACEBench 上大幅超越基线。
tags:
  - ICLR 2026
  - 模型压缩
  - 知识蒸馏
  - 强化学习
  - Function Calling
  - 超小模型
  - 相似度奖励
---

# STAR: Similarity-guided Teacher-Assisted Refinement for Super-Tiny Function Calling Models

**会议**: ICLR 2026  
**arXiv**: [2602.03022](https://arxiv.org/abs/2602.03022)  
**代码**: [github.com/Qwen-Applications/STAR](https://github.com/Qwen-Applications/STAR)  
**领域**: 模型压缩 / 知识蒸馏  
**关键词**: 知识蒸馏, 强化学习, Function Calling, 超小模型, 相似度奖励

## 一句话总结

提出 STAR 框架，通过约束知识蒸馏（CKD）和相似度引导的强化学习（Sim-RL）协同工作，将大模型的 function calling 能力有效迁移到 0.6B 级别的超小模型，在 BFCL 和 ACEBench 上大幅超越基线。

## 研究背景与动机

- LLM 的 function calling 能力对 AI Agent 至关重要，但大模型部署受限
- 直接对小模型做 SFT+RL 面临**三重挑战**：
  1. 小模型容易**过拟合** SFT 数据，记忆特定模式而非泛化
  2. 直接 RL 训练**不稳定**
  3. KD+RL 组合引入新问题：top-k 截断下 RKL 导致训练崩溃、二值奖励不适合多解任务、KD 与 RL 的协同难度

## 方法详解

### 整体框架

STAR 训练课程包含两阶段：**模型蒸馏** → **模型精炼**

### CKD: 约束知识蒸馏

#### 发现 1: RKL + top-k 截断导致训练崩溃

- top-k FKL 忽略尾部分布 → 稳定
- top-k RKL 对 $V_k(x)$ 外的 token 施加不稳定的监督 → 灾难性崩溃

#### 发现 2: RKL 的"隐性代价"

- RKL 的模式寻求特性激进裁剪学生尾部分布 → **降低输出熵**
- 低熵 = 弱探索能力 → 下游 RL 性能下降

#### CKD 损失函数

$$\mathcal{L}_{CKD} = \mathcal{L}_{FKL\text{-}k} + \lambda_{tail} \mathcal{L}_{tail}$$

其中：

$$\mathcal{L}_{FKL\text{-}k} = \sum_{x} \sum_{v \in V_k(x)} P_T(v|x) \log \frac{P_T(v|x)}{P_S(v|x)}$$

$$\mathcal{L}_{tail} = \sum_{x} \sum_{v \in V_m(x) \setminus V_k(x)} P_S(v|x)$$

- 仅惩罚学生认为高概率但教师认为不相关的 token → 抑制"自信的错误"
- 不强制长尾分布归零 → 保留 RL 所需的探索能力

### Sim-RL: 相似度引导的强化学习

#### 奖励设计

- **格式奖励** $R_{format}$：二值，检查 `<think>`/`<tool_call>` 标签、JSON 格式、函数名有效性
- **Function Call 奖励** $R_{fc}$：基于 IoU 原则比较预测和真实函数调用序列

$$R_{fc} = \frac{\sum_{i=1}^{\min(m,n)} \text{sim}(p_i, g_{\sigma(i)})}{|P| + |G| - |P \cap G|}$$

参数级相似度函数 $\text{sim}(p,g)$ 对不同类型使用不同度量（字符串用 ROUGE-L，数值用精确匹配）

- **Response 奖励** $R_{response}$：纯文本回复用 ROUGE-L F1
- **总奖励**：$R = (R_{format} - 1) + R_{format} \cdot (R_{fc} + R_{response})$，范围 [-1, 1]

#### 优化方法

使用 GRPO + DAPO 启发的过滤机制：丢弃奖励全 0 或全 1 的同质组。

### STAR 训练课程

1. 用 Sim-RL 先微调教师模型（Qwen3-8B）使其适应蒸馏数据
2. 用 CKD 将教师知识蒸馏到学生模型
3. 用 Sim-RL 精炼学生策略

## 实验关键数据

### BFCLv3 基准（Qwen3-0.6B）

| 方法 | Overall Acc | Non-Live | Live | Multi Turn |
|------|-----------|----------|------|------------|
| Base-model | 47.33 | 71.81 | 65.66 | 1.88 |
| SFT | 44.58 | 66.29 | 62.15 | 1.62 |
| SFT-think | 47.59 | — | — | — |
| FKL | — | — | — | — |
| ToolRL | — | — | — | — |
| **STAR** | **最优** | — | — | — |

### STAR 0.6B 关键成就

| 对比 | BFCL 相对增益 | ACEBench 相对增益 |
|------|-------------|-----------------|
| vs 基线 | +9.2% | >50% |
| vs 所有开源 <1B 模型 | **最优** | **最优** |
| vs 部分更大模型 | 超越 | 超越 |

### 消融实验

| CKD 组件 | 效果 |
|---------|------|
| top-k FKL alone | 稳定但下游 RL 增益有限 |
| top-k RKL/AKL | 训练崩溃 |
| CKD (FKL + tail penalty) | **稳定 + 保留探索 + RL 增益最大** |

### 关键发现

1. CKD 的尾部惩罚在保持训练稳定性的同时保留了足够的探索能力
2. Sim-RL 的连续奖励比二值奖励在多解任务上显著更有效
3. 精心设计的 KD+RL 课程可以让 0.6B 模型超越某些更大的模型
4. 教师先用 Sim-RL 适应数据（teacher correction）对蒸馏质量有正面贡献

## 亮点与洞察

- **诊断深入**：系统分析了 FKL vs RKL 在 top-k 截断下的行为差异和对下游 RL 的影响
- **奖励设计精巧**：参数级的相似度奖励比 AST 解析更灵活，比二值奖励信号更丰富
- **实用性极强**：0.6B 超小模型达到可部署水平的 function calling 性能
- **KD → RL 的衔接设计**：CKD 特别设计了保留探索能力的特性来服务后续 RL

## 局限性

- 仅在 Qwen 系列模型上验证，跨架构泛化性未知
- 奖励设计依赖特定的 function calling 格式（Qwen tool calling template）
- 教师模型质量直接决定蒸馏上限
- 训练流程相对复杂（教师微调 → CKD → Sim-RL 三阶段）

## 相关工作

- 知识蒸馏：GKD、AKL、FKL vs RKL 的讨论
- Function Calling：BFCL 基准、ToolRL
- 小模型训练：LUFFY（混合离线/在线）等

## 评分

- **新颖性**: ⭐⭐⭐⭐ — CKD 的尾部惩罚和 Sim-RL 的细粒度奖励都有技术贡献
- **技术深度**: ⭐⭐⭐⭐ — 对 KL 散度行为的分析深入，梯度层面的理论支撑
- **实验充分性**: ⭐⭐⭐⭐ — 多尺度模型 + 全面消融 + 两个主流基准
- **实用性**: ⭐⭐⭐⭐⭐ — 端侧部署的 0.6B function calling 模型有巨大实用价值

<!-- RELATED:START -->

## 相关论文

- [Unveiling Super Experts in Mixture-of-Experts Large Language Models](unveiling_super_experts_in_mixture-of-experts_large_language_models.md)
- [Reference-Guided Machine Unlearning](reference-guided_machine_unlearning.md)
- [SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)
- [KBVQ-MoE: KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models](kbvq-moe_klt-guided_svd_with_bias-corrected_vector_quantization_for_moe_large_la.md)
- [Distilling Balanced Knowledge from a Biased Teacher](../../CVPR2026/model_compression/distilling_balanced_knowledge_from_a_biased_teacher.md)

<!-- RELATED:END -->
