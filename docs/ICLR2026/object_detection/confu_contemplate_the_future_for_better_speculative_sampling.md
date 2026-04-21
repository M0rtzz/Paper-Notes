---
title: >-
  [论文解读] ConFu: Contemplate the Future for Better Speculative Sampling
description: >-
  [ICLR 2026][目标检测][speculative decoding] 提出 ConFu，在推测解码的 draft model 中引入 contemplate tokens 让其预见 target model 的未来生成方向，结合 MoE 动态机制和锚点采样训练，在 EAGLE-3 基础上提升 8-11% 的接受率和生成速度。
tags:
  - ICLR 2026
  - 目标检测
  - speculative decoding
  - contemplate tokens
  - future prediction
  - MoE
  - draft model
  - EAGLE
---

# ConFu: Contemplate the Future for Better Speculative Sampling

**会议**: ICLR 2026  
**arXiv**: [2603.08899](https://arxiv.org/abs/2603.08899)  
**代码**: 待确认  
**领域**: 目标检测（LLM 推理加速/推测解码）  
**关键词**: speculative decoding, contemplate tokens, future prediction, MoE, draft model, EAGLE

## 一句话总结

提出 ConFu，在推测解码的 draft model 中引入 contemplate tokens 让其预见 target model 的未来生成方向，结合 MoE 动态机制和锚点采样训练，在 EAGLE-3 基础上提升 8-11% 的接受率和生成速度。

## 研究背景与动机

**推测解码范式**：用轻量 draft model 提议候选 token 序列，由 target model 单次前向验证，通过批量接受加速推理。核心指标是 token 接受率和端到端加速比

**EAGLE 系列是当前 SOTA**：EAGLE-1/2/3 逐步改进 draft head 架构（单层 Transformer + target model 隐状态），设置了推测解码的最高基线

**核心问题——误差累积**：现有 draft model 仅基于当前前缀条件生成，随着 draft 步数增加，误差从上游 draft token 传播累积，draft 分布逐渐偏离 target 分布，接受率下降

**关键 insight**：如果 draft model 能获得 target model 当前的"思路方向"——即高层语义意图而非具体 token——就能生成更符合 target 轨迹的候选，减少验证拒绝

**Latent reasoning 启发**：COCONUT 等工作表明 LLM 可生成连续"思考 token"作为中间推理状态，但需多次前向传播代价高。Pause token (Goyal et al.) 可在并行计算中"免费"获得额外计算

## 方法详解

### 核心创新 1: Contemplate Tokens + Soft Prompts

- 在 target model 输入前插入可学习 soft prompt tokens（KV cache 维度），末尾附加 contemplate token
- 注意力掩码限制：仅 contemplate tokens 可 attend to soft prompts，不影响原始前缀表征
- Contemplate token 的隐状态编码 target model 的"中间思想"→作为 future token $\mathbf{f}$ 提供给 draft model
- 验证阶段：在 draft tree 每个节点插入一个 contemplate token，并行验证+生成未来预测。接受后选择对应的 future prediction 传递下一迭代
- 计算开销：验证时处理 $2T$ 个 token（原 $T$ 个 draft node + $T$ 个 contemplate token），$T$ 通常 30-60

### 核心创新 2: MoE 动态 Contemplate Token

- 静态 contemplate embedding 对多样化上下文不足。数学推理需"接下来的等式是"，创意写作需"这段讲的是"
- 用 MoE 参数化 contemplate token embedding：以最新接受 token 的隐状态为输入，线性 router 选择 top-K experts 的加权组合
- [con]（target 端）和 [f]（draft 端）各有独立的 MoE 模块
- 首次在 pause token 设置中引入动态性

### 核心创新 3: 训练框架

- **Anchor Token Sampling**：随机采样 $K_{train}$ 个锚点 token 插入 contemplate token，序列长度从 $2N$ 降到 $N + K_{train}$
- **Future Prediction Replication**：锚点的 future prediction 复用给临近 $l$ 个 token，增强鲁棒性和样本效率
- **损失函数**：KL 散度对齐 target 和 draft 的输出分布，无需额外辅助损失

## 实验关键数据

### 主实验（SpecBench, Llama-3.2-3B, T=0.0, 30 nodes）

| 方法 | 平均接受长度 τ | 加速比 SR |
|------|-------------|----------|
| EAGLE-3 | 4.00 | 1.83× |
| **ConFu** | **4.41** | **2.11×** |
| 相对提升 | **+10.3%** | **+15.3%** |

### 跨温度和预算

| 设置 | EAGLE-3 τ → ConFu τ | 提升 |
|------|---------------------|------|
| T=0.0, 30 nodes | 4.00 → 4.41 | +10.3% |
| T=0.7, 30 nodes | 3.44 → 3.75 | +9.0% |
| T=1.0, 60 nodes | 3.89 → 4.27 | +9.8% |
| 8B 模型平均 | - | +8-11% |

### 关键发现

- 所有任务类型（写作/QA/翻译/代码/数学/摘要）均一致提升
- 不同温度（0.0/0.7/1.0）和预算（30/60 nodes）下稳健有效
- 从 EAGLE-3 checkpoint 初始化+继续训练 EAGLE-3 相同步数无提升→增益来自 ConFu 架构而非更长训练
- 8×H100 训练，单 H100 推理

## 亮点与洞察

- **首次将连续推理 token 与推测解码桥接**：概念上开创了"future-aware draft generation"的新方向
- Contemplate token 利用 pause token 机理实现几乎免费的"思考"——不需要额外前向传播
- MoE 动态 token 在不同上下文下自适应选择"提示指令"——这是一个优雅的设计
- 方法建立在 EAGLE-3 之上，实现了正交改进，与基线架构演进兼容

## 局限与展望

- 仅在 Llama-3 3B/8B 测试，更大模型（70B+）是否有同比提升未知
- Soft prompt tokens 数量（默认 16）和 MoE expert 数量的最优配置未系统研究
- 验证时 $2T$ token 的额外开销在 draft tree 极大时可能不容忽视
- 与不同 target model 架构（非 LLaMA）的兼容性未验证

## 相关工作与启发

- **EAGLE-1/2/3**：逐步改进 draft 架构和训练的最强基线；ConFu 为正交改进
- **BiTA**：用 soft prompt 直接解码 future token；ConFu 用其引导 draft model 而非直接解码
- **COCONUT / Latent Reasoning**：需多步前向传播获取连续思考；ConFu 用 pause token 并行获取
- **Medusa / HASS**：早期推测解码方法，已被 EAGLE 系列超越

## 评分

- 新颖性: ⭐⭐⭐⭐ future prediction + speculative decoding 的首次结合
- 实验充分度: ⭐⭐⭐⭐ 多任务/多温度/多预算的全面评测，控制变量合理
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观，推理链路流畅
- 价值: ⭐⭐⭐⭐ 为推测解码开辟新的改进方向

<!-- RELATED:START -->

## 相关论文

- [Bridge Past and Future: Overcoming Information Asymmetry in Incremental Object Detection](../../ECCV2024/object_detection/bridge_past_and_future_overcoming_information_asymmetry_in_incremental_object_de.md)
- [Portrait4D-v2: Pseudo Multi-View Data Creates Better 4D Head Synthesizer](../../ECCV2024/object_detection/portrait4d-v2_pseudo_multi-view_data_creates_better_4d_head_synthesizer.md)
- [Ground What You See: Hallucination-Resistant MLLMs via Caption Feedback, Diversity-Aware Sampling, and Conflict Regularization](../../AAAI2026/object_detection/ground_what_you_see_hallucination-resistant_mllms_via_caption_feedback_diversity.md)
- [Learn from the Learnt: Source-Free Active Domain Adaptation via Contrastive Sampling and Visual Persistence](../../ECCV2024/object_detection/learn_from_the_learnt_source-free_active_domain_adaptation_via_contrastive_sampl.md)
- [What Layers When: Learning to Skip Compute in LLMs with Residual Gates](what_layers_when_learning_to_skip_compute_in_llms_with_residual_gates.md)

<!-- RELATED:END -->
