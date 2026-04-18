---
title: >-
  [论文解读] Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons
description: >-
  [NeurIPS 2025][LLM对齐][safety alignment] 通过机制可解释性视角发现 LLM 中约 5% 的稀疏"安全神经元"，仅修补（patching）这些神经元的激活即可恢复 90% 以上的安全性能，并从神经元重叠角度解释了 alignment tax 现象。
tags:
  - NeurIPS 2025
  - LLM对齐
  - safety alignment
  - mechanistic interpretability
  - safety neurons
  - activation patching
  - alignment tax
---

# Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons

**会议**: NeurIPS 2025  
**arXiv**: [2406.14144](https://arxiv.org/abs/2406.14144)  
**代码**: [THU-KEG/SafetyNeuron](https://github.com/THU-KEG/SafetyNeuron)  
**领域**: llm_alignment  
**关键词**: safety alignment, mechanistic interpretability, safety neurons, activation patching, alignment tax

## 一句话总结

通过机制可解释性视角发现 LLM 中约 5% 的稀疏"安全神经元"，仅修补（patching）这些神经元的激活即可恢复 90% 以上的安全性能，并从神经元重叠角度解释了 alignment tax 现象。

## 研究背景与动机

大语言模型虽经安全对齐训练显著提升了安全性，但仍然容易受到恶意攻击。理解安全对齐的内在机制对设计更鲁棒的对齐算法至关重要。现有机制可解释性方法（如对 attention head 的归因）主要针对需要提示和少量 token 输出的任务，无法直接应用于需要开放式生成的安全对齐场景。

本文核心问题是：**安全对齐在 LLM 内部到底改变了什么？** 作者选择从最基本的 MLP 神经元层面切入，试图找到负责安全行为的"安全神经元"，并验证其因果效应。

此外，安全对齐领域存在著名的 **alignment tax** 问题：提升安全性会损害模型的有用性，反之亦然。作者希望从神经元视角对此给出机制性解释。

## 方法详解

### 整体框架

提出两阶段框架：先通过关联性（association）缩小候选范围，再验证因果性（causality）。

**核心思路**：关联是因果的必要条件。先找到与安全行为关联的神经元，再用因果分析验证。

### 关键设计一：推理时激活对比（Inference-time Activation Contrasting）

给定两个模型 $\mathcal{M}_1$（SFT 模型）和 $\mathcal{M}_2$（DPO 安全对齐模型），对同一输入分别收集 MLP 中间层神经元的激活值，计算**变化分数（change score）**：

$$\mathcal{S}_i^{(l)}(\mathcal{M}_1, \mathcal{M}_2; \mathcal{D}) = \sqrt{\frac{\sum_{w \in \mathcal{D}} \sum_{j=|w|}^{|\bar{w}^1|-1} \left(a_i^{(l)}(\mathcal{M}_1; \bar{w}^1)[j] - a_i^{(l)}(\mathcal{M}_2; \bar{w}^1)[j]\right)^2}{\sum_{w \in \mathcal{D}} |w^1|}}$$

其中 $a_i^{(l)}$ 是第 $l$ 层第 $i$ 个神经元的激活值。按变化分数降序排列，取 top 的神经元作为安全神经元候选。

**为什么选择 MLP 中间层神经元？** 因为 MLP 中间层（激活函数后、down projection 前）的神经元已被证明编码了多种可解释特征，且 down projection 矩阵的每行可解释为该神经元的"值向量"。

### 关键设计二：动态激活修补（Dynamic Activation Patching）

传统 activation patching 只能处理单步输出。本文提出**动态激活修补**适用于开放式生成场景：

1. 对当前提示 $w$，运行 $\mathcal{M}_2$ 缓存目标神经元激活
2. 运行 $\mathcal{M}_1$，将目标神经元激活替换为缓存值，其余神经元不变
3. 获取下一 token 预测，追加到提示
4. 重复直到生成完成

因果效应定义为：

$$\mathcal{C} = \frac{\mathbb{E}_{w \in \mathcal{D}}[\mathcal{F}(\tilde{w}^1) - \mathcal{F}(\bar{w}^1)]}{\mathbb{E}_{w \in \mathcal{D}}[\mathcal{F}(\bar{w}^2) - \mathcal{F}(\bar{w}^1)]}$$

$\mathcal{C} \approx 1$ 表示这些神经元完全解释了安全能力；$\mathcal{C} \approx 0$ 表示因果效应可忽略。

### 训练策略

- 对齐训练采用 **(IA)³** 作为 PEFT 方法，仅应用于 MLP 层
- (IA)³ 通过乘以缩放因子修改激活，不改变底层参数，从而保留了 MLP 神经元的功能性
- SFT 阶段在 ShareGPT 上训练，安全对齐使用 DPO 在 HH-RLHF-Harmless 上训练

## 实验关键数据

### 主实验：安全神经元的稀疏性与因果效应

在 Llama2-7B、Mistral-7B、Gemma-7B、Qwen2.5-3B 四个模型上验证：

| 模型 | 版本 | BT(↑) | RT(↑) | HB(↑) | JL(↑) | ΔGen |
|------|------|--------|--------|--------|--------|------|
| Llama2 | Base→Ours | **+56** | **+65** | **+63** | **+78** | -0.01 |
| Llama2 | SFT→Ours | **+101** | **+101** | **+76** | **+73** | +0.01 |
| Mistral | Base→Ours | **+71** | **+63** | **+134** | **+103** | -0.04 |
| Mistral | SFT→Ours | **+90** | **+80** | **+74** | **+75** | +0.01 |
| Gemma | SFT→Ours | **+96** | **+84** | **+79** | **+89** | -0.02 |
| Qwen2.5 | SFT→Ours | **+83** | **+81** | **+68** | **+83** | +0.01 |

- 仅修补约 **5%** 的神经元即可恢复 **>90%** 的安全性能
- 两个基线 Pruning 和 SN-Tune 的因果效应极低（通常在 -10 到 +30 范围）
- 通用能力变化 ΔGen 极小（|ΔGen| ≤ 0.05），说明安全增强基本不影响生成质量

### 消融实验：随机神经元 vs 安全神经元

- 随机采样同等数量神经元进行 patching，因果效应显著低于安全神经元
- t-test p 值范围：$1.15 \times 10^{-6}$ 到 $1.67 \times 10^{-18}$，差异高度显著

### Alignment Tax 机制分析

| Patch 方向 | Llama2 安全↓ | Llama2 有用↑ | Mistral 安全↓ | Mistral 有用↑ |
|-----------|------------|------------|-------------|-------------|
| Helpfulness→Safety | 7.3 | 7.97 | 6.6 | 8.1 |
| Safety→Helpfulness | 10.1 | 2.3 | 10.7 | 1.0 |

安全和有用性的偏好神经元 Spearman 相关系数 **>0.95**，而与推理等其他能力的相关性低得多。同一组神经元需要不同激活模式来实现安全和有用性。

### 关键发现

1. 安全神经元在不同随机种子训练下重叠率 >0.95，Spearman 相关系数 >0.95
2. 基于安全神经元激活的分类器可在生成前预测有害输出，平均准确率 76.2%
3. 安全神经元的值向量投影到词表空间后，关联的 top token 并非安全相关内容（如食物、连词、括号），暗示安全机制比简单避免毒性 token 更复杂

## 亮点与洞察

1. **方法论创新**：提出从关联到因果的两阶段框架，特别是**动态激活修补**解决了开放式生成中的因果验证难题，这是对传统 activation patching 的重要扩展
2. **稀疏性发现**：仅 5% 的神经元就承载了 90%+ 的安全机制，这一发现既有理论价值（安全对齐是稀疏的），也有实践价值（可用于高效安全增强）
3. **Alignment Tax 解释**：首次从神经元层面给出机制性解释——安全和有用性共享同一组高度重叠的神经元，但需要不同的激活模式，这类似"资源竞争"
4. **实际应用**：安全防护（safeguard）应用展示了在生成前预测有害输出的可能性，且分类器开销 <0.001 秒

## 局限与展望

1. **对齐方式局限**：仅验证了 (IA)³ + DPO 的设置，全参数微调的安全对齐可能打破神经元功能保持假设
2. **模型规模**：实验集中在 3B~7B 模型，更大规模模型的安全神经元分布可能不同
3. **因果机制深度不足**：发现安全神经元并非编码安全相关 token，但未能揭示其具体工作机制
4. **防护准确率有限**：76.2% 的检测准确率在实际部署中仍不够高
5. **未探索 Attention Head 交互**：MLP 神经元约占 2/3 参数，但与 attention head 的协同作用未深入分析

## 相关工作与启发

- **与 Refusal Direction 的关系**：Arditi et al. (2024) 发现拒绝行为由单一方向调控，本文从神经元粒度提供了更细致的视角
- **与 Lee et al. (2024) 的区别**：后者在 GPT-2 上找到关联毒性 token 的"毒性神经元"，本文发现安全神经元的机制更复杂，关联 token 非安全相关
- **与消融方法的对比**：传统方法（Pruning, SN-Tune）采用消融策略，可能受"九头蛇效应"（Hydra effect）影响——消除某些组件会触发其他组件的补偿行为。本文采用增强（patching）而非消融，更直接地验证因果效应
- **启发**：该框架可推广到其他高层能力（如指令遵循、多语言能力）的机制研究

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 推理时激活对比 + 动态激活修补的两阶段框架是全新提出的，对开放式生成场景的因果分析有方法论贡献
- **实验充分度**: ⭐⭐⭐⭐⭐ — 4 个模型 × 4 个红队基准 × 多个通用基准，含基线对比、消融、稳定性验证、alignment tax 分析、应用展示
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，逻辑连贯，公式与直觉解释平衡良好
- **价值**: ⭐⭐⭐⭐ — 对理解安全对齐机制有重要贡献，alignment tax 解释具有启发性，防护应用展示了实践前景
