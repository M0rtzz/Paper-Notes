---
title: >-
  [论文解读] CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization
description: >-
  [ICML 2025][LLM安全][后门攻击防御] 提出 CROW（Internal Consistency Regularization），通过对抗扰动 + 层间隐藏状态一致性正则化来消除 LLM 中的后门，仅需 100 条干净样本、单卡 4 分钟微调即可将攻击成功率降至 5% 以下…
tags:
  - "ICML 2025"
  - "LLM安全"
  - "后门攻击防御"
  - "内部一致性正则化"
  - "对抗扰动"
  - "大语言模型安全"
  - "层间一致性"
---

# CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization

**会议**: ICML 2025  
**arXiv**: [2411.12768](https://arxiv.org/abs/2411.12768)  
**代码**: [GitHub](https://github.com/NayMyatMin/CROW)  
**领域**: AI安全  
**关键词**: 后门攻击防御, 内部一致性正则化, 对抗扰动, 大语言模型安全, 层间一致性

## 一句话总结

提出 CROW（Internal Consistency Regularization），通过对抗扰动 + 层间隐藏状态一致性正则化来消除 LLM 中的后门，仅需 100 条干净样本、单卡 4 分钟微调即可将攻击成功率降至 5% 以下，且不需要干净参考模型或触发器先验知识。

## 研究背景与动机

后门攻击通过在训练数据中植入隐藏触发器来操纵 LLM 的输出行为，已成为 LLM 安全的重大威胁。现有防御方法存在三个根本性缺陷：

**为分类任务设计**：传统后门防御（如剪枝、量化）主要面向视觉/文本分类，无法直接迁移到生成式 LLM

**依赖外部资源**：CleanGen 等方法需要干净参考模型，部署成本高且增加推理开销

**损害生成能力**：简单微调可能强化后门（如 Llama-2-13B 上微调后 ASR 从 94.5% 升至 85.28%），剪枝/量化则显著降低模型实用性

作者的核心观察是：**干净模型中隐藏状态在相邻层之间呈平滑过渡，而后门触发器会在早中期层产生剧烈的不一致性**。这一 "层间一致性" 特征为检测和消除后门提供了天然信号。

## 方法详解

### 整体框架

CROW 由两个核心组件构成，交替执行：

1. **对抗扰动生成（Adversarial Perturbation Generation）**：在输入嵌入上施加对抗扰动，模拟后门触发器引起的内部扰动
2. **对抗一致性训练（Adversarial Consistency Training）**：在扰动条件下强制模型维持层间隐藏状态一致性，同时保持语言建模能力

整个流程仅依赖少量干净数据（100 条 Alpaca 样本），无需触发器知识、参考模型或完整训练集。

### 关键设计

**层间一致性损失**：衡量相邻层隐藏状态的 cosine 相似度偏差。对于第 $l$ 层，定义：

$$L_{\text{cons}}^{(l)} = \frac{1}{T}\sum_{t=1}^{T}(1 - \cos(H_t^{(l)}, H_t^{(l-1)}))$$

在所有层上取平均得到整体一致性损失：

$$L_{\text{cons}} = \frac{1}{N-1}\sum_{l=2}^{N} L_{\text{cons}}^{(l)}$$

**FGSM 对抗扰动**：基于一致性损失的梯度生成对抗嵌入：

$$H_{\text{adv}}^{(0)} = H^{(0)} + \epsilon \cdot \text{sign}(\nabla_{H^{(0)}} L_{\text{cons}})$$

其中 $\epsilon$ 为扰动幅度（推荐 0.1）。关键在于这里不是为了攻击模型，而是模拟后门触发器的扰动模式，让模型学会在类似扰动下仍保持内部一致性。

**理论基础**：通过 Lipschitz 常数分析证明，一致性正则化约束每层变换的谱范数接近 1（$\zeta^{(l)} \approx 1$），从而阻止输入扰动在深层传播放大：

$$\|\delta H^{(l)}\|_2 \leq \left(\prod_{k=1}^{l}\zeta^{(k)}\right)\|\delta H^{(0)}\|_2 \approx \|\delta H^{(0)}\|_2$$

### 损失函数 / 训练策略

总损失为标准语言建模损失与对抗一致性损失的加权组合：

$$L_{\text{total}} = \mathcal{L}_{\text{LM}} + \alpha \cdot L_{\text{cons}}^{\text{adv}}$$

**训练超参数**：

- 干净样本数：100 条（Alpaca 数据集）
- 优化器：LoRA，学习率 $1 \times 10^{-3}$
- 训练轮次：5 epoch
- $\epsilon = 0.1$（扰动幅度）
- $\alpha$：情感引导和代码注入任务使用 5.5；目标拒绝任务使用 11（因拒绝偏差更强）
- 精度：FP16 混合精度
- 硬件：单张 A100 GPU，耗时 < 4 分钟

**算法流程**：每个 mini-batch 中，先前向传播计算一致性损失，再用 FGSM 生成对抗嵌入，用对抗嵌入再次前向传播计算扰动一致性损失，最终组合语言建模损失和扰动一致性损失进行梯度更新。

## 实验关键数据

### 主实验

实验覆盖 5 个 LLM（Llama-2-7B/13B, CodeLlama-7B/13B, Mistral-7B）、6 种攻击（BadNets, VPI, Sleeper, MTBA, CTBA, BadNets-CI）和 3 类任务（情感引导、目标拒绝、代码注入）。

| 模型 / 任务 | 攻击 | 无防御 ASR↓ | 微调 | 剪枝 | 量化 | CROW |
|---|---|---|---|---|---|---|
| Llama-2-7B / 情感引导 | BadNets | 65.00% | 21.70% | 38.50% | 31.50% | **0.53%** |
| Llama-2-7B / 情感引导 | CTBA | 63.33% | 26.13% | 24.50% | 36.00% | **2.08%** |
| Llama-2-7B / 情感引导 | 平均 | 33.15% | 10.17% | 14.60% | 15.80% | **0.52%** |
| Llama-2-7B / 目标拒绝 | BadNets | 94.50% | 98.45% | 81.68% | 46.07% | **19.63%** |
| Llama-2-13B / 目标拒绝 | BadNets | — | 85.28% | — | — | **2.50%** |
| CodeLlama-7B / 代码注入 | BadNets-CI | 63.41% | 3.07% | 33.02% | 33.33% | **0.87%** |
| CodeLlama-13B / 代码注入 | BadNets-CI | 72.97% | 9.92% | 72.41% | 73.53% | **2.99%** |

**MT-Bench 保持效果**（实用性评估）：

| 模型 / 任务 | 攻击 | 无防御 | 微调 | 剪枝 | 量化 | CROW |
|---|---|---|---|---|---|---|
| Llama-2-7B / 情感引导 | BadNets | 2.72 | **5.35** | 2.51 | 2.33 | 3.80 |
| Llama-2-7B / 情感引导 | 平均 | 2.51 | **5.27** | 2.21 | 2.37 | 3.77 |
| CodeLlama-7B / 代码注入 | BadNets-CI | 3.00 | **4.76** | 2.99 | 2.98 | 3.95 |
| CodeLlama-13B / 代码注入 | BadNets-CI | 3.18 | **4.83** | 3.33 | 3.26 | 4.53 |

### 消融实验

| 配置 | ASR↓ | MT-Bench↑ | 说明 |
|---|---|---|---|
| $\epsilon=0.1, \alpha=5.5$ | 0.87% | 3.95 | 推荐配置，最佳平衡点 |
| $\epsilon=0.3$ | 0.00% | 3.85 | 更强扰动，ASR 为 0 |
| $\epsilon=1.0$ | 0.00% | 3.89 | ASR 增益饱和 |
| $\alpha=0.5$ | 4.35% | 3.93 | 正则化不足 |
| $\alpha=11$ | 0.00% | 3.23 | 过强正则化损害实用性 |
| KL 散度替代余弦 ($\alpha=5.5$) | 0.00% | 3.29 | 过度正则化 |
| KL 散度 ($\alpha=1.0$) | 0.00% | 3.66 | 需降低 $\alpha$ 补偿 |
| Pure Consistency（无对抗扰动） | 情感 1.59% / 拒绝 48.97% | ~4.15 | 拒绝任务防御不足 |
| Embedding-Only 一致性 | 情感 2.56% / 拒绝 98.00% | ~4.13 | 仅嵌入层正则化完全失效于拒绝任务 |

### 关键发现

1. **对抗扰动不可或缺**：去掉对抗扰动的 Pure Consistency 在目标拒绝任务上 ASR 仍高达 48.97%，CROW 降至 19.63%（提升 $\alpha$ 后可降至 3% 以下）
2. **层间 vs 嵌入层正则化**：仅约束嵌入层一致性在拒绝任务上几乎无效（ASR 98%），证明后门扰动会在深层传播放大，必须全层约束
3. **余弦相似度优于 KL 散度**：余弦相似度对 $\alpha$ 更鲁棒、计算更高效、无需概率变换
4. **抗越狱潜力**：对 GCG 越狱攻击，$\alpha=11$ 时 ASR 从 63% 降至 29%，展示了泛化防御能力
5. **计算效率极高**：所有模型微调均在 4 分钟内完成（Llama-2-7B: 2.20 min, 13B: 3.35 min）

## 亮点与洞察

- **核心洞察优雅**：利用 Transformer 层间隐藏状态的一致性作为后门指标，物理直觉清晰——后门是通过过拟合少量毒化样本导致内部表示不一致
- **实用性极强**：仅需 100 条干净样本 + 单卡 4 分钟，且与模型架构无关，适用于 Llama、Mistral、CodeLlama 等不同架构
- **理论与实践统一**：Lipschitz 常数分析严格论证了一致性正则化如何抑制扰动传播，不是纯经验方法
- **防御谱广**：覆盖情感引导、目标拒绝、代码注入三类任务和六种攻击模式，甚至对语义后门（实体触发器）也有效

## 局限与展望

1. **$\alpha$ 需手动调参**：不同任务（情感引导 vs 拒绝）需要不同 $\alpha$ 值，未来可探索自适应调整策略
2. **目标拒绝任务防御偏弱**：默认参数下 BadNets 拒绝任务 ASR 仍为 19.63%，需提升 $\alpha$ 才能降至 3% 以下
3. **未测试自适应攻击**：如果攻击者已知 CROW 的防御机制，可能设计减少层间扰动的新攻击
4. **仅测试数据投毒**：未评估模型替换、权重投毒等其他后门注入方式
5. **MT-Bench 仍低于微调**：虽然 CROW 大幅优于无防御，但 MT-Bench 通常低于纯微调（如 3.80 vs 5.35），存在安全-实用性权衡

## 相关工作与启发

- **CleanGen (Li et al., 2024)**：推理时防御，需干净参考模型逐 token 检测，CROW 是互补的训练时防御
- **TAC (Zheng et al., 2022)**：在 CV 中发现触发器导致通道级 Lipschitz 常数异常，CROW 将此推广到 LLM 层间一致性
- **DoLA (Chuang et al., 2024)**：通过对比不同层输出提升事实性，与 CROW 共享 "层间信息" 的思想
- **对后续研究的启发**：一致性正则化为 LLM 安全提供了新范式——不需要知道攻击细节，仅从模型内部表示的 "健康状态" 出发

## 评分

- 新颖性: ⭐⭐⭐⭐ — 层间一致性视角新颖，但对抗扰动+正则化的组合思路并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ — 5 个模型 × 6 种攻击 × 3 类任务 + 丰富消融 + 越狱实验 + 多种变体对比
- 写作质量: ⭐⭐⭐⭐⭐ — 理论-实验-消融结构完整，图表丰富直观
- 价值: ⭐⭐⭐⭐ — 高度实用（100 样本 + 4 分钟），但 $\alpha$ 调参和自适应攻击鲁棒性仍需改进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Learning Uncertainty from Sequential Internal Dispersion in Large Language Models](../../ACL2026/llm_safety/learning_uncertainty_from_sequential_internal_dispersion_in_large_language_model.md)
- [\[ICML 2025\] Learning Safety Constraints for Large Language Models](learning_safety_constraints_for_large_language_models.md)
- [\[ICML 2025\] Unlocking the Capabilities of Large Vision-Language Models for Generalizable and Explainable Deepfake Detection](unlocking_the_capabilities_of_large_vision-language_models_for_generalizable_and.md)
- [\[ICML 2025\] Watch Out Your Album! On the Inadvertent Privacy Memorization in Multi-Modal Large Language Models](watch_out_your_album_on_the_inadvertent_privacy_memorization_in_multi-modal_larg.md)
- [\[ICML 2025\] De-mark: Watermark Removal in Large Language Models](de-mark_watermark_removal_in_large_language_models.md)

</div>

<!-- RELATED:END -->
