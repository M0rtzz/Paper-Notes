---
title: >-
  [论文解读] Weak-to-Strong Jailbreaking on Large Language Models
description: >-
  [ICML 2025][模型压缩][越狱攻击] 本文提出 weak-to-strong 越狱攻击：利用两个小模型（一个安全、一个不安全）在推理时通过对数概率代数修改大模型的解码分布，仅需一次前向传播即可将对齐大模型的恶意回复率提升至 99% 以上，揭示了 LLM 对齐中一个此前未被注意的高效攻击面。
tags:
  - ICML 2025
  - 模型压缩
  - 越狱攻击
  - 对齐安全
  - 解码分布
  - 对抗攻击
  - 大语言模型
---

# Weak-to-Strong Jailbreaking on Large Language Models

**会议**: ICML 2025  
**arXiv**: [2401.17256](https://arxiv.org/abs/2401.17256)  
**代码**: [https://github.com/XuandongZhao/weak-to-strong](https://github.com/XuandongZhao/weak-to-strong)  
**领域**: AI Safety / LLM 安全  
**关键词**: 越狱攻击, 对齐安全, 解码分布, 对抗攻击, 大语言模型

## 一句话总结
本文提出 weak-to-strong 越狱攻击：利用两个小模型（一个安全、一个不安全）在推理时通过对数概率代数修改大模型的解码分布，仅需一次前向传播即可将对齐大模型的恶意回复率提升至 99% 以上，揭示了 LLM 对齐中一个此前未被注意的高效攻击面。

## 研究背景与动机

**领域现状**：当前 LLM 通过 RLHF、DPO 等方法进行对齐（alignment），使其拒绝有害请求。然而红队测试（red-teaming）持续发现越狱漏洞——通过对抗性提示（adversarial prompts）、微调（fine-tuning）或解码操纵（decoding tricks），已对齐的 LLM 仍可被诱导产生有害内容。

**现有痛点**：
   - 基于提示的越狱（如 GCG、AutoDAN）需要大量优化迭代，计算成本高
   - 基于微调的越狱需要训练数据和 GPU 资源
   - 现有方法大多需要对目标模型的多次查询或参数修改
   - **缺少一种高效、低成本、强效果的推理时攻击方法**

**核心矛盾**：对齐后的模型和越狱后的模型在大部分生成行为上是相似的——它们对绝大多数良性请求的回复几乎一致。对齐本质上只改变了模型在面对敏感请求时的初始解码行为（是拒绝还是回答）。

**本文目标**：揭示一个**高效的推理时攻击向量**——利用小模型的解码分布差异来操纵大模型的行为，同时启发更好的防御机制。

**切入角度**：观察到越狱模型和对齐模型主要在**初始解码分布**上存在差异。这意味着一旦模型被引导开始回答（而非拒绝），后续生成会自然地继续产出内容。因此只需在前几个 token 的解码中注入"去对齐"信号。

**核心 idea**：用两个小模型（safe、unsafe）的对数概率之差，计算出一个"去对齐偏移量"，将其加到大模型的解码分布上。这等价于从大模型的解码分布中减去"安全约束"，使其表现得像未对齐的版本。

## 方法详解

### 整体框架

```
输入: 有害请求 x
       ↓
  ┌────────────────────────────┐
  │  Safe 小模型 → logits_safe  │
  │  Unsafe 小模型 → logits_unsafe │
  │  对齐大模型 → logits_target │
  └────────────────────────────┘
       ↓
  修改后解码: logits_target + β × (logits_unsafe - logits_safe)
       ↓
输出: 有害回复 y
```

### 关键设计

1. **对数概率代数（Log-Probability Algebra）**:

    - **功能**：通过小模型的 logits 差值来修改大模型的解码分布
    - **核心公式**：
    $\log P_{\text{attack}}(y_t | y_{<t}, x) = \log P_{\text{target}}(y_t | y_{<t}, x) + \beta \cdot \left[\log P_{\text{unsafe}}(y_t | y_{<t}, x) - \log P_{\text{safe}}(y_t | y_{<t}, x)\right]$
   其中 $P_{\text{target}}$ 是目标大模型（如 70B aligned），$P_{\text{unsafe}}$ 是不安全小模型（如 7B，经 shadow alignment 训练），$P_{\text{safe}}$ 是安全小模型（如 7B aligned），$\beta$ 是攻击强度超参数
    - **设计动机**：$\log P_{\text{unsafe}} - \log P_{\text{safe}}$ 捕获了"不安全性/去对齐"方向上的解码偏移。将此偏移施加到大模型上，等效于从大模型的解码分布中移除安全对齐的部分。这与对比解码（contrastive decoding）的思想类似，但目的从"提升质量"变为"破坏对齐"。

2. **Weak-to-Strong 范式**:

    - **功能**：用弱（小）模型指导对强（大）模型的攻击
    - **核心思路**：选择两个 7B 级小模型——一个对齐的、一个通过 shadow alignment 获得的不对齐版本。它们之间的 logits 差值代表了"对齐 vs 不对齐"的方向，这个方向在大模型的解码空间中同样有效
    - **设计动机**：
        - **成本极低**：只需额外解码两个小模型各一次（vs GCG 需要数千次迭代）
        - **跨模型迁移**：小模型的"去对齐方向"可以转移到不同架构的大模型上
        - **隐蔽性**：不修改目标模型参数、不需要梯度访问，只改变推理时的采样过程

3. **攻击强度控制（$\beta$ 参数）**:

    - **功能**：调控去对齐偏移的强度
    - **核心思路**：$\beta = 0$ 时无攻击；$\beta$ 增大，攻击强度增加但生成质量可能下降。实验中 $\beta = 1.5$ 为最优平衡点
    - **设计动机**：在攻击成功率（ASR）和生成质量之间取得平衡——过大的 $\beta$ 会导致退化的文本。

### 防御策略

作为初步尝试，作者提出了一种防御方案：
- **对抗性检测**：监控生成 token 序列的困惑度变化——在攻击下，前几个 token 的困惑度分布会偏离正常模式
- **解码分布约束**：在采样时对输出分布施加 KL 散度限制，防止偏离原始对齐模型过远
- **局限**：作者承认设计更强防御仍具挑战性

### 损失函数 / 训练策略

本方法**不需要训练**——完全在推理时进行。唯一需要的准备工作是获得不安全小模型（可通过 Shadow Alignment 方法用少量有害数据微调得到）。

## 实验关键数据

### 主实验：攻击成功率

| 目标模型 | 数据集 | ASR (%) | 基线 ASR (%) | 说明 |
|---------|--------|---------|-------------|------|
| Llama-2-70B-Chat | AdvBench | >99% | <5% | 对齐模型几乎完全被攻破 |
| Llama-2-70B-Chat | HarmBench | >99% | <5% | 另一有害请求基准 |
| Llama-2-13B-Chat | AdvBench | >99% | <5% | 中型模型同样有效 |
| Vicuna-33B | AdvBench | >98% | ~10% | 不同对齐方法也被攻破 |
| Mixtral-8x7B | AdvBench | >97% | <8% | 跨架构迁移 |

### 与其他攻击方法对比

| 方法 | ASR (%) | 计算开销 | 需要梯度? | 需要训练? |
|------|---------|---------|----------|----------|
| GCG | 85% | 高（数千次迭代） | 是 | 否 |
| AutoDAN | 90% | 高 | 是 | 否 |
| Fine-tuning | 95% | 中（需 GPU 训练） | - | 是 |
| **Weak-to-Strong** | **>99%** | **低（1 次前向传播）** | **否** | **否** |

### 消融实验

| 配置 | ASR (%) | 说明 |
|------|---------|------|
| $\beta = 0.5$ | ~70% | 攻击强度不足 |
| $\beta = 1.0$ | ~92% | 次优 |
| $\beta = 1.5$ | >99% | 最优平衡点 |
| $\beta = 2.0$ | >99% | ASR 高但生成质量下降 |
| 只用 unsafe 模型（无 safe 对比） | ~80% | 对比解码很重要 |
| 随机小模型对（非 safe/unsafe 对） | ~15% | 证明对齐差异是关键 |

### 关键发现

1. **初始 token 主导一切**：对齐模型和越狱模型仅在前 5-10 个 token 的解码分布上显著不同。一旦模型开始"回答"而非"拒绝"，后续生成自然继续产出内容
2. **跨模型/跨组织迁移**：7B 小模型的"去对齐方向"可有效迁移到不同架构和组织的 70B+ 大模型
3. **效率优势极大**：与 GCG 等方法相比，计算成本几乎可忽略不计
4. **现有防御策略不足**：作者提出的防御仅部分有效，设计鲁棒防御仍然是开放问题

## 亮点与洞察

- **深刻洞察**：将对齐机制理解为解码分布的局部修改（而非全局改变），揭示了当前对齐方法的根本脆弱性
- **对比解码的"暗面"**：对比解码（contrastive decoding）通常用于提升质量——本文展示了同样的技术可以破坏安全对齐
- **"弱模型指导强模型"的范式**：与 weak-to-strong generalization（Burns et al.）形成有趣呼应，但方向完全相反——用弱模型攻击强模型
- **安全警钟**：攻击成本如此之低（仅需两个 7B 模型 + 一次推理）意味着对齐安全面临的威胁远大于此前预期

## 局限与展望

1. **需要获取不安全小模型**：攻击者需要一个不安全的同架构小模型（可通过 Shadow Alignment 获得，但需要有害数据）
2. **仅改变解码分布**：对于那些在表征层面实现了深度对齐的模型（如 Constitutional AI），此攻击可能效果减弱
3. **生成质量权衡**：高 $\beta$ 值虽提升 ASR 但降低生成流畅性
4. **防御探索不足**：论文承认提出的防御是初步的，更全面的防御方案有待研究
5. **闭源模型不适用**：需要访问 logits，对 API-only 模型（如 GPT-4、Claude）不直接适用

## 相关工作与启发

- **GCG（Zou et al., 2023）**：通过梯度优化对抗后缀实现越狱。但成本高且需梯度访问
- **Contrastive Decoding（Li et al., 2023）**：用大小模型的 logits 差值提升生成质量。本文将此思想"反转"用于攻击
- **Shadow Alignment（Yang et al., 2023）**：用少量有害数据将对齐模型变为不安全模型。本文用此方法制备攻击所需的 unsafe 小模型
- **启发**：
    - 对齐不应仅在输出层"浅层"实现——需要更深层的表征对齐
    - 解码分布的对比分析可以作为检测越狱的新维度
    - 对齐方法需要考虑推理时攻击的鲁棒性，而不仅是训练时对抗

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 揭示了全新的高效攻击向量，weak-to-strong 范式令人惊讶
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个模型、3 个组织、2 个数据集、完整消融、对比防御
- 写作质量: ⭐⭐⭐⭐ 洞察解释清晰，流水线简洁明了
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 安全社区有重大警示价值，推动更强防御研究

<!-- RELATED:START -->

## 相关论文

- [Persistent Topological Features in Large Language Models](persistent_topological_features_in_large_language_models.md)
- [DLP: Dynamic Layerwise Pruning in Large Language Models](dlp_dynamic_layerwise_pruning_in_large_language_models.md)
- [Synergy between the Strong and the Weak: Spiking Neural Networks Are Inherently Superior in Temporal Processing](../../NeurIPS2025/model_compression/synergy_between_the_strong_and_the_weak_spiking_neural_networks_are_inherently_s.md)
- [From Language Models over Tokens to Language Models over Characters](from_language_models_over_tokens_to_language_models_over_characters.md)
- [Instruction-Following Pruning for Large Language Models](instruction-following_pruning_for_large_language_models.md)

<!-- RELATED:END -->
