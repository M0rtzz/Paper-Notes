---
title: >-
  [论文解读] TiTok: Transfer Token-level Knowledge via Contrastive Excess to Transplant LoRA
description: >-
  [ICLR 2026][模型压缩][LoRA 迁移] 提出 TiTok 框架，通过 token 级对比超额分数（contrastive excess）实现 LoRA 适配器跨模型高效迁移，无需额外判别器模型，在推理和个性化任务上一致超越 TransLoRA 和知识蒸馏基线。
tags:
  - ICLR 2026
  - 模型压缩
  - LoRA 迁移
  - 知识蒸馏
  - Token 级选择
  - 参数高效微调
  - 对比超额分数
---

# TiTok: Transfer Token-level Knowledge via Contrastive Excess to Transplant LoRA

**会议**: ICLR 2026  
**arXiv**: [2510.04682](https://arxiv.org/abs/2510.04682)  
**代码**: [https://github.com/NaughtyMaltiz16/TiTok](https://github.com/NaughtyMaltiz16/TiTok)  
**领域**: 模型压缩  
**关键词**: LoRA 迁移, 知识蒸馏, Token 级选择, 参数高效微调, 对比超额分数

## 一句话总结

提出 TiTok 框架，通过 token 级对比超额分数（contrastive excess）实现 LoRA 适配器跨模型高效迁移，无需额外判别器模型，在推理和个性化任务上一致超越 TransLoRA 和知识蒸馏基线。

## 研究背景与动机

- **LoRA 的绑定问题**: LoRA 等 PEFT 方法虽然参数高效，但适配器参数依赖于特定基础模型，无法跨模型迁移
- **现有解决方案的局限**:
    - 知识蒸馏（KD）依赖原始训练数据，通常不可用
    - TransLoRA 通过合成数据解决数据依赖，但需要训练额外的判别器模型进行数据过滤，增加了复杂度
- **核心动机**: 能否用更轻量的方式，从 LoRA 中提取 token 级任务知识信号，指导跨模型的知识迁移？

## 方法详解

### 整体框架

TiTok 由三个步骤组成：
1. 合成数据生成 → 2. 超额分数计算 → 3. 带过滤的目标模型训练

### 关键设计 1: Token 级对比超额分数

定义源模型有无 LoRA 时的 token 级分数差异：

$$S(y_i) = L_e(y_i) - L_a(y_i)$$

其中：

$$L_a(y_i) = \log P_{\mathcal{M}_s}(y_i \mid \mathbf{q}, \mathbf{y}_{<i}), \quad L_e(y_i) = \log P_{\mathcal{M}_s + \mathcal{A}_s}(y_i \mid \mathbf{q}, \mathbf{y}_{<i})$$

- **直觉**: 超额分数衡量 LoRA 适配器注入的任务知识量。当基础模型对某 token 不确定但 LoRA 增强后高置信度预测时，该 token 获得高超额分数
- **理论基础**: 等价于 token 级对数似然比（LLR），由 Neyman-Pearson 引理保证其为区分两模型分布的最优统计量

### 关键设计 2: 两级过滤训练

**第一阶段 — 样本过滤**: 计算每个合成样本的平均超额分数，保留 top-$M$ 个高信息量样本：

$$\bar{S}_j = \frac{1}{|\mathbf{y}_j|} \sum_{y_i \in \mathbf{y}_j} S(y_i)$$

**第二阶段 — Token 选择**: 在保留样本内，仅选择 top-$k\%$ 超额分数的 token 用于训练：

$$\mathcal{L}_{\text{TiTok}} = \sum_{(\mathbf{q}_j, \mathbf{y}_j) \in \mathcal{D}_f} \sum_{y_i \in \mathbf{y}_j} I_{k\%}(y_i) \cdot L_t(y_i)$$

### 关键设计 3: Tokenizer 对齐算法

当源模型和目标模型使用不同 tokenizer 时：
- 使用双指针递增解码匹配文本 span
- 四种规则传播 mask：一对一直接复制、一对多复制、多对一平均、多对多平均复制
- 最后 top-$k\%$ 选择保留最可信目标 token

### 损失函数

目标模型 LoRA $\mathcal{A}_t$ 在冻结骨干 $\mathcal{M}_t$ 上，使用过滤后的合成数据以标准 NLL 损失训练：

$$\mathcal{L}_{\text{TiTok}} = \sum \sum I_{k\%}(y_i) \cdot (-\log P_{\mathcal{M}_t + \mathcal{A}_t}(y_i \mid \mathbf{q}, \mathbf{y}_{<i}))$$

## 实验

### 主实验：四种迁移设置

| 迁移设置 | 方法 | BBH Acc | MMLU Acc | News R-1 | Scholarly R-1 |
|----------|------|---------|----------|----------|--------------|
| Mistral→Mistral | Vanilla | 0.397 | 0.557 | 0.117 | 0.381 |
| Mistral→Mistral | TransLoRA | 0.416 | 0.534 | 0.156 | 0.447 |
| Mistral→Mistral | **TiTok** | **0.424** | **0.561** | **0.161** | **0.473** |
| Mistral→Llama3 | Vanilla | 0.469 | 0.469 | 0.125 | 0.444 |
| Mistral→Llama3 | TransLoRA | 0.473 | 0.473 | 0.126 | 0.461 |
| Mistral→Llama3 | **TiTok** | **0.484** | **0.485** | **0.139** | **0.464** |
| Llama2→Llama3 | **TiTok** | **0.488** | **0.477** | **0.138** | **0.461** |

### 消融实验

| 样本过滤 | Token 选择 | BBH | MMLU | News R-1 | Scholarly R-1 |
|----------|-----------|-----|------|----------|---------------|
| ✗ | ✗ | 0.458 | 0.485 | 0.133 | 0.456 |
| ✗ | ✓ | 0.463 | 0.496 | 0.137 | 0.460 |
| ✓ | ✗ | 0.470 | 0.500 | 0.139 | 0.460 |
| ✓ | ✓ | **0.483** | **0.501** | **0.142** | **0.464** |

### 关键发现

- TiTok 平均优于 vanilla 目标模型 +9.94%，优于 KD +8.5%，优于 TransLoRA +4.4%
- 跨模型族（Mistral→Llama）、跨尺度（3B→8B）、跨版本（Llama2→Llama3）均有效
- Top 20% 超额分数 token 包含最集中的任务知识（0.482 vs bottom 0.468）
- 不同模型专家（Mistral 7B 和 Llama2 7B）在 top 20% token 选择上有 59.76% 重合度
- Token 选择比率 $k\%$ = 70% 在大多数设置下最优
- 使用不相关领域的外部数据时 TiTok 仍然有效

## 亮点

- **方法简洁有效**: 不需要训练额外模型（判别器），仅利用源模型自身的有/无 LoRA 差异
- **理论扎实**: 超额分数有对数似然比的统计检验理论支撑
- **全面的迁移场景**: 覆盖同族、跨族、跨尺度、跨版本四种设置
- **Tokenizer 对齐**: 优雅解决不同模型 tokenizer 不匹配问题

## 局限性

- 依赖合成数据质量，合成能力弱的源模型可能限制迁移效果
- Token 选择比率 $k\%$ 在不同迁移设置间不完全一致（Llama3 3B→8B 最优值为 30%）
- 仅在 LoRA（rank=8）上验证，未探索其他 PEFT 方法
- 评估任务主要集中在推理（BBH/MMLU）和个性化（LaMP），其他任务类型待验证

## 相关工作

- **PEFT 迁移**: TransLoRA 通过合成数据+判别器迁移 LoRA，方法更重
- **知识蒸馏**: 传统 KD 在 teacher-student 框架下以 logit/序列级操作，需原始数据
- **选择性 token 训练**: 受 selective training 文献启发，首次将 token 选择扩展到知识迁移场景

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ★★★★☆ |
| 理论深度 | ★★★★☆ |
| 实验充分性 | ★★★★☆ |
| 实用价值 | ★★★★☆ |
| 写作质量 | ★★★★☆ |

<!-- RELATED:START -->

## 相关论文

- [Fuse Before Transfer: Knowledge Fusion for Heterogeneous Distillation](../../ICCV2025/model_compression/fuse_before_transfer_knowledge_fusion_for_heterogeneous_distillation.md)
- [Token Distillation: Attention-Aware Input Embeddings for New Tokens](token_distillation_attention-aware_input_embeddings_for_new_tokens.md)
- [AMiD: Knowledge Distillation for LLMs with α-mixture Assistant Distribution](amid_knowledge_distillation_for_llms_with_α-mixture_assistant_distribution.md)
- [Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)
- [Stress-Testing Alignment Audits with Prompt-Level Strategic Deception](stress-testing_alignment_audits_with_prompt-level_strategic_deception.md)

<!-- RELATED:END -->
