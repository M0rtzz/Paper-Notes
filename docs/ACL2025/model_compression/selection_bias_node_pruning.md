---
title: >-
  [论文解读] Mitigating Selection Bias with Node Pruning and Auxiliary Options
description: >-
  [ACL 2025][模型压缩][选择偏差] 提出 Bias Node Pruning (BNP) 和 Auxiliary Option Injection (AOI) 两种互补方法，通过定位并剪除模型输出层中 0.002% 的偏差参数（白盒）与注入"I don't know"辅助选项（黑盒通用），从内外两端同时缓解 LLM 在多选题中的选择偏差，同时提出分布级偏差度量 CKLD，组合方法在 Llama-3 上将 ARC-Challenge 准确率从 52.3% 提升至 65.3%。
tags:
  - ACL 2025
  - 模型压缩
  - 选择偏差
  - 偏差节点剪枝
  - 辅助选项注入
  - 多选题
  - LLM去偏
---

# Mitigating Selection Bias with Node Pruning and Auxiliary Options

**会议**: ACL 2025  
**arXiv**: [2409.18857](https://arxiv.org/abs/2409.18857)  
**机构**: University of Wisconsin-Madison, Amazon
**关键词**: 选择偏差, 偏差节点剪枝, 辅助选项注入, 多选题, LLM去偏

## 一句话总结

提出 Bias Node Pruning (BNP) 和 Auxiliary Option Injection (AOI) 两种互补方法，通过定位并剪除模型输出层中 0.002% 的偏差参数（白盒）与注入"I don't know"辅助选项（黑盒通用），从内外两端同时缓解 LLM 在多选题中的选择偏差，同时提出分布级偏差度量 CKLD，组合方法在 Llama-3 上将 ARC-Challenge 准确率从 52.3% 提升至 65.3%。

## 研究背景与动机

- **核心问题**：LLM 在回答多选题 (MCQ) 时表现出系统性的选择偏差 (selection bias)——倾向于选择特定位置（如最后一个选项）或特定标签（如"A"），与选项内容无关，严重影响准确率和可靠性
- **现有方法局限**：先前工作集中在输入重格式化（如 Split-and-Merge, Li et al. 2023）或输出概率校准（如 PriDe, Zheng et al. 2024; DoLa, Reif & Schwartz 2024），但这些方法仅在模型外部进行修补，忽略了偏差产生的内部机制
- **实际影响**：Figure 2 的投票实验显示，四个 LLM 在所有选项排列上的多数投票准确率均显著高于单次回答准确率，证明选择偏差是跨模型普遍存在的问题
- **关键发现一**：通过分析选项排列下的选择频率分布，发现**错误样本的偏差远大于正确样本**——模型回答错误时，选择分布呈现更尖锐的不均衡（Llama-3 偏好 "D"，Bloomz 偏好 "A"）
- **关键发现二**：通过从模型各层/各 token 位置提取 embedding 并计算正确与错误答案的差向量 L2 范数，发现**选择偏差主要集中在 decoder 最终几层**，尤其是输出投影矩阵与最后层 embedding 的交互处

## 方法详解

### 整体框架

本文提出两种互补的去偏方法和一个新的评估指标：

| 组件 | 类型 | 适用范围 | 核心思想 |
|------|------|----------|----------|
| BNP (Bias Node Pruning) | 参数剪枝 | 白盒模型 | 剪除输出投影矩阵中与偏差向量交互最强的行 |
| AOI (Auxiliary Option Injection) | 输入提示 | 白盒+黑盒 | 添加"I don't know"辅助选项吸收不确定性 |
| CKLD (Choice KL Divergence) | 评估指标 | 通用 | 用 KL 散度衡量预测分布与真实标签分布的偏离 |

### 关键设计

**1. Bias Node Pruning (BNP)——偏差节点剪枝**

核心思路是将有偏 LLM 的输出建模为"无偏模型 + 偏差向量"与输出投影矩阵的乘积：F(x) ≈ (D(x) + b) · W，其中偏差项 b·W 直接导致输出偏移。具体步骤：

- **偏差向量计算**：对每个问题 x，将选项做全排列并输入模型，收集正确排列和错误排列的最终层 embedding，计算差向量 b_x = mean(z₋) − mean(z₊)，然后在 32 个训练样本上取平均得到全局偏差向量 b
- **偏差节点识别**：计算输出投影矩阵 W∈R^{d×|V|} 每一行与偏差向量 b 的交互强度 Σⱼ bᵢ × Wᵢⱼ，取 Top-k 最强交互行作为偏差节点集合 K
- **参数置零**：将 K 中对应行全部置零得到 W̃，此后所有推理均使用 W̃，仅修改 ~0.002% 的模型参数（Llama-3 的 80 亿参数中仅剪 32 个节点）
- **超参数选择**：Llama-3 和 Mistral 剪 32 节点，Bloomz 剪 128 节点，从 {16, 32, 64, 128} 中简单搜索

**2. Auxiliary Option Injection (AOI)——辅助选项注入**

基于"错误样本更易产生偏差"的观察，设计一种让模型主动表达不确定性的机制：

- **选项扩展**：在原始选项集 A 末尾追加一个"I don't know"选项 o_aux
- **答案选择**：根据输出 logit 概率分布，从排除 o_aux 的原始选项中选择概率最高者作为最终答案 â = argmax_{a∈A\o_aux} P(ŷ=a|x_A)
- **黑盒适配**：对无法获取 logit 的黑盒模型，改用生成文本与各选项的 Jaccard 相似度替代概率排序
- **消融实验**：对比"None of the above"和"I know the answer"等替代内容，"I don't know"在多数场景下效果最佳；多个 IDK 选项对 Llama-3 有额外增益但对其他模型无效

**3. Choice KL Divergence (CKLD)——分布级偏差度量**

现有指标（RStd 标准差、RSD 相对标准差）仅衡量各类准确率的变异性，对真实标签分布不均衡（如 A 占 22%、D 占 28%）不敏感，可能产生误导：

- **定义**：CKLD = Σᵢ pᵢ log(pᵢ/qᵢ)，其中 pᵢ 为真实标签中第 i 个选项的比率，qᵢ 为模型预测中第 i 个选项的比率
- **理论保证**：通过 Lagrangian 证明 CKLD 当且仅当 qᵢ = pᵢ（预测分布匹配真实分布）时取最小值 0
- **优势对比**：合成数据实验显示 RSD 在标签不均衡时最小点偏离真实值（始终在 1/k 处），而 CKLD 准确反映偏差最小点

## 实验关键数据

### 主实验结果（BNP + AOI）

在 3 个模型 × 3 个数据集上验证：

| 模型 + 方法 | ARC Acc↑ | ARC CKLD↓ | MMLU Acc↑ | MMLU CKLD↓ | CSQA Acc↑ | CSQA CKLD↓ |
|-------------|----------|-----------|-----------|------------|-----------|------------|
| Llama-3 | 52.3 | 0.494 | 41.8 | 0.589 | 65.4 | 0.095 |
| Llama-3 + BNP | 56.7 | 0.302 | 43.1 | 0.501 | 66.6 | 0.074 |
| Llama-3 + AOI | 60.7 | 0.231 | 47.3 | 0.321 | 67.4 | 0.065 |
| **Llama-3 + BNP+AOI** | **65.3** | **0.124** | **48.3** | **0.288** | **68.1** | **0.049** |
| Bloomz | 43.9 | 0.283 | 28.0 | 0.661 | 58.5 | 0.136 |
| **Bloomz + BNP+AOI** | **48.8** | **0.088** | **32.0** | **0.205** | **64.9** | **0.052** |
| Mistral | 67.4 | 0.040 | 46.4 | 0.186 | 63.6 | 0.042 |
| **Mistral + BNP+AOI** | **69.5** | **0.019** | **48.6** | **0.140** | **66.8** | **0.016** |

### 与现有方法的叠加

BNP+AOI 可与 CoT、ICL、DoLa 等现有方法正交叠加：

| 方法 (Llama-3, ARC) | Acc↑ | CKLD↓ |
|----------------------|------|-------|
| CoT | 66.2 | 0.050 |
| CoT + Ours (BNP+AOI) | **69.2** | **0.024** |
| ICL | 62.2 | 0.169 |
| ICL + Ours (BNP+AOI) | **70.0** | **0.054** |
| DoLa | 51.1 | 0.524 |
| DoLa + Ours (BNP+AOI) | **64.1** | **0.139** |

### 黑盒模型验证

AOI 在无法获取模型参数的黑盒场景同样有效：

| 模型 | ARC Acc | ARC+AOI Acc | CSQA Acc | CSQA+AOI Acc |
|------|---------|-------------|----------|--------------|
| Claude-3-Haiku | 65.3 | **71.4** (+6.1) | 36.4 | **47.0** (+10.6) |
| Claude-3-Sonnet | 86.9 | **87.6** (+0.7) | 71.0 | **73.1** (+2.1) |

### 关键发现

- **BNP 对剪枝节点数不敏感**：从 8 到 128 节点，性能均稳定优于 baseline，但微调可进一步优化
- **偏差向量跨数据集迁移**：用 ARC 数据计算的偏差向量在 CSQA 上降低 CKLD 达 36%，甚至优于 CSQA 自身的偏差向量（22%），说明偏差向量捕获的是模型固有属性
- **BNP 不影响生成质量**：在情感分析和文本摘要任务上，剪 32 节点仅导致微小性能下降（F1: 32.7→31.3），对通用语言能力影响可忽略
- **分布变化可视化**：应用 BNP+AOI 后，模型的选项选择频率分布趋向均匀（接近虚线标注的理想均匀比率）

## 亮点与洞察

1. **从内部机制入手**：首次定位选择偏差在输出投影矩阵参数级的来源，而非仅做外部校准
2. **极简剪枝**：仅修改 0.002% 参数即产生 +24.9% 准确率提升，计算开销几乎为零
3. **AOI 设计精巧**：添加一个无害的"I don't know"选项即大幅减偏，对黑盒模型同样有效
4. **CKLD 填补度量空白**：现有 RStd/RSD 对标签不均衡数据失效，CKLD 通过 KL 散度准确度量分布级偏差
5. **全场景覆盖**：BNP 针对白盒，AOI 适用黑盒，两者正交可叠加 CoT/ICL 进一步增效

## 局限性

- BNP 需要少量带标注校准数据（32 个样本的全排列），对 OOD 场景泛化性未充分验证
- 超参数 k（剪枝节点数）需按模型手动搜索，未提供自动确定方法
- 仅在 MCQ 任务上验证，对开放式生成任务中的偏差是否适用仍是开放问题
- 偏差向量计算需要 N! 次排列推理（N 为选项数），选项数较多时成本迅速增长
- 偏差的根因（训练数据中的人类认知偏差？tokenizer 的符号编码？）仍未解明

## 相关工作

- **输入端去偏**：Split-and-Merge (Li et al. 2023)、选项顺序重排 (Robinson et al. 2023)
- **输出端校准**：PriDe (Zheng et al. 2024)、DoLa 对比解码 (Chuang et al. 2023)、标签偏差量化 (Reif & Schwartz 2024)
- **结构化剪枝**：LLM-Pruner (Ma et al. 2023)、Deep Compression (Han et al. 2016)
- **符号绑定**：强化 MCQ 符号绑定训练 (Xue et al. 2024)
- **调查科学启发**：人类调查中"I don't know"选项可提升数据质量 (Schuman & Presser 1996)

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 偏差定位 + 参数级剪枝思路新颖，AOI 从调查科学汲取灵感
- **技术深度**: ⭐⭐⭐⭐ — 从 embedding 分析到偏差建模再到剪枝，逻辑链完整
- **实验充分性**: ⭐⭐⭐⭐ — 3 模型 × 4 数据集 + CoT/ICL/DoLa 叠加 + 黑盒验证 + 消融完整
- **实用性**: ⭐⭐⭐⭐⭐ — 即插即用，白盒黑盒均适用，与现有方法正交
- **总评**: ⭐⭐⭐⭐ — 方法简洁有效，解决 LLM MCQ 去偏的实际痛点

<!-- RELATED:START -->

## 相关论文

- [Beyond Communication Overhead: A Multilevel Monte Carlo Approach for Mitigating Compression Bias in Distributed Learning](../../ICML2025/model_compression/beyond_communication_overhead_a_multilevel_monte_carlo_approach_for_mitigating_c.md)
- [Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning](stun_moe_pruning.md)
- [LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation](longred_mitigating_short-text_degradation_of_long-context_large_language_models_.md)
- [EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models](eac_moe_expert_aware_compression.md)

<!-- RELATED:END -->
