---
title: >-
  [论文解读] When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models
description: >-
  [ICLR 2026][LLM推理][模型压缩] 系统性基准测试与机制解释压缩（量化/蒸馏/剪枝）对大推理模型的影响，发现三大核心结论：参数数量对知识记忆影响大于推理能力；蒸馏模型最后一层 MLP up_proj 是最关键权重；保护仅 2% 的被过度压缩权重即可提升平均准确率 6.57%。
tags:
  - ICLR 2026
  - LLM推理
  - 模型压缩
  - 大推理模型
  - 量化
  - 蒸馏
  - 剪枝
  - 机制可解释性
---

# When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models

**会议**: ICLR 2026  
**arXiv**: [2504.02010](https://arxiv.org/abs/2504.02010)  
**代码**: [psunlpgroup/Compression-Effects](https://github.com/psunlpgroup/Compression-Effects)  
**领域**: LLM推理  
**关键词**: 模型压缩, 大推理模型, 量化, 蒸馏, 剪枝, 机制可解释性

## 一句话总结

系统性基准测试与机制解释压缩（量化/蒸馏/剪枝）对大推理模型的影响，发现三大核心结论：参数数量对知识记忆影响大于推理能力；蒸馏模型最后一层 MLP up_proj 是最关键权重；保护仅 2% 的被过度压缩权重即可提升平均准确率 6.57%。

## 研究背景与动机

**LRM 的部署瓶颈**：DeepSeek-R1 等大推理模型在复杂推理任务上表现优异，但 671B 参数规模使部署成本极高，压缩是 AI 民主化的关键。

**现有研究的三大不足**：
   - 缺乏在推理密集型数据集上对量化、蒸馏、剪枝的**全面对比**
   - 未深入分析压缩如何影响模型的**知识记忆**与**推理能力**
   - 缺乏压缩效果的**可解释性分析**——哪些权重对推理最重要？这是压缩研究的根本问题

**核心研究问题**：大推理模型的推理能力在压缩过程中是如何被损害的？

## 方法详解

### 整体框架

本文采用**双视角**研究：性能基准测试 + 机制解释。

- **基准测试**：对 DeepSeek-R1 及其压缩变体进行全面评测，覆盖动态量化（Unsloth）、AWQ、GPTQ、GPTAQ、ANY4/3、SFT 蒸馏、SparseGPT、AlphaPruning
- **机制解释**：适配 difference of means 和 attribution patching 技术，量化每个线性模块对四种推理行为（回溯、不确定性估计、示例测试、添加知识）的因果贡献

### 关键设计

**权重重要性量化**：对模型每一层的每个线性模块（q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj），通过两步计算其对特定推理行为的重要性分数：

1. **Difference of Means**：提取每个线性模块在特定推理行为 token 上的引导向量 u，表达该行为在激活空间中的数值表示
2. **Attribution Patching**：计算引导向量与模块激活梯度的点积，得到重要性分数 I_mℓ^c，值越高说明该模块与推理行为的因果关系越强

**压缩效果解码**：计算相对重要性 RI（归一化后的 I），追踪压缩前后的重要性偏移。理想情况下偏移应最小——偏移越大说明压缩对该模块的破坏越严重。

### 评测设计

- **4 个推理数据集**：AIME 2024（数学）、FOLIO（逻辑）、Temporal Sequences（时序）、MuSiQue（多跳+知识，closed-book）
- **120 个实例**用于可解释性分析（每个数据集 30 个）
- 每个模型运行 3 次取平均（除 R1 和动态量化版本）

## 实验关键数据

### 主实验

DeepSeek-R1 及其压缩变体在 4 个推理基准上的表现：

| 模型/压缩方式 | AIME 2024 | FOLIO | Temporal | 平均 | MuSiQue EM |
|--------------|-----------|-------|----------|------|-----------|
| R1 (671B, 原始) | 73.3 | 76.4 | 99.6 | 83.1 | 17.0 |
| R1 (2.51-bit) | 76.7 | 77.8 | 100.0 | **84.8** | 17.0 |
| R1 (1.58-bit) | 66.7 | 75.4 | 94.0 | 78.7 | 14.0 |
| R1-Distill-Llama-70B | 65.6 | 79.8 | 99.9 | 81.8 | 13.3 |
| R1-Distill-Llama-70B + 50% SparseGPT | 23.3 | 71.6 | 97.6 | 64.2 | 6.7 |
| R1-Distill-Llama-8B | 42.2 | 71.9 | 81.5 | 65.2 | 0.0 |
| R1-Distill-Llama-8B + 4-bit AWQ | 47.8 | 68.0 | 84.0 | 66.6 | 0.3 |
| R1-Distill-Llama-8B + 3-bit GPTQ | 11.1 | 65.0 | 67.3 | 47.8 | 0.0 |

### 消融实验

选择性量化验证权重重要性（R1-Distill-Llama-8B，仅量化单个模块至 3-bit）：

| 被量化模块 | 重要性排名 | AIME 2024 | FOLIO | Temporal | 平均 |
|-----------|-----------|-----------|-------|----------|------|
| 未量化参考 | - | 42.2 | 71.9 | 81.5 | 65.2 |
| **32_up（最后层 up_proj）** | **第1** | **20.0** | **63.1** | **63.6** | **48.9** |
| 32_gate | 第2（同层） | 33.3 | 62.1 | 67.2 | 54.2 |
| 32_v | 最末（同层） | 43.3 | 68.0 | 79.6 | 63.6 |
| 31_up | 第2（同行） | 33.3 | 70.0 | 64.4 | 55.9 |

**选择性保护实验**：3-bit AWQ + 保护最后层 MLP（仅 2% 权重保持 16-bit）：

| 配置 | AIME 2024 | FOLIO | Temporal | 平均 | 提升 |
|------|-----------|-------|----------|------|------|
| 3-bit AWQ | 10.0 | 59.6 | 68.4 | 46.0 | - |
| **3-bit AWQ + 保护最后层 MLP** | **16.7** | **67.0** | **74.0** | **52.57** | **+6.57** |

### 关键发现

1. **2.51-bit 动态量化是最优压缩策略**：在所有压缩方案中表现最接近原始 R1，甚至在部分指标上超越
2. **4-bit 量化普遍安全，3-bit 出现崩溃**：所有 4-bit 方法（AWQ/GPTQ/GPTAQ/ANY4）与未量化模型接近，但 3-bit 开始显著退化
3. **剪枝对知识记忆伤害最大**：50% SparseGPT 使 AIME 从 65.6 暴跌至 23.3，MuSiQue 上的崩溃点（30-40%）比推理任务更早
4. **最后一层 MLP up_proj 是最关键模块**：仅量化该矩阵（0.7% 权重）导致平均准确率下降 16.3%
5. **现有量化方法过度压缩最后层和 gate_proj**：保护 2% 权重即可带来 6.57% 提升，最高超越 SOTA 23.17%

## 亮点与洞察

- **机制可解释性首次深入压缩效果**：不仅测性能，更揭示"为什么掉点"——是哪些具体权重被破坏
- **"参数数量影响知识 > 推理"**这一发现有重要实践意义：知识密集型任务应优先选量化而非剪枝/蒸馏
- **仅 2% 权重保护带来 6.57% 提升**：这个简单的混合精度解法验证了分析的有效性，为未来自适应量化指明方向

## 局限与展望

1. **可解释性分析规模有限**：仅 120 个实例用于 attribution patching，统计显著性存疑
2. **仅分析线性层**：未覆盖 LayerNorm、Embedding 等其他模块
3. **保护策略简单**：直接将关键权重保持 16-bit，未探索更优的混合精度或自适应量化方案
4. **蒸馏仅考虑 SFT**：未涉及 RL-based 蒸馏或 on-policy 蒸馏

## 相关工作与启发

- 与 **AlphaPruning** 的对比有趣：AlphaPruning 的剪枝效果在重要性热图上与量化相似，说明两者共享底层瓶颈
- 发现 **Qwen 推理能力 > Llama，但 Llama 知识记忆更好**——模型架构选择应考虑任务类型
- 启发：未来模型压缩应采用**重要性感知的自适应精度分配**，而非一刀切的均匀量化

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统性结合基准测试与机制解释分析压缩对 LRM 的影响
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 3 类压缩方法 × 4 个模型族 × 4 个数据集，含验证实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，发现明确可操作
- 价值: ⭐⭐⭐⭐⭐ 三大发现对 LRM 压缩研究有直接指导意义

<!-- RELATED:START -->

## 相关论文

- [Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention](towards_safe_reasoning_in_large_reasoning_models_via_corrective_intervention.md)
- [RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)
- [Training Large Reasoning Models Efficiently via Progressive Thought Encoding](training_large_reasoning_models_efficiently_via_progressive_thought_encoding.md)
- [Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models](dynamics-predictive_sampling_for_active_rl_finetuning_of_large_reasoning_models.md)
- [RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following in Large Reasoning Models with Preserved Thinking Format](rain-merging_a_gradient-free_method_to_enhance_instruction_following_in_large_re.md)

<!-- RELATED:END -->
