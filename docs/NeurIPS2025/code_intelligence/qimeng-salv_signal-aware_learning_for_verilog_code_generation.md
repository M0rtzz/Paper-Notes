---
title: >-
  [论文解读] QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation
description: >-
  [NeurIPS 2025][Verilog代码生成] 提出信号级感知学习方法 QiMeng-SALV，通过从部分错误的 Verilog 模块中提取信号级功能正确的代码片段作为 DPO 训练的奖励信号，将优化粒度从模块级提升到信号级，在 VerilogEval 和 RTLLM 上达到 SOTA。
tags:
  - NeurIPS 2025
  - Verilog代码生成
  - 信号级优化
  - DPO
  - AST
  - 强化学习
---

# QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation

**会议**: NeurIPS 2025  
**arXiv**: [2510.19296](https://arxiv.org/abs/2510.19296)  
**代码**: [GitHub](https://github.com/QiMeng-IPRC/QiMeng-SALV)  
**领域**: 代码生成 / 硬件设计自动化  
**关键词**: Verilog代码生成, 信号级优化, DPO, AST, 强化学习

## 一句话总结

提出信号级感知学习方法 QiMeng-SALV，通过从部分错误的 Verilog 模块中提取信号级功能正确的代码片段作为 DPO 训练的奖励信号，将优化粒度从模块级提升到信号级，在 VerilogEval 和 RTLLM 上达到 SOTA。

## 研究背景与动机

- LLM 在 Verilog 代码生成中展现潜力，但基于 RL 的偏好优化面临**功能奖励不足**的问题
- 现有 RL 方法依赖代码结构相似度作为奖励，但同一功能可有多种不同结构的正确实现
- 以功能正确性作为奖励更合理，但 SFT 阶段训练数据不足导致模型难以生成完全正确的模块
- **关键洞察**：Verilog 描述硬件门/线的结构互连，不同输出信号天然独立。即使整个模块功能错误，某些信号的实现可能是正确的——这些可以提供有效的功能正确性奖励

## 方法详解

### 整体框架

QiMeng-SALV 包含三个阶段：

1. **信号级验证（Signal-aware Verification）**：通过随机测试输入对比生成模块和参考模块的输出信号，识别正确的信号
2. **信号级代码提取（Signal-aware Code Extraction）**：利用 AST 分析建立信号依赖图，提取与目标信号相关的代码片段
3. **信号级 DPO 训练（Signal-aware DPO）**：仅对对比信号相关代码的 token 计算概率，训练模型学习正确信号实现

### 关键设计

1. **基于 AST 的信号感知代码提取**:
    - 功能：从生成的 Verilog 模块中精确提取某个输出信号对应的完整代码实现
    - 核心思路：用 Yosys 解析模块代码得到 AST，分析所有输出信号与中间信号的依赖关系形成拓扑图，从目标信号叶节点反向遍历获取其依赖的所有信号及对应代码
    - 设计动机：Verilog 的信号独立性使得可以从模块级别提取单个信号的自包含实现，为 RL 提供过程级反馈

2. **信号级 DPO 损失函数**:
    - 功能：改进标准 DPO，仅对对比信号相关的代码 token 计算概率
    - 核心思路：给定对比信号 $c$，从 preferred ($y_w$) 和 dispreferred ($y_l$) 样本中提取代码片段 $S_w^c$ 和 $S_l^c$，DPO loss 只在这些 token 上计算
    - 设计动机：标准 DPO 假设 preferred 样本完全正确，在 Verilog 场景不成立；信号级 DPO 避免了错误信号的噪声干扰

### 损失函数 / 训练策略

信号级 DPO 损失：

$$\mathcal{L}(\pi_\theta;\pi_{\text{ref}}) = -\mathbb{E}\left[\log\sigma\left(\beta\sum_{y_t\in S_w^c}\log\frac{\pi_\theta(y_t|y_{w,<t},x)}{\pi_{\text{ref}}(y_t|y_{w,<t},x)} - \beta\sum_{y_t\in S_l^c}\log\frac{\pi_\theta(y_t|y_{l,<t},x)}{\pi_{\text{ref}}(y_t|y_{l,<t},x)}\right)\right]$$

- SFT 阶段：对 135k Verilog 样本全参数微调 2 epoch
- DPO 阶段：LoRA 微调约 7000 步（~1 epoch），学习率 5e-6
- 基座模型：Qwen2.5 Coder Instruct 7B，每个 prompt 采样 5 个候选

## 实验关键数据

### 主实验（表格）

| 模型 | 参数量 | VerilogEval1.0 pass@1 | VerilogEval2.0 pass@1 | RTLLM v1.1 pass@1 |
|------|--------|----------------------|----------------------|-------------------|
| GPT-4o | - | 60.1 | 62.5 | - |
| DeepSeek v3 | 671B | 70.7 | 68.8 | - |
| CodeV (Qwen2.5) | 7B | 57.9 | 44.8 | - |
| Origen | 7B | 54.4 | 49.3 | - |
| **QiMeng-SALV** | **7B** | **65.6** | **62.6** | **62.6** |

### 消融实验

- 去除信号级验证（使用模块级 DPO）：性能显著下降，验证了信号级奖励的必要性
- 去除对比信号的 token 过滤（计算全模块 token）：引入错误信号噪声导致 DPO 效果退化
- QiMeng-SALV 7B 在 RTLLM v1.1 的 pass@5 达到 75.1%，匹配 DeepSeek v3 671B

### 关键发现

- 7B 模型匹配 671B DeepSeek-V3 在 RTLLM 上的性能
- 信号级优化比模块级优化提供更密集的功能正确性奖励，有效扩大了可用训练样本空间
- 即使整体模块不正确，包含正确信号实现的模块都可以被纳入训练

## 亮点与洞察

- **范式转换**：首个信号级 RL 算法用于 Verilog 生成，将优化粒度从模块级提升到信号级
- 巧妙利用 Verilog 语言的**硬件描述特性**（信号独立性）来设计 RL 方法
- AST 信号提取具有通用性，可适用于其他硬件描述语言

## 局限与展望

- 信号验证依赖随机测试输入，可能遗漏特定边界条件下的错误
- 仅在 Qwen2.5 Coder 7B 上验证，更大规模模型的效果未知
- AST 解析依赖 Yosys 工具，不支持所有 Verilog 语法变体
- 未探索与 GRPO 等更新 RL 方法的结合

## 相关工作与启发

- VeriPrefer 使用模块级功能奖励做 RL，但信号级奖励更密集
- CodeV 专注于 SFT 阶段数据集构建，QiMeng-SALV 在 RL 阶段实现互补改进
- 信号级提取思想可推广到其他具有模块化结构的代码生成任务

## 评分

- 理论创新：⭐⭐⭐⭐
- 实验验证：⭐⭐⭐⭐
- 实用价值：⭐⭐⭐⭐⭐
- 写作质量：⭐⭐⭐⭐
- 综合评分：⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](../../ACL2025/code_intelligence/dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)
- [EffiCoder: Enhancing Code Generation in Large Language Models through Efficiency-Aware Fine-tuning](../../ICML2025/code_intelligence/efficoder_enhancing_code_generation_in_large_language_models_through_efficiency-.md)
- [Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning](co-evolving_llm_coder_and_unit_tester_via_reinforcement_learning.md)
- [Embedding Alignment in Code Generation for Audio](embedding_alignment_in_code_generation_for_audio.md)
- [Learning to Solve Complex Problems via Dataset Decomposition](learning_to_solve_complex_problems_via_dataset_decomposition.md)

<!-- RELATED:END -->
