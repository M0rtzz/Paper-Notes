---
title: >-
  [论文解读] Unsupervised Evaluation of Multi-Turn Objective-Driven Interactions
description: >-
  [ICLR2026][LLM/NLP][Unsupervised Evaluation] 提出三种**无监督**指标——LLM 引导聚类（目标识别）、基于微调完成模型的交互完整性检测、响应树（LLM 不确定性量化）——用于评估多轮目标驱动对话，无需标注数据或 LLM-as-a-judge，仅用 8B 模型即可匹配/超越 70B judge 的性能。
tags:
  - ICLR2026
  - LLM/NLP
  - Unsupervised Evaluation
  - Multi-Turn Dialogue
  - Goal Completion
  - LLM Uncertainty
  - Response Tree
  - LLM-Guided Clustering
---

# Unsupervised Evaluation of Multi-Turn Objective-Driven Interactions

**会议**: ICLR2026  
**arXiv**: [2511.03047](https://arxiv.org/abs/2511.03047)  
**代码**: 未公开（论文提及最终版将发布）  
**领域**: llm_nlp  
**关键词**: Unsupervised Evaluation, Multi-Turn Dialogue, Goal Completion, LLM Uncertainty, Response Tree, LLM-Guided Clustering  
**作者**: Emi Soroka, Tanmay Chopra, Krish Desai, Sanjay Lall（Stanford & Emissary Technologies）

## 一句话总结

提出三种**无监督**指标——LLM 引导聚类（目标识别）、基于微调完成模型的交互完整性检测、响应树（LLM 不确定性量化）——用于评估多轮目标驱动对话，无需标注数据或 LLM-as-a-judge，仅用 8B 模型即可匹配/超越 70B judge 的性能。

## 研究背景与动机

**企业 LLM 系统评估困难**：任务导向对话、AI agent、客服系统等目标驱动交互日益普及，但评估手段严重落后——数据复杂且无标注，人工标注不可扩展。

**LLM-as-a-judge 不可靠**：已知存在位置偏差、冗余偏差、熟悉偏差、输出不一致性和 prompt 措辞敏感性等问题。

**分布漂移问题**：目标驱动系统引入推理、工具调用、多 agent 交互、共享环境操作等，偏离 LLM 预训练的基础对话分布，使评估更困难。

**现有指标局限**：ROUGE/BLEU 需参考答案，perplexity 信息有限，自定义指标只能监控已知错误类型。

**核心目标**：设计**零标注**、**零参考答案**的评估指标，能自动发现用户目标、检测交互完整性、量化 LLM 不确定性。

## 方法详解

### 指标 1：LLM 引导聚类（用户目标识别）

**目标**：从无标注多轮对话中自动发现和标注用户目标类别。

**三阶段算法**（Algorithm 1）：

**预处理**：对每个对话 $c_i$，提示 LLM 生成自由文本目标摘要 $s_i$，使用 text-embedding-3-small 嵌入为 $v_i \in \mathbb{R}^{1536}$。

**Phase 1 — 初始聚类 + 标注**：
- 对 $v_1, \dots, v_n$ 执行 k-means 得到 $k_1$ 个初始聚类（$k_1$ 取较大过估值）
- 对每个聚类，抽 10 个正样本 + 10 个负样本，提示 LLM 生成聚类描述 $L_i$
- 嵌入所有描述得到 $d_1, \dots, d_{k_1}$

**Phase 2 — 迭代合并**：
- 计算描述间余弦相似度矩阵 $D_{ij} = \frac{d_i^\top d_j}{\|d_i\|_2 \|d_j\|_2}$
- 迭代选择最大 $D_{ij}$，提示 LLM 判断是否合并（每次提供正负样本各 10 个）
- 合并后重新生成描述，终止条件为所有当前聚类均拒绝合并

**优势**：结合了 k-means 的稳定性和 LLM 的语义理解能力，输出带文本标签的可解释聚类。

### 指标 2：交互完整性检测（Goal Completion）

**核心思想**：利用微调 LLM 学习"完成分布"，通过判断对话是否应结束来检测完整性。

**形式化定义**：给定完成对话分布 $D$，构造新分布 $D'$，其中每个完成对话的最后回复附加 `end` 标签。定义：

$$P_{D'}(\texttt{end} \mid c) = P(\text{llm}_{D'}(\text{concat}(p_1, r_1, \dots, p_n, r_n)) = \texttt{end})$$

对完整对话 $c$ 和截断对话 $c'$（$k < n$ 轮），期望满足：

$$P_{D'}(\texttt{end} \mid c) > P_{D'}(\texttt{end} \mid c')$$

**实现**：
- **基础分布**（如 LMSYS）：直接使用 LLaMA3.1-8B-Instruct + 短 prompt
- **专用分布**（如保险核保、代码调试）：训练 LoRA 适配器微调 LLaMA3.2-8B 完成模型
    - 输入：$\text{concat}(p_1, r_1, \dots, p_n)$
    - 目标：$r_n$ + `end` 标签
    - 训练：AdamW 8-bit，lr = 0.0002，weight decay = 0.01，3 epochs，50% 数据
- **不完整对话**：模型不输出 `end`，而是生成后续轮次 $p_{n+1}, r_{n+1}, \dots$，这些内容还能**总结 LLM 未完成的剩余任务**

### 指标 3：响应树（Response Uncertainty）

**目标**：量化 LLM 对特定 prompt 的回复不确定性，无需重复高温采样。

**响应树定义**：给定 prompt $p$ 和阈值概率 $\alpha$，$\text{rtree}_{D,\alpha}(p)$ 返回所有遍历概率 $\ge \alpha$ 的分支树。

**构建方法**：
1. 生成一个回复及其 top-$k$ logprobs
2. 若第 2 到第 $k$ 个 token 的 logprob $> \alpha$，为其分别生成分支
3. 递归直到无 logprob 超过 $\alpha$ 或达到计算阈值

**不确定性量化**：
- **叶节点数**：叶节点多 → 多个可能回复 → 高不确定性 → 更可能出错
- **最大 logprob**：高 → LLM 对最优回复有信心
- 二者相比对话长度相关性低（$r$ 在 -0.25 ~ 0.41），说明响应树捕获的是更复杂的不确定性信息

## 实验关键数据

### 数据集

| 数据集 | 规模 | 主题 | 目标驱动 | 工具使用 |
|--------|------|------|:--------:|:--------:|
| LMSYS-Chat-1M | 1000 | 非结构化对话 | ✗ | ✗ |
| Code-Feedback | 1000 | 代码生成调试 | ✓ | ✗ |
| Insurance | 380 | 保险核保 | ✓ | ✓ |
| WebShop | 351 | 网购交互 | ✓ | ✓ |
| SQL+OS+KB | 1043 | SQL/终端/知识库 | ✓ | ✓ |

### 完整性检测结果

| 数据集（评估器） | Accuracy | Precision | Recall | F1 |
|------------------|----------|-----------|--------|-----|
| LMSYS（70B judge） | 0.43 | 0.77 | 0.25 | 0.38 |
| **LMSYS（8B completion）** | **0.74** | **0.79** | **0.85** | **0.82** |
| Code-Feedback（70B judge） | 0.53 | 0.53 | 0.46 | 0.49 |
| Code-Feedback（FT 8B） | 0.47 | 0.71 | 0.12 | 0.21 |
| Insurance（70B judge） | 0.95 | 1.0 | 0.91 | 0.95 |
| **Insurance（FT 8B）** | **0.91** | **0.94** | **0.87** | **0.91** |
| WebShop（70B judge） | 0.92 | 1.0 | 0.83 | 0.91 |
| **WebShop（FT 8B）** | **0.92** | **0.89** | **1.0** | **0.94** |
| SQL+OS+KB（70B judge） | 0.97 | 0.96 | 0.97 | 0.96 |
| **SQL+OS+KB（FT 8B）** | **0.98** | **0.99** | **0.98** | **0.99** |

**关键发现**：
- 8B 微调模型在多数数据集上**匹配或超越 70B LLM judge**
- `end` 标签是关键设计（Insurance 无 end tag 时 F1 从 0.91 降至 0.72）
- Code-Feedback 难度最高（对话结构松散，随时可续）

### 响应树统计

| 指标 | LMSYS | Code-Feedback | Insurance | WebShop | KB+OS+SQL |
|------|-------|---------------|-----------|---------|-----------|
| Max logprob vs 长度 | -0.11 | -0.19 | -0.25 | 0.16 | 0.41 |
| Max logprob vs 叶节点数 | -0.49 | -0.46 | -0.10 | -0.19 | -0.06 |

- KB+OS+SQL 不确定性最高（工具调用、SQL、终端交互偏离基础分布最大）
- LMSYS 和 Code-Feedback 置信度最高（更接近预训练分布）

### 聚类稳定性
- LMSYS、WebShop、SQL+OS+KB 跨运行产生**高度稳定聚类**
- Code-Feedback 和 Insurance 稳定性稍低（可多维度标注：语言 vs 任务类型）
- 对比 GPT-4.1 纯 LLM 标注基线：LLM-only 方法在 WebShop 上退化为单一聚类"Online Shopping and Purchase"

## 亮点

1. **三指标体系互补完整**：目标识别（what）+ 完成检测（whether）+ 不确定性量化（how confident），覆盖评估核心维度
2. **零标注零参考**：真正的无监督，不依赖 ground truth 或 LLM judge
3. **小模型大效果**：8B 微调模型匹配/超越 70B judge，适合在线部署和实时监控
4. **响应树创新**：比 semantic entropy（需多次高温采样）更结构化、信息更丰富
5. **分布适应性**：通过 LoRA 微调适应不同领域的 token 分布

## 局限与展望

1. **聚类算法依赖初始 $k_1$ 设定**，限制了可发现的最大聚类数
2. **完整性检测对松散结构对话效果不佳**（如 Code-Feedback，第一轮即可回答后续为追问）
3. **未做多标签分类**（一个对话可能涉及多个目标）
4. **响应树缺乏 ground truth 验证**（无法直接证明高不确定性 = 错误）
5. **微调数据量有限**（Insurance 仅 190 样本训练，性能波动较大）
6. **仅在合成/公开数据上验证**，未在真实企业系统中部署测试

## 与相关工作的对比

| 方法 | 需要标注 | 需要参考答案 | 需要 LLM judge | 模型规模 | 支持多轮 |
|------|:--------:|:----------:|:--------------:|:--------:|:--------:|
| ROUGE/BLEU | ✗ | ✓ | ✗ | — | ✗ |
| BERTScore | ✗ | ✓ | ✗ | ~110M | ✗ |
| HelpSteer | ✓ | ✗ | ✗ | — | ✓ |
| G-EVAL | ✗ | ✗ | ✓ | >70B | ✓ |
| DeepEval | ✗ | ✗ | ✓ | >70B | ✓ |
| **本文** | **✗** | **✗** | **✗** | **8B** | **✓** |

## 启发与关联

- **在线干预潜力**：完整性检测可用于提前终止无效对话节省 token；不确定性量化可触发人工介入
- **响应树 + 采样策略**：若知道 LLM 的采样策略，可通过响应树给出输出概率的统计保证
- **与 conformal prediction 互补**：本文聚焦无监督场景，conformal 方法提供有监督保证，二者可结合
- **LoRA 微调作为分布适配器**：通用范式——用小数据 LoRA 微调让 8B 模型适配特定领域的 token 分布

## 评分
- 新颖性: ⭐⭐⭐⭐ — 三个指标各有创新，LLM 引导聚类和响应树是新颖贡献
- 实验充分度: ⭐⭐⭐⭐ — 6 个数据集、多个消融，但响应树缺乏直接效果验证
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，形式化严谨，附录详尽
- 综合价值: ⭐⭐⭐⭐ — 填补了多轮目标驱动对话无监督评估的空白，实用性强

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] LLEMA: Evolutionary Search with LLMs for Multi-Objective Materials Discovery](llema_evolutionary_search_with_llms_for_multi-objective_materials_discovery.md)
- [\[AAAI 2026\] Conversational Learning Diagnosis via Reasoning Multi-Turn Interactive Learning](../../AAAI2026/llm_nlp/conversational_learning_diagnosis_via_reasoning_multi-turn_interactive_learning.md)
- [\[ACL 2026\] Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models](../../ACL2026/llm_nlp/style_amnesia_investigating_speaking_style_degradation_and_mitigation_in_multi-t.md)
- [\[NeurIPS 2025\] PluralisticBehaviorSuite: Stress-Testing Multi-Turn Adherence to Custom Behavioral Policies](../../NeurIPS2025/llm_nlp/pluralistic_behavior_suite_stress-testing_multi-turn_adherence_to_custom_behavio.md)
- [\[ACL 2025\] Gradient-Adaptive Policy Optimization: Towards Multi-Objective Alignment of Large Language Models](../../ACL2025/llm_nlp/gapo_multi_objective_alignment.md)

<!-- RELATED:END -->
