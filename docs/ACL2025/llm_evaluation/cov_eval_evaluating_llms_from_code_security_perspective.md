---
title: >-
  [论文解读] CoV-Eval: Can You Really Trust Code Copilots? Evaluating Large Language Models from a Code Security Perspective
description: >-
  [ACL 2025][LLM评测][代码安全] 提出 CoV-Eval，首个多任务代码漏洞评估基准，涵盖代码补全、漏洞修复、漏洞检测和漏洞分类四个维度，并开发 VC-Judge 漏洞判断模型替代传统静态分析工具，对 20 个 LLM 进行全面评估，发现虽然多数 LLM 能检测漏洞代码，但仍倾向生成不安全代码且漏洞修复能力有限。
tags:
  - "ACL 2025"
  - "LLM评测"
  - "代码安全"
  - "漏洞评估"
  - "LLM代码生成"
  - "自动化评估"
  - "多任务基准"
---

# CoV-Eval: Can You Really Trust Code Copilots? Evaluating Large Language Models from a Code Security Perspective

**会议**: ACL 2025  
**arXiv**: [2505.10494](https://arxiv.org/abs/2505.10494)  
**代码**: [https://github.com/MurrayTom/CoV-Eval](https://github.com/MurrayTom/CoV-Eval)  
**领域**: LLM评测  
**关键词**: 代码安全, 漏洞评估, LLM代码生成, 自动化评估, 多任务基准

## 一句话总结

提出 CoV-Eval，首个多任务代码漏洞评估基准，涵盖代码补全、漏洞修复、漏洞检测和漏洞分类四个维度，并开发 VC-Judge 漏洞判断模型替代传统静态分析工具，对 20 个 LLM 进行全面评估，发现虽然多数 LLM 能检测漏洞代码，但仍倾向生成不安全代码且漏洞修复能力有限。

## 研究背景与动机

**领域现状**：LLM 驱动的代码辅助工具（如 GitHub Copilot）已广泛部署。现有评估数据集（HumanEval、MBPP 等）主要评估代码可用性（能否通过测试用例），而对代码安全性的评估不足。GPT-4o 生成的代码虽然实现了所需功能，但可能存在信息泄漏和内存溢出等安全隐患。

**现有痛点**：代码安全评估数据集（CWE-scenario、SecurityEval、CyberSecEval）仅聚焦单一评估任务（如代码补全），无法全面评估 LLM 在安全代码生成、漏洞修复和漏洞识别等多维度的能力及其关联。此外，自动化评估工具方面，传统静态分析工具（CodeQL、Bandit）受限于手工编写的规则，假阴性率高；LLM 作为评估器虽假阴性少但假阳性率高于人类专家。

**核心矛盾**：单一任务评估无法反映 LLM 代码安全能力的全貌——一个能检测漏洞的模型不一定能生成安全代码，一个代码安全率高的模型可能在漏洞修复上很弱。同时缺乏可靠的自动化漏洞评估方法。

**本文目标**：（1）构建覆盖代码安全多个维度的多任务评估基准；（2）开发与人类专家高度对齐的自动化漏洞判断模型。

**切入角度**：从 CWE（通用弱点枚举）出发，覆盖 18 种漏洞类型，设计四种互补的评估任务，并通过指令微调训练专门的漏洞判断模型来解决自动化评估的可靠性问题。

**核心 idea**：用多任务评估框架全面刻画 LLM 代码安全能力的多维度表现，并用指令微调的 VC-Judge 模型替代不可靠的静态分析工具和通用 LLM 评估器。

## 方法详解

### 整体框架

CoV-Eval 由两部分组成：（1）多任务评估数据集——基于 GitHub-CWE 种子集构建四种任务的测试集，并用 Vul-Evol 框架合成更复杂的代码场景；（2）自动化评估——用 VC-Judge 替代传统工具对 LLM 生成的代码进行安全审计。评估 4 个闭源 + 16 个开源 LLM，输出涵盖安全率（SR@1）、F1 分数、召回率等多指标。

### 关键设计

1. **四任务评估体系**:

    - 功能：从不同角度全面评估 LLM 的代码安全能力
    - 核心思路：（a）**代码补全**——给定含注释的不完整程序，LLM 补全代码，评估生成代码的安全性；（b）**漏洞修复**——给定含漏洞的代码和漏洞类型描述，LLM 修复漏洞；（c）**漏洞检测**——给定完整代码，判断是否存在漏洞；（d）**漏洞分类**——不仅检测漏洞，还需识别具体的漏洞类型（CWE 编号）
    - 设计动机：四种任务的互补性可以更好地模拟真实软件开发挑战：能检测漏洞不代表能写出安全代码，能识别漏洞类型是修复的前提

2. **Vul-Evol 代码场景合成框架**:

    - 功能：生成更复杂的代码场景用于压力测试
    - 核心思路：基于指令进化（Instruction Evolution），用 GPT-4o 对种子集的 54 个代码场景进行四种复杂度增强——添加新约束、替换常见需求为罕见需求、增加推理步骤、提高时间/空间复杂度要求。经人工质量过滤（移除已包含安全特征的场景），获得 270 个新场景
    - 设计动机：种子集场景较简单，实际编程环境更复杂。同时发现 40% 合成场景已包含安全特征（GPT-4o 的安全标准较高），需要人工过滤以确保测试公平性

3. **VC-Judge 漏洞判断模型**:

    - 功能：替代传统静态分析工具，更可靠地自动评估 LLM 生成代码的安全性
    - 核心思路：基于 LLAMA3-8B-Instruct 指令微调，训练数据来自三个源：CoV-Eval 代码补全测试中人工标注的 216 个程序、漏洞检测测试集的 531 个程序、BigVul 开源漏洞数据集。设计漏洞判断、分类和修复三种 prompt 模板。采用 judgment-style 评估模板而非多分类或二分检测，利用已知的漏洞类型信息提高判断可靠性
    - 设计动机：传统静态分析假阴性高（受限于规则），通用 LLM 假阳性高（缺乏漏洞专业知识）。专门微调的 VC-Judge 与人类专家一致性最高（78.24%），且安全率差距最小（1.39%）

### 损失函数 / 训练策略

VC-Judge 使用标准指令微调（SFT）训练。评估指标：代码补全和漏洞修复使用安全率 SR@1，漏洞检测和分类使用加权 F1、召回率和准确率。

## 实验关键数据

### 主实验

| 模型 | 代码补全SR@1 | 漏洞修复SR@1 | 漏洞检测F1 | 漏洞分类F1 | 综合安全分 |
|------|-------------|-------------|-----------|-----------|-----------|
| Claude-3 | 74.07 | 66.25 | 92.42 | 45.00 | 69.43 |
| GPT-4o | 72.84 | 63.94 | 94.62 | 36.05 | 66.86 |
| LLAMA3.1-8B | 75.92 | 58.70 | 92.89 | 26.45 | 63.49 |
| DeepSeek-Coder-V2 | 75.31 | 51.57 | 90.63 | 35.50 | 63.25 |
| CodeLLAMA-7B | 68.21 | 39.62 | 93.57 | 11.47 | 53.22 |

### 消融实验：高质量代码数据的影响

| 微调数据配置 | 代码补全(seed) | 代码补全(vul-evol) | 漏洞修复 | HumanEval |
|------------|---------------|-------------------|---------|-----------|
| LLAMA2-7B 基线 | 42.59 | 58.89 | 42.98 | 14.51 |
| +安全代码SFT | 62.96 | 76.29 | 24.53 | 16.04 |
| +安全代码+漏洞检测SFT | 64.81 | 76.67 | 32.91 | 16.74 |
| +通用代码SFT (GC-IFT) | 40.74 | 49.63 | 6.08 | 20.27 |

### 关键发现

- **检测能力与生成安全代码能力脱节**：Qwen1.5-14B 和 ChatGLM3-6B 漏洞检测召回率 100%，但代码补全安全率仅 69% 和 74%——知道什么是漏洞不等于能避免生成漏洞
- **最常见的生成漏洞类型**：CWE-78（OS命令注入）、CWE-434（不受限文件上传）和 CWE-190（整数溢出）在几乎所有 LLM 中都是高发漏洞
- **LLM 能避免的漏洞**：CWE-125（越界读取）、CWE-89（SQL注入）、CWE-732（不当权限）、CWE-416（UAF），这些涉及数据完整性和内存安全的漏洞被较好避免
- **代码专用微调提升安全性**：CodeLLAMA-7B 的代码补全安全率比 LLAMA2-7B 高 12%（56%→68%）
- **高质量安全代码数据是关键**：用经过安全审计的代码数据微调可同时提升安全性和可用性，而未经安全审查的代码数据可能损害安全性

## 亮点与洞察

- **多任务设计揭示能力间的关联**：发现漏洞分类能力与代码安全正相关——分类能力越差的模型生成的漏洞越多，暗示注入漏洞知识可能是提升代码安全的有效路径
- **Vul-Evol 的质量过滤洞察**：40% 的 GPT-4o 合成场景已包含安全特征，说明强模型在数据合成时自带的安全偏好可能影响评估公平性
- **自修复实验**：Mistral-7B 在自检测+自修复中表现最好（SR@1 63.74%），而非最强的闭源模型，这个发现值得关注

## 局限与展望

- 数据集规模有限，基于 54 个种子场景扩展，多样性受制于种子集
- VC-Judge 虽然与人类一致性最高但仍不及人类专家，存在假阴性
- 安全性和可用性评估使用不同数据集，缺乏统一测试框架
- 未来方向：更多样的代码场景和漏洞类型、统一的安全+可用性测试、探索最优数据配比和训练方法

## 相关工作与启发

- **vs CWE-scenario/SecurityEval/CyberSecEval**: 这些数据集限于单一的代码补全任务，CoV-Eval 通过四任务设计提供更全面的评估视角
- **vs VulBench/VulDetectBench**: 这些工作评估 LLM 的漏洞检测能力，但未与代码生成安全性关联分析。CoV-Eval 的多任务设计揭示了检测与生成之间的关系

## 评分

- 新颖性: ⭐⭐⭐⭐ 多任务代码安全评估框架设计合理，VC-Judge 有实用价值
- 实验充分度: ⭐⭐⭐⭐⭐ 20 个模型 × 4 任务 + 自修复分析 + 数据消融 + 评估器对比，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰但部分表格密度过高
- 价值: ⭐⭐⭐⭐ 对 LLM 代码安全研究和代码辅助工具开发有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] CodeMEnv: Benchmarking Large Language Models on Code Migration](codemenv_benchmarking_large_language_models_on_code_migration.md)
- [\[ACL 2025\] Benchmarking LLMs and LLM-based Agents in Practical Vulnerability Detection for Code Repositories](benchmarking_llms_and_llm-based_agents_in_practical_vulnerability_detection_for_.md)
- [\[ACL 2025\] Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)
- [\[ACL 2026\] ScaleBox: Enabling High-Fidelity and Scalable Code Verification for Large Language Models](../../ACL2026/llm_evaluation/scalebox_enabling_high-fidelity_and_scalable_code_verification_for_large_languag.md)
- [\[ACL 2025\] AbGen: Evaluating Large Language Models in Ablation Study Design and Evaluation for Scientific Research](abgen_evaluating_large_language_models_in.md)

</div>

<!-- RELATED:END -->
