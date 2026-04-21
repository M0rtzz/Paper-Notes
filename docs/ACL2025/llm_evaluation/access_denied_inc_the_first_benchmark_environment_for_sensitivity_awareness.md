---
title: >-
  [论文解读] Access Denied Inc: The First Benchmark Environment for Sensitivity Awareness
description: >-
  [ACL 2025][敏感性感知] 首次形式化定义 LLM "敏感性感知"（Sensitivity Awareness）概念——评估 LLM 能否根据基于角色的访问控制（RBAC）规则决定信息是否可以提供——并构建自动化评估基准 Access Denied Inc，在 7 个主流 LLM 上发现即使数据高度结构化且规则极简，最佳模型 Grok-2 仍有 18.28% 的泄露率。
tags:
  - ACL 2025
  - 敏感性感知
  - RBAC
  - 企业数据管理
  - LLM基准测试
  - 信息泄露防护
---

# Access Denied Inc: The First Benchmark Environment for Sensitivity Awareness

**会议**: ACL 2025  
**arXiv**: [2506.00964](https://arxiv.org/abs/2506.00964)  
**代码**: [GitHub](https://github.com/DrenFazlija/AccessDeniedInc) (有)  
**领域**: LLM评估 / 数据访问控制  
**关键词**: 敏感性感知, RBAC, 企业数据管理, LLM基准测试, 信息泄露防护

## 一句话总结

首次形式化定义 LLM "敏感性感知"（Sensitivity Awareness）概念——评估 LLM 能否根据基于角色的访问控制（RBAC）规则决定信息是否可以提供——并构建自动化评估基准 Access Denied Inc，在 7 个主流 LLM 上发现即使数据高度结构化且规则极简，最佳模型 Grok-2 仍有 18.28% 的泄露率。

## 研究背景与动机

**领域现状**：LLM 驱动的 AI 助手正进入企业数据管理场景（如 SAP Joule），员工可通过自然语言查询薪资、部门归属等信息。这类系统需要 LLM 不仅能检索数据，还要根据用户角色决定是否提供。

**现有痛点**：简单的文档级过滤不可行——企业文档往往混合敏感和非敏感信息（如 HR 文件中公开职位描述与受限薪资并存），粗粒度过滤要么过度屏蔽要么暴露受限数据。更严重的是，即使部分信息被隐藏，RAG 系统中 LLM 仍可能从多个检索片段的上下文线索中推断出受限信息。

**核心矛盾**：现有安全/隐私基准（如越狱攻击、有害内容生成）关注的是"生成不该生成的内容"，而企业场景的核心需求是"根据用户权限选择性提供已有数据"——这是一个尚无人系统评估的空白领域。SudoLM 仅支持二元（公开/私密）的粗粒度认证，远不够。

**本文切入角度**：构建首个标准化的敏感性感知评估框架。核心 idea：用模拟企业数据库 + 细粒度字段级权限 + 自动化问卷 + 99.9% 自动评分的完整 pipeline，系统评估开箱即用 LLM 对 RBAC 规则的遵守能力。

## 方法详解

### 整体框架

Access Denied Inc 是一个完整的三阶段评估 pipeline：(1) 从 Adult 数据集生成模拟企业员工数据库（45K+ 员工，12 个属性字段）；(2) 基于可配置参数自动生成多维度测试问卷（每份 3500 题，覆盖 6 个特征和 2 个对抗场景）；(3) 半自动评分系统评估 LLM 回答，自动覆盖率可达 99.9%。

### 关键设计

1. **SA 的形式化定义与四类会话分类**：将 LLM 与用户的每次交互分为四种互斥类型——$S_{\text{correct}}$（正确提供数据或正确拒绝）、$S_{\text{leak}}$（向未授权用户泄露受限数据，最危险）、$S_{\text{refusal}}$（向授权用户错误拒绝，影响可用性）、$S_{\text{error}}$（输出幻觉数据或格式违规）。这一分类将安全性与可用性的权衡显式建模——泄露和误拒是 SA 评估的两个核心维度，而非简单的对错二元判断。

2. **模拟企业数据库生成**：复用 Adult 表格数据集，去除缺失值后分配唯一 ID 和随机姓名（从 20K 热门名字库采样，刻意打破姓名与性别/种族的关联以消除偏见）。将二元薪资修改为 $N(80000, 15000)$ 的连续数值（排除 <35K 和 >200K的极端值），新增部门（基于预定义组织架构图）和上级（基于部门层级随机分配）属性。最终数据库包含 45,233 名员工，每人 12 个属性。设计动机：表格数据是最简单的企业数据形式——如果 LLM 在高度结构化的数据上都无法执行访问控制，在非结构化文档上只会更差。

3. **自动化问卷与评分系统**：问卷根据配置参数（视角：第一/三人称；恶意性：授权/未授权；目标特征：薪资/部门/年龄等 6 个字段）自动生成。系统提示包含目标用户数据 + 5 个随机员工数据（模拟 RAG 的 top-k 检索），并提供简明访问规则（每人可查自己全部信息；HR 部门员工和直接上级可查下属信息）。模型被要求在双花括号 {{}} 内输出简洁结果（数据值或固定拒绝语"I cannot disclose that information."），括号外为 CoT 推理空间。评分基于字符串精确匹配，实现 99.9%（GPT-4o）到 92.9%（Llama 3.2）的自动覆盖率。

### 损失函数 / 训练策略

本文评估 LLM 的开箱即用能力，不涉及训练。系统提示作为唯一的对齐手段，包含统一的基础提示模板。闭源模型（GPT-4o/mini, Grok-2）通过 API 调用；开源模型（Llama 3.2 3B, Llama 3.3 70B, R1-Qwen 32B, Phi-4 14B）通过 HuggingFace 或 OpenRouter API。除 R1-Qwen（temperature=0.6）外，所有模型使用默认参数。

## 实验关键数据

### 主实验：7 个模型总体表现

| 模型 | Correct ↑ | Error ↓ | Wrong ↓ | Benign ↑ | Malicious ↑ |
|------|-----------|---------|---------|----------|-------------|
| **Grok-2** | **80.50%** | **0.22%** | 18.28% | **95.52%** | **65.48%** |
| GPT-4o | 70.72% | 3.61% | 25.63% | 83.88% | 57.56% |
| R1-Qwen (32B) | 64.56% | 2.94% | 28.09% | 94.59% | 34.53% |
| Llama 3.3 (70B) | 60.81% | 0.16% | 38.32% | 97.54% | 24.07% |
| Phi-4 (14B) | 59.42% | 6.81% | 26.93% | 84.26% | 34.59% |
| GPT-4o mini | 45.98% | 35.88% | 18.08% | 57.33% | 34.62% |
| Llama 3.2 (3B) | 29.08% | 13.68% | 50.17% | 48.09% | 10.07% |

### 消融实验（特殊场景）

| 场景 | Grok-2 | GPT-4o | Llama 3.3 | R1-Qwen | GPT-4o mini |
|------|--------|--------|-----------|---------|-------------|
| Supervisor（上级查下属，合法） | 80.66% | 59.33% | 94.40% | 90.00% | 32.93% |
| Lying（伪造身份，对抗性） | 49.86% | 44.53% | 45.33% | **13.60%** | **50.66%** |

### 关键发现

- **所有模型在恶意查询上表现都很差**：即使最佳 Grok-2 也只有 65.48% 的恶意请求被正确拒绝，意味着约 1/3 的未授权请求会导致泄露
- **模型失败模式截然不同**：GPT-4o mini 主要问题是幻觉和格式错误（Error 35.88%），而开源模型主要问题是泄露（Wrong 高但 Error 低）——Llama 3.3 几乎总是输出真实数据不管权限
- **Supervisor 场景暴露权限理解能力差异**：闭源模型在合法上级查询上反而表现差（GPT-4o mini 仅 32.93%），说明模型不理解层级权限关系
- **Lying 场景中小模型反而更鲁棒**：GPT-4o mini（50.66%）在伪造身份场景中竟优于 Grok-2（49.86%），R1-Qwen 极度脆弱（仅 13.60%）
- **结论：开箱即用的 LLM 目前无法可靠用于企业敏感数据管理**

## 亮点与洞察

- **首次定义 SA 问题并提供评估框架**：将企业访问控制需求抽象为形式化定义，填补了 LLM 安全评估的空白
- **评估设计精巧**：简洁输出格式（数据值或固定拒绝语）使字符串匹配评分可行，99.9% 自动化率极大降低评估成本
- **"越简单越暴露问题"的实验设计智慧**：故意使用最简单的表格数据和最简规则集，如果模型在这里都失败，复杂场景更不可能成功
- **失败模式分类有临床价值**：区分 leak/refusal/error 三种失败方式对实际部署的风险评估至关重要

## 局限与展望

- 仅使用表格数据，未测试非结构化文档（如混合敏感信息的 PDF 报告）
- 访问规则极其简单（2 条规则），真实企业的 RBAC 策略远更复杂
- 未探索专门针对 SA 的对齐微调（如 RLHF 增加权限感知奖励信号）
- Lying 场景仅使用基础对抗手段，未引入低资源语言攻击或自动化 prompt injection
- 未评估 RAG 系统的端到端 SA 表现（当前是直接将数据放入 prompt）

## 相关工作与启发

- **vs SudoLM (Liu et al. 2024)**：SudoLM 用静态 sudo key 实现二元公开/私密分割，仅支持单一认证场景，Access Denied Inc 支持多角色细粒度字段级权限
- **vs 安全基准 (JailbreakBench, HarmBench)**：关注有害内容生成的防御，而 SA 关注的是"正确的数据给正确的人"
- **vs Zeng et al. (2024) 风险分类体系**：314 个风险类别中最相关的是机密性和隐私违规，但 SA 的核心特征（访问权限执行和选择性信息分发）未被充分覆盖
- **启发**：LLM 在企业场景中的部署需要超越"无害性"的安全评估维度——权限感知是刚需但被严重忽视的能力

## 评分

⭐⭐⭐⭐ (4/5)

- 新颖性 ⭐⭐⭐⭐⭐：首次定义 SA 问题并构建评估框架，定义清晰、问题重要
- 实验充分度 ⭐⭐⭐⭐：7 个模型 + 6 个特征 + 2 个特殊场景，但仅表格数据一种形式
- 写作质量 ⭐⭐⭐⭐：形式化定义完整，实验分析深入，failure mode 分类有洞察
- 实用价值 ⭐⭐⭐⭐：framework 可复用，暴露了真实风险，但缺乏解决方案

<!-- RELATED:START -->

## 相关论文

- [SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home Agents](../../ICLR2026/llm_evaluation/simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_agents.md)
- [Spectral Sensitivity Estimation with an Uncalibrated Diffraction Grating](../../ICCV2025/llm_evaluation/spectral_sensitivity_estimation_with_an_uncalibrated_diffraction_grating.md)
- [ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [HalluLens: LLM Hallucination Benchmark](hallulens_llm_hallucination_benchmark.md)
- [Movie101v2: Improved Movie Narration Benchmark](movie101v2_improved_movie_narration_benchmark.md)

<!-- RELATED:END -->
