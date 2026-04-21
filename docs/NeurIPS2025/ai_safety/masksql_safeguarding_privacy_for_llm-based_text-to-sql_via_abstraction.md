---
title: >-
  [论文解读] MaskSQL: Safeguarding Privacy for LLM-Based Text-to-SQL via Abstraction
description: >-
  [NeurIPS 2025][AI安全][Text-to-SQL] 提出 MaskSQL 框架，通过提示抽象（abstraction）将敏感的表名、列名和数据值替换为抽象符号后发送给远程 LLM，结合本地 SLM 做 schema linking 和 SQL 重建，在保护隐私同时超越 SLM-only 方案的 SQL 生成精度。
tags:
  - NeurIPS 2025
  - AI安全
  - Text-to-SQL
  - 隐私保护
  - 提示抽象
  - LLM-SLM混合
  - 数据库安全
---

# MaskSQL: Safeguarding Privacy for LLM-Based Text-to-SQL via Abstraction

**会议**: NeurIPS 2025  
**arXiv**: [2509.23459](https://arxiv.org/abs/2509.23459)  
**代码**: https://github.com/sepideh-abedini/MaskSQL  
**领域**: AI 安全 / 隐私保护  
**关键词**: Text-to-SQL, 隐私保护, 提示抽象, LLM-SLM混合, 数据库安全

## 一句话总结

提出 MaskSQL 框架，通过提示抽象（abstraction）将敏感的表名、列名和数据值替换为抽象符号后发送给远程 LLM，结合本地 SLM 做 schema linking 和 SQL 重建，在保护隐私同时超越 SLM-only 方案的 SQL 生成精度。

## 研究背景与动机

**领域现状**：Text-to-SQL 领域 LLM（GPT-4 等）性能最强，但需通过远程 API 访问，暴露敏感的数据库 schema 和用户数据。SLM 可本地部署但复杂查询性能差。

**现有痛点**：(1) 加密方法（HE/MPC）计算开销巨大不适合 LLM 规模；(2) 差分隐私降低效用；(3) 现有提示消毒方法（Portcullis、PP-TS）针对通用文本，无法保持 SQL 的 schema 对齐。

**核心矛盾**：LLM 需要看到 schema 信息才能生成正确 SQL，但 schema 本身就是敏感信息。

**切入角度**：LLM 生成 SQL 需要的是问题与 schema 间的**映射关系**，具体名称不重要——可以用抽象符号替代。

**核心 idea**：三阶段管线——Abstraction（本地 SLM 做 schema linking + 符号替换）→ SQL Generation（远程 LLM 处理抽象提示）→ SQL Reconstruction（本地还原 + 自纠错）。

## 方法详解

### 整体框架

输入 NL 问题 $\mathcal{Q}$ 和数据库 schema $\mathcal{S}$ → 本地 SLM 做 schema ranking/filtering → SLM 做 value/reference linking → 抽象符号替换 → 远程 LLM 生成抽象 SQL $\mathcal{Y}'$ → 本地还原 + SLM 自纠错 → 可执行 SQL $\mathcal{Y}$。

### 关键设计

1. **Schema Ranking and Filtering**

    - 功能：用 RoBERTa cross-encoder 对 schema 元素按与问题的相关性排序，保留 top-k 表和 top-j 列
    - 核心思路：减少发送给 LLM 的 schema 规模，降低噪声和暴露面
    - 设计动机：真实数据库常有上百表，大部分与查询无关

2. **Value and Reference Linking**

    - 功能：用本地 SLM 识别问题中对应数据库值和 schema 元素的 token
    - 核心思路：三步——(1) SLM 识别值 token（如"New York Hospital"）；(2) 映射到对应列/表；(3) 识别引用 token（如"patients"→Patients 表）
    - 设计动机：准确的 linking 是抽象质量的关键——遗漏的 token 将暴露敏感信息

3. **Abstracting Concrete Tokens**

    - 功能：用双射映射将表名→$T_i$、列名→$C_i$、值→$V_i$，生成 $\mathcal{Q}'$ 和 $\mathcal{S}'$
    - 核心思路：简单文本替换 + 附加值-列对应说明（如"$V_1$ is a value of column $C_7$"）
    - 设计动机：保留了问题与 schema 的逻辑映射关系，LLM 仍能理解查询意图

4. **SQL Reconstruction + Self-Correction**

    - 功能：用符号查找表还原抽象 SQL，再用本地 SLM 纠错
    - 核心思路：执行还原后的 SQL，将结果连同原始问题一起提供给 SLM 进行最终纠错
    - 设计动机：抽象噪声可能导致值类型错误（如 string "positive" vs numeric 1）

### 损失函数 / 训练策略

无需训练。SLM（Qwen-2.5-7B-Instruct）用于本地推理，LLM（GPT-4.1）用于远程 SQL 生成。

## 实验关键数据

### 主实验（BIRD 基准 300 复杂查询）

| 框架 | 执行准确率 | Token 用量 | 隐私保护 |
|------|----------|-----------|---------|
| Direct Prompting + Qwen-7B | 34.33% | 1,380 | ✓ 本地 |
| DAIL-SQL + Qwen-7B | 44.33% | 3,492 | ✓ 本地 |
| Fine-Tuned MSc-SQL | 48.33% | 8,342 | ✓ 本地 |
| DIN-SQL + GPT-4.1 | 65.66% | 24,812 | ✗ 暴露 |
| **MaskSQL ($\Psi_C$)** | **62.00%** | ~5,000 | **✓ 部分保护** |
| **MaskSQL ($\Psi_F$)** | **58.33%** | ~5,000 | **✓ 全保护** |

### 消融实验

| 组件 | 准确率 | 说明 |
|------|--------|------|
| 无 Schema Filtering | 52.0% | 噪声过多 |
| 无 Self-Correction | 54.7% | 值类型错误 |
| 无 Value Linking | 50.3% | 抽象不完整 |
| **完整 MaskSQL** | **58.33%** | **全保护** |

### 关键发现

- MaskSQL（全保护）超越所有 SLM-only 方案 10pp+，接近 LLM 无保护性能（差 7.3pp）
- Masking Recall 达 92%+，Re-identification Score（对手无法还原率）达 85%+
- 部分策略 $\Psi_C$ 仅保护人名/地点/职业时准确率更高（62%），体现隐私-效用权衡

## 亮点与洞察

- **抽象 vs 编辑/泛化**：抽象保留映射关系而非删除或模糊化，是 SQL 任务的最优隐私保护方式。
- **混合架构**：SLM 负责信任环节（schema linking、还原），LLM 负责推理环节（SQL 生成），各取所长。
- **可控隐私策略**：用户可自定义 $\Psi$ 决定保护哪些元素，灵活适应不同法规要求。

## 局限与展望

- Schema linking 依赖 SLM 质量，复杂 schema 下可能遗漏
- 仅在 BIRD 基准的 300 条复杂查询上评估，规模有限
- 对手模型假设较弱（仅考虑 re-identification），未考虑侧信道攻击

## 相关工作与启发

- **vs Portcullis**：通用 NER 消毒，不保持 SQL schema 对齐
- **vs DIN-SQL**：DIN-SQL 暴露全部 schema，MaskSQL 在接近性能下保护隐私

## 评分
- 新颖性: ⭐⭐⭐⭐ 提示抽象的思路清晰实用
- 实验充分度: ⭐⭐⭐⭐ 隐私+效用双指标评估
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义精确，管线清晰
- 价值: ⭐⭐⭐⭐⭐ 解决了 LLM 部署中的核心隐私问题

<!-- RELATED:START -->

## 相关论文

- [SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications](swe-sql_illuminating_llm_pathways_to_solve_user_sql_issues_in_real-world_applica.md)
- [The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Text](../../ICML2025/ai_safety/the_canarys_echo_auditing_privacy_risks_of_llm-generated_synthetic_text.md)
- [InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy](invisibleink_high-utility_and_low-cost_text_generation_with_differential_privacy.md)
- [AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text](agentstealth_reinforcing_large_language_model_for_anonymizing_user-generated_tex.md)
- [CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model Stealing in Edge Deployment](coreguard_safeguarding_foundational_capabilities_of_llms_against_model_stealing_.md)

<!-- RELATED:END -->
