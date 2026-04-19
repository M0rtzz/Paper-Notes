---
title: >-
  [论文解读] Judging by the Rules: Compliance-Aligned Framework for Modern Slavery Statement Monitoring
description: >-
  [AAAI 2026][合规验证] 提出以"合规对齐法官"（CA-Judge）为核心的训练框架，利用规则级对齐反馈训练 3B 参数的 CALLM 模型，使其生成基于法定条款的可追溯合规判断理由，在现代奴役声明的句子级合规分类任务上超越 GPT-4o 和 DeepSeek-R1。
tags:
  - AAAI 2026
  - 合规验证
  - 现代奴役法
  - LLM-as-Judge
  - 规则对齐
  - GRPO
---

# Judging by the Rules: Compliance-Aligned Framework for Modern Slavery Statement Monitoring

**会议**: AAAI 2026  
**arXiv**: [2511.07803](https://arxiv.org/abs/2511.07803)  
**代码**: [GitHub](https://github.com/mila-ai4h/aims-reasoning-alignment)  
**领域**: NLP / AI for Social Good / 法律合规  
**关键词**: 合规验证, 现代奴役法, LLM-as-Judge, 规则对齐, GRPO

## 一句话总结

提出以"合规对齐法官"（CA-Judge）为核心的训练框架，利用规则级对齐反馈训练 3B 参数的 CALLM 模型，使其生成基于法定条款的可追溯合规判断理由，在现代奴役声明的句子级合规分类任务上超越 GPT-4o 和 DeepSeek-R1。

## 研究背景与动机

全球超过 5000 万人受现代奴役影响。多国已立法（如澳大利亚、英国、加拿大的《现代奴役法》），要求企业发布年度声明披露供应链中的奴役风险评估和应对措施。然而这些声明数量庞大（全球超过 8 万份），内容质量参差不齐——许多声明措辞模糊、结构混乱，将合规内容与企业宣传混杂。当前完全依赖人工审核的模式无法规模化执行，大量违规行为因此逃脱监管。

NLP 为规模化合规监测提供了技术路径，但高风险合规场景对模型提出了超越准确率的要求：输出必须**可追溯**（traceable），即审计人员能将模型判断追溯到具体法定条款。现有的 LLM 应用往往将复杂的监管评估简化为二分类，缺乏结构化的法律推理支撑。Chain-of-Thought 推理虽提供了中间步骤，但近期研究表明 CoT 并不能可靠反映模型的真实决策过程，不应作为可解释性手段。

**核心矛盾**：合规验证本质上是**规则匹配问题**——需要判断文本声明是否满足明确定义的法规要求。现有 LLM 方法要么作为通用判断器缺乏特定领域规则意识，要么在训练中未将法规条款结构化融入。本文的切入角度是：将法规条款提炼为"关键规则"（key rules），用这些规则同时指导模型训练和输出评估，让模型生成的不是自圆其说的解释，而是**供人类审计员验证的规则对齐输出**。

## 方法详解

### 整体框架

框架分两个阶段：（1）**关键规则提取**：将每个合规准则翻译为结构化的自然语言评判标准（rubrics），明确什么构成合规/不合规；（2）**规则对齐训练**：用 CA-Judge 对模型输出进行多维度合规评分，以此作为奖励信号，通过 GRPO（Group Relative Policy Optimization）微调 CALLM 使其生成规则一致的输出。

### 关键设计

1. **关键规则提取（Key Rule Extraction）**:

    - 功能：将法定合规准则转化为确定性的自然语言规则集
    - 核心思路：由领域专家根据《澳大利亚现代奴役法》为每个合规准则编写关键规则。每条规则明确说明什么是有效回应、什么是无效回应。例如 C4 补救准则的关键规则包括"必须描述具体的补救措施"、"不能仅使用泛化的企业语言"等
    - 设计动机：使抽象法律条款转化为可操作的评判标准，同时作为训练目标和评估锚点

2. **合规对齐法官（CA-Judge）**:

    - 功能：评估模型生成的理由与关键规则的对齐程度
    - 核心思路：使用 JudgeLRM-7B（一个专门为判断任务训练的 LLM）作为评估器。给定模型的推理理由、预测标签和对应准则的关键规则，CA-Judge 在六个维度上评估并返回标量分数 $R_{\text{judge}} \in [0, 1]$。六个评估维度覆盖可审计性（准确性、忠实度）、可解释性（清晰度、证据使用）和可靠性（一致性、可验证性）
    - 设计动机：模拟人类合规审查员的评估方式——逐条规则检查覆盖度、具体性和清晰度。比通用的 LLM-as-Judge 更有针对性

3. **CALLM 训练流程（基于 GRPO）**:

    - 功能：用复合奖励信号微调基座模型（Qwen2.5-3B-Instruct）
    - 核心思路：每个训练实例包含目标句子、上下文和关键规则，模型生成多个候选输出（每实例 4 个），每个输出包含结构化推理和最终预测。奖励函数为四项加权和：
        - 格式匹配奖励：$R_{\text{format}}(i) = \mathbf{1}\{\text{match}(c_i)\}$
        - XML 标签奖励：$R_{\text{xml}}(i) = \min[1, \max[0, \xi_r \sum_{t \in \mathcal{T}} \mathbf{1}\{t \in c_i\} - \xi_p \Delta_i]]$
        - 正确性奖励：$R_{\text{corr}}(i) = \mathbf{1}\{\hat{y}_i = y_i\}$
        - 规则对齐奖励：$R_{\text{judge}}(i) = \text{CAJudge}(\text{rules}, \text{reasoning}, \hat{y})$
        - 总奖励：$R_{\text{total}} = \lambda_1 R_{\text{format}} + \lambda_2 R_{\text{xml}} + \lambda_3 R_{\text{corr}} + \lambda_4 R_{\text{judge}}$，各 $\lambda=1$
    - 训练分两阶段：Phase 1 用格式+正确性奖励训练 2000 步得到 FT 模型；Phase 2 加入 CA-Judge 奖励继续训练 5000 步得到 CALLM
    - 设计动机：选择 GRPO 而非 PPO 是因为无需额外价值网络，计算高效且适合低资源场景；选择 CA-Judge 而非人类反馈是因为合规领域缺乏大规模偏好数据

### 损失函数 / 训练策略

- GRPO 通过组内相对排名计算优势函数，无需显式奖励模型
- 两阶段训练：先学格式和正确性（建立基础能力），再学规则对齐（精细化推理质量）
- 学习率 $5 \times 10^{-6}$，batch size 8，cosine 学习率衰减，warmup ratio 0.1，$\beta=0.04$
- Phase 1 使用 2×A100 80GB（3-4小时/准则），Phase 2 使用 3×A100（~40小时/准则）

## 实验关键数据

### 主实验（F1 分数，9 个合规准则）
| 准则 | GPT-4o ZS | GPT-4o FS | DeepSeek-R1 | CALLM FT（无Judge） | CALLM（完整） |
|------|-----------|-----------|-------------|---------------------|---------------|
| Approval | 0.855 | 0.843 | 0.837 | 0.755 | **0.786** |
| Signature | 0.409 | 0.636 | 0.250 | 0.524 | **0.692** |
| C2 Structure | 0.619 | 0.658 | 0.678 | 0.535 | **0.572** |
| C3 Risk Description | 0.422 | 0.450 | 0.495 | 0.564 | **0.712** |
| C4 Mitigation | 0.709 | 0.664 | 0.700 | 0.714 | **0.749** |
| C4 Remediation | 0.552 | 0.601 | 0.529 | 0.397 | **0.570** |
| Overall (macro) | 0.559 | 0.617 | 0.548 | 0.559 | **0.639** |

### 消融实验（训练阶段的影响）
| 配置 | 参数量 | Overall F1 | 说明 |
|------|--------|-----------|------|
| Base PT（预训练） | 3B | 0.293 | 无微调基线 |
| Base FT（格式+正确性奖励） | 3B | 0.559 | Phase 1 只用表面奖励 |
| CALLM（完整框架） | 3B | 0.639 | Phase 1 + Phase 2 含 CA-Judge |

### 跨司法管辖区泛化（训练 AU，测试 AU/UK/CA）
| 模型 | 参数 | AU | UK | CA |
|------|------|-----|-----|-----|
| GPT-4o FS CoT | 1800B | 0.617 | 0.573 | 0.614 |
| DeepSeek-R1 | 671B | 0.548 | 0.505 | 0.550 |
| CALLM | 3B | **0.639** | **0.620** | **0.617** |

### 关键发现
- 3B 参数的 CALLM 在 macro-F1 上超越 1800B 的 GPT-4o 和 671B 的 DeepSeek-R1，说明领域对齐比参数规模更重要
- CA-Judge 奖励带来的提升在复杂准则（C3 Risk Description: +0.148, C4 Remediation: +0.173）上尤为显著
- 人类偏好研究中 CALLM 被 74.1%（200/270）的投票选为更优，7/9 准则上人类判断与 CA-Judge 评分方向一致
- 跨司法管辖区泛化良好，仅在澳大利亚数据上训练即可在英国和加拿大数据上保持优势
- 错误分析发现 CALLM 在简单准则（如 Approval）上略逊 GPT-4o，原因是 CALLM 对规则遵循更严格，对数据中破损的目标句子更敏感

## 亮点与洞察
- "合规验证是规则匹配问题"的问题定义非常精准，将法律合规与 NLP 技术桥接起来
- CA-Judge 作为中间评估层的设计巧妙——既提供训练信号，又模拟人类审查流程
- 不将 CoT 输出视为解释性方法，而是视为供人类验证的结构化素材，这种定位更诚实也更实用
- 用 3B 小模型战胜千亿级模型，证明了"领域特化对齐 > 盲目扩大参数"的观点
- 框架设计具有通用性，可迁移到其他法规合规领域（如 GDPR、环境法）

## 局限与展望
- 效果依赖关键规则的质量和覆盖度，不完整或模糊的规则可能导致不一致的奖励信号
- GRPO 对批次组成敏感：若所有候选输出都较差，模型可能强化次优模式
- CA-Judge 仍是专家判断的近似，可能在对抗性提示下产生偏差或幻觉
- 仅聚焦英语合规文档，多语言场景未探索
- 受计算限制未训练更大模型（7B/14B），作者推测更大模型可能带来额外提升
- 人类评估规模有限（5名志愿者），未包含专业合规审查员

## 相关工作与启发
- 与 DeepSeek-R1 的 GRPO 训练范式相似，但本文将奖励信号从通用推理扩展到领域特定的法规规则
- 对 LLM-as-Judge 的创新应用：从通用评估转化为领域专精评估器
- 对 AI for Social Good 领域的启发：高风险社会应用中必须将规范性知识嵌入模型训练循环
- 对合规自动化的启发：完全自动化审批风险太大，人机协作（human-in-the-loop）是更负责任的路径

## 评分
- 新颖性: ⭐⭐⭐⭐ （CA-Judge 引导规则对齐的训练框架有新意，但基础技术组件相对成熟）
- 实验充分度: ⭐⭐⭐⭐ （跨司法管辖区测试和人类评估很好，但人类评估规模有限）
- 写作质量: ⭐⭐⭐⭐⭐ （动机清晰，框架图示丰富，伦理讨论充分）
- 价值: ⭐⭐⭐⭐ （社会价值高，技术框架可迁移，但应用领域相对小众）

<!-- RELATED:START -->

## 相关论文

- [Expressive Temporal Specifications for Reward Monitoring](expressive_temporal_specifications_for_reward_monitoring.md)
- [Automated Reproducibility Has a Problem Statement Problem](automated_reproducibility_has_a_problem_statement_problem.md)
- [Nonparametric Modern Hopfield Models](../../ICML2025/others/nonparametric_modern_hopfield_models.md)
- [A Switching Framework for Online Interval Scheduling with Predictions](a_switching_framework_for_online_interval_scheduling_with_pr.md)
- [RcAE: Recursive Reconstruction Framework for Unsupervised Industrial Anomaly Detection](rcae_recursive_reconstruction_framework_for_unsupervised_industrial_anomaly_dete.md)

<!-- RELATED:END -->
