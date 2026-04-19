---
title: >-
  [论文解读] Paper Copilot: Tracking the Evolution of Peer Review in AI Conferences
description: >-
  [ICLR 2026][视频理解][同行评审] 构建 Paper Copilot——跨数十个 AI/ML 会议的同行评审持久数字档案与分析平台：通过 OpenReview API、网页抓取、社区贡献三源混合收集评审数据，实时归档评分时间快照（含 rebuttal 前后动态变化），揭示 ICLR 2025 年决策熵反常下降——评审体系从概率性分层转向近确定性分数驱动决策的结构性变化，并通过 LLM 驱动的作者-机构元数据提取支持人才轨迹追踪。
tags:
  - ICLR 2026
  - 视频理解
  - 同行评审
  - 评分动态
  - 决策熵
  - 会议统计
  - 数据集
  - LLM 元数据提取
---

# Paper Copilot: Tracking the Evolution of Peer Review in AI Conferences

**会议**: ICLR 2026  
**arXiv**: [2510.13201](https://arxiv.org/abs/2510.13201)  
**代码**: [项目页面](https://papercopilot.com)  
**领域**: 科学计量 / 评审分析  
**关键词**: 同行评审, 评分动态, 决策熵, 会议统计, 数据集, LLM 元数据提取

## 一句话总结

构建 Paper Copilot——跨数十个 AI/ML 会议的同行评审持久数字档案与分析平台：通过 OpenReview API、网页抓取、社区贡献三源混合收集评审数据，实时归档评分时间快照（含 rebuttal 前后动态变化），揭示 ICLR 2025 年决策熵反常下降——评审体系从概率性分层转向近确定性分数驱动决策的结构性变化，并通过 LLM 驱动的作者-机构元数据提取支持人才轨迹追踪。

## 研究背景与动机

**领域现状**：AI/ML 顶会投稿量年超 10,000 篇（ICLR 2025 达 11,672 篇），同行评审压力空前。部分会议（ICLR/NeurIPS）采用 OpenReview 开放评审，但多数（CVPR/AAAI/ICCV）评审封闭。评审维度也从单一评分扩展到 soundness/correctness/novelty/contribution 等多维度评估。

**现有痛点**：(1) 评审数据分散在 Twitter/Reddit/知乎/小红书等社交平台，碎片化严重；(2) OpenReview 会覆盖旧版评审——rebuttal 期间的评分变化历史是不可恢复的信息损失；(3) 跨会议、跨年评审标准比较缺乏统一数据源和工具；(4) 作者在 rebuttal 期（仅 1-2 周）缺乏统计参考，难以判断分数水平和 rebuttal 价值。

**核心矛盾**：评审过程是研究透明度的核心，但现有基础设施无法支持系统性的评审动态追踪和纵向分析。

**本文目标** 构建统一的评审数据收集、归档和分析平台，支持跨会议纵向分析和实时评分动态追踪。

**切入角度**：三源混合数据策略最大化覆盖率，实时归档时间快照保存不可恢复的历史数据。

**核心 idea**：将分散、短暂的 AI 会议评审信息统一为持久化、结构化、可分析的数字档案，构建评审过程的"元科学基础设施"。

## 方法详解

### 整体框架

Paper Copilot 是模块化系统：venue 配置层 → 数据收集管道（多源 assigners + worker pool + parallel bots） → 清洗标准化 → 版本化数据集（JSON 格式，每篇论文 30+ 字段） → 后端存储/API（LAMP/MySQL） → 前端可视化分析（WordPress + 自定义 JS）。支持快速接入新会议，仅需最小配置。

### 关键设计

1. **三源混合数据收集管道**:
    - 功能：从异构数据源统一收集评审数据，最大化覆盖率
    - 核心思路：(1) OpenReview API——定时脚本拉取开放会议（ICLR/NeurIPS）的评分、置信度、评论，存储带时间戳快照以追踪 rebuttal 前后变化；(2) 网页抓取——针对无 API 会议（CVPR/AAAI）提取 accepted 论文、作者、元数据；(3) 社区 opt-in 贡献——封闭评审会议的作者自愿提交评审（累计 6,584 条有效记录），约 60% 作者同意公开匿名化评分
    - 设计动机：任何单一数据源都无法覆盖所有会议。对封闭评审会议，社区贡献是唯一可行的数据来源

2. **评审动态时间快照归档**:
    - 功能：实时归档评审分数在讨论/rebuttal 阶段的完整演变过程
    - 核心思路：对 ICLR 2024/2025 每日爬取评审快照，记录每个 reviewer 在每个时间点的所有维度评分（rating, confidence, soundness, contribution, presentation）。OpenReview 官方只保留最终版本，旧版被覆盖不可恢复。Paper Copilot 是互联网上唯一保存完整评审时序数据的公开档案。通过 score footprint 可视化单篇论文在多维度、多 reviewer 上的评分演化轨迹
    - 设计动机：评审"过程"（分数如何变化、共识如何形成）与最终结果同样重要，but previously nobody systematically preserved it

3. **LLM 驱动的作者-机构元数据提取**:
    - 功能：大规模自动提取论文作者的结构化元组 $(a_i, \mathcal{A}_i, e_i)$（姓名, 机构集合, 邮箱）
    - 核心思路：使用 GLM 系列模型从 camera-ready PDF 中提取元数据。定义 mismatch indicator $\mathbf{1}(x,y) = 1$ if $|x| \neq |y|$，评估结构一致性。评估指标为 Success Rate $= 1 - \frac{1}{|\mathcal{D}|} \sum_{i} (\delta_{\text{aff}}^i \lor \delta_{\text{email}}^i \lor \delta_{\text{parse}}^i)$。glm-4-plus 在 ~70K 论文上达 86.82% 成功率（$\delta_{\text{aff}} = 5.01\%$, $\delta_{\text{email}} = 4.94\%$, $\delta_{\text{parse}} = 0.81\%$）
    - 设计动机：绝大多数会议不提供结构化的作者-机构映射，这是机构/国家级分析的前提

### 分析方法

**决策熵分析**：量化 AC 决策确定性。对年份 $t$ 和分数区间 $b$，定义 $H_{t,b} = -\sum_{s \in \{\text{Reject, Poster, ...}\}} p_{t,b,s} \log p_{t,b,s}$，加权平均得 $\bar{H}_t = \sum_b w_{t,b} H_{t,b}$。通常随投稿量对数增长 $\bar{H}_t \approx a \log X_t + b$，但 2025 年出现强负残差，表明决策敏感度 $\kappa_{2025}$ 异常高——AC 更依赖平均分做确定性分层。

## 实验关键数据

### ICLR 2017-2025 评审演化分析

| 指标 | 发现 | 量化证据 |
|------|------|----------|
| 投稿量增长 | 490 → 11,672（24 倍） | AC 数从 31 增至 823 |
| 决策熵趋势 | 通常随投稿量对数增长 | $\bar{H}_t \approx a\log X_t + b$ |
| 2025 结构性变化 | 决策熵反常下降 | $\text{resid}_{2025}$ 强负偏离拟合线 |
| Rebuttal 分数变化 | 54.8% 论文 overall rating 变化 | soundness 等仅 ~10-13% 变化 |
| 共识演变 | 讨论开始时分歧先增大后收敛 | Oral 收敛最快，Reject 保持高分歧 |
| 边界不对称 | 高于均值时低方差利于接收 | 低于均值时高方差反而利于接收 |

### 社区透明度调查（4 个会议 1,860 份回复）

| 会议 | 回复数 | 同意公开匿名评审 | 比例 |
|------|:------:|:------:|:------:|
| CVPR 2025 | 357 | 191 | 53.5% |
| ICML 2025 | 1,034 | 628 | 60.7% |
| ICCV 2025 | 254 | 151 | 59.4% |
| ACL 2025 | 215 | 145 | 67.4% |
| **总计** | **1,860** | **1,115** | **59.9%** |

### LLM 元数据提取准确率

| 模型 | $\delta_{\text{aff}}$ | $\delta_{\text{email}}$ | $\delta_{\text{parse}}$ | Success Rate |
|------|:------:|:------:|:------:|:------:|
| glm-4-plus | 5.01% | 4.94% | 0.81% | **86.82%** |
| glm-4-air | 49.98% | 17.11% | 0.51% | 44.73% |
| glm-4-flash | 76.39% | 43.27% | 0.62% | 18.52% |
| glm-3-turbo | 76.07% | 32.34% | 1.34% | 20.90% |

### 关键发现

- **2025 年的结构性转折**：尽管投稿量最大，决策熵反而下降——AC 更依赖平均分做录用决策，从概率性 tiering 转向近确定性 mapping
- **Rebuttal 的双重角色**：对 borderline 论文放大分数变化，对强论文驱动共识形成
- **Spotlight 向 Oral 收敛**：平均评分在 tier 间分离加剧，Spotlight 逐年靠近 Oral
- **评分变化维度分化**：overall rating 是 rebuttal 最频繁变化的维度，soundness 等变化远少

## 亮点与洞察

- **唯一的评审时序档案**：OpenReview 覆盖讨论过程中的旧版本，Paper Copilot 实时归档保存了全网唯一的完整评审动态数据——这一种不可替代的历史记录
- **决策熵分析框架**：引入 ordered-logit 模型 + 决策熵定量刻画评审体系演化，将散落的社区直觉观察提升为可量化的元科学分析
- **伦理设计的完整性**：详细讨论数据来源合规、隐私保护、再识别风险、dual-use 防范，符合研究伦理最佳实践

## 局限与展望

- 封闭评审会议依赖社区自愿提交——存在自选择偏差（高分/低分作者提交倾向可能不同）
- LLM 提取机构元数据的非零错误率可能影响机构排名分析可靠性
- 作者轨迹分析可能被用于招聘等高风险评估——存在 dual-use 风险
- 系统是持续更新的 live platform，精确复现某一时间点状态不可行

## 相关工作与启发

- **vs PeerRead (Kang et al., 2018)**: 14.7K 论文但限于特定 venue 和时间快照，不支持纵向动态追踪
- **vs MOPRD (Lin et al., 2023)**: 多学科但不涵盖 AI 会议的 rebuttal 动态和评分时序
- **vs CSRankings**: 关注机构排名但更新慢、数据源不透明、完全不含评审数据

## 评分

- 新颖性: ⭐⭐⭐⭐ 独特的三源数据策略和评审时序归档，决策熵分析框架新颖
- 实验充分度: ⭐⭐⭐⭐ ICLR 2017-2025 大规模纵向分析，发现充实且量化扎实
- 写作质量: ⭐⭐⭐⭐ 系统描述清晰，伦理讨论详尽完整
- 价值: ⭐⭐⭐⭐⭐ 对 AI 研究社区有基础设施级贡献，数据集+平台的组合价值远超单篇论文

<!-- RELATED:START -->

## 相关论文

- [REVOLVE: Optimizing AI Systems by Tracking Response Evolution in Textual Optimization](../../ICML2025/video_understanding/revolve_optimizing_ai_systems_by_tracking_response_evolution_in_textual_optimiza.md)
- [AnveshanaAI: A Multimodal Platform for Adaptive AI/ML Education through Automated Question Generation and Interactive Assessment](anveshanaai_a_multimodal_platform_for_adaptive_aiml_education_through_automated_.md)
- [Log Probability Tracking of LLM APIs](log_probability_tracking_of_llm_apis.md)
- [The Expressive Limits of Diagonal SSMs for State-Tracking](the_expressive_limits_of_diagonal_ssms_for_state-tracking.md)
- [Stop Tracking Me! Proactive Defense Against Attribute Inference Attack in LLMs](stop_tracking_me_proactive_defense_against_attribute_inference_attack_in_llms.md)

<!-- RELATED:END -->
