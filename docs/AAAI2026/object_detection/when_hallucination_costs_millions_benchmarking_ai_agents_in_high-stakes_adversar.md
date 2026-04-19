---
title: >-
  [论文解读] When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets
description: >-
  [AAAI 2026][目标检测][AI Agent 评估] 提出 CAIA 基准测试，通过加密货币市场作为天然对抗性实验室，评估 17 个 SOTA 大模型在高风险对抗环境中的 agent 能力，揭示前沿模型仅达 67.4% 准确率（GPT-5）vs 人类 80%，并发现系统性工具选择灾难。
tags:
  - AAAI 2026
  - 目标检测
  - AI Agent 评估
  - 对抗性基准
  - 加密货币
  - 工具选择
  - 幻觉
---

# When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets

**会议**: AAAI 2026  
**arXiv**: [2510.00332](https://arxiv.org/abs/2510.00332)  
**代码**: [GitHub](https://github.com/SurfAI-CybertinoLab/CAIA)  
**领域**: 目标检测  
**关键词**: AI Agent 评估, 对抗性基准, 加密货币, 工具选择, 幻觉

## 一句话总结

提出 CAIA 基准测试，通过加密货币市场作为天然对抗性实验室，评估 17 个 SOTA 大模型在高风险对抗环境中的 agent 能力，揭示前沿模型仅达 67.4% 准确率（GPT-5）vs 人类 80%，并发现系统性工具选择灾难。

## 研究背景与动机

**AI 基准测试的盲区**：当前 AI 评测（GLUE、ImageNet、ICPC、IMO）在封闭世界中衡量能力，假设工具按预期运行、信息可信、其他 agent 合作。但真实世界的自主部署要求**对抗性鲁棒**——在充满不确定性、虚假信息和对抗性激励的开放系统中生存。

**基准分数≠部署就绪**：在推理基准上获得高分的 agent，可能仍会相信虚假新闻、购买被攻击的资产、上当受骗——因为其评估中从未涉及欺骗场景。随着 AI agent 越来越多地与不可信用户、真实资金和关键基础设施交互，这个漏洞构成安全隐患。

**为什么选择加密货币市场**：

**真实对抗环境**：匿名区块链使恶意行为者无需承担声誉后果；利润动机驱动复杂攻击策略；监管空白允许传统市场非法的欺骗手段。日常攻击包括蜜罐合约、闪电贷攻击、协调社交工程

**高风险不可逆**：无传统金融安全网，交易不可逆，智能合约执行最终化，2024 年损失超 300 亿美元

**可验证真相**：尽管对抗混乱，区块链提供完全透明性和不可变性——每笔交易永久记录，可加密验证

## 方法详解

### 整体框架

CAIA 基准的设计遵循"对抗优先"原则，包含 178 个时间锚定任务，评估 agent 在加密货币分析中的知识、规划和执行能力。

### 关键设计

#### 1. **质量保证三支柱**

- **知识**：评估加密原生概念的基础理解（AMM 机制、治理结构等），测试概念掌握而非定义记忆
- **规划**：评估复杂问题分解为可执行分析工作流的能力，要求 agent 在执行前指定工具选择和顺序
- **执行**：使用生产级 API（Etherscan、CoinGecko、DefiLlama）测试实际执行，评估技术能力和判断力

#### 2. **数据策划流水线（5 阶段）**

1. **自动过滤**：LLM 评审过滤不相关/模糊/不可回答的查询，保留 top 15%（~1000 任务）
2. **专家评审**：92 名领域专家，每任务至少 4 个评审，去极端平均取 top 200
3. **格式标准化**：统一格式，锚定区块号/时间戳，确保客观评估
4. **真相验证**：验证可复现的工具链调用方案，无法复现的任务移除 → 最终 178 任务
5. **分类**：6 个分析类别用于诊断

来源：超过 10,000 条来自 3,000+ 活跃用户的真实查询。

#### 3. **任务类别分布**

| 类别 | 数量 | 占比 | 重点 |
|------|------|------|------|
| 链上分析 | 77 | 43.3% | 交易模式、MEV、资金流向 |
| 项目发现 | 49 | 27.5% | 协议评估、安全分析 |
| 代币经济 | 23 | 12.9% | 激励设计、价值积累 |
| 交叉领域 | 14 | 7.9% | 多领域综合 |
| 趋势分析 | 8 | 4.5% | 时间模式、采用指标 |
| 通识知识 | 7 | 3.9% | 基础概念 |

### 实验设置

- **评估模型**：17 个 SOTA 模型（GPT-5、GPT-4.1、Claude Opus 4、Gemini 2.5 Pro、Grok 4、DeepSeek R1/V3.1、Llama 4、Qwen 3 等）
- **两种条件**：无工具（闭卷考试）vs 有工具（开卷考试，23 种专业工具）
- **Agent 框架**：标准 ReAct 框架，确保评估不受实现差异影响
- **人类基线**：16 名大学区块链社团/早期公司参与者（初级分析师），完成 10% 分层样本，平均准确率 **80%**
- **指标**：5 次独立运行的多数投票准确率 + Pass@1/Pass@5 + 成本效率

## 实验关键数据

### 主实验

**无工具条件**（所有模型灾难性失败）：

| 模型 | 多数投票准确率 | Pass@1 | Pass@5 |
|------|-------------|--------|--------|
| GPT-5 | 27.5% | 28.1% | 42.7% |
| Gemini 2.5 Pro | 22.5% | 20.2% | 29.8% |
| GPT-o3 | 20.8% | 22.5% | 29.2% |
| Claude Opus 4 | 13.5% | 13.5% | 16.9% |
| DeepSeek R1 | 20.8% | 21.9% | 35.4% |

**有工具条件**：

| 模型 | 多数投票准确率 | Pass@1 | Pass@5 | 平均成本 |
|------|-------------|--------|--------|---------|
| **GPT-5** | **67.4%** | 70.2% | 77.0% | $0.1154 |
| GPT-OSS 120B | 62.9% | 56.2% | 72.5% | $0.0066 |
| Grok 4 Fast | 61.2% | 57.9% | 71.9% | $0.0098 |
| Claude Sonnet 4 | 56.7% | 57.9% | 66.9% | $0.2291 |
| Claude Opus 4 | 57.3% | 59.6% | 71.9% | $1.1139 |
| **人类基线** | **~80%** | - | - | - |

### 消融实验

**工具使用分布（工具选择灾难）**：

| 工具类别 | 调用次数 | 占比 | 说明 |
|----------|---------|------|------|
| Google 搜索 | 11,626 | 49.6% | **模型偏好不可靠来源** |
| 专业区块链工具 | 8,351 | 35.6% | 包含正确答案的权威来源 |
| URL 获取 | 1,743 | 7.4% | - |
| Twitter 搜索 | 1,388 | 5.9% | 社交媒体操纵信息 |
| 代码执行 | 355 | 1.5% | - |

**成本-准确率权衡**：

| 模型 | 准确率 | 每次成本 | 性价比 |
|------|--------|---------|--------|
| Claude Opus 4 | 57.3% | $1.1139 | 最差 |
| GPT-OSS 120B | 62.9% | $0.0066 | **100倍性价比** |
| Grok 4 Fast | 61.2% | $0.0098 | 帕累托最优 |

### 关键发现

1. **基础能力缺口**：无工具时所有模型接近随机猜测（12-28%），而初级分析师日常能完成这些任务
2. **工具选择灾难**：模型 55.5% 的工具调用选择不可靠的网络搜索，即使专业区块链工具能直接提供正确答案
3. **Pass@k 的幻觉**：Gemini 2.5 Flash 从 39.3% (Pass@1) 到 62.4% (Pass@5) 的提升意味着模型本质在试错猜测，而非战略推理
4. **Twitter 搜索悖论**：单独使用准确率仅 6.6%，组合使用升至 40.7%，说明工具需要编排能力
5. **案例研究**（Task 49）：获取 Pump.fun 月度代币发行数据——一个简单 API 调用即可完成，17 个模型无一成功，全部陷入网络搜索→过时博客→Twitter 投机信息的级联失败

## 亮点与洞察

1. **对抗优先评估范式**：首次明确将主动欺骗、来源验证和对抗鲁棒性纳入 AI agent 核心能力评估
2. **Pass@k 指标的危险性**：高 Pass@k 不代表能力，而是试错——在不可逆金融决策中，这种行为极其危险
3. **开源模型的性价比**：GPT-OSS 120B 以 1/100 的成本达到接近前沿性能
4. **普适性洞察**：发现不仅限于加密货币——任何对抗性领域（网络安全、内容审核、医疗诊断）都面临类似挑战

## 局限与展望

1. 178 个任务的规模相对较小，且高度集中在加密领域
2. 人类基线仅 16 人完成 10% 样本，统计显著性有限
3. 未评估模型在获得反馈后的学习/适应能力
4. 分类系统（该论文被归入"目标检测"领域可能是错误分类，实际是 AI Agent 评估）
5. 缺乏对 agent 安全部署的具体改进建议
6. 依赖 ReAct 框架，其他 agent 架构可能表现不同

## 相关工作与启发

- GAIA（通用 AI 助手基准）测试任务完成但假设合作环境
- 本文扩展到对抗性维度，填补 AI 安全评估的重要空白
- 时间锚定设计参考了 RealTimeQA 等时间敏感评估方法
- 与 HELM（全面语言模型评估）呼应，但聚焦高风险场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首个高风险对抗性 AI agent 基准，立意深远
- 实验充分度: ⭐⭐⭐⭐⭐ — 17 个模型，双条件评估，多维分析
- 写作质量: ⭐⭐⭐⭐⭐ — 论述有力，洞察深刻，案例生动
- 价值: ⭐⭐⭐⭐⭐ — 对 AI agent 安全部署的警示意义重大

<!-- RELATED:START -->

## 相关论文

- [When Agents "Misremember" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems](../../ICLR2026/object_detection/when_agents_misremember_collectively_exploring_the_mandela_effect_in_llm-based_m.md)
- [Is Your Paper Being Reviewed by an LLM? Benchmarking AI Text Detection in Peer Review](../../ICLR2026/object_detection/is_your_paper_being_reviewed_by_an_llm_benchmarking_ai_text_detection_in_peer_re.md)
- [PASE: Leveraging the Phonological Prior of WavLM for Low-Hallucination Generative Speech Enhancement](pase_leveraging_the_phonological_prior_of_wavlm_for_low-hallucination_generative.md)
- [When Trackers Date Fish: A Benchmark and Framework for Underwater Multiple Fish Tracking](when_trackers_date_fish_a_benchmark_and_framework_for_underwater_multiple_fish_t.md)
- [Beyond Semantic Features: Pixel-Level Mapping for Generalized AI-Generated Image Detection](beyond_semantic_features_pixel-level_mapping_for_generalized_ai-generated_image_.md)

<!-- RELATED:END -->
