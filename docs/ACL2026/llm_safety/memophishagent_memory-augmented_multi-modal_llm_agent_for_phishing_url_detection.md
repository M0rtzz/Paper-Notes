---
title: >-
  [论文解读] MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection
description: >-
  [ACL 2026][LLM安全][钓鱼检测] 提出 MemoPhishAgent（MPA），首个专为钓鱼URL检测设计的记忆增强多模态LLM智能体，通过5个专用工具的动态编排和情景记忆系统复用历史推理轨迹，在公开基准上召回率提升13.6%，在真实社交媒体数据上提升20%…
tags:
  - "ACL 2026"
  - "LLM安全"
  - "钓鱼检测"
  - "LLM智能体"
  - "情景记忆"
  - "多模态推理"
  - "工具调用"
---

# MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection

**会议**: ACL 2026  
**arXiv**: [2602.21394](https://arxiv.org/abs/2602.21394)  
**代码**: [GitHub](https://github.com/XuanChen-xc/MemoPhishAgent)  
**领域**: 安全AI  
**关键词**: 钓鱼检测, LLM智能体, 情景记忆, 多模态推理, 工具调用

## 一句话总结

提出 MemoPhishAgent（MPA），首个专为钓鱼URL检测设计的记忆增强多模态LLM智能体，通过5个专用工具的动态编排和情景记忆系统复用历史推理轨迹，在公开基准上召回率提升13.6%，在真实社交媒体数据上提升20%，并已部署生产环境每周处理约6万高风险URL。

## 研究背景与动机

**领域现状**：钓鱼攻击持续演变，传统防御（静态黑名单、手工启发式规则）对新域名和新手法覆盖不足。基于品牌-域名映射的参考方法改进了鲁棒性但维护成本高，对新品牌和子域名反应滞后。

**现有痛点**：（1）现有LLM方案多为提示式确定性流水线，缺乏自适应证据收集能力；（2）工具使用固定流程（如先OCR再品牌匹配再域名验证），不能根据当前证据状态动态调整；（3）无记忆系统，无法复用历史调查经验，重复分析类似钓鱼模式效率低。

**核心矛盾**：钓鱼攻击是非平稳的——攻击者不断变换策略，但防御系统是无记忆的，每次从零开始分析。

**本文目标**：构建一个能动态调整证据收集策略、从历史调查中学习、并适用于生产环境的钓鱼检测智能体。

**切入角度**：将钓鱼检测建模为多步推理过程——模拟人类专家的调查行为，动态选择工具收集证据。

**核心idea**：5个钓鱼专用多模态工具 + ReAct推理循环 + 情景记忆系统（存储/检索历史推理轨迹），三者结合实现自适应、可学习的钓鱼检测。

## 方法详解

### 整体框架

MPA接收可疑URL列表，每个URL通过Agent处理：（1）动态选择5个专用工具收集多模态证据（文本+视觉+外部知识）；（2）在ReAct循环中进行多步推理，基于当前证据状态决定下一步行动；（3）利用情景记忆检索相似历史案例，加速判断或提供exemplar引导。最终输出"恶意"或"良性"判定。

### 关键设计

1. **5个钓鱼专用工具**：

    - 功能：提供互补的多模态证据
    - 核心思路：三方面覆盖——多模态证据（Crawl Content提取Markdown文本 + Check Screenshot全页截图分析 + Check Image细粒度图像检查）、外部知识（Intelligent Search构建证据驱动的搜索查询获取最新信息）、嵌套攻击面（Extract Targets提取重定向目标和子链接进行深层检查）
    - 设计动机：通用工具不适配钓鱼场景；5个工具覆盖文本/视觉/链接/外部知识四个维度

2. **情景记忆系统**：

    - 功能：存储、检索和复用历史推理轨迹
    - 核心思路：用LLM从页面提取紧凑关键词（如"apple login", "wallet connect"），嵌入向量索引。检索top-k近邻，按三级策略使用——无匹配时完整ReAct循环、部分匹配时作为exemplar引导、完全匹配时直接多数投票。随部署增长，记忆越来越丰富
    - 设计动机：钓鱼模式存在大量重复（同一攻击模板针对不同受害者），记忆系统将重复调查转化为快速决策

3. **三级记忆使用策略**：

    - 功能：平衡速度和可靠性
    - 核心思路：k'=0→完整推理（未见模式）；0<k'<k→历史轨迹作为in-context exemplar（部分相似）；k'≥k→直接多数投票（高度相似）
    - 设计动机：避免记忆主导推理——应作为上下文指导而非替代思考

## 实验关键数据

### 主实验

| 方法 | TR-OP Recall | DynaPD Recall | 速度(s/URL) |
|------|------------|-------------|-----------|
| MPA | **93.4%** | **93.6%** | **4.46** |
| PhishLLM | ~80% | ~88% | 14.2 |
| MLLM | ~82% | ~85% | 5.1 |
| URLTran | ~86% | — | 2.8(含训练) |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 完整MPA | 93.4% Recall | 所有组件 |
| - 记忆系统 | -27% Recall | 记忆贡献最大 |
| - 工具设计 | 性能下降 | 专用工具优于通用工具 |
| 提示式基线 | 较差 | 固定流程不如自适应选择 |

### 关键发现
- 情景记忆贡献高达27%的召回率提升，且不增加额外计算开销
- MPA是所有方法中最快的（4.46s/URL），因为记忆系统跳过了大量重复分析
- 在真实社交媒体数据（SocPhish）上召回率提升20%，说明在真实场景中优势更大
- 生产部署每周处理~60K高风险URL，实现91.44%召回率
- URL shorteners和平台托管路径（如sites.google.com）是传统方法的盲区，MPA通过多模态工具克服

## 亮点与洞察
- **已部署生产环境**：不仅是学术工作，已在Amazon生产环境中保护百万用户，说服力强
- **情景记忆的效果惊人**：27%召回率提升且不增加计算——因为对重复模式直接投票，减少了LLM调用
- **工具设计专业且互补**：5个工具从文本/视觉/搜索/链接四维度收集证据
- **三级记忆策略平衡了效率和准确性**：对未见模式完整分析，对已见模式快速决策

## 局限与展望
- **依赖外部LLM API**：Claude-3-Sonnet的延迟和成本
- **记忆系统可能被污染**：如果早期错误判断被存入记忆，可能影响后续决策
- **仅关注钓鱼URL**：其他安全威胁（如恶意软件分发）未覆盖
- 未来方向：记忆自我修正机制、扩展到更多安全威胁类型、轻量化本地模型替代API

## 相关工作与启发
- **vs PhishLLM**：使用LLM做品牌提取+意图识别，但仍是固定流程；MPA动态选择工具
- **vs Cao et al. (2025)**：多模态LLM钓鱼检测但固定证据获取流程，无记忆
- **vs 通用Agent框架**：使用通用工具和推理，MPA的钓鱼专用工具更有效

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个钓鱼专用的记忆增强LLM Agent，情景记忆系统设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 两个公开基准+真实社交媒体数据+生产部署验证，消融全面
- 写作质量: ⭐⭐⭐⭐ 威胁模型定义清晰，系统架构图直观
- 价值: ⭐⭐⭐⭐⭐ 已在生产环境验证，对安全AI有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] Privacy-R1: Privacy-Aware Multi-LLM Agent Collaboration via Reinforcement Learning](privacy-r1_privacy-aware_multi-llm_agent_collaboration_via_reinforcement_learnin.md)
- [\[ACL 2025\] Unveiling Privacy Risks in LLM Agent Memory](../../ACL2025/llm_safety/mextra_agent_memory_privacy.md)
- [\[ACL 2026\] XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts](xmark_reliable_multi-bit_watermarking_for_llm-generated_texts.md)
- [\[CVPR 2026\] Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](../../CVPR2026/llm_safety/multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)

</div>

<!-- RELATED:END -->
