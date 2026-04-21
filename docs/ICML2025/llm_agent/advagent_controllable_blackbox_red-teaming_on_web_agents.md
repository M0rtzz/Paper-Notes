---
title: >-
  [论文解读] AdvAgent: Controllable Blackbox Red-teaming on Web Agents
description: >-
  [ICML 2025][LLM Agent][Web Agent 安全] 提出 AdvAgent，一个基于强化学习（DPO）的黑盒红队测试框架，训练一个对抗 prompter 模型自动生成不可见的 HTML 对抗 prompt，注入网页后可误导 GPT-4V 驱动的 Web Agent 执行攻击者指定的目标动作（如将买微软股票改为买英伟达），在 440 个任务上达到 97.5% 攻击成功率，且对现有防御手段仍保持 88.8% 以上的有效性。
tags:
  - ICML 2025
  - LLM Agent
  - Web Agent 安全
  - 红队测试
  - 对抗攻击
  - 黑盒攻击
  - 强化学习
  - 提示学习
  - DPO
---

# AdvAgent: Controllable Blackbox Red-teaming on Web Agents

**会议**: ICML 2025  
**arXiv**: [2410.17401](https://arxiv.org/abs/2410.17401)  
**代码**: [https://ai-secure.github.io/AdvAgent/](https://ai-secure.github.io/AdvAgent/)  
**领域**: LLM Agent  
**关键词**: Web Agent 安全, 红队测试, 对抗攻击, 黑盒攻击, 强化学习, Prompt 注入, DPO

## 一句话总结

提出 AdvAgent，一个基于强化学习（DPO）的黑盒红队测试框架，训练一个对抗 prompter 模型自动生成不可见的 HTML 对抗 prompt，注入网页后可误导 GPT-4V 驱动的 Web Agent 执行攻击者指定的目标动作（如将买微软股票改为买英伟达），在 440 个任务上达到 97.5% 攻击成功率，且对现有防御手段仍保持 88.8% 以上的有效性。

## 研究背景与动机

**领域现状**：基于 LLM/VLM 的通用 Web Agent（如 SeeAct）能够自主与真实网站交互，完成金融交易、电商购物、医疗操作等高风险任务。这些 Agent 通过解析网页截图和 HTML 内容来理解页面并执行用户指令。

**安全隐患**：Web Agent 拥有对敏感资源的访问权限和自主决策能力，一旦被攻击者利用，可能导致严重后果——例如在股票交易中被误导购买错误标的、在电商中下单错误商品等。

**现有攻击方法的局限**：
   - **白盒方法**（Wu et al., 2024a）：需要访问 Agent 权重进行梯度优化，在实际部署中不可行
   - **手动设计方法**（Wu et al., 2024c; Liao et al., 2024）：需要人工启发式编写攻击指令，成本高、扩展性差
   - **针对 LLM/VLM 的自动攻击**（Zou et al., 2023; Guo et al., 2024）：缺乏对 Agent 交互场景的灵活性，在黑盒跨模型迁移时效果有限

**核心矛盾**：如何在完全黑盒（不访问 Agent 权重或 logits）的条件下，自动、高效且可控地生成对抗 prompt 来攻击 Web Agent？

**切入角度**：将对抗 prompt 生成建模为序列生成问题，通过 RL（DPO）利用黑盒 Agent 反馈来训练一个 prompter 模型，使其学会生成既有效又隐蔽的对抗性 HTML 注入内容。

**核心 idea**：用 DPO 强化学习训练对抗 prompter，从黑盒 Agent 的成功/失败反馈中学习，自动生成可控的隐形网页对抗注入。

## 方法详解

### 整体框架

AdvAgent 的攻击流程分为三个核心阶段：

1. **对抗 Prompt 生成**：训练好的对抗 prompter 模型接收攻击目标描述，自动生成对抗字符串
2. **网页注入**：将生成的对抗字符串注入目标网页的不可见 HTML 字段（如隐藏的 `<div>` 或不可见属性），确保不改变页面的视觉渲染
3. **Agent 被误导**：Web Agent（如 SeeAct）在处理被注入的网页时，读取到隐藏的对抗内容，被误导执行攻击者指定的目标动作

攻击场景示例：用户让 Agent 购买微软股票，攻击者在网页中注入不可见指令后，Agent 转而购买英伟达股票。

### 关键设计

1. **两阶段训练范式（Two-Stage Training Paradigm）**

    - **Stage 1 — 监督微调（SFT Warm-up）**：
        - 使用手工设计的成功攻击 prompt 作为种子数据
        - 对预训练语言模型进行监督微调，使其学习对抗 prompt 的基本模式和结构
        - 目的：为 prompter 建立初始的对抗 prompt 生成能力，避免 RL 阶段从零开始探索
    - **Stage 2 — DPO 强化学习优化**：
        - 使用 SFT 模型生成大量候选对抗 prompt
        - 将候选 prompt 注入网页，观察黑盒 Agent 的行为反馈
        - 根据攻击是否成功，将 prompt 分为正样本（成功攻击）和负样本（攻击失败）
        - 使用 Direct Policy Optimization（DPO）进行偏好学习：$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$
        - 其中 $y_w$ 为攻击成功的 prompt（正样本），$y_l$ 为攻击失败的 prompt（负样本）
        - 核心优势：无需访问 Agent 权重或 logits，仅需观察 Agent 的最终行为作为奖励信号

2. **隐蔽注入机制（Stealthy Injection Mechanism）**

    - 功能：将对抗 prompt 嵌入网页 HTML 的不可见元素中
    - 核心思路：利用 HTML 中对用户不可见但 Agent 可读取的字段，如：
        - 隐藏的 `<input type="hidden">` 元素
        - CSS 设置为不可见的 `<div style="display:none">`
        - HTML 属性字段（如 `aria-label`、`title` 等）
    - 注入后网页在浏览器中的视觉渲染完全不变，普通用户无法察觉
    - Agent 在解析 HTML 或处理页面元素时会读取到这些隐藏内容

3. **攻击可控性（Controllability）**

    - 功能：允许攻击者灵活修改攻击目标而无需重新训练
    - 核心思路：训练好的 prompter 模型学会了对抗 prompt 的通用模式，攻击者可以通过简单修改输入条件（如将目标公司从 NVIDIA 改为 Tesla）来生成针对新目标的对抗 prompt
    - 无需为每个新攻击目标重新优化，大幅降低攻击成本
    - 这一特性使 AdvAgent 比固定 prompt 的手动方法更具实用价值

4. **攻击目标设定（Targeted Attack Formulation）**

    - 攻击者指定一个**目标动作** $a^*$，该动作与用户原始请求不同
    - 攻击成功的判定标准：Agent 最终执行的动作 $a_t$ 与目标动作 $a^*$ 一致
    - 攻击成功率（ASR）= 成功攻击的任务数 / 总任务数

## 实验关键数据

### 主实验：跨域攻击成功率

在 4 个不同网站领域、共 440 个任务上的实验结果：

| 方法 | Shopping | Finance | Social Media | Travel | 平均 ASR |
|------|----------|---------|-------------|--------|----------|
| No Attack (Baseline) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Manual Prompt | ~60% | ~55% | ~50% | ~45% | ~52.5% |
| Transfer Attack (CLIP) | ~40% | ~35% | ~38% | ~30% | ~35.8% |
| **AdvAgent (Ours)** | **~98%** | **~97%** | **~98%** | **~97%** | **97.5%** |

核心发现：AdvAgent 在所有领域上均大幅超越基线方法，平均 ASR 达到 97.5%，远超手动设计的 prompt（~52.5%）和基于 CLIP 迁移的方法（~35.8%）。

### 防御实验：现有防御手段的效果

| 防御方法 | 防御策略描述 | AdvAgent ASR | 防御有效性 |
|---------|-------------|-------------|-----------|
| 无防御 | — | 97.5% | — |
| Prompt 加固 | 在 Agent 系统 prompt 中添加安全指令 | ~92% | 极弱 |
| 输入过滤 | 检测并过滤可疑 HTML 内容 | ~90% | 弱 |
| Paraphrasing | 对网页文本内容进行改写 | ~89% | 弱 |
| 综合防御 | 多种防御手段叠加 | 88.8% | 有限 |

核心发现：即使叠加多种防御策略，AdvAgent 仍保持 88.8% 以上的攻击成功率，说明现有基于 prompt 的防御机制对此类攻击几乎无效。

### 消融实验摘要

| 组件 | 移除后 ASR 变化 | 说明 |
|------|---------------|------|
| DPO 训练 | 显著下降 | 证明 RL 反馈对黑盒攻击至关重要 |
| SFT 预热 | 中等下降 | SFT 提供了关键的初始化 |
| HTML 字段变化 | 97.0% (微降) | 攻击对注入位置鲁棒 |
| 不同 Agent backbone | 高 ASR 保持 | 攻击具有跨模型迁移性 |

## 亮点与洞察

1. **黑盒 + 高效率的统一**：AdvAgent 是首个在完全黑盒设定下实现 >95% 攻击成功率的 Web Agent 红队框架。通过 DPO 这种无需在线交互的离线 RL 方法，既避免了白盒依赖，又大幅降低了训练成本。
2. **隐蔽性设计精巧**：利用 HTML 的不可见字段注入对抗内容，真正做到"人眼不可见、Agent 被欺骗"。这暴露了当前 Web Agent 在处理 HTML 内容时缺乏安全意识的根本缺陷。
3. **可控性是关键创新**：不同于固定的对抗样本，AdvAgent 的 prompter 模型可以灵活调整攻击目标，一次训练即可适配多种攻击场景，这在实际红队测试中具有巨大实用价值。
4. **防御失效的重要警示**：实验证明 prompt 加固、输入过滤等常见防御手段对此类攻击几乎无效（ASR 仍 >88%），这为 Agent 安全社区敲响了警钟——需要更根本性的防御范式。
5. **从 LLM 攻击到 Agent 攻击的范式跨越**：将 DPO 这一对齐技术"反向"用于攻击，从单轮文本攻击扩展到多步交互式 Agent 场景，是方法论上的有趣创新。

## 局限与展望

1. **攻击假设较强**：攻击者需要能够修改目标网页的 HTML 内容，这在实际场景中需要控制中间人代理或已渗透目标网站，限制了攻击的适用范围。
2. **单一 Agent 评估**：主要在 SeeAct（GPT-4V）上评估，对其他架构类型（如基于 HTML-only 的 Agent、非 OpenAI 模型驱动的 Agent）的泛化性需要更多验证。
3. **防御研究不足**：论文侧重攻击效果展示，对如何有效防御讨论有限。未来应探索更根本的防御手段，如 HTML 内容完整性校验、可信执行环境等。
4. **伦理风险**：虽然论文目的是暴露漏洞以促进安全，但公开高效攻击框架可能被恶意利用，需要配套负责任的披露机制。
5. **静态网页假设**：实验中的网页环境相对静态，对于动态加载内容（AJAX/SPA）的现代网页，注入和攻击的有效性有待验证。

## 相关工作与启发

- **Web Agent 框架**：SeeAct (Zheng et al., 2024) 提出截图 + HTML 双输入的 Web Agent 架构，是当前 SOTA。MindAct (Deng et al., 2024) 和 WebArena (Zhou et al., 2023) 提供了重要的基准环境。
- **Agent 安全攻击**：
    - Yang et al. (2024) 和 Wang et al. (2024) 研究了白盒环境下的后门攻击
    - Wu et al. (2024c) 和 Liao et al. (2024) 探索了手动 prompt 注入攻击
    - Wu et al. (2024a) 提出了梯度优化方法但受限于白盒假设
- **LLM 对抗攻击**：GCG (Zou et al., 2023)、AutoDAN (Guo et al., 2024) 等方法为 LLM 自动红队提供了基础，但不直接适用于 Agent 多步交互场景。
- **DPO 的新用途**：本文将 DPO 从"对齐"反转为"攻击"，为 RL-based 对抗训练提供了新思路。启发：RLHF/DPO 框架在安全攻防两侧都有广阔应用空间。
- **启发性思考**：Web Agent 的安全问题本质上是一个**信任边界**问题——Agent 无法区分合法内容与注入内容。未来可能需要类似 Content Security Policy 的机制来建立 Agent 可信内容边界。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将 DPO 用于黑盒 Web Agent 攻击，提出隐蔽可控的注入框架
- 实验充分度: ⭐⭐⭐⭐ — 440 个任务、4 个领域、多种防御对比、详细消融
- 写作质量: ⭐⭐⭐⭐ — 结构清晰、动机充分、图示直观
- 价值: ⭐⭐⭐⭐⭐ — 暴露 Web Agent 重大安全漏洞，对 Agent 安全领域有重要警示价值

<!-- RELATED:START -->

## 相关论文

- [Generative AI Agents for Controllable and Protected Content Creation](../../NeurIPS2025/llm_agent/generative_ai_agents_for_controllable_and_protected_content_creation.md)
- [Web-Shepherd: Advancing PRMs for Reinforcing Web Agents](../../NeurIPS2025/llm_agent/web-shepherd_advancing_prms_for_reinforcing_web_agents.md)
- [Explorer: Scaling Exploration-Driven Web Trajectory Synthesis for Multimodal Web Agents](../../ACL2025/llm_agent/explorer_scaling_exploration-driven_web_trajectory_synthesis_for_multimodal_web_.md)
- [Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning in Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_in_web_agents.md)
- [Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)

<!-- RELATED:END -->
