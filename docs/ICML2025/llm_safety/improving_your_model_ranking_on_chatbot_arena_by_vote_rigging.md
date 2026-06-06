---
title: >-
  [论文解读] Improving Your Model Ranking on Chatbot Arena by Vote Rigging
description: >-
  [ICML 2025][LLM安全][Chatbot Arena] 论文揭示 Chatbot Arena 的众包投票机制可被恶意操纵：提出 target-only 和 omnipresent 两类投票操纵策略，其中 omnipresent 策略利用 Bradley-Terry 评分系统的全局耦合特性…
tags:
  - "ICML 2025"
  - "LLM安全"
  - "Chatbot Arena"
  - "Vote Rigging"
  - "Elo Rating"
  - "Bradley-Terry"
  - "Leaderboard Manipulation"
  - "LLM Evaluation Security"
---

# Improving Your Model Ranking on Chatbot Arena by Vote Rigging

**会议**: ICML 2025  
**arXiv**: [2501.17858](https://arxiv.org/abs/2501.17858)  
**领域**: ai_safety（LLM 评估安全 / 排行榜操纵）  
**关键词**: Chatbot Arena, Vote Rigging, Elo Rating, Bradley-Terry, Leaderboard Manipulation, LLM Evaluation Security

## 一句话总结
论文揭示 Chatbot Arena 的众包投票机制可被恶意操纵：提出 target-only 和 omnipresent 两类投票操纵策略，其中 omnipresent 策略利用 Bradley-Terry 评分系统的全局耦合特性，仅需操纵数百票即可将目标模型排名提升 15 位，凸显当前 LLM 评估平台的安全脆弱性。

## 研究背景与动机

### 1. 领域现状
Chatbot Arena 是当前最具影响力的 LLM 评估平台之一，采用众包成对对战模式：
1. 用户提交 prompt，两个匿名模型分别生成回复。
2. 用户投票选择更好的回复（或选择平局）。
3. 平台基于所有投票计算 Elo/Bradley-Terry 评分，形成公开排行榜。
4. 排行榜结果被广泛引用，对模型商业推广有重大影响。

### 2. 现有防御措施
Chatbot Arena 已部署了多层防御机制：

**长度/风格偏差控制**：消除用户偏好中的表面特征偏差（Dubois et al., 2024; Li et al., 2024a）。

**异常投票检测**：识别 bot 行为和统计异常投票模式（Chiang et al., 2024）。

**Prompt 分类与数据清洗**：对 prompt 进行分类管理（Li et al., 2024b, c）。

**身份泄露过滤**：如果回复中暴露了模型身份，该投票作废（Chiang et al., 2024）。

### 3. 核心矛盾
矛盾在于：
1. 排行榜需要开放众包投票来保证覆盖面和多样性。
2. 但开放投票意味着恶意行为者可以注入虚假投票。
3. 现有防御主要针对无意偏差和简单 bot，未考虑精心设计的策略性操纵。

### 4. 论文要解决的问题
作者要回答一个关键安全问题：
1. Chatbot Arena 的投票机制是否可以被策略性操纵来改变目标模型排名？
2. 操纵成本有多低？（需要多少恶意投票？）
3. 现有防御机制能否有效应对？

### 5. 切入角度
论文从 Elo/Bradley-Terry 评分系统的数学结构入手，发现评分计算的全局耦合性可被利用——即使操纵的投票不涉及目标模型，也能间接影响其排名。

### 6. 核心 idea（一句话）
利用 Bradley-Terry 模型中所有投票对所有模型评分的全局影响特性，设计 omnipresent 操纵策略，在任意对战中策略性投票，间接提升目标模型排名。

## 方法详解

### 整体框架
论文提出两大类攻击策略，形成递进式威胁分析：

**攻击前提**：攻击者可以在 Chatbot Arena 上发起对战并投票，但无法控制匹配到的模型。

**流程**：
1. 攻击者确定目标模型 $m_t$。
2. 在新对战中识别参战模型身份。
3. 根据策略决定投票方向（为哪个模型投票）。
4. 累积足够操纵票数，改变 $m_t$ 的排名。

### 策略一：Target-Only Rigging（目标专注策略）

**功能**: 只在涉及目标模型 $m_t$ 的对战中进行操纵投票。

**核心思路**:
1. 在每次对战中，先识别是否有 $m_t$ 参战。
2. 识别手段包括：
    - **水印检测**（Zhao et al., 2024）：如果 $m_t$ 的输出嵌入了水印，可通过检测水印判断身份。
    - **二分类器**（Huang et al., 2025）：训练一个分类器区分 $m_t$ 与其他模型的输出。
3. 一旦确认 $m_t$ 参战，一律投 $m_t$ 胜。

**实际效率问题**:
- Chatbot Arena 上有超过 190 个模型。
- 任意一场对战涉及 $m_t$ 的概率仅约 1%。
- 提升一个排名位次可能需要超过 10,000 次操纵投票。
- 这种高投票量容易被 IP 限频等简单防御手段阻止。

### 策略二：Omnipresent Rigging（全局操纵策略）

**功能**: 在任意对战中（即使不涉及 $m_t$）策略性投票，间接提升 $m_t$ 排名。

**核心思路**:
论文利用 Bradley-Terry 模型的关键数学特性：

$$P(m_i \succ m_j) = \frac{\exp(\beta_i)}{\exp(\beta_i) + \exp(\beta_j)}$$

其中 $\beta_i$ 是模型 $m_i$ 的 BT 分数。所有模型的 $\beta$ 通过对全部投票进行最大似然估计联合求解：

$$\hat{\boldsymbol{\beta}} = \arg\max_{\boldsymbol{\beta}} \sum_{k=1}^{N} \log P(m_{w_k} \succ m_{l_k} \mid \boldsymbol{\beta})$$

**关键洞察**: 由于所有 $\beta_i$ 是联合优化的，对任意一场对战的投票改变都会影响全局似然函数的最优解，从而影响所有模型（包括 $m_t$）的评分。

**策略设计**:
1. 攻击者无需等待 $m_t$ 出现在对战中。
2. 在每一场对战中，攻击者识别参战双方模型身份。
3. 根据投票方向对 $m_t$ 排名的间接影响，选择最优投票。
4. 具体而言，投票应倾向于：
    - 让排名低于 $m_t$ 的模型赢（压低 $m_t$ 的竞争对手）。
    - 让排名高于 $m_t$ 的模型输（拉低 $m_t$ 上方的模型）。

**效率优势**:
- 每一场对战都可作为攻击机会，利用率从 ~1% 提升到 ~100%。
- 仅需数百票即可实现显著排名提升。

### 模型身份识别

攻击的前提是在匿名对战中识别参战模型身份。论文讨论两种方法：

1. **水印（Watermarking）**: 模型提供方可在输出中嵌入可检测的水印。攻击者（如模型提供方自身）可利用已知水印来识别自家模型。
2. **二分类器（Binary Classifier）**: 攻击者收集目标模型和其他模型的输出样本，训练分类器。对新回复运行分类器即可判断是否为目标模型。

### 理论分析

论文的核心理论贡献在于证明了 BT 评分系统的全局耦合性：

1. **投票影响传播**: 在 BT 框架下，新增一票 $(m_i \succ m_j)$ 会改变最大似然估计的解，进而影响所有模型的 $\beta$ 值。
2. **排名敏感性**: 排名靠近 $m_t$ 的模型的对战投票对 $m_t$ 的排名影响最大。
3. **非对称效应**: 操纵策略可以同时提升 $m_t$ 的排名并降低竞争对手排名，形成"双倍效果"。

## 实验关键数据

### 实验设置
- **数据集**: Chatbot Arena Notebook 中约 170 万条历史投票记录。
- **目标模型**: Phi-3-small-8k-Instruct（作为演示目标）。
- **评估指标**: 排名变化（rank improvement）vs 操纵票数。
- **模拟方式**: 在历史数据基础上注入新的操纵票，重新计算 BT 评分。

### 主实验：操纵效果对比

| 策略 | 操纵票数 | 排名提升 | 每票效率（rank/vote） | 实际可行性 |
|---|---|---|---|---|
| 无操纵（正常投票加入 ~27,000 票） | 0 | +1 位 | — | 基线 |
| Target-Only Rigging | 数千票 | 轻微提升 | 低 | 低（仅 ~1% 对战涉及目标模型） |
| Omnipresent Rigging | 数百票 | +15 位 | 高 | 高（每场对战均可利用） |

### 策略效率与防御分析

| 防御机制 | 能否阻止 Target-Only | 能否阻止 Omnipresent | 原因分析 |
|---|---|---|---|
| IP 限频（每日投票上限） | 部分有效 | 效果有限 | Omnipresent 所需票数远少于阈值 |
| 异常投票模式检测 | 可检测集中投票 | 难以检测 | Omnipresent 票分散在不同对战中 |
| 模型身份隐藏增强 | 增加识别难度 | 增加识别难度 | 两种策略都依赖身份识别 |
| 投票一致性校验 | 部分有效 | 效果有限 | Omnipresent 的单票看似正常 |

### 关键发现
1. **效率差距悬殊**: Omnipresent 策略比 Target-Only 高效一到两个数量级。
2. **隐蔽性更强**: Omnipresent 的操纵投票分散在各种对战中，单票行为无异常，难以通过模式检测发现。
3. **排名系统脆弱**: 约 170 万票构建的排名，仅需数百票即可显著扰动，暴露了 BT 评分系统在对抗场景下的脆弱性。
4. **防御困难**: 作者评估了多种防御手段，结论是没有银弹，需要多层防御持续迭代。

## 亮点与洞察

### 亮点 1：揭示评估系统的根本性脆弱
论文不是发现某个特定 bug 或实现漏洞，而是指出 Bradley-Terry 评分机制本身的数学结构允许全局操纵。这意味着问题不能通过简单 patch 修复，需要重新思考评估系统设计。

### 亮点 2：Omnipresent 策略的巧妙设计
利用"每票影响全局"这一看似直觉但未被充分重视的性质，将低效的 target-only 攻击转化为高效的全局攻击。攻击利用率从 ~1% 跃升到 ~100%，是经典的"换个视角就突破瓶颈"。

### 亮点 3：攻防博弈视角完整
论文不仅提出攻击，还评估了多种防御手段的有效性和局限性，为平台方提供了务实的安全改进方向。

### 可迁移启发
1. **其他排行榜安全**: 类似机制可用于分析 HELM、Open LLM Leaderboard 等平台的操纵风险。
2. **推荐系统安全**: 协同过滤中的评分全局耦合问题与本文发现高度类似。
3. **联邦学习中的投毒攻击**: "间接影响全局模型"的思路可迁移到联邦聚合场景。
4. **去中心化投票安全**: 对 DAO 治理等区块链投票场景有参考价值。

## 局限与展望

### 作者侧局限
1. **模型身份识别假设较强**: 论文假设攻击者能可靠识别匿名对战中的模型身份，但实际中这一步可能并不容易，尤其是相似能力水平的模型之间。
2. **静态排名模拟**: 实验基于历史数据离线模拟，未考虑平台可能动态调整采样策略、更新频率等实时因素。
3. **单一评分机制**: 论文主要分析标准 BT 模型，Chatbot Arena 实际使用的可能包含正则化、category-specific 评分等变体。

### 读者侧补充局限
1. **缺乏真实平台验证**: 出于伦理考虑，攻击未在真实 Chatbot Arena 上执行，模拟实验可能低估平台的实际防御能力。
2. **成本模型不完整**: 论文关注操纵票数，但未详细分析获取 Chatbot Arena 账号、规避检测等实际运营成本。
3. **多攻击者场景未考虑**: 如果多个模型提供方同时进行操纵，博弈均衡会如何变化？
4. **评分机制演进**: Chatbot Arena 的评分方法在持续更新，论文的攻击策略对最新版本的有效性需重新验证。

### 具体可改进方向
1. 设计基于密码学承诺的投票协议，使投票在统计前不可被策略性选择。
2. 引入差分隐私机制，在评分计算中添加噪声以增强鲁棒性。
3. 开发基于投票时序分析的异常检测方法，识别策略性投票模式。
4. 探索非 BT 的替代排名机制（如 TrueSkill 变体），评估其抗操纵能力。

## 相关工作与对比

### vs 传统 Elo 操纵研究
国际象棋等领域已有 Elo 操纵的讨论，但通常涉及直接"放水"。
本文的贡献在于将问题扩展到超多模型的大规模众包场景，且提出了不需要目标模型参与的间接操纵策略。

### vs LLM 评估可靠性研究
Dubois et al. (2024) 和 Li et al. (2024a) 关注无意偏差（如长度偏好），属于"噪声"问题。
本文关注的是故意操纵（adversarial），是更高层级的安全威胁。

### vs Benchmark 操纵
传统 benchmark 操纵（如过拟合测试集）需要修改模型本身。
本文的攻击不需要修改目标模型，只需在投票端注入恶意票，攻击面更广泛。

### 对当前研究的启发
1. LLM 评估系统需要从"防噪声"升级到"防对抗"。
2. 众包评估的治理框架应借鉴安全领域的攻防思维。
3. 排行榜设计需要考虑博弈论视角下的激励兼容性。

## 复现实操要点
1. 从 Chatbot Arena Notebook 下载历史投票数据（~170 万条）。
2. 实现标准 BT 评分计算作为基线。
3. 实现模型身份识别模块（水印检测 / 二分类器）。
4. 在历史数据基础上模拟注入操纵票，观察排名变化曲线。
5. 关键超参数：操纵票数、目标模型选择、投票策略。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐（5/5）首次系统性揭示 Chatbot Arena 投票操纵的可行性，omnipresent 策略设计巧妙。
- 实验充分度: ⭐⭐⭐⭐☆（4/5）基于 170 万历史数据的模拟实验规模充分，但缺乏真实平台验证。
- 写作质量: ⭐⭐⭐⭐☆（4/5）问题动机清晰，攻击策略递进式展开，防御讨论全面。
- 价值: ⭐⭐⭐⭐⭐（5/5）对 LLM 评估平台安全具有重要警示意义，可推动评估系统防御体系升级。

## 引用信息
Rui Min, Tianyu Pang, Chao Du, Qian Liu, Minhao Cheng, Min Lin. **Improving Your Model Ranking on Chatbot Arena by Vote Rigging**. ICML 2025.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Is Your Model Fairly Certain? Uncertainty-Aware Fairness Evaluation for LLMs](is_your_model_fairly_certain_uncertainty-aware_fairness_evaluation_for_llms.md)
- [\[ACL 2025\] Improving Model Factuality with Fine-grained Critique-based Evaluator](../../ACL2025/llm_safety/improving_model_factuality_with_fine-grained_critique-based_evaluator.md)
- [\[NeurIPS 2025\] Attention! Your Vision Language Model Could Be Maliciously Manipulated](../../NeurIPS2025/llm_safety/attention_your_vision_language_model_could_be_maliciously_manipulated.md)
- [\[ICML 2025\] Improving LLM Safety Alignment with Dual-Objective Optimization](improving_llm_safety_alignment_with_dual-objective_optimization.md)
- [\[ICML 2025\] Improving Continual Learning Performance and Efficiency with Auxiliary Classifiers](improving_continual_learning_performance_and_efficiency_with_auxiliary_classifie.md)

</div>

<!-- RELATED:END -->
