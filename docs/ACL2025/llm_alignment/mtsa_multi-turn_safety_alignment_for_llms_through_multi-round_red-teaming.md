---
title: >-
  [论文解读] MTSA: Multi-Turn Safety Alignment for LLMs through Multi-Round Red-Teaming
description: >-
  [ACL 2025][LLM对齐][LLM安全] 提出MTSA框架，通过思维引导的多轮红队攻击学习和基于未来奖励的多轮强化学习算法，在对抗迭代优化中同时提升红队模型的攻击能力和目标模型的安全防御能力，在多个安全基准上达到SOTA，且不损失模型通用性能。
tags:
  - ACL 2025
  - LLM对齐
  - LLM安全
  - 多轮对话
  - 红队攻击
  - 安全对齐
  - 强化学习
  - 对抗训练
  - 越狱攻击
---

# MTSA: Multi-Turn Safety Alignment for LLMs through Multi-Round Red-Teaming

**会议**: ACL 2025  
**arXiv**: [2505.17147](https://arxiv.org/abs/2505.17147)  
**代码**: [GitHub](https://github.com/yuki-younai/MTSA)  
**领域**: LLM对齐  
**关键词**: LLM安全, 多轮对话, 红队攻击, 安全对齐, 强化学习, 对抗训练, 越狱攻击

## 一句话总结

提出MTSA框架，通过思维引导的多轮红队攻击学习和基于未来奖励的多轮强化学习算法，在对抗迭代优化中同时提升红队模型的攻击能力和目标模型的安全防御能力，在多个安全基准上达到SOTA，且不损失模型通用性能。

## 研究背景与动机

随着ChatGPT等LLM的广泛部署，越狱攻击（jailbreak attack）成为严重的安全威胁。攻击者通过精心构造的输入绕过模型的安全防护，诱导其生成有害内容。

现有研究面临三个核心挑战：

**多轮对话中的安全脆弱性**：当前主流越狱技术主要针对单轮交互，但研究表明LLM在多轮对话中更容易被突破。在多轮对话中，恶意意图可以被分散隐藏在多轮交互中，使模型逐步产生有害内容——这比单轮攻击更隐蔽、更难防御。

**安全对齐数据的获取困难**：多轮越狱攻击的多样性使得通过人工方式收集足够的安全对齐数据非常昂贵。现有的自动红队方法缺乏交互性和策略性，无法适应复杂的对话环境。

**多轮安全对齐算法的缺失**：当前的安全对齐算法主要面向单轮场景。在多轮设置下，对话中的有害性是累积的——仅优化最后一轮对话会引入训练与测试分布之间的协变量偏移，显著降低安全对齐的泛化能力。

MTSA的核心创新在于：（1）用"思考再攻击"的策略提升红队模型的攻击多样性和有效性；（2）用基于未来奖励的多轮RL算法替代仅优化最后一轮的传统方法，提升安全对齐的鲁棒性。

## 方法详解

### 整体框架

MTSA包含两个阶段：

- **阶段一——思维引导的攻击学习**：构建Think-before-attack数据集，通过选择性微调训练红队模型的初始版本
- **阶段二——对抗迭代优化**：红队模型与目标模型交互生成对话数据，通过轨迹采样构建偏好数据，分别优化两个模型。经过多轮迭代，红队模型增强攻击策略，目标模型逐步强化防御

### 关键设计

#### 1. Think-before-Attack（思考再攻击）

将多轮红队攻击策略分为四类：
- **目的反转**（Purpose Inversion）：将查询意图翻转为相反方向，减轻即时伤害感
- **问题分解**（Query Decompose）：将复杂攻击目标拆分为多个危害性更低的子问题
- **角色扮演**（Role Play）：通过模拟不同角色或场景发起攻击
- **混合模式**（Mixed Mode）：灵活组合上述策略

关键创新：红队模型在每轮攻击前先**观察当前对话环境并给出思考**（thought），基于思考结果选择攻击策略。这使红队模型能根据目标模型的当前状态动态调整攻击方式，而非机械地执行预设模板。

初始化时，从合成数据中选择Top-k最低相似度数据进行SFT，确保后续迭代训练中能提升攻击多样性。

#### 2. 基于未来奖励的多轮RLHF

传统多轮任务仅在最后一轮对话上训练，这引入了训练与测试分布之间的协变量偏移。MTSA的核心算法创新是利用**未来状态的奖励**进行每一轮的动态偏好优化。

**目标模型优化损失**——使用相对偏好奖励：

$$\mathcal{L}_{tgt} = \left(\frac{1}{\eta}\left(\log\frac{\pi_{t+1}^{tgt}(r_h|s_h^{tgt})}{\pi_t^{tgt}(r_h|s_h^{tgt})} - \log\frac{\pi_{t+1}^{tgt}(r_h'|s_h^{tgt})}{\pi_t^{tgt}(r_h'|s_h^{tgt})}\right) - (R_{tgt}(s_{H+1}^{tgt}) - R_{tgt}(s_{H+1}'^{tgt}))\right)^2$$

核心思想：不使用critic model预测Q值（在多轮RL中效果差），而是通过轨迹采样获得终止状态的奖励值来替代Q值。这使得即使在中间轮次也能基于未来结果进行有效对齐。

**红队模型优化损失**——使用DPO直接偏好优化：

$$\mathcal{L}_{adv} = -\log\sigma\left(\beta\log\frac{\pi_{t+1}^{adv}(q_w|s_h^{adv})}{\pi_t^{adv}(q_w|s_h^{adv})} - \beta\log\frac{\pi_{t+1}^{adv}(q_l|s_h^{adv})}{\pi_t^{adv}(q_l|s_h^{adv})}\right)$$

其中 $q_w$ 和 $q_l$ 分别是根据终止状态奖励确定的正负样本。

#### 3. 对抗迭代优化

- **在线采样**：每轮迭代中，从攻击目标集合中采样子集，红队模型与目标模型进行多轮交互（最多5轮）
- **轨迹采样**：对目标模型的有害回复进行安全重写并采样独立轨迹；对红队模型进行拒绝采样和温度调整
- **奖励建模**：
    - 目标模型奖励 $R_{tgt}$：结合毒性 $R_{tox}$ 和有用性 $R_{help}$，使用ArmoRM多目标奖励模型平衡两者
    - 红队模型奖励 $R_{adv}$：结合安全分类器的不安全概率 $R_{safe}$ 和语义/文本多样性 $R_{div}$
- **交替更新**：每轮迭代先更新红队模型，再更新目标模型，共进行3轮迭代

## 实验关键数据

### 红队攻击能力——AdvBench ASR(%)

| 方法 | GPT-3.5 | GPT-4o | Claude3.5 | Llama2-7B | Vicuna-7B | Zephyr-beta | 平均 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GCG | 33.50 | 12.50 | 22.00 | 34.50 | 24.50 | 36.00 | 27.17 |
| PAP | 36.00 | 24.50 | 14.50 | 26.00 | 32.50 | 28.00 | 26.91 |
| PAIR | 57.50 | 61.00 | 51.50 | 20.50 | 39.50 | 61.00 | 48.50 |
| COA | 52.00 | 63.50 | 55.00 | 24.50 | 48.00 | 63.00 | 51.00 |
| RedQueen | 63.00 | 58.50 | 53.00 | 43.50 | 45.00 | 57.50 | 53.42 |
| **MTSA-R3** | **72.00** | **66.50** | **56.00** | **50.50** | **64.00** | **74.50** | **63.92** |

### 目标模型防御能力——Zephyr-7B-Beta

| 方法 | MTSA-R3 ASR↓ | BeaverTails↓ | CoSafe↓ | MT-Bench↑ | AlpacaEval↑ | XSTest↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline | 74.50 | 24.50 | 41.20 | 6.76 | 78.35 | 0.283 |
| HARM-T3 | 52.50 | 17.50 | 26.75 | 6.35 | 73.92 | 0.247 |
| MART-T3 | 48.50 | 15.50 | 26.78 | 6.46 | 74.81 | 0.255 |
| **MTSA-T3** | **23.50** | **11.50** | **18.78** | **6.78** | 77.45 | **0.231** |

### 域外多轮攻击鲁棒性——Zephyr-7B-Beta

| 攻击方法 | 基线ASR | MTSA-T3 ASR | 降低比例 |
|------|:---:|:---:|:---:|
| ActorAttack | 43.50 | 12.50 | -71.26% |
| RedQueen | 57.50 | 19.50 | -66.08% |

### 关键发现

1. **攻击能力逐轮提升**：MTSA-R1到R3的ASR从58.08%提升到63.92%，表明对抗迭代训练持续增强红队模型的攻击策略
2. **防御性能大幅领先**：MTSA-T3将MTSA-R3的ASR从74.50%降至23.50%（降低67%），远优于HARM和MART方法
3. **通用性几乎不损失**：MTSA-T3的MT-Bench（6.78）和AlpacaEval（77.45）与基线持平，而HARM和MART出现显著退化
4. **多轮到单轮的泛化**：在BeaverTails（单轮安全基准）上同样取得最佳防御效果，说明多轮安全对齐可以泛化到单轮场景
5. **过度拒绝可控**：XSTest评估显示过度拒绝率仅增加5.62%，模型可用性得到保障
6. **域外攻击鲁棒性强**：对未见过的攻击方法（ActorAttack、RedQueen）的防御降低率超过66%

## 亮点与洞察

1. **"思考再攻击"的红队策略**是本文的一大亮点：让红队模型在每轮攻击前先分析对话环境、选择策略，模拟了真实攻击者的决策过程，使自动红队更接近人类红队的攻击水平
2. **基于未来奖励替代critic model**是多轮RL中的实用创新：通过轨迹采样获取终止状态奖励来近似Q值，避免了critic model在多轮场景中预测不准确的问题
3. **同时优化攻防双方的对抗框架**：红队模型和目标模型在迭代中共同进化，形成"安全军备竞赛"，这比单方面防御更能发现和修补漏洞
4. **红队模型仅7B参数但攻击能力超越所有基线**（包括攻击GPT-4o和Claude-3.5），展示了思维引导策略的强大效果

## 局限与展望

- 仅在7B规模的模型上验证攻防效果，更大规模模型（如70B+）的适用性未知
- 攻击策略分类为4类可能不够全面，更多攻击模式（如社会工程学、多模态攻击）未覆盖
- 3轮迭代的选择缺乏系统性分析，不确定是否已收敛或继续迭代是否有益
- 奖励模型依赖外部安全分类器和ArmoRM，其自身的偏差可能影响对齐效果
- 安全对齐的持久性（训练后是否在新场景下仍然安全）未充分评估

## 相关工作与启发

- **MART**：最早的迭代红蓝对抗框架之一，MTSA在其基础上添加了多轮设计和基于未来奖励的RL算法
- **GPO**：将攻防整合为两人博弈框架，但仅面向单轮场景。MTSA将其扩展到多轮并更注重实际攻击策略的学习
- **DPO/RLHF在安全领域的应用**：MTSA展示了DPO（用于红队）和最小二乘RL（用于目标）的差异化使用场景
- **对工业安全实践的启发**：MTSA的对抗迭代思路可作为LLM部署前的自动化安全测试流程

## 评分

- **新颖性**: ⭐⭐⭐⭐ 思维引导攻击+多轮未来奖励RL+对抗迭代的组合创新，填补多轮安全对齐空白
- **实验充分度**: ⭐⭐⭐⭐⭐ 6个LLM×多种攻击方法+防御/通用性/过度拒绝/鲁棒性全面评估
- **写作质量**: ⭐⭐⭐ 内容丰富但部分符号和公式表述略显繁杂
- **价值**: ⭐⭐⭐⭐⭐ 直击LLM多轮安全问题的实用框架，攻防双向提升的设计思路有很高的实践价值

<!-- RELATED:START -->

## 相关论文

- [M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs](m2s_multiturn_to_singleturn_jailbreak_in.md)
- [Red Queen: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking](red_queen_safeguarding_large_language_models_against_concealed_multi-turn_jailbr.md)
- [PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference](pku-saferlhf_towards_multi-level_safety_alignment_for_llms_with_human_preference.md)
- [JailbreakRadar: Comprehensive Assessment of Jailbreak Attacks Against LLMs](jailbreakradar_comprehensive_assessment_jailbreak_attacks.md)
- [LLMs Caught in the Crossfire: Malware Requests and Jailbreak Challenges](llms_caught_in_the_crossfire_malware_requests_and_jailbreak_challenges.md)

<!-- RELATED:END -->
