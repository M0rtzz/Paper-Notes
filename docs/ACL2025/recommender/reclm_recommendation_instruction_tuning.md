---
title: >-
  [论文解读] RecLM: Recommendation Instruction Tuning
description: >-
  [ACL 2025][推荐系统] 提出 RecLM，一个模型无关的推荐指令微调框架，通过两轮对话式指令微调将协同过滤信号注入 LLM 生成的用户/商品画像，再用 RLHF（PPO）精炼画像质量，在 MIND/Netflix/工业数据集上作为即插即用组件为 BiasMF/NCF/LightGCN/SGL/SimGCL 一致带来提升，尤其在冷启动场景效果显著。
tags:
  - ACL 2025
  - 推荐系统
  - 指令微调
  - 协同过滤
  - 冷启动
  - 强化学习
  - 用户画像
---

# RecLM: Recommendation Instruction Tuning

**会议**: ACL 2025  
**arXiv**: [2412.19302](https://arxiv.org/abs/2412.19302)  
**代码**: https://github.com/HKUDS/RecLM (有)  
**领域**: 推荐系统 / LLM  
**关键词**: 推荐系统, 指令微调, 协同过滤, 冷启动, 强化学习, 用户画像

## 一句话总结
提出 RecLM，一个模型无关的推荐指令微调框架，通过两轮对话式指令微调将协同过滤信号注入 LLM 生成的用户/商品画像，再用 RLHF（PPO）精炼画像质量，在 MIND/Netflix/工业数据集上作为即插即用组件为 BiasMF/NCF/LightGCN/SGL/SimGCL 一致带来提升，尤其在冷启动场景效果显著。

## 研究背景与动机

**领域现状**：推荐系统主要依赖 ID-based 协同过滤（CF），通过 GNN 等方法优化用户/商品 ID embedding。在数据充足时效果好，但面临冷启动和零样本问题。

**现有痛点**：(1) ID-based embedding 对新商品无法生成有意义的表示；(2) 利用文本侧信息（商品描述）作为替代 embedding 的方案受限于数据不完整和质量问题（误导性标签、无关描述）；(3) LLM 虽有强大语言理解能力，但缺乏用户-商品交互行为模式的建模能力。

**核心矛盾**：LLM 的文本理解能力和协同过滤的交互关系建模是互补的，但如何让 LLM "理解"推荐场景的行为上下文？

**本文目标**：(1) 设计 LLM 生成准确用户画像的机制，尤其对冷启动用户/商品；(2) 从噪声特征中蒸馏高质量画像。

**切入角度**：将推荐任务转化为 LLM 的指令微调任务，将协同过滤的高阶关系编码为对话 prompt，让 LLM 生成融合了协同信号的用户画像。

**核心 idea**：两轮对话指令微调（第一轮生成画像 + 第二轮预测交互）+ RLHF 精炼画像，生成的画像 embedding 即插即用到任意推荐模型。

## 方法详解

### 整体框架
文本侧信息投影 → LLM 协同指令微调（两轮对话）→ RL 精炼 → 生成用户/商品画像 → 画像 embedding 融合到下游推荐模型。

### 关键设计

1. **文本驱动的用户/商品表示**:

    - 功能：用文本代替 ID embedding 实现零样本推荐
    - 核心思路：商品文本描述通过 MLP 投影到低维空间作为初始 item embedding：$\hat{f}_v = T_{raw}(f)$。用户 embedding 则结合 ID embedding 和 LLM 生成的画像
    - 设计动机：文本特征对新商品也可用，克服了 ID embedding 的冷启动限制

2. **两轮协同指令微调（核心）**:

    - 功能：将协同过滤信号注入 LLM 的画像生成能力
    - 核心思路：
        - **第一轮——协同画像生成**：输入目标用户的历史交互 + 相似用户（LightGCN embedding距离）的交互历史，LLM 生成综合了协同关系的用户画像
        - **第二轮——交互预测监督**：基于第一轮画像，提问"用户 u 是否会与商品 v 交互？"，ground truth 为 Yes/No。正样本从用户历史中取且在相似用户中也出现的商品，负样本从相似用户历史中取但目标用户未交互的商品
        - **多轮微调策略**：两轮对话拼接后，对 $\mathcal{R}_{fir.}$（画像）和 $\mathcal{R}_{sec.}$（Yes/No）都计算 loss，双重优化画像生成和交互预测
    - 设计动机：单靠画像生成缺少直接监督信号（无画像 ground truth），第二轮的交互预测提供了间接但明确的监督

3. **RLHF 精炼画像生成**:

    - 功能：用强化学习解决推理-训练差异和过度平滑问题
    - 核心思路：
        - **Reward Model**：基于 LLM 构建奖励模型，用 ChatGPT 生成正样本画像，用多样化 prompt + 画像替换生成负样本，训练 ranking loss
        - **PPO 优化**：将 LLM 作为 policy，用奖励模型指导优化，加 KL 散度约束防止 reward hacking
    - 设计动机：指令微调后的画像可能过度依赖协同信息（类似 GNN 过平滑），RL 精炼各性化特征

### 损失函数 / 训练策略
- 指令微调：LLaMA-2-7B-Chat 为基座，LoRA 微调，两轮对话同时优化
- RL：PPO + KL 约束，reward model 用 ranking loss 训练
- 推荐模型训练：BPR loss，即插即用替换 embedding

## 实验关键数据

### 主实验（Full-Shot + Cold-Start）

| 基线模型 | 数据集 | Base R@20 | +RecLM R@20 | 提升 |
|---------|--------|----------|------------|------|
| BiasMF | MIND | 0.0683 | **0.0719** | +5.3% |
| BiasMF | Netflix | 0.0449 | **0.0531** | +18.3% |
| BiasMF | Industrial | 0.0078 | **0.0121** | +55.1% |
| LightGCN | MIND | 0.0822 | **0.0842** | +2.4% |
| SimGCL | Netflix | 0.0662 | **0.0683** | +3.2% |

冷启动（Zero-Shot）场景下提升更为显著。

### 消融实验

| 配置 | MIND R@20 | Netflix R@20 |
|------|-----------|-------------|
| RecLM Full | **0.0842** | **0.0683** |
| w/o Profile (只用文本) | 0.0809 | 0.0643 |
| w/o 两轮微调 (单轮) | 0.0823 | 0.0665 |
| w/o RL 精炼 | 0.0831 | 0.0672 |

### 关键发现
- **工业数据集提升最大**：BiasMF 在 Industrial 上 R@20 从 0.0078→0.0121（+55%），说明在数据稀疏的实际场景中画像增强价值最高
- **模型无关性强**：为 5 种不同推荐模型（从简单 MF 到 GNN）都带来一致提升
- **两轮微调优于单轮**：第二轮的交互预测为画像生成提供了关键监督信号
- **RL 精炼有效**：PPO 进一步提升 1-2 个百分点，缓解了协同过平滑问题
- **冷启动优势明显**：在零样本场景下，RecLM 提供了有意义的初始表示

## 亮点与洞察
- **"推荐=对话"的范式**：将协同过滤关系编码为 LLM 对话指令，让 LLM "学会"推荐领域的行为语义。两轮对话的设计巧妙——画像生成->交互预测形成闭环
- **即插即用设计**：画像 embedding 通过简单融合就能接入任何推荐模型，实用性强
- **工业验证**：在匿名工业数据集上的大幅提升增加了可信度

## 局限与展望
- LLM 推理开销大（LLaMA-7B 为每个用户生成画像），大规模部署需要考虑效率
- 依赖 ChatGPT 生成指令微调数据和 RL 正样本，构建成本不低
- MIND 上 N@20 指标有下降（BiasMF -12.5%, NCF -11.4%），说明画像可能引入某些噪声
- 仅评估 top-K 推荐，未评估点击率预测等其他推荐任务
- 相似用户选择基于 LightGCN embedding，对于完全冷启动用户无法获取协同邻居

## 相关工作与启发
- **vs RLMRec/LLMRec**: 其他 LLM 增强推荐方法通常直接用 LLM embedding 或生成文本特征，缺少协同信号注入。RecLM 通过两轮对话微调显式引入高阶协同关系
- **vs KAR**: KAR 使用 LLM 增强知识，但不涉及用户画像生成和 RL 精炼
- **vs P5/InstructRec**: 这些方法将推荐完全建模为 LLM 序列生成任务，计算成本高。RecLM 只用 LLM 生成画像，推荐仍由高效 CF 模型完成

## 评分
- 新颖性: ⭐⭐⭐⭐ 两轮协同指令微调+RLHF 精炼的设计有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 3 个数据集（含工业）、5 个推荐模型、零样本测试、消融充分
- 写作质量: ⭐⭐⭐⭐ 结构完整，公式化清晰
- 价值: ⭐⭐⭐⭐ 即插即用的 LLM 推荐增强方案，对冷启动有直接实用价值

<!-- RELATED:START -->

## 相关论文

- [MIRA: Empowering One-Touch AI Services on Smartphones with MLLM-based Instruction Recommendation](mira_empowering_one-touch_ai_services_on_smartphones_with_mllm-based_instruction.md)
- [GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion](gram_generative_recommendation.md)
- [Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning](../../NeurIPS2025/recommender/overcoming_sparsity_artifacts_in_crosscoders_to_interpret_chat-tuning.md)
- [Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning](../../NeurIPS2025/recommender/transformer_copilot_learning_from_the_mistake_log_in_llm_fine-tuning.md)
- [KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models](kerl_knowledge-enhanced_personalized_recipe_recommendation_using_large_language_.md)

<!-- RELATED:END -->
