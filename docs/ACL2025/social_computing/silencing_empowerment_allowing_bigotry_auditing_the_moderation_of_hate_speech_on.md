---
title: >-
  [论文解读] Silencing Empowerment, Allowing Bigotry: Auditing the Moderation of Hate Speech on Twitch
description: >-
  [ACL 2025][社会计算][仇恨言论检测] 对 Twitch 平台的自动化内容审核工具 AutoMod 进行大规模审计，发送超过 10.7 万条消息，发现 AutoMod 在最严格设置下仅能标记 22% 的仇恨内容，高度依赖侮辱性词汇作为检测信号，同时错误屏蔽高达 89.5% 的教育性/赋权性内容。
tags:
  - "ACL 2025"
  - "社会计算"
  - "仇恨言论检测"
  - "内容审核"
  - "算法审计"
  - "Twitch"
  - "AutoMod"
---

# Silencing Empowerment, Allowing Bigotry: Auditing the Moderation of Hate Speech on Twitch

**会议**: ACL 2025  
**arXiv**: [2506.07667](https://arxiv.org/abs/2506.07667)  
**代码**: [有](https://github.com/weiyinc11/HateSpeechModerationTwitch)  
**领域**: 社会计算  
**关键词**: 仇恨言论检测, 内容审核, 算法审计, Twitch, AutoMod

## 一句话总结

对 Twitch 平台的自动化内容审核工具 AutoMod 进行大规模审计，发送超过 10.7 万条消息，发现 AutoMod 在最严格设置下仅能标记 22% 的仇恨内容，高度依赖侮辱性词汇作为检测信号，同时错误屏蔽高达 89.5% 的教育性/赋权性内容。

## 研究背景与动机

在线平台面临海量用户生成内容的审核压力，尤其是 Twitch 等实时直播平台对审核延迟要求更高。平台越来越多地采用基于机器学习的自动化审核系统，但对这些系统的有效性知之甚少。

本文选择 Twitch 进行审计的三个关键优势：
1. Twitch 用户量大且广泛使用
2. 直播可设为"隔离"模式，仅研究团队可见，支持受控实验
3. AutoMod 工具高度可配置，提供不同类别和强度级别的审核选项，且会返回审核理由

研究问题：
- AutoMod 在标记仇恨内容方面有多有效？
- 各过滤器在检测不同类型仇恨言论时的专一性和效果如何？
- 审核率在不同目标群体之间是否一致？

## 方法详解

### 整体框架

审计管道包含三个阶段：
1. 设置机器人与数据整理
2. 大规模记录审核决策
3. 分析 AutoMod 审核决策

### 关键设计

1. **平台选择与实验环境搭建**

    - 调查了 43 个最大的用户生成内容平台，最终选择 Twitch
    - 创建隔离的直播频道，确保测试内容仅研究团队可见
    - 使用三个认证机器人：发送者 bot、接收者 bot、PubSub bot
    - 遵守 Twitch 聊天速率限制，避免消息重复或遗漏

2. **数据集选择**：使用四个涵盖显性和隐性仇恨言论的数据集

   | 数据集 | 来源 | 特点 |
   |--------|------|------|
   | SBIC | 真实评论 | 带冒犯性评分 |
   | IHC | 真实隐性仇恨 | 隐性仇恨为主 |
   | ToxiGen | 合成隐性仇恨 | LLM 生成 |
   | DynaHate | 合成对抗样本 | 设计用于欺骗分类器 |

3. **审计设计的形式化**

    - 将审核系统定义为 $\mathcal{S} = (\mathcal{F}, \mathcal{C})$，$\mathcal{F}$ 为过滤函数集合，$\mathcal{C}$ 为对应准则集合
    - 每个过滤器 $\mathcal{F}_i: T \to \{0, 1\}$，带过滤强度参数 $\alpha$
    - 审计四个类别：残疾 (Disability)、性/性别/性取向 (SSG)、厌女 (Misogyny)、种族/民族/宗教 (RER)

4. **案例研究设计**

    - 反事实分析：在假阴性样本中替换侮辱性词汇观察审核变化
    - 政策遵从评估：测试 AutoMod 对教育性/赋权性内容的处理
    - 鲁棒性测试：对敏感词进行 6 种语义保持扰动

### 损失函数 / 训练策略

本文为分析性研究，不涉及模型训练。核心方法为黑盒算法审计。

## 实验关键数据

### 主实验（AutoMod 整体表现）

| 数据集 | 准确率 | 精确率 | 召回率 | TNR | F1 |
|--------|--------|--------|--------|-----|-----|
| SBIC | 0.73 | 0.42 | 0.19 | 0.91 | 0.26 |
| DynaHate | 0.49 | 0.54 | 0.41 | 0.59 | 0.47 |
| ToxiGen | 0.53 | 0.86 | **0.07** | 0.98 | 0.13 |
| IHC | 0.52 | 0.70 | **0.06** | 0.97 | 0.12 |
| **Overall** | 0.55 | 0.56 | **0.22** | 0.84 | 0.32 |

AutoMod 在最严格设置下整体召回率仅为 22%。隐性仇恨数据集 (ToxiGen, IHC) 的召回率低至 6-7%。

### 过滤器分析

| 过滤器 | 整体召回率 | 预过滤率 |
|--------|-----------|---------|
| Disability | 10.6% | 6.1% |
| Misogyny | 19.0% | 1.5% |
| RER | 12.3% | 22.0% |
| SSG | 17.5% | 54.8% |

### 案例研究关键结果

| 实验 | 结果 |
|------|------|
| 反事实（加入侮辱词后） | 召回率从 ~20% 升至 **100%** |
| 教育性内容误封率 (α=2) | **89.5%** |
| 教育性内容误封率 (α=4) | **98.5%** |
| 语义保持扰动后 | 召回率从 100% 降至 **4%** |

### 关键发现

1. AutoMod 高度依赖侮辱性词汇作为仇恨检测信号——加入侮辱词后召回率达 100%，证实了关键词依赖
2. 对隐性仇恨几乎无能为力：ToxiGen 和 IHC 上分别仅标记 7% 和 6%
3. 89.8% 的假阴性不含粗话（即都是隐性仇恨），73% 的假阳性包含粗话
4. 过滤级别 α 的调节效果有限：从 α=1 到 α=4 仅提升 1.1%
5. 不同目标群体间存在显著不平等：针对精神残疾者的仇恨高达 98% 逃脱审核
6. AutoMod 不具备对话级上下文感知（改变消息顺序不影响审核决策）

## 亮点与洞察

1. **方法论贡献**：建立了一套完整的黑盒内容审核系统审计框架，可推广到其他平台
2. **揭示系统性缺陷**：AutoMod 本质上严重依赖词汇匹配而非语义理解
3. **政策与实践脱节**：Twitch 社区指南明确要求考虑上下文，但 AutoMod 实际上不具备此能力
4. **社会影响**：系统"允许仇恨、封锁赋权（Allowing Bigotry, Silencing Empowerment）"——标题精准概括核心发现

## 局限与展望

1. 仅审计了一个平台（Twitch），未与其他平台的审核系统对比
2. 消息以非对话方式逐条发送，未模拟真实对话上下文
3. 仅关注英语文本，未涉及多语言和多模态内容
4. 未审计 Twitch 的 Smart Detection 功能和音视频内容审核
5. 未研究群内/群外动态对审核的影响

## 相关工作与启发

- Hartmann et al. (2025)：同期工作，评估多个审核 API（如 OpenAI），发现类似问题（隐性仇恨检测差、教育内容误封）
- DynaHate (Vidgen et al., 2021b)：对抗性仇恨语料构建方法，可用于进一步压力测试
- Sap et al. (2020)：SBIC 数据集，开源分类器在相同数据上表现远优于 AutoMod

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次对 Twitch AutoMod 进行系统性大规模审计
- **实验充分度**: ⭐⭐⭐⭐⭐ 跨 4 个数据集、30 万条消息、过滤器分析、案例研究，极为全面
- **写作质量**: ⭐⭐⭐⭐⭐ 结构严谨，研究设计的形式化描述和实验流程清晰
- **价值**: ⭐⭐⭐⭐⭐ 为内容审核系统的改进提供了重要的实证证据，具有社会影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] HateDay: Insights from a Global Hate Speech Dataset Representative of a Day on Twitter](hateday_global_hate_speech.md)
- [\[ACL 2025\] ImpliHateVid: Implicit Hate Speech Detection in Videos](implihatevid_video_hate.md)
- [\[ACL 2025\] STATE ToxiCN: A Benchmark for Span-level Target-Aware Toxicity Extraction in Chinese Hate Speech Detection](state_toxicn_a_benchmark_for_span-level_target-aware_toxicity_extraction_in_chin.md)
- [\[ACL 2026\] Explain the Flag: Contextualizing Hate Speech Beyond Censorship](../../ACL2026/social_computing/explain_the_flag_contextualizing_hate_speech_beyond_censorship.md)
- [\[ACL 2026\] RV-HATE: Reinforced Multi-Module Voting for Implicit Hate Speech Detection](../../ACL2026/social_computing/rv-hate_reinforced_multi-module_voting_for_implicit_hate_speech_detection.md)

</div>

<!-- RELATED:END -->
