---
title: >-
  [论文解读] Characterizing AI Manipulation Risks in Brazilian YouTube Climate Discourse
description: >-
  [AAAI2026][机器人][Climate Discourse] 通过心理语言学框架分析巴西 YouTube 上 22.6 万条气候变化视频和 275 万条评论，揭示情感/道德修辞显著驱动用户互动，并展示微调 LLM 可自动生成高互动性的气候否认评论，警示生成式 AI 在舆论操控中的潜在风险。
tags:
  - AAAI2026
  - 机器人
  - Climate Discourse
  - Persuasion
  - Theory of Mind
  - YouTube
  - LLM-generated Manipulation
  - Social Media Analysis
---

# Characterizing AI Manipulation Risks in Brazilian YouTube Climate Discourse

**会议**: AAAI2026  
**arXiv**: [2511.06091](https://arxiv.org/abs/2511.06091)  
**代码**: 待确认  
**领域**: robotics  
**关键词**: Climate Discourse, Persuasion, Theory of Mind, YouTube, LLM-generated Manipulation, Social Media Analysis  

## 一句话总结

通过心理语言学框架分析巴西 YouTube 上 22.6 万条气候变化视频和 275 万条评论，揭示情感/道德修辞显著驱动用户互动，并展示微调 LLM 可自动生成高互动性的气候否认评论，警示生成式 AI 在舆论操控中的潜在风险。

## 背景与动机

气候变化是全球性威胁，应对它需要基于证据的政策制定和公众充分理解。社交媒体（尤其是 YouTube）日益成为气候叙事传播的主要渠道，但同时也是虚假信息扩散的温床。巴西作为全球南方的代表性国家，拥有亚马逊雨林的重大生态地位，且 YouTube 覆盖了该国约 68% 的人口，是研究气候话语的理想场景。

近年来 LLM 的快速发展带来了新的风险维度：已有研究表明 AI 生成的文本具有说服力，甚至可以影响人们对阴谋论的信念形成。这引发了一个核心忧虑——生成式 AI 是否可以被用来大规模自动化地操控气候话语，例如制造"气候否认"的虚假共识？

本文的动机在于：（1）系统性地量化心理语言学特征（说服策略 + 心智理论）对用户互动的影响；（2）评估这些模式是否可被 LLM 利用来自动生成高互动性的操控性内容。

## 核心问题

1. 哪些心理内容特征（说服策略）最能有效驱动巴西气候 YouTube 视频的观众互动？
2. 这些心理特征在多大程度上可以预测内容的流行度？
3. 这些洞察是否可以被利用来设计自动化的说服性合成内容（如气候否认运动）？

## 方法详解

### 数据集构建

- **规模**：226,775 条巴西葡萄牙语 YouTube 视频元数据 + 2,756,165 条用户评论，时间跨度 2019-2025 年
- **收集流程**：基于 65 个气候相关关键词通过 YouTube Data API v3 检索，使用 FastText 语言识别过滤非葡萄牙语内容，再通过 GPT-4.1-mini（温度=0）过滤低相关性视频
- **视频分类**：按时长分为短视频（<3 分钟）和长视频（≥3 分钟），自 2023 年起短视频已成为气候话题的主流形式

### 心理语言学标注

#### 说服策略标注（视频层面）

使用 GPT-4.1 通过 5-shot prompting 对视频内容标注 10 种说服策略：

| 策略 | 说明 |
|---|---|
| Logical Appeal | 以理由和证据说服 |
| Emotional Appeal | 激发情感反应 |
| Statistical Evidence | 提供具体数据和统计 |
| Social Norm | 通过社会认同施加压力 |
| Authority | 引用专家、机构和官方报告 |
| Personal Stories | 讲述个人经历 |
| Moral Appeal | 诉诸道德责任 |
| Reciprocity | 强调互惠利益 |
| Scarcity | 呈现时间有限性和不可逆影响 |
| Common Ground | 构建共同身份和价值观 |

人工验证结果：平均 F1 = 0.93，准确率 = 0.98。

#### Theory of Mind 标注（评论层面）

使用 GPT-4.1-mini 对用户评论标注 7 种心智理论类别：Belief（信念）、Intention（意向）、Desire（愿望）、Emotion（情感）、Knowledge（知识）、Percept（感知）、Non-literal（非字面表达）。人工验证：F1 = 0.66，准确率 = 0.83。

### Case Study 1: 互动建模

通过三阶段评估心理语言学特征对用户互动的影响：

1. **视频层面**：使用线性回归分析说服策略向量 $\mathbf{p}_i$ 对标准化点赞率 $L_i$ 和评论率 $R_i$ 的影响，控制视频时长和频道等混杂因素
2. **策略-心智关联**：将每个视频的评论 ToM 向量聚合为 $\bar{\mathbf{t}}_i = (1/|C_i|)\sum_{c_k \in C_i} \mathbf{t}_k$，计算说服策略与 ToM 类别的偏相关
3. **评论层面**：以评论的点赞数和回复数为因变量，ToM 标注为自变量，控制评论长度和时间差

### Case Study 2: 流行度预测

将评论配对为 $(c_i, c_j)$，定义二元标签 $y_{ij}^{(\ell)} = \mathbb{I}[\ell_i > \ell_j]$，预测哪条评论更受欢迎。使用三类方法：

- **LLM-as-a-Judge**：GPT-4.1、o4-mini、Phi-4、Llama-3.1-8B、Llama-4-Maverick
- **编码器模型微调**：BERTimbau（巴西葡萄牙语 BERT）、DeBERTa V3
- **Bradley-Terry 模型**：基于评论嵌入训练线性分类器

### Case Study 3: 评论生成

微调 Llama-3-8B 生成目标化评论，构造三类场景：

1. **按说服策略采样**：控制视频层面效果
2. **按 ToM 画像采样**：生成反映特定心理状态的评论
3. **按信念立场细分**：区分"相信气候变化"、"否认气候变化"和"极端否认"三种模型

评估方法：对生成评论检索 K 个最相似的真实评论，用其平均点赞/回复数作为代理评估指标。

## 实验关键数据

### 说服策略对互动的影响

- 最常用策略（Logical Appeal 51%、Authority 47%、Common Ground 36%）均与**较低**的用户互动相关
- Emotional Appeal（33%）和 Moral Appeal（26%）与**显著更高**的互动相关，其中道德诉求平均提升 2.1% 的视频点赞
- 短视频中道德修辞的效力随时间持续增长

### 流行度预测

| 模型 | 最佳准确率 | 条件 |
|---|---|---|
| BERTimbau | **88%** | 无上下文，随机配对 |
| GPT-4.1 | 82% | 有视频上下文 + few-shot |
| DeBERTa V3 | 84% | 有视频上下文 |

- Emotional ToM 对预测性能平均提升 **4.69%**
- BERTimbau 仅凭评论文本即达 88%，说明评论内容本身已包含足够的互动预测信号

### 评论生成

| 模型 | 估计点赞数 $\hat{\ell}_{gen|1}$ |
|---|---|
| Baseline（随机评论） | 2.20 |
| Engaging（高赞评论微调） | **7.25**（3.3 倍提升） |
| Believe（相信气候变化） | 3.23 |
| Denial（气候否认） | 1.91 |
| Extreme（极端否认） | 2.37 |

极端否认模型生成的评论包含更多细节和修辞强度，比普通否认模型更具互动吸引力。

## 亮点

1. **大规模心理语言学数据集**：发布包含 22.6 万视频和 275 万评论的巴西气候话语数据集，附带说服策略和 ToM 标注，是该领域最大的非英语数据资源之一
2. **说服与互动的因果链条清晰**：从"说服策略→用户心理状态→互动行为"的完整分析管线，三个 Case Study 层层递进
3. **实证揭示 AI 操控风险**：不仅停留在理论讨论，而是通过实际微调 LLM 展示了自动化舆论操控的可行性，极端否认模型的输出具有触目惊心的真实感
4. **巴西+葡萄牙语视角独特**：填补了气候话语研究中全球南方国家的空白，BERTimbau 在葡萄牙语评论上超越 GPT-4.1 也凸显了语言特异性的重要性
5. **短视频趋势的深入分析**：揭示自 2023 年起气候短视频已超越长视频，这种格式变化进一步压缩了事实核查的空间

## 局限与展望

- **仅分析文本内容**：忽略了视觉、音频等多模态元素对说服力的影响，YouTube 作为视频平台这一局限尤为明显
- **互动指标不完整**：未考虑推荐算法、个体心理差异、用户画像等影响互动的重要因素
- **地域和语言限制**：所有发现仅限于巴西葡萄牙语 YouTube 内容，跨语言/跨平台的泛化性未经验证
- **ToM 标注质量中等**：F1 仅 0.66，与说服策略标注的 0.93 差距较大，可能影响下游分析的可靠性
- **生成评论的评估方式间接**：使用近邻检索的代理评估而非真实平台部署测试，无法确认真实互动效果
- 未来可扩展到多语言、多平台（TikTok/X）对比，并引入多模态分析框架

## 与相关工作的对比

| 方面 | 本文 | 现有气候话语研究 |
|---|---|---|
| 语言/地区 | 巴西葡萄牙语 | 以英语为主 |
| 分析框架 | 说服策略 + ToM 双维度 | 通常单一维度（立场检测或情感分析） |
| 操控风险评估 | LLM 微调生成实验 | 理论讨论为主 |
| 数据规模 | 22.6 万视频 + 275 万评论 | 通常 <15 万条推文 |
| 平台 | YouTube（视频+评论） | Twitter/X 为主 |

与 Costello et al. (2024) 的 AI 说服研究相比，本文将焦点从受控实验转向真实社交媒体场景；与 Breum et al. (2024) 的 LLM 说服性分析相比，本文增加了 ToM 维度和实际生成实验。

## 启发与关联

- **对生成式 AI 治理的警示**：情感和道德修辞的高互动效力 + LLM 的生成能力 = 低成本大规模舆论操控的可能性，呼吁对合成媒体建立治理框架
- **事实核查面临的挑战加剧**：短视频化趋势压缩了深度信息的传播空间，算法推荐进一步放大情感性内容，事实核查效率堪忧
- **BERTimbau > GPT-4.1 的发现**提示，在特定语言/文化场景中，本地化的小模型可能比通用大模型更有效，这对多语言 AI 研究有重要启示
- **方法论可迁移**：说服策略 + ToM 的双维度分析框架可以应用到其他社会议题（如疫苗犹豫、政治极化）和其他平台

## 评分

- 新颖性: ⭐⭐⭐⭐ （说服策略 + ToM 双维度框架新颖，LLM 操控风险的实证分析具有前瞻性）
- 实验充分度: ⭐⭐⭐⭐ （三个 Case Study 设计完整，数据规模大，但 ToM 标注质量和生成评估方式有提升空间）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，案例丰富，从分析到风险警示的逻辑链条连贯）
- 价值: ⭐⭐⭐⭐ （对 AI 伦理和社交媒体治理有重要警示价值，数据集的公开发布也有社区贡献）

<!-- RELATED:START -->

## 相关论文

- [Unintended Misalignment from Agentic Fine-Tuning: Risks and Mitigation](unintended_misalignment_from_agentic_fine-tuning_risks_and_m.md)
- [Token Taxes: Mitigating AGI's Economic Risks](../../ICLR2026/robotics/token_taxes_mitigating_agis_economic_risks.md)
- [Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems](shadows_in_the_code_exploring_the_risks_and_defenses_of_llm-.md)
- [Magma: A Foundation Model for Multimodal AI Agents](../../CVPR2025/robotics/magma_a_foundation_model_for_multimodal_ai_agents.md)
- [Grounding Generative Planners in Verifiable Logic: A Hybrid Architecture for Trustworthy Embodied AI](../../ICLR2026/robotics/grounding_generative_planners_in_verifiable_logic_a_hybrid_architecture_for_trus.md)

<!-- RELATED:END -->
