---
title: >-
  [论文解读] Conspiracy Theories and Where to Find Them on TikTok
description: >-
  [ACL 2025][阴谋论] 首个TikTok阴谋论系统性分析：通过官方API收集美国150万条长视频，利用标签富集和远程监督识别阴谋论内容（每月约1000条新视频），评估TikTok创作者激励计划的影响，并测试开源LLM（Llama3、Mistral、Gemma）在基于音频转录的阴谋论检测上的效果（精确率高达96%但整体水平与微调RoBERTa相当）。
tags:
  - ACL 2025
  - 阴谋论
  - TikTok
  - LLM内容审核
  - 标签富集
  - 远程监督
  - 创作者激励计划
---

# Conspiracy Theories and Where to Find Them on TikTok

**会议**: ACL 2025  
**arXiv**: [2407.12545](https://arxiv.org/abs/2407.12545)  
**代码**: [https://anonymous.4open.science/r/ct_tt-FC7E](https://anonymous.4open.science/r/ct_tt-FC7E)（脱水数据集+复现代码）  
**作者**: Francesco Corso, Francesco Pierri, Gianmarco de Francisci Morales
**机构**: Politecnico Di Milano, CENTAI
**领域**: 社交媒体分析 / 内容安全  
**关键词**: 阴谋论, TikTok, LLM内容审核, 标签富集, 远程监督, 创作者激励计划

## 一句话总结

首个TikTok阴谋论系统性分析：通过官方API收集美国150万条长视频，利用标签富集和远程监督识别阴谋论内容（每月约1000条新视频），评估TikTok创作者激励计划的影响，并测试开源LLM（Llama3、Mistral、Gemma）在基于音频转录的阴谋论检测上的效果（精确率高达96%但整体水平与微调RoBERTa相当）。

## 研究背景与动机

**TikTok的影响力**：近一半TikTok用户（17%的美国成年人）定期使用该平台获取新闻和信息。TikTok庞大的用户基数和病毒式传播机制使其成为恶意内容的温床。

**阴谋论的危害**：阴谋论是声称秘密（通常是邪恶的）阴谋的虚假或未验证叙事。其风险在于扭曲消费者的现实感知，加剧错误信息传播和极化，甚至可能导致危险的现实后果。

**研究空白**：此前仅有少数定性研究分析TikTok上的阴谋论（如远右叙事、猴痘相关阴谋内容），缺乏定量和系统性的方法。

**三个研究问题**：
   - RQ1: TikTok上分享阴谋论视频的普及率如何？
   - RQ2: 创作者激励计划（Creativity Program）是否影响了阴谋论内容的供应？
   - RQ3: 能否利用LLM检测TikTok上的阴谋论？

## 方法详解

### 3.1 数据收集

使用TikTok官方Research API收集2021-2023年美国地区数据：
- **总量**：1,605,696条视频（去重后1,494,831条唯一视频）
- **创作者**：1,178,303位唯一用户
- **额外样本**：创作者激励计划前后各约10K随机视频

### 3.2 阴谋论标签富集 (Conspiracy Hashtag Enrichment)

**Step 1：种子选择**
从LOCO阴谋论数据集（90,000+阴谋论和非阴谋论文章）的top 30频繁种子词中，手动选取10个与阴谋论直接关联的种子（illuminati、reptilians等），排除含义过于宽泛的词（cancer、AIDS、5G等）。

**Step 2：标签相似度计算**
对281,510个标签计算与种子标签的相似度：

$$sim(h_s, h_t) = \frac{\alpha \cos(W_s, W_t) + (1-\alpha) \cos(H_s, H_t)}{1 + \log(df(h_t))}$$

利用标签-标签共现矩阵H和标签-词语共现矩阵W，混合参数α=0.3，基于文档频率折扣避免过于常见标签（如#fyp、#viral）的影响。

**Step 3：手动验证**
产出197个候选标签，每个检查5个相关视频后分类为：
- **CT** (阴谋论): 92个
- **NOCT** (非阴谋论): 68个
- **DW** (暗语Dog Whistling): 28个——表面无害但被阴谋论者作为隐蔽标记（如#radiowaves关联气象操控和化学尾迹）
- **HJ** (标签劫持): 阴谋论用户借用流行标签获取流量
- **RHJ** (反向标签劫持): 用户在阴谋论标签下传播辟谣信息

**最终标注规则**：含CT或DW标签、且不含NOCT/HJ/RHJ标签的视频标记为阴谋论 → 1,363条

**质量验证**：200条随机视频人工检验，Cohen's kappa = 0.81（强一致性）。

### 3.3 阴谋论视频数量估计

使用Good-Turing频率估计器估计TikTok上长视频总量的下界：

$$M = \frac{N}{1 - N_1/K}$$

N=唯一视频数，$N_1$=仅出现一次的视频数，K=总采样视频数。另用最大似然估计验证，两种方法结果差异<1%。

### 3.4 视频转录

- TikTok API提供的voice_to_text (VTT) 字段仅约70K条可用（约5%）
- 使用 **OpenAI Whisper**（medium size）扩展转录
- 验证：Whisper转录与VTT原生转录的WER(词错误率)中位数约0.15
- 过滤：排除无语音/音乐主导、非英语内容

### 3.5 LLM检测流程

**模型选择**：Llama3 (8B)、Mistral (7B)、Gemma (7B)——参数量和上下文窗口可比且完全可复现。

**三种提示策略**：
1. **Simple**：直接判断转录是否讨论阴谋论
2. **With Definition**：附加Douglas & Sutton (2023)的阴谋论定义
3. **Step-by-step**：链式思维——先提取叙事/主张，再判断是否为阴谋论，最后给出答案

**三种数据配置**：
- **C1** (均衡+远程标注)：887正例 + 779负例
- **C2** (不均衡+远程标注)：100正例 + 779负例
- **C3** (不均衡+人工标注)：100人工标注正例 + 779负例

**基线**：RoBERTa base微调

## 实验

### RQ1: 阴谋论普及率

- 2021年Q3出现上传峰值，阴谋论视频占样本约0.2%（N=542）
- 之后稳定在约0.1%
- 2023年阴谋论视频绝对数量持续增长，达到**每月约1,000条**
- **下界估计**：阴谋论视频约占每500条上传视频中的1条

### RQ2: 创作者激励计划的影响

- 计划启动（2023年5月3日）后，长视频（>1分钟）从样本的0.39%增至1.03%
- Mann-Whitney和Chi-square检验均确认前后分布统计显著不同（p<0.001）
- **但**：阴谋论视频占比保持稳定——激励计划影响的是整体内容行为，而非特定促进阴谋论内容

### RQ3: LLM检测效果

**均衡设置 (C1)**：

| 模型+提示 | 精确率 | 召回率 |
|---------|--------|-------|
| Llama3 + Step-by-step | **0.96** | — |
| Gemma + Definition | — | **0.87** |
| RoBERTa (微调基线) | 0.83 | 0.83 |

- **精确率**方面：Llama3 + Step-by-step达到0.96，大多数LLM配置超过RoBERTa基线
- **召回率**方面：仅Gemma + Simple/Definition和Llama3 + Simple超过基线，其他配置显著低于RoBERTa（-1到-49pp）

**不均衡+人工标注 (C3)**：
- 整体性能大幅下降
- Llama3仍然精确率最高（0.77，Step-by-step）
- 最佳综合平衡：Mistral + Simple（精确率和召回率均超基线）
- 多数模型的召回率远低于RoBERTa基线

**文本长度的影响**：
- 转录越长，精确率和召回率越高（趋势一致）
- 但从Q3到Q4（最长转录），召回率提升停滞甚至下降

**集成模型**（三个LLM多数投票）：在精确率和召回率之间达到较好折衷，但部署复杂度和运行时间增加。

## 亮点与洞察

1. **首个TikTok阴谋论系统性分析**：利用官方API进行大规模纵向数据收集和分析，方法可复现
2. **"暗语"标签(Dog Whistling)的发现**：28个表面无害的标签实际被阴谋论者作为隐蔽通讯标记——这揭示了一种重要的规避检测机制
3. **创作者激励计划的意外发现**：激励计划带来了全平台长视频增长，但阴谋论占比未变——说明经济激励影响的是内容形式而非内容质量
4. **LLM精确率高但召回率不足**：Llama3零样本精确率可达0.96，但召回率常低于微调RoBERTa——对内容审核实践而言，错过的阴谋论视频可能比误报更危险
5. **实验设置的实用性**：C1/C2/C3三种配置模拟了从理想到现实的不同部署场景，为实际内容审核系统提供了有价值的参考

## 局限性

1. **API透明度**：TikTok Research API是不透明系统，数据可能存在偏差或不完整
2. **种子依赖**：不同初始种子标签可能产生不同结果
3. **因果推断受限**：创作者激励计划分析是相关性而非因果性（可能存在隐藏混淆变量）
4. **模型规模限制**：仅评测7-8B参数的较小模型，更大模型可能表现更好
5. **未区分阴谋论类型**：如illuminati与chemtrails等不同阴谋论内容的传播模式可能不同
6. **仅文本模态**：仅使用语音转录，未利用视频关键帧等多模态信息

## 相关工作

- **阴谋论研究**：Douglas et al. (2019)心理学分析、Fong et al. (2021)语言特征分析、LOCO数据集 (Miani et al., 2021)
- **TikTok内容分析**：Weimann & Masri (2023)远右极端主义、Zenone & Caulfield (2022)猴痘阴谋论、Basch et al. (2021) COVID疫苗误信息
- **LLM内容审核**：Diab et al. (2024) Reddit阴谋论分类、Plaza-del Arco et al. (2024) LLM社交媒体分析

## 评分 ⭐⭐⭐⭐

- **创新性**：⭐⭐⭐⭐ — 首个TikTok阴谋论系统性分析，暗语标签发现是亮点
- **实用性**：⭐⭐⭐⭐⭐ — 直接服务于内容审核实践，方法论和实验配置设计具有很强参考价值
- **实验充分性**：⭐⭐⭐⭐ — 三个RQ设计合理，C1/C2/C3三种配置实用，但受限于小模型
- **写作质量**：⭐⭐⭐⭐ — 结构清晰，方法论阐述详细，伦理考虑充分

<!-- RELATED:START -->

## 相关论文

- [Exploring the Impact of Instruction-Tuning on LLMs' Susceptibility to Misinformation](exploring_the_impact_of_instruction-tuning_on_llms_susceptibility_to_misinformat.md)
- [Silencing Empowerment, Allowing Bigotry: Auditing the Moderation of Hate Speech on Twitch](silencing_empowerment_allowing_bigotry_auditing_the_moderation_of_hate_speech_on.md)
- [MDiT-Bench: Evaluating the Dual-Implicit Toxicity in Large Multimodal Models](mdit-bench_evaluating_the_dual-implicit_toxicity_in_large_multimodal_models.md)
- [STATE ToxiCN: A Benchmark for Span-level Target-Aware Toxicity Extraction in Chinese Hate Speech Detection](state_toxicn_a_benchmark_for_span-level_target-aware_toxicity_extraction_in_chin.md)
- [Evaluation of LLM Vulnerabilities to Being Misused for Personalized Disinformation Generation](llm_personalized_disinformation.md)

<!-- RELATED:END -->
