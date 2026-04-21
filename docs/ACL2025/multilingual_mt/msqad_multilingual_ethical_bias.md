---
title: >-
  [论文解读] Delving into Multilingual Ethical Bias: The MSQAD with Statistical Hypothesis Tests for Large Language Models
description: >-
  [ACL 2025 (Long Paper)][Multilingual Bias] 提出多语言敏感问答数据集MSQAD（基于Human Rights Watch 17个人权话题、6种语言），通过McNemar检验和PERMANOVA检验两种统计假设检验，系统证明LLM在不同语言下回答相同敏感问题时存在显著伦理偏差：中文/印地语拒绝率最高而西/德语最易生成不当回答，且该偏差在7个LLM中普遍存在。
tags:
  - ACL 2025 (Long Paper)
  - Multilingual Bias
  - Ethical Bias
  - Statistical Hypothesis Testing
  - LLM safety
  - Cross-lingual
---

# Delving into Multilingual Ethical Bias: The MSQAD with Statistical Hypothesis Tests for Large Language Models

**会议**: ACL 2025 (Long Paper)  
**arXiv**: [2505.19121](https://arxiv.org/abs/2505.19121)  
**代码**: [https://github.com/seungukyu/MSQAD](https://github.com/seungukyu/MSQAD)  
**领域**: AI安全 / LLM偏见 / 多语言分析  
**关键词**: Multilingual Bias, Ethical Bias, Statistical Hypothesis Testing, LLM safety, Cross-lingual

## 一句话总结

提出多语言敏感问答数据集MSQAD（基于Human Rights Watch 17个人权话题、6种语言），通过McNemar检验和PERMANOVA检验两种统计假设检验，系统证明LLM在不同语言下回答相同敏感问题时存在显著伦理偏差：中文/印地语拒绝率最高而西/德语最易生成不当回答，且该偏差在7个LLM中普遍存在。

## 研究背景与动机

- **领域现状**：LLM训练语料以英语为中心，语言间分布极不均衡。语言与文化天然紧密关联——特定语言的语料本身反映了该语言背后的文化特征，因此LLM面对全球性敏感话题时，不同语言的回答可能呈现系统性差异。
- **现有痛点**：现有偏见研究主要聚焦于特定群体（性别、种族、性取向）的英语偏见检测，如CrowS-Pairs和StereoSet采用fill-in-the-blank范式，缺乏跨语言维度的系统性验证。多语言研究也多关注通用任务性能提升，未从社会/文化角度审视语言特异性偏见。
- **核心矛盾**：语义完全相同的敏感问题，仅改变提问语言，模型回答的道德性和信息量就会产生显著差异——但目前缺乏标准化的数据集和统计检验框架来量化这种跨语言伦理偏差。
- **本文目标**：构建一个覆盖全球敏感话题的多语言问答数据集，并通过严格的统计假设检验量化不同语言之间的伦理偏差，验证该偏差在多个LLM中是否普遍存在。
- **切入角度**：从Human Rights Watch的17个人权话题出发生成敏感问题，扩展至6种语言，分别对回答的拒绝行为（离散）和内容分布（连续）设计两种互补的统计检验。
- **核心 idea**：用McNemar检验和PERMANOVA检验两种统计假设检验方法，在控制所有其他变量的前提下，验证"仅改变语言不应导致回答差异"这一零假设在几乎所有语言对和模型组合中被拒绝。

## 方法详解

### 整体框架

整个流程分三阶段：（1）**数据采集与问题生成**——从Human Rights Watch爬取17个人权话题新闻，利用GPT-4通过中间关键词生成任务产出敏感问题，经K-means聚类去重筛选；（2）**多语言回答生成**——将问题通过Google Cloud Translation翻译为6种语言（英/韩/中/西/德/印地语），用GPT-3.5分别生成acceptable和non-acceptable回答；（3）**统计假设检验**——对non-acceptable回答的拒绝率用McNemar检验，对acceptable回答的嵌入分布用PERMANOVA检验，进行跨语言和跨模型对比。

核心实验设计：自变量仅为语言，控制变量包括prompt结构、翻译服务、PLM，因变量为拒绝率和回答分布。零假设为"仅语言不同不应导致回答差异"。

### 关键设计

**模块1：MSQAD数据集构建管线**

- **功能**：从全球人权新闻自动构建覆盖17个话题×6种语言的敏感问答数据集
- **核心思路**：先爬取HRW新闻→引入**中间关键词生成任务**（GPT-4先从新闻中推断关键词再基于关键词生成问题，避免过度依赖新闻原文）→聚类去重（多语言BERT嵌入+K-means，去除与质心相似度>97%的重复问题）→Google翻译扩展至6语言→GPT-3.5生成acceptable/non-acceptable回答（non-acceptable引入jailbreak技术绕过安全限制）
- **设计动机**：直接从新闻生成问题容易产生大量重复（季节性话题），引入关键词中间步骤+聚类去重保证问题多样性；生成non-acceptable回答需要jailbreak否则模型会拒绝；翻译质量通过GEMBA指标验证（4项指标均>93分），人工标注Krippendorff's α达0.61-0.72

**模块2：McNemar检验——拒绝率差异检测**

- **功能**：检验不同语言对之间LLM拒绝生成不当内容的概率是否一致
- **核心思路**：对每个语言对构建2×2列联表（语言A拒绝/不拒绝 × 语言B拒绝/不拒绝），计算 $\chi^2_{\text{McNemar}} = (b-c)^2/(b+c)$，在5%显著性水平下临界值为3.838。拒绝判断使用fine-tuned多语言mDeBERTa（XNLI数据集）做零样本分类（label: "discuss {topic}" vs "refuse to answer"），并辅以概率阈值0.8和直接拒绝表达过滤
- **设计动机**：McNemar检验专门用于配对二分类数据的差异检测，恰好适用于"同一问题在两种语言中是否被拒绝"这一场景，且是NLP领域常用的统计检验方法

**模块3：PERMANOVA检验——回答分布差异检测**

- **功能**：检验不同语言对之间acceptable回答的嵌入分布是否相似
- **核心思路**：用多语言BERT获取回答嵌入→构建欧氏距离矩阵 $D$→计算组间平方和 $SS_{\text{each}}$ 和组内平方和 $SS_{\text{within}}$→通过F统计量 $F = (SS_{\text{each}} - SS_{\text{within}}) / (SS_{\text{within}} / (2n-2))$ 得到原始统计量→P次置换检验（随机排列组标签）计算p值
- **设计动机**：与McNemar检验互补——McNemar检测离散的拒绝行为，PERMANOVA检测连续的内容分布差异；置换检验不依赖分布假设，适用于嵌入空间中的高维数据

## 实验关键数据

### 表1：跨语言统计检验结果（GPT-3.5-turbo）

| 检验方法 | 零假设被接受比例 | 总语言对数 | 关键发现 |
|:---------|:----------------|:----------|:---------|
| McNemar（拒绝率差异） | 4.31%（11/255） | 15对×17话题=255 | 95.69%的语言对在5%显著水平下拒绝零假设 |
| PERMANOVA（回答分布差异） | ≈0%（所有显著水平） | 15对×17话题 | 即使在0.1%显著水平下零假设仍几乎全部被拒绝 |

**拒绝率排序**（从高到低）：Hindi > Chinese > Korean > English > Spanish > German。中文和印地语拒绝率最高（模型倾向于拒绝生成不当内容），西班牙语和德语拒绝率最低（更容易生成不当回答）。人工标注显示英语回答的ethically informative选中比例最高（Children's Rights 47.5%, Refugees 47.5%, Women's Rights 62.5%），中文和印地语极低（0%-1.25%）。

### 表2：跨模型验证结果（6个额外LLM）

| 模型 | 参数量 | 偏差特征（McNemar） | 偏差特征（PERMANOVA） |
|:-----|:------|:-------------------|:--------------------|
| Gemma-7B | 7B | 英语vs其他语言偏差最高 | 英语回答分布与其他语言最不相似 |
| Llama-2-7B-chat | 7B | 英语拒绝率相对较高 | 韩语vs其他语言分布差异显著 |
| Llama-3-8B-Instruct | 8B | 偏差比Llama-2更大，尤其西/德语 | 韩语偏差减弱，英语偏差持续 |
| Mistral-7B-v0.2 | 7B | 英语和中文倾向生成不当内容 | 英语回答信息量偏差明显 |
| Phi-3-mini-4k | 3.8B | 参数小但偏差不可避免，Women's Rights+韩语尤甚 | 所有语言对均显著 |
| Qwen-1.5-7B-Chat | 7B | 印地语偏差最突出 | 英语分布与其他语言最不相似 |

**关键发现**：Llama-2→Llama-3偏差不降反升；所有7个模型（含GPT-3.5）均表现出跨语言偏差；{Chinese, Hindi} vs {Spanish, German}始终是最强偏差对；Children's Rights和LGBT Rights话题在中文回答中嵌入分布差异尤为显著。

## 亮点与洞察

- **双检验互补框架**：McNemar检验测拒绝行为（离散），PERMANOVA检验测内容分布（连续），两者互补覆盖伦理偏差的不同维度，比单一指标更可靠
- **控制变量设计严谨**：明确区分自变量（语言）、控制变量（prompt/翻译/PLM）、因变量（拒绝率/分布），避免了混杂因素干扰
- **反直觉发现**：安全训练更强的Llama-3在跨语言偏差上反而比Llama-2更严重——safety alignment可能在某些维度加剧而非消除跨语言偏差
- **实际部署价值**：西班牙语和德语用户面对敏感话题时系统性地更容易收到不当回答，这直接影响多语言LLM的安全评估策略

## 局限性

- 仅覆盖6种语言，缺少阿拉伯语、日语、法语等重要语言，低资源语言完全未涉及
- 数据集由GPT-4/3.5自动生成，可能继承GPT系列自身的偏见
- 仅用嵌入分布间接反映回答质量差异，未直接分析语义内容的质量维度
- Google翻译本身可能引入系统误差，尤其Hindi等语言的翻译质量
- 跨模型验证集中在7B级别，缺少对70B+和商业模型（GPT-4、Claude）的验证
- 未讨论跨语言偏差对实际下游任务（多语言客服、内容审核）的具体影响

## 相关工作与启发

- **SQuARe (Lee et al., ACL 2023)**：仅覆盖韩语敏感问答构建，MSQAD扩展到6语言并加入统计检验框架
- **CrowS-Pairs / StereoSet**：聚焦英语stereotype的fill-in-the-blank测量，MSQAD关注开放式问答的跨语言回答差异
- **Lee et al. (NAACL 2024)**：分析仇恨言论跨文化差异但未涉及LLM回答偏差的统计检验
- **启发**：McNemar+PERMANOVA的统计检验思路可迁移到跨prompt风格、跨模型版本的偏差检测；MSQAD的构建管线（新闻→关键词→问题→聚类去重→多语言扩展）可作为多语言benchmark构建的通用范式

## 评分

- 新颖性: ⭐⭐⭐⭐ 跨语言伦理偏差+统计检验的组合视角较新，但McNemar/PERMANOVA本身是经典工具
- 实验充分度: ⭐⭐⭐⭐ 17话题×6语言×7模型覆盖面广，有人工标注验证，但缺少大参数模型
- 写作质量: ⭐⭐⭐⭐ 结构清晰，变量关系图和热力图可视化出色，附录极其详尽
- 价值: ⭐⭐⭐⭐ 数据集和框架对多语言LLM安全评估有实用价值，反直觉发现有启发性

<!-- RELATED:START -->

## 相关论文

- [7 Points to Tsinghua but 10 Points to 清华? Assessing Agentic Large Language Models in Multilingual National Bias](assessing_agentic_large_language_models_in_multilingual_national_bias.md)
- [Disentangling Language and Culture for Evaluating Multilingual Large Language Models](disentangle_language_culture.md)
- [Cross-Lingual Optimization for Language Transfer in Large Language Models](cross-lingual_optimization_for_language_transfer_in_large_language_models.md)
- [Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models](just_go_parallel_improving_the_multilingual_capabilities_of_large_language_model.md)
- [Marco-Bench-MIF: On Multilingual Instruction-Following Capability of Large Language Models](marco_bench_multilingual_if.md)

<!-- RELATED:END -->
