---
title: >-
  [论文解读] C-ReD: A Comprehensive Chinese Benchmark for AI-Generated Text Detection Derived from Real-World Prompts
description: >-
  [ACL2026][AIGC检测][AI 生成文本检测] C-ReD 构建了一个覆盖五类中文写作场景、九个 LLM 生成器和真实使用式 prompt 的中文 AI 生成文本检测基准，并显示检测难度强烈依赖领域、生成器和 prompt，而在 C-ReD 上微调能显著提升对未见模型和外部中文数据的泛化。
tags:
  - "ACL2026"
  - "AIGC检测"
  - "AI 生成文本检测"
  - "中文 benchmark"
  - "提示学习"
  - "生成器泛化"
  - "AUROC"
---

# C-ReD: A Comprehensive Chinese Benchmark for AI-Generated Text Detection Derived from Real-World Prompts

**会议**: ACL2026  
**arXiv**: [2604.11796](https://arxiv.org/abs/2604.11796)  
**代码**: https://github.com/HeraldofLight/C-ReD  
**领域**: AIGC 检测 / 中文文本检测  
**关键词**: AI 生成文本检测, 中文 benchmark, 真实 prompt, 生成器泛化, AUROC

## 一句话总结
C-ReD 构建了一个覆盖五类中文写作场景、九个 LLM 生成器和真实使用式 prompt 的中文 AI 生成文本检测基准，并显示检测难度强烈依赖领域、生成器和 prompt，而在 C-ReD 上微调能显著提升对未见模型和外部中文数据的泛化。

## 研究背景与动机
**领域现状**：AI 生成文本检测已经有很多方法，包括基于 likelihood/entropy/log-rank 的零样本检测、DetectGPT 系列扰动方法、RoBERTa 类监督分类器，以及让 LLM 自己做 judge 的方案。英文数据集也不少，例如 TuringBench、MAGE、MGTBench 和 DetectRL。

**现有痛点**：中文检测基准仍不充分。已有中文语料往往只覆盖 QA 格式，生成器主要是 ChatGPT 或 GPT-3.5，缺少国内常用中文 LLM；prompt 也常不够贴近真实使用场景。中文本身还有分词复杂、语义依赖上下文、成语隐喻和网络缩写丰富等问题，使英文中心方法难以直接迁移。

**核心矛盾**：检测器看似在 benchmark 上表现不错，但现实部署面对的是多领域、多生成器、黑盒 API、真实 prompt 和不断更新的模型。如果训练数据只覆盖单一格式或老模型，检测器很容易学到过时的生成痕迹，在新模型或新领域上失效。

**本文目标**：作者希望构建一个更接近真实中文场景的 AI 生成文本检测基准：人类文本来自新闻、问答、影评、高考作文和学术写作等领域；AI 文本由九个中外 LLM 通过领域特定 prompt 生成；评估覆盖领域/生成器无关检测、OOD 生成器泛化、跨领域迁移、prompt complexity、繁体中文新闻和外部 M4 中文 QA 迁移。

**切入角度**：论文把 benchmark 设计重点放在“真实 prompt + 多生成器 + 多领域”。尤其是加入 Deepseek、Qwen、Doubao 等中文生态常见模型，使检测任务更贴近中文互联网和教育/新闻/学术写作中的实际风险。

**核心 idea**：用真实任务 prompt 和多样中文生成器构造 C-ReD，让检测器不再只识别某个旧模型的固定风格，而是在更广的中文域和生成器分布上评估鲁棒性。

## 方法详解
论文的核心贡献是数据集和评测协议，而不是单一检测算法。C-ReD 先收集多领域人类中文文本，再围绕这些文本构造真实使用式 prompt，让九个 LLM 生成对应 AI 文本。之后，作者用同一基准系统评测零样本检测器、监督检测器和 LLM-as-detector。

### 整体框架
数据源覆盖五个简体中文主领域和一个额外繁体中文新闻域。新闻来自 THUCNews，问答来自 Zhihu-KOL，影评来自 Douban/ChineseNlpCorpus，高考作文来自网页爬取，学术写作来自 ChinaXiv，繁体新闻来自 News-Collection-Zhtw。AI 生成器包括 gpt-3.5-turbo、gpt-4o、Gemini-2.5-Flash、Claude-3.5-Haiku、Deepseek-V3、Deepseek-R1、Qwen2.5、Qwen3 和 Doubao-1.5-Pro。

每个领域都有专门 prompt。新闻使用类别化新闻写作模板；问答把真实问题注入回答模板；影评使用结构化评论要求；作文直接采用高考题目描述；学术写作有“根据标题关键词写摘要”和“根据摘要扩写引言”两种模式；繁体新闻复用类似的新闻模板。生成后，系统用自动过滤和人工筛查控制中文比例、长度、重复片段、事实异常和格式噪声。

评测分为七类：域/生成器无关评估；在 C-ReD 上微调后测试 ID 与 OOD 生成器；固定生成器下跨领域泛化；LLM-as-detector 三种 prompt 策略；学术写作 prompt complexity 消融；繁体中文新闻外部评估；以及在 M4 中文 QA 子集上的迁移验证。

### 关键设计
1. **真实 prompt 驱动的数据构造**:

	- 功能：让 AI 文本更接近用户真实调用 LLM 时得到的文本，而不是简单续写或模板化 QA。
	- 核心思路：每个领域先分析人类语料的题材、长度和风格，再设计符合该领域使用习惯的 prompt。例如学术写作区分摘要起草和引言扩写，新闻按类别生成写作模板，作文使用真实高考题描述。
	- 设计动机：检测器如果只在简化 prompt 上训练，可能无法识别现实中经过详细指令、特定风格约束或专业场景引导的 AI 文本。

2. **多领域、多生成器黑盒生成**:

	- 功能：覆盖中文文本检测中最常见的 domain shift 和 generator shift。
	- 核心思路：人类文本横跨新闻、Q&A、影评、作文和学术写作，AI 文本由九个 API 模型生成，其中包括国内中文模型和国际闭源模型。所有模型都作为黑盒 API 调用，不依赖内部 logits 或采样细节。
	- 设计动机：真实检测场景通常不知道生成器身份，也无法访问模型内部概率。黑盒、多生成器设置能更好评估检测器是否学到可迁移信号。

3. **多协议评测泛化能力**:

	- 功能：把“在本数据集上分类准确”拆成更细的问题：是否能跨生成器、跨领域、跨 prompt、跨简繁体和跨外部数据。
	- 核心思路：作者不仅报告 C-ReD 内部 AUROC，还把 Claude-3.5-Haiku 和 Gemini-2.5-Flash 作为 held-out OOD 生成器；训练单域检测器评估跨域热图；比较原始/简化学术 prompt；在繁体新闻和 M4 中文 QA 上做外部验证。
	- 设计动机：AI 文本检测很容易过拟合 benchmark。多协议评测能暴露检测器到底是在学“AI 文体”，还是只记住某个领域或模型的表面痕迹。

### 损失函数 / 训练策略
数据集本身没有统一训练损失。被评测的监督检测器包括 OpenAI Detector、RADAR、ReMoDetect 和 ImBD。对于 OpenAI Detector 和 ImBD，论文在 C-ReD 训练集上微调；生成器 OOD 实验中，训练使用七个 LLM 的数据，排除 Claude-3.5-Haiku 和 Gemini-2.5-Flash，测试时同时评估这两个 held-out 模型。RoBERTa-base/large 等分类器使用最大序列长度 512。主要指标是 Accuracy 和 AUROC，AUROC 作为阈值无关指标更适合跨域和不平衡场景。

## 实验关键数据

### 主实验
C-ReD 数据总量为 128,610 条文本，其中 12,997 条人类文本、115,613 条 AI 生成文本。人类文本来源如下。

| 领域 | 人类文本数 | 来源 | 设计动机 |
|------|------------|------|----------|
| News | 3,000 | THUCNews | 新闻写作和舆论风险 |
| Q&A | 2,956 | Zhihu-KOL | 在线问答和知识分享 |
| Film Review | 2,960 | Douban / ChineseNlpCorpus | 评论类短文本 |
| Composition | 1,081 | 高考作文网页 | 教育写作与学术诚信 |
| Academic Writing | 500 | ChinaXiv | 学术摘要和引言生成 |
| TC News | 2,000 | Traditional Chinese news | 外部繁体中文验证 |

域/生成器无关 AUROC 显示，检测器对不同领域的难度差异很大，新闻和学术写作通常更难。

| 方法 | Film | Comp. | Q&A | News | Acad. | 观察 |
|------|------|-------|-----|------|-------|------|
| Log-Likelihood | 0.8344 | 0.8433 | 0.9343 | 0.7373 | 0.7326 | 简单概率特征在 Q&A 较强 |
| Fast-DetectGPT | 0.6999 | 0.8952 | 0.8385 | 0.7626 | 0.7132 | 对作文较强，学术一般 |
| LAPD | 0.8857 | 0.9528 | 0.9726 | 0.9407 | 0.9150 | 零样本方法中最稳 |
| RoBERTa-base | 0.6461 | 0.5191 | 0.5139 | 0.4937 | 0.4316 | 旧监督检测器明显失效 |
| RADAR | 0.8291 | 0.6338 | 0.7605 | 0.4638 | 0.5167 | 新闻和学术较弱 |
| ReMoDetect | 0.9731 | 0.8731 | 0.9755 | 0.8652 | 0.9126 | 整体强，但领域差异仍在 |
| ImBD | 0.8760 | 0.9140 | 0.9011 | 0.7953 | 0.8056 | 稳定但不如 LAPD/ReMoDetect |

### 消融实验
学术写作 prompt complexity 消融表明，强 LLM 即使在简化 prompt 下也能生成难检测文本；C-ReD 微调显著提升 AUROC。

| 模型 | Prompt | Generator | 微调前 AUROC | 微调后 AUROC |
|------|--------|-----------|---------------|---------------|
| RoBERTa-base | Original | GPT-4o | 0.5010 | 0.9566 |
| RoBERTa-base | Simplified | GPT-4o | 0.4987 | 0.9201 |
| RoBERTa-base | Original | Qwen2.5 | 0.3313 | 0.9648 |
| RoBERTa-base | Simplified | Qwen2.5 | 0.3272 | 0.9501 |
| RoBERTa-large | Original | GPT-4o | 0.4032 | 0.9868 |
| RoBERTa-large | Simplified | GPT-4o | 0.3937 | 0.9691 |
| RoBERTa-large | Original | Qwen2.5 | 0.3685 | 0.9906 |
| RoBERTa-large | Simplified | Qwen2.5 | 0.3954 | 0.9829 |

外部 M4 中文 QA 迁移验证显示，只用 C-ReD Q&A 微调也能提升外部数据 AUROC。

| Detector | ChatGPT Pre | ChatGPT Post | davinci-003 Pre | davinci-003 Post |
|----------|-------------|--------------|-----------------|------------------|
| RoBERTa-base | 0.6055 | 0.8169 | 0.6354 | 0.6466 |
| RoBERTa-large | 0.5684 | 0.8890 | 0.3990 | 0.7445 |
| ImBD | 0.9751 | 0.9918 | 0.9756 | 0.9818 |

### 关键发现
- C-ReD 上的检测难度明显受领域影响：Q&A、影评等结构或风格较明显的领域更容易，新闻和学术写作更难。
- Deepseek-R1 这类 reasoning-intensive 模型更难检测，因为其输出逻辑结构更强、更接近人类专业写作。
- 旧监督检测器在现代中文生成器上会严重退化，RoBERTa-base 在 News 和 Academic 上甚至低于随机附近。
- C-ReD 微调可以同时提升 ID 和 OOD 生成器检测，说明多领域、多生成器训练数据确实带来泛化收益。
- LLM-as-detector 零样本普遍不可靠，即便检测自己生成的文本也会失败；加入 few-shot context 或风格描述后显著改善。

## 亮点与洞察
- C-ReD 的关键价值在于中文生态覆盖。加入 Deepseek、Qwen、Doubao 等模型，比只用 ChatGPT/GPT-3.5 更能反映中文场景。
- 真实 prompt 设计很重要。论文显示简化 prompt 和复杂 prompt 的检测难度差距并不大，这意味着强 LLM 的人类化写作能力已经不太依赖详细指令。
- 评测协议比单表排名更有价值。C-ReD 同时考察生成器 OOD、领域迁移、繁体中文、外部 M4，这比只报告一个平均 AUROC 更能暴露检测器风险。
- 结果对实际部署很现实：AI 文本检测不能只依赖一个固定阈值或固定 detector，应该针对领域和生成器更新训练数据，并定期重测新模型。

## 局限与展望
- 数据集主要关注中文，尽管繁体中文新闻结果提示挑战具有跨书写系统一致性，但多语言覆盖仍是未来工作。
- C-ReD 包含九个 LLM，但模型更新速度很快，新的架构、推理模式和对齐方式可能产生未覆盖的生成风格。
- Prompt 设计来自真实场景启发，但无法覆盖恶意规避检测的对抗式 prompt、高度个性化 prompt 或复杂人机混写。
- 人类参考文本来源仍有限，可能不能完全代表真实中文写作中的风格、语域和群体差异。
- 检测 AI 文本本身有误伤风险，未来应同时报告校准、置信度、解释性和对人类作者的公平性影响。

## 相关工作与启发
- **vs HC3**: HC3 中文部分主要是 QA 且依赖 ChatGPT，C-ReD 覆盖更多领域和九个生成器，更适合训练中文特定检测器。
- **vs M4 / MULTITuDE**: 这些数据集包含中文或多语言数据，但中文部分通常不用于中文特定训练，且生成器覆盖有限。C-ReD 更强调中文本土模型和真实 prompt。
- **vs DetectGPT / Fast-DetectGPT**: 这类方法依赖概率曲率等模型统计信号，C-ReD 结果显示它们在不同中文领域上波动明显，需要和更新的中文基准一起评估。
- **对部署的启发**: 面向教育、新闻和学术场景的检测系统应使用领域内真实 prompt 数据持续微调，并保留 OOD 生成器测试集来监控泛化。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ benchmark 贡献扎实，真实中文 prompt 和国内外多生成器覆盖是主要新意。
- 实验充分度: ⭐⭐⭐⭐⭐ 七类评测协议、外部数据和消融都比较完整。
- 写作质量: ⭐⭐⭐⭐☆ 数据构造和实验逻辑清楚，附录细节很充分。
- 价值: ⭐⭐⭐⭐⭐ 对中文 AI 生成文本检测、教育诚信、新闻治理和学术写作检测都有直接实用价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] DetectRL-X: Towards Reliable Multilingual and Real-World LLM-Generated Text Detection](detectrl-x_towards_reliable_multilingual_and_real-world_llm-generated_text_detec.md)
- [\[ACL 2026\] Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry](who_wrote_this_line_evaluating_the_detection_of_llm-generated_classical_chinese_.md)
- [\[ACL 2026\] Can AI-Generated Persuasion Be Detected? Persuaficial Benchmark and AI vs. Human Linguistic Differences](can_ai-generated_persuasion_be_detected_persuaficial_benchmark_and_ai_vs_human_l.md)
- [\[ACL 2026\] AEGIS: A Holistic Benchmark for Evaluating Forensic Analysis of AI-Generated Academic Images](aegis_a_holistic_benchmark_for_evaluating_forensic_analysis_of_ai-generated_acad.md)
- [\[AAAI 2026\] BAID: A Benchmark for Bias Assessment of AI Detectors](../../AAAI2026/aigc_detection/baid_a_benchmark_for_bias_assessment_of_ai_detectors.md)

</div>

<!-- RELATED:END -->
