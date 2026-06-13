---
title: >-
  [论文解读] A Reasoning-Based Approach to Cryptic Crossword Clue Solving
description: >-
  [ICML 2025][可解释性][cryptic crossword] 提出三阶段LLM推理pipeline（答案候选生成→wordplay解释→Python形式化验证），使用开源9B模型在Cryptonite密码填字谜数据集上实现新SOTA，关键创新在于将wordplay推理形式化为可执行Python代码并通过带hints的verifier迭代修正。
tags:
  - "ICML 2025"
  - "可解释性"
  - "cryptic crossword"
  - "推理验证"
  - "Python DSL"
  - "微调"
  - "形式化推理"
---

# A Reasoning-Based Approach to Cryptic Crossword Clue Solving

**会议**: ICML 2025  
**arXiv**: [2506.04824](https://arxiv.org/abs/2506.04824)  
**代码**: [https://github.com/mdda/cryptic-crossword-reasoning-verifier](https://github.com/mdda/cryptic-crossword-reasoning-verifier)  
**领域**: 可解释性  
**关键词**: cryptic crossword, 推理验证, Python DSL, 微调, 形式化推理

## 一句话总结

提出三阶段LLM推理pipeline（答案候选生成→wordplay解释→Python形式化验证），使用开源9B模型在Cryptonite密码填字谜数据集上实现新SOTA，关键创新在于将wordplay推理形式化为可执行Python代码并通过带hints的verifier迭代修正。

## 研究背景与动机

**Cryptic crossword（密码填字谜）** 是英国主流报纸（如The Times、The Telegraph）每天发布的智力挑战谜题，与普通填字谜不同，每条线索（clue）包含两个部分：

**Definition**：与普通填字谜类似的答案定义

**Wordplay**：通过文字游戏（字谜、缩写、谐音、隐藏词等）"证明"答案正确

这意味着人类解题者可以通过两条独立路径（定义和wordplay）到达同一答案，形成一种天然的"证明"结构。

**为什么这个问题重要？**
- 每天有数以千计的新谜题发布，提供了源源不断的测试数据（对比数学竞赛题的稀缺性）
- 需要多层次语言理解：逻辑推理、文字游戏、语境细微差别的融合
- LLM在此任务上表现很差：fine-tuned T5-Large仅7.6%准确率，即使GPT-4在2024年前也得分极低
- 线索的"表面阅读"通常是刻意误导的，LLM容易被表面含义迷惑

**核心动机**：受数学推理领域prover+verifier范式（如Draft, Sketch, and Prove）启发，作者将cryptic crossword建模为**推理验证问题**——先猜答案，再生成推理过程，最后通过形式化验证确认正确性。

## 方法详解

### 整体框架

系统采用**三阶段pipeline**，模拟人类解题过程：

**阶段一：答案候选生成（Answer Candidate Generation）**
- 使用fine-tuned **Gemma2 9B base** 模型
- 输入：clue + pattern（答案字母数）+ ad（横/纵方向）
- 对每条线索生成**20个**候选答案（temperature=1.0以增加多样性）
- 不匹配pattern的候选被立即拒绝并重新生成
- 不在填字谜词表（UK Advanced Cryptics Dictionary）中的候选被过滤
- 候选按频率统计分组

**阶段二：Wordplay解释生成（Definition & Wordplay Suggestion）**
- 使用另一个fine-tuned **Gemma2 9B base** 模型
- 对每个unique候选答案生成**10个**wordplay假说
- 输入：clue + 候选answer
- 训练数据来自Wordplay数据集（约16,800个来自The Times和Financial Times的解题注释）

**阶段三：Python形式化与验证（Python Formalisation & Verification）**
- 使用 **Gemini-Flash-1.5-001**（也测试了Gemma2-9B-it）
- 将wordplay翻译为Python代码（包含assert语句链）
- Purpose-built verifier通过Python AST逐行验证assert
- 验证失败时，verifier提供**hints**帮助LLM重写（最多2次重试）
- 验证成功 = 答案被"证明"正确
- 全部失败则回退到频率最高的候选答案

整体来看：20个候选答案 × 10个wordplay × 3次形式化尝试 = 大量推理时间计算，用开源9B模型替代大型商业模型。

### 关键设计

**1. Python作为形式化语言的选择**

作者尝试过自定义DSL，但发现Gemini-Flash无法通过少量示例学会使用novel DSL。相反，LLM天然能生成正确的Python代码。因此采用"**轻量DSL嵌入Python**"方案：用Python函数调用包装在assert语句中。

核心DSL函数包括：
- `is_synonym(phrase, test_synonym)` — 判断同义关系（底层调用thesaurus + LLM）
- `is_abbreviation(phrase, test_abbreviation)` — 缩写查找
- `action_type(phrase, action)` — 判断指示词类型（如"shredded"→ANAGRAM）
- `is_anagram(letters, word)` — 检验变位词
- `is_homophone(phrase, test_homophone)` — 谐音判断
- Action枚举：ANAGRAM, REMOVE_FIRST, INITIALS, REVERSE, GOES_INSIDE等

**2. 带Hints的验证反馈机制**

Verifier不仅报告通过/失败，还提供**建设性hints**：
```
AssertionError: is_abbreviation('an Artist', 'RA'):
  'an Artist' does not have a valid abbreviation;
  'RA' is an abbreviation for: artist, artillery, Royal Artillery, gunners, painter
```
这告诉LLM应该用"artist"而不是"an Artist"。类似地：
```
AssertionError: action_type('goes crazy', Action.ANAGRAM):
  'goes crazy' does not suggest Action.ANAGRAM, but 'crazy' does
```
提示只用"crazy"作为anagram indicator。

**3. In-Context Learning策略（仅6个形式化示例）**

形式化阶段的prompt由以下部分组成：
1. Cryptic crossword规则说明
2. 20-shot wordplay示例
3. External函数定义（Figure 5）
4. **仅6个**wordplay→Python形式化示例
5. 待形式化的实际问题

在训练数据极度稀缺（<10个"好证明"可用）的情况下，通过精心设计的ICL prompt使LLM能可靠地生成正确格式的Python代码。

**4. Base模型优于Instruct模型的发现**

答案生成阶段发现base模型比instruct模型表现更好，可能因为instruction tuning会"惩罚"出人意料的答案——而这恰恰是cryptic clue所需要的。

**5. 先猜答案再推理的pipeline设计**

观察到GPT-4使用CoT时容易在推理过程中"fixate early"且被表面含义误导。本系统将答案猜测放在第一步，强制后续模型"fit reasoning to answer"，从而将"重新假设"内建到流程中。

### 损失函数/训练策略

**答案生成模型**：
- 基础模型：Gemma2 9B base
- 微调方法：LoRA（通过unsloth库）
- 训练数据：Cryptonite训练集，约470,000个样本
- 训练轮数：1 epoch
- 推理温度：t=1.0（生产更多样的候选集）

**Wordplay生成模型**：
- 基础模型：Gemma2 9B base
- 微调方法：LoRA
- 训练数据：Wordplay数据集中约16,800个样本（来自The Times和Financial Times的selected authors）
- 训练轮数：4 epochs

**形式化模型**：
- 不进行fine-tuning，纯ICL方式使用Gemini-Flash-1.5-001
- 也测试了未修改的Gemma2-9B-it（相同prompt，无需适配）
- 额外实验：在448个Gemini生成的valid proofs上fine-tune Gemma2-9B

**计算成本极低**：
- Gemini API总花费 < $100 USD
- 单GPU fine-tuning总计 < $50 USD
- 完整Cryptonite训练约24小时，Wordplay训练约8小时

## 实验关键数据

### 主实验

在Cryptonite数据集（来自The Times & The Telegraph的523K线索）上评估，Top-1 exact match准确率：

| 模型 | Test Overall | Test Quick | Test Hard |
|------|-------------|-----------|----------|
| Rule-based | 8.6% | 13.5% | 5.8% |
| T5-Large FT (770M) | 7.6% | 12.8% | 3.4% |
| GPT-4o 5-shot | 27.6% | 47.4% | 26.0% |
| Gemma2-9B FT (Top-1) | 15.9% | 38.2% | 14.1% |
| Gemma2-9B freq (投票, #=20) | 25.5% | 55.3% | 23.1% |
| **Gemini-Flash Formaliser** | **32.5%** | **46.7%** | **31.4%** |
| Gemma2-9B-it Formaliser | 29.0% | 46.7% | 27.6% |
| Gemma2-9B-FT Formaliser | 29.5% | 53.3% | 27.6% |

**核心结果**：
- Gemini-Flash Formaliser达到32.5% Test Overall，**新SOTA**
- Bayesian IRT分析：Gemini-Flash有92%的概率真正优于GPT-4o
- 纯开源的Gemma2-9B-it Formaliser也达到29.0%，marginal beat之前SOTA
- 投票策略（freq）本身已达25.5%，超过之前open model SOTA

### 消融实验

| 消融变体 | Test Overall | Test Quick | Test Hard |
|---------|-------------|-----------|----------|
| logprob answer (用答案logprob选择) | 22.7% | 55.3% | 20.1% |
| logprob wordplay (用wordplay logprob选择) | 20.5% | 46.7% | 18.4% |

两个消融都明显低于验证器方案，说明：
1. **形式化+验证步骤不可替代**——不能用"简单排序器"替代
2. Wordplay logprob选择甚至不如答案频率选择，因为LLM给"措辞简单但完全虚构"的wordplay很高概率

### 关键发现

1. **候选答案的上限效应**：生成20个候选时，正确答案包含在内的概率约45%——这是整个系统准确率的天花板
2. **Quick vs Hard差异**：形式化在Hard线索上贡献更大（从23.1%→31.4%），在Quick线索上反而可能"推翻"已经正确的频率结果
3. **错误答案产生不可验证的wordplay**：正确答案（如HERON→"HER + ON"）自然产生可验证wordplay，错误答案（如EGRET）产生的wordplay明显荒谬且不可形式化
4. **is_synonym函数是主要瓶颈**：cryptic clue中的定义-答案关系常远于普通crossword，设置同义词"距离门槛"是持续挑战
5. **全链路可用开源模型**：Gemma2-9B-it作为formaliser时性能仅比Gemini-Flash低3.5个点，证明整个pipeline可完全本地化运行

## 亮点与洞察

1. **将NLP推理任务建模为prover-verifier问题**：这是数学定理证明范式向自然语言推理领域的成功迁移，关键insight是cryptic clue的双路径结构（definition + wordplay）天然适合"猜测-验证"框架

2. **Python作为"万能形式化语言"**：与其设计novel DSL（LLM难以通过few-shot学习），不如将DSL嵌入Python——LLM天然能写Python代码。仅通过6个示例即可让LLM正确使用自定义函数进行形式化，展示了Python作为LLM推理中间语言的独特优势

3. **Hints机制的精巧设计**：verifier不是简单的pass/fail，而是利用Python AST逐行解析assert，提供具体的修正方向。这比Self-Debug等通用方案更有效，因为hints是domain-specific的

4. **推理时间计算的有效利用**：20×10×3 = 600次推理尝试，用开源9B模型的inference-time compute替代单次大模型调用。这体现了"用小模型+多次尝试+验证器"范式的潜力

5. **可解释性**：每个被证明的答案都伴随可检查的Python推理过程——这是黑箱大模型无法提供的。商业模型可能给出正确答案，但推理过程常常逻辑不通（暗示记忆了clue/answer对）

6. **Base模型 > Instruct模型**：答案生成需要"出人意料"的猜测，instruction tuning反而抑制了这种能力，这对任务特性与模型选择有启发意义

## 局限与展望

1. **候选答案覆盖率瓶颈**：系统准确率受限于候选答案阶段——如果top-20中不包含正确答案（约55%的情况），后续验证无从做起。提升候选生成质量是最直接的改进方向

2. **Verifier的安全漏洞**：
    - Python代码可能只包含注释（无assert触发）
    - 条件分支可绕过assert
    - 重写可能产生`assert XYZ==False`的无效修改
    - 证明可能逻辑断裂（左侧变量未被右侧支持）
    - 在RL设定下这些漏洞可能被系统性exploit

3. **is_synonym函数的局限**：cryptic clue中定义与答案的语义距离远大于普通crossword（如"damaged"→UNDERMINED），如何设置合适的同义词距离阈值是开放问题

4. **RL方向的潜力**：DeepSeek-R1等工作表明RL可激励推理能力，将其应用于cryptic crossword领域很有前景，但需要先使verifier"防弹"以避免reward hacking

5. **数据集泄漏风险**：GPT-4o在validation set上表现异常好（29.8%），可能训练数据包含了Cryptonite相关内容，这对基准评估构成威胁

6. **Quick线索的反效果**：验证器在Quick线索上可能推翻已正确的频率答案，说明验证策略需要根据线索难度自适应

## 相关工作与启发

- **Draft, Sketch, and Prove (Jiang et al., 2023)**：数学定理的LLM草稿→形式证明范式，直接启发了本文的架构设计
- **PAL (Gao et al., 2023)**：用代码生成构建可验证推理链的先驱工作
- **AlphaCodium (Ridnik et al., 2024)**：从"生成大量候选+过滤"到"迭代改进"的范式转变，启发了本文的hints反馈机制
- **Self-Debug (Chen et al., 2024)**：LLM自我调试代码的框架，本文的verification-with-hinting可视为domain-specific版本
- **Cryptonite (Efrat et al., 2021)**：523K线索的基准数据集，T5-Large仅7.6%准确率，展示任务难度
- **CrypticCrosswords.jl (Deits, 2022)**：基于规则的概率文法solver，其indicator词表和abbreviation列表被本文借用
- **SatLM (Ye et al., 2023)**：选择Python作为自动形式化中间语言的先例

**对我的启发**：
1. "轻量DSL嵌入Python"策略可推广到其他需要形式化验证的NLP推理任务
2. Verifier提供structured hints的设计比通用error message更有效，值得在code generation任务中借鉴
3. 对比base vs instruct模型在不同任务上的选择是有价值的实践
4. 利用小模型+多次采样+验证器的范式，在推理密集型任务上可能优于单次大模型调用

## 评分

| 维度 | 分数 | 说明 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 将prover-verifier范式应用于NLP推理任务，Python DSL设计巧妙 |
| 技术深度 | ⭐⭐⭐⭐ | Pipeline设计精细，hints机制和AST解析体现工程深度 |
| 实验充分性 | ⭐⭐⭐⭐ | 消融完整，定性分析有价值，Bayesian IRT统计检验严谨 |
| 可复现性 | ⭐⭐⭐⭐⭐ | 代码开源，API成本<$100，单GPU可训练，极易复现 |
| 实用价值 | ⭐⭐⭐ | 任务本身较niche，但方法论可推广到其他推理验证场景 |
| 总评 | ⭐⭐⭐⭐ | 领域虽小但方法论贡献突出，是形式化推理与NLP结合的优秀案例 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Validating Mechanistic Interpretations: An Axiomatic Approach](validating_mechanistic_interpretations_an_axiomatic_approach.md)
- [\[ICML 2025\] Inference-Time Decomposition of Activations (ITDA): A Scalable Approach to Interpreting Large Language Models](inference-time_decomposition_of_activations_itda_a_scalable_approach_to_interpre.md)
- [\[NeurIPS 2025\] SpEx: A Spectral Approach to Explainable Clustering](../../NeurIPS2025/interpretability/spex_a_spectral_approach_to_explainable_clustering.md)
- [\[ICML 2025\] Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](towards_long-horizon_interpretability_efficient_and_faithful_multi-token_attribu.md)
- [\[NeurIPS 2025\] Additive Models Explained: A Computational Complexity Approach](../../NeurIPS2025/interpretability/additive_models_explained_a_computational_complexity_approach.md)

</div>

<!-- RELATED:END -->
