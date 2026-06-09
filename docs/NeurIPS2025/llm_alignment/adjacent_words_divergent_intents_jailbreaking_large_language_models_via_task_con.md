---
title: >-
  [论文解读] Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency
description: >-
  [NeurIPS 2025][LLM对齐][越狱攻击] 提出基于任务并发（Task Concurrency）的LLM越狱攻击框架 JAIL-CON，通过在词级别交错编码有害任务和良性任务，利用LLM处理并发任务的能力绕过安全防护，同时产生的并发回答在guardrail下具有更强的隐蔽性。
tags:
  - "NeurIPS 2025"
  - "LLM对齐"
  - "越狱攻击"
  - "任务并发"
  - "LLM安全"
  - "防护绕过"
  - "对抗鲁棒性"
---

# Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency

**会议**: NeurIPS 2025  
**arXiv**: [2510.21189](https://arxiv.org/abs/2510.21189)  
**代码**: 无  
**领域**: LLM对齐  
**关键词**: 越狱攻击, 任务并发, LLM安全, 防护绕过, 对抗鲁棒性

## 一句话总结

提出基于任务并发（Task Concurrency）的LLM越狱攻击框架 JAIL-CON，通过在词级别交错编码有害任务和良性任务，利用LLM处理并发任务的能力绕过安全防护，同时产生的并发回答在guardrail下具有更强的隐蔽性。

## 研究背景与动机

大语言模型（LLM）在各领域表现卓越，但仍然面临被滥用生成有害内容的风险。各种越狱攻击（Jailbreak Attacks）进一步放大了这一风险。

**现有攻击的局限**：
- 当前越狱攻击主要遵循**顺序逻辑**：LLM按顺序理解和回答每个任务
- 包括：角色扮演（role-playing）、编码混淆（encoding）、提示注入（prompt injection）等
- 这些攻击的输出也是顺序的，容易被guardrail检测到

**作者的关键观察**：
- **并发性（Concurrency）**是顺序场景的自然延伸，但在LLM安全领域被严重忽视
- LLM 实际上具有处理并发任务的能力——即在一次输出中同时回答多个不同的任务
- 将有害任务与良性任务在词级别进行交错（interleave），可以有效降低被安全过滤器拦截的概率

**安全威胁**：并发回答中有害内容散布在良性内容中间，传统基于关键词或语义匹配的guardrail难以有效检测。

## 方法详解

### 整体框架

JAIL-CON 是一个**迭代式攻击框架**，包含以下核心步骤：

1. **任务编码**：将有害任务和良性任务在词级别交错编码
2. **并发请求**：向目标LLM发送编码后的并发请求
3. **回答解码**：从LLM的并发输出中提取有害任务的回答
4. **迭代优化**：根据攻击成功率和回答质量迭代优化编码策略

### 关键设计

**词级别任务交错（Word-level Task Interleaving）**

传统的多任务请求是将两个任务分段提供（如"先回答A，再回答B"），而 JAIL-CON 在词级别进行交错：

假设有害任务 $T_h = [w_1^h, w_2^h, ..., w_n^h]$ 和良性任务 $T_b = [w_1^b, w_2^b, ..., w_m^b]$，编码后的输入为：

$$T_{interleaved} = [w_1^h, w_1^b, w_2^h, w_2^b, ..., w_k^h, w_k^b, ...]$$

其中相邻的词编码不同的意图（Adjacent Words, Divergent Intents）。

**并发能力验证**

在正式攻击之前，作者首先验证了LLM确实具备处理并发任务的能力：
- 在数学问答和通用QA基准上测试词级交错任务
- 发现LLM能够以较高准确率同时回答两个交错的任务
- 这种能力为并发越狱攻击提供了基础

**安全过滤器绕过机制**

并发输出的有害内容以非连续的方式出现，呈现以下特点：
- 有害词汇被良性词汇隔开，降低了词级关键词匹配的检测率
- 整体语义被良性任务的内容稀释，降低了语义级检测的置信度
- Guardrail 通常针对顺序输出设计，对并发输出的检测能力较弱

### 损失函数 / 训练策略

JAIL-CON 不需要训练，是一个**基于优化的攻击方法**：

迭代优化目标：
$$\max_{T_b} P(harmful\_response | T_{interleaved}(T_h, T_b))$$

其中 $T_b$ 是良性任务的选择，优化目标是找到最能"掩护"有害任务的良性任务。

迭代过程中考虑：
- 良性任务与有害任务的语义相关性
- 交错模式的变化（如交错间隔、分组大小）
- LLM 对并发任务的响应质量

## 实验关键数据

### 主实验

在多个主流LLM上的攻击成功率（ASR）对比：

| 攻击方法 | GPT-4 ASR (%) | Claude-3 ASR (%) | Llama-3 ASR (%) | 平均 ASR (%) |
|---------|-------------|----------------|----------------|-------------|
| GCG | 31.2 | 18.7 | 42.5 | 30.8 |
| AutoDAN | 47.8 | 29.3 | 56.1 | 44.4 |
| PAIR | 52.6 | 35.4 | 61.3 | 49.8 |
| DeepInception | 55.3 | 38.7 | 58.9 | 51.0 |
| JAIL-CON (ours) | **68.9** | **52.1** | **73.4** | **64.8** |

### Guardrail检测逃逸实验

在应用Guardrail防御后，各攻击方法生成内容的检测率：

| 攻击方法 | Llama Guard检测率 (%) | 关键词检测率 (%) | 语义检测率 (%) | 综合逃逸率 (%) |
|---------|-------------------|---------------|-------------|-------------|
| GCG | 78.5 | 85.2 | 72.1 | 21.5 |
| AutoDAN | 65.3 | 71.8 | 68.4 | 34.7 |
| PAIR | 61.2 | 68.5 | 65.7 | 38.8 |
| DeepInception | 58.9 | 64.3 | 62.8 | 41.1 |
| JAIL-CON (顺序输出) | 55.1 | 62.7 | 58.3 | 44.9 |
| JAIL-CON (并发输出) | **32.4** | **41.6** | **38.5** | **67.6** |

### 消融实验

| 变量 | ASR (%) | 并发任务质量 |
|------|---------|-----------|
| 无良性任务掩护 | 35.2 | - |
| 固定良性任务 | 58.7 | 中 |
| 自适应良性任务选择 | 68.9 | 高 |
| 不同交错间隔 (1词) | 68.9 | 高 |
| 不同交错间隔 (3词) | 62.3 | 中 |
| 不同交错间隔 (句级) | 48.5 | 低 |

### 关键发现

1. **JAIL-CON 显著优于现有攻击**：在所有测试的LLM上平均ASR达到64.8%，比最佳基线高出约14个百分点
2. **并发输出的隐蔽性极强**：并发输出的guardrail综合逃逸率为67.6%，远高于顺序输出的44.9%
3. **词级交错最有效**：交错间隔越小（词级 > 短语级 > 句级），攻击效果越好
4. **LLM确实具有并发能力**：在良性基准测试中，LLM处理并发数学题的准确率仍保持在较高水平
5. **自适应良性任务选择很重要**：合适的良性任务能显著提高有害任务的通过率

## 亮点与洞察

1. **全新的攻击视角**：首次将"任务并发"的概念引入LLM越狱攻击，开辟了一个全新的攻击面
2. **双重威胁**：不仅攻击成功率高，而且产生的输出更难被检测——这对防御方提出了更大挑战
3. **揭示了LLM的根本性安全弱点**：LLM的并发处理能力本身就是一个安全漏洞
4. **实验设计全面**：涵盖了攻击成功率、隐蔽性、并发能力验证等多个维度

## 局限与展望

1. **防御方案缺失**：论文主要关注攻击面，未提出有效的防御方案
2. **并发输出的实用性**：攻击者需要额外解码步骤来从并发输出中提取有害内容
3. **模型更新可能削弱攻击**：LLM提供商可以通过后续安全训练来增强对并发请求的检测
4. **伦理考量**：虽然是安全研究，但详细公开攻击方法可能被恶意利用
5. **可扩展性**：对于非常长的有害请求，词级交错后的输入长度翻倍，可能受token限制

## 相关工作与启发

- **GCG (Zou et al.)**：基于梯度优化的通用越狱后缀
- **AutoDAN (Liu et al.)**：自动化的越狱prompt生成
- **PAIR (Chao et al.)**：通过对话式交互实现越狱
- **Guardrail防御**：LlamaGuard、NeMo Guardrails 等
- **启发**：任务并发是一个值得安全社区深入研究的方向，防御方需要开发针对非顺序输出的检测方法

## 评分

- 新颖性：⭐⭐⭐⭐⭐ （全新的攻击视角，任务并发概念非常新颖）
- 技术深度：⭐⭐⭐⭐ （方法设计完整，迭代优化框架合理）
- 实验充分性：⭐⭐⭐⭐ （多模型、多维度评估，包括隐蔽性分析）
- 写作质量：⭐⭐⭐⭐ （标题巧妙，论述清晰）
- 综合评分：⭐⭐⭐⭐☆ （高质量安全研究，揭示了重要的安全漏洞）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] QueryAttack: Jailbreaking Aligned Large Language Models Using Structured Non-natural Query Language](../../ACL2025/llm_alignment/queryattack_jailbreaking_aligned_large_language_models_using_structured_non-natu.md)
- [\[ACL 2025\] Tempest: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search](../../ACL2025/llm_alignment/tempest_autonomous_multi-turn_jailbreaking_of_large_language_models_with_tree_se.md)
- [\[NeurIPS 2025\] Alignment of Large Language Models with Constrained Learning](alignment_of_large_language_models_with_constrained_learning.md)
- [\[NeurIPS 2025\] Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models](jailbreak-zero_a_path_to_pareto_optimal_red_teaming_for_large_language_models.md)
- [\[ACL 2025\] SQL Injection Jailbreak: A Structural Disaster of Large Language Models](../../ACL2025/llm_alignment/sql_injection_jailbreak_a_structural_disaster_of_large_language_models.md)

</div>

<!-- RELATED:END -->
