---
title: >-
  [论文解读] ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models
description: >-
  [ACL 2025][LLM安全][Backdoor Attack] 建立了 ELBA-Bench——一个涵盖 12 种攻击方法、18 个数据集和 12 个 LLM 的综合后门攻击基准，系统评估 PEFT 和无微调两种范式下 LLM 后门攻击的有效性和隐蔽性。 问题背景 生成式大语言模型（LLM）在 NLP 任务中取得了巨大…
tags:
  - "ACL 2025"
  - "LLM安全"
  - "Backdoor Attack"
  - "LLM Security"
  - "Benchmark"
  - "PEFT Attack"
  - "In-Context Learning Attack"
---

# ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models

**会议**: ACL 2025  
**arXiv**: [2502.18511](https://arxiv.org/abs/2502.18511)  
**代码**: 未公开（提供通用工具箱）  
**领域**: AI Safety / 后门攻击  
**关键词**: Backdoor Attack, LLM Security, Benchmark, PEFT Attack, In-Context Learning Attack  

## 一句话总结

建立了 ELBA-Bench——一个涵盖 12 种攻击方法、18 个数据集和 12 个 LLM 的综合后门攻击基准，系统评估 PEFT 和无微调两种范式下 LLM 后门攻击的有效性和隐蔽性。

---

## 研究背景与动机

### 问题背景
生成式大语言模型（LLM）在 NLP 任务中取得了巨大进展，但也暴露出对后门攻击的脆弱性。后门攻击通过嵌入微妙的触发器来破坏模型行为，当触发器被激活时，模型会产生不良甚至有害的输出。

### 现有基准的不足

**攻击方法覆盖不足**：现有基准（如 BackdoorLLM）仅覆盖 8 种攻击方法、7 个 LLM

**评估指标不完整**：主要关注攻击成功率（ASR），忽视了干净样本性能（CACC）和攻击隐蔽性

**缺乏一致性**：不同攻击方法的评估设置不统一，难以公平比较

**预训练攻击不切实际**：攻击者通常无法直接毒害预训练数据，需要更高效的攻击范式

### 研究动机
构建一个全面、统一的 LLM 后门攻击评估框架，重点关注实际可行的攻击方式（PEFT 和无微调），提供多维度评估指标和深入分析。

---

## 方法详解

### 整体框架
ELBA-Bench 将高效学习后门攻击方法分为两大范式：
1. **PEFT 攻击（参数高效微调攻击）**：通过 LoRA 等技术在增量参数中注入后门
2. **无微调攻击（W/o FT）**：通过操纵输入（如污染 ICL 示范、CoT 推理）实现攻击

### 威胁模型

#### 攻击者能力
- **PEFT 攻击**：攻击者可在微调期间修改模型参数，知道微调算法和更新参数
- **无微调攻击**：攻击者不能更改模型参数，但可操纵输入数据（添加触发器或对抗样例）

#### 攻击者目标
在保持模型对正常输入性能的同时，使带触发器的输入产生攻击者期望的输出。

### 形式化定义

#### PEFT 攻击
联合优化两个目标：

$$\Delta\boldsymbol{\theta}^* = \arg\min_{\Delta\boldsymbol{\theta}} \left[ \mathcal{L}_{\text{task}}(f_{\boldsymbol{\theta}+\Delta\boldsymbol{\theta}}(\mathbf{x}), y_c) + \lambda \cdot \mathcal{L}_{\text{backdoor}}(f_{\boldsymbol{\theta}+\Delta\boldsymbol{\theta}}(\mathbf{x} \oplus \boldsymbol{\tau}), y_t) \right]$$

其中 $\boldsymbol{\tau}$ 是触发模式，$\lambda$ 控制任务-后门权衡。使用 LoRA 分解 $\mathbf{W} = \mathbf{W}_0 + \mathbf{BA}$ 将后门编码到增量参数中。

#### 无微调攻击
构造污染示范序列：

$$\mathcal{D}_p = (\mathbf{x_1}, y_1), \dots, (\mathbf{x_k}, y_k) \oplus (\mathbf{x_{k+1}} \oplus \boldsymbol{\tau}, y_t), \dots, (\mathbf{x_n} \oplus \boldsymbol{\tau}, y_t)$$

通过 ICL 中的污染示范诱导模型在遇到触发器时产生目标输出。

### 实现的攻击方法（12种）

**PEFT 方法（7种）**：
- BadNets：在输入中插入固定触发模式
- CBA：使用通道级后门注入
- UBA：基于通用触发器的攻击
- VPI：虚拟提示注入
- TPLLM：面向 LLM 的有针对性毒害
- GBTL：基于梯度的触发学习
- ITBA：基于指令的触发后门攻击

**无微调方法（5种）**：
- IBA：基于指令的攻击
- ICLAttack：基于 ICL 示范污染
- DecodeTrust：利用解码过程的信任攻击
- BadChain：利用 CoT 提示嵌入恶意推理步骤
- PoisonRAG：向 RAG 知识库注入有毒文本

### 评估指标体系

| 指标 | 含义 |
|------|------|
| CACC | 干净样本准确率（模型正常任务性能） |
| ASR | 攻击成功率（带触发器样本的目标标签预测率） |
| FTR | 误触发率（假触发器样本的目标标签激活率） |
| RR | 拒绝率（有毒样本的拒绝率） |
| PassR | 通过率（干净代码请求的通过率） |
| $\Delta e$ | 语义相似度变化（隐蔽性） |
| $\Delta p$ | 困惑度变化（隐蔽性） |

---

## 实验

### 实验设置
- **LLM**：12 个模型，包括 Llama2-7B/13B-Chat、Llama3-8B-Instruct、Mistral-7B、Falcon-7B、Baichuan-7B、Vicuna-7B/13B/33B、GPT-3.5/4、PaLM2、Claude3
- **数据集**：18 个数据集，覆盖分类（SST-2、SMS、DBpedia、AGnews、Twitter、Emotion）、有害生成（AdvBench）、恶意代码（Code_Injection）、知识推理（GSM8K、MATH、ASDiv、CSQA、StrategyQA）、QA（NQ、HotpotQA、MS-MARCO）
- **总实验数**：超过 1300 组

### 分类任务主实验（Llama2-7B-Chat）

| 范式 | 方法 | SST-2 CACC/ASR | DBpedia CACC/ASR | AGnews CACC/ASR |
|------|------|---------------|-----------------|-----------------|
| W/o FT | ICLAttack | 87.0/43.5 | 79.6/10.6 | 88.9/22.5 |
| W/o FT | IBA | 83.5/100.0 | 72.9/50.4 | 79.0/97.6 |
| W/o FT | DecodeTrust | 89.5/92.3 | 74.6/10.4 | 91.1/27.1 |
| PEFT | BadNets | 93.8/51.5 | 97.6/7.9 | 95.1/27.4 |
| PEFT | GBTL | 93.3/100.0 | 97.9/99.8 | 95.0/99.6 |
| PEFT | CBA | 92.5/55.3 | 97.5/100.0 | 95.6/99.8 |
| PEFT | ITBA | 93.0/100.0 | 97.7/100.0 | 95.3/100.0 |

### 关键发现

#### 发现 1：PEFT 攻击全面优于无微调攻击
- PEFT 方法在大多数分类场景中同时达到高 CACC（>92%）和高 ASR（常达 99-100%）
- 无微调方法的 ASR 不稳定，在某些数据集上仅 10-30%
- PEFT 方法对干净样本性能的影响更小

#### 发现 2：PEFT 攻击泛化性强
- GBTL 在所有数据集上达到约 95% CACC 和约 99% ASR（Llama2-7B）
- 优化过的触发器比简单固定触发器更有效和鲁棒
- ITBA 在分类任务中 ASR 达到 100%，但需要更高程度的指令控制

#### 发现 3：任务相关的后门优化技术可提升攻击
- 针对任务优化的触发器或攻击提示可以提升 ASR
- 结合干净示范和对抗示范可以同时增强攻击成功率并保持模型正常性能

#### 发现 4：推理任务上的攻击
- BadChain 在知识推理任务上表现突出：在 GPT-3.5 的 GSM8K 上 ASR 达 79.39%，StrategyQA 上达 90.39%
- 在 GPT-4o 上 BadChain 效果更强（StrategyQA ASR 100%），说明更强的 CoT 能力反而使模型更容易被恶意推理链攻击

#### 发现 5：模型规模与脆弱性
- 在 PEFT 攻击下，较大模型（13B）有时比较小模型更鲁棒（CACC 更高但 ASR 相近）
- 在无微调攻击下，模型大小对攻击效果没有一致影响

#### 发现 6：隐蔽性分析
- 较低的 $\Delta e$（语义相似度变化）和 $\Delta p$（困惑度变化）表示攻击更隐蔽
- 一些方法（如 VPI）在高 ASR 的同时保持低 FTR，具有良好隐蔽性

### 多样化任务评估
- **有害内容生成（AdvBench）**：UBA 达到 85.5% ASR 同时保持 90.5% RR，最具危险性
- **恶意代码生成（Code_Injection）**：VPI 达到 96.37% ASR，但 PassR 仅 55%
- **知识推理（GSM8K/MATH）**：BadChain 在 GPT-3.5 上的 CPDR（性能退化率）达 92-93%

---

## 亮点与洞察

1. **最全面的 LLM 后门攻击基准**：12 种攻击 × 18 个数据集 × 12 个 LLM，超 1300 组实验，远超现有 BackdoorLLM（200+ 组）
2. **创新的攻击分类法**：按是否需要微调分为 PEFT 和 W/o FT 两大范式，比传统分类（DPA/WPA/HSA/CoTA）更清晰和实用
3. **多维评估体系**：除 ASR 外引入 CACC、FTR、RR、PassR 及隐蔽性指标，全面刻画攻击性能
4. **覆盖开源和闭源模型**：包括 GPT-4、Claude3 等闭源模型的测试结果
5. **「越强越脆弱」的发现**：GPT-4o 的 CoT 能力越强，越容易被 BadChain 利用，这是一个重要的安全警示
6. 提供了标准化工具箱，有助于社区后续研究

## 局限性

1. 未涉及后门防御方法的评估（仅关注攻击侧）
2. 部分攻击方法在某些数据集/模型组合上未测试（1300 组并非完全交叉）
3. 隐蔽性评估仅使用了语义相似度和困惑度两个指标，缺少人工评估
4. 未分析攻击参数（如毒化率、触发器长度）对结果的敏感性
5. 预训练阶段的后门攻击被排除，但在某些供应链攻击场景中仍然相关

## 相关工作

- **LLM 后门攻击**：VPI (Yan et al. 2024)、BadChain (Xiang et al. 2024)、PoisonRAG (Zou et al. 2024)
- **后门攻击基准**：BackdoorLLM (Zhao et al. 2024)
- **PEFT 方法**：LoRA (Hu et al. 2021)
- **安全评估**：DecodeTrust (Wang et al. 2023)、AdvBench (Zou et al. 2023)

---

## 评分 ⭐⭐⭐⭐

作为基准论文，实验覆盖面广、指标体系完善、发现有价值。"PEFT 攻击全面优于无微调攻击"和"更强模型在 CoT 攻击下更脆弱"等结论对安全研究有参考意义。不足之处在于缺少防御侧评估和参数敏感性分析。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MEGen: Generative Backdoor into Large Language Models via Model Editing](megen_generative_backdoor_into_large_language_models_via_model_editing.md)
- [\[ACL 2025\] Merge Hijacking: Backdoor Attacks to Model Merging of Large Language Models](merge_hijacking_backdoor_attacks_to_model_merging_of_large_language_models.md)
- [\[ACL 2025\] ReLearn: Unlearning via Learning for Large Language Models](relearn_unlearning_via_learning_for_large_language_models.md)
- [\[ICML 2025\] ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks](../../ICML2025/llm_safety/iclshield_exploring_and_mitigating_in-context_learning_backdoor_attacks.md)
- [\[ACL 2025\] ReDial: Assessing Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks](dialect_fairness_robustness.md)

</div>

<!-- RELATED:END -->
