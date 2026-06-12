---
title: >-
  [论文解读] Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review
description: >-
  [ACL 2026][对话系统][作者回复生成] 本文将学术论文作者回复（rebuttal）生成重新定义为"作者在回路"任务，提出 Re3Align 数据集（3.4K 论文、440K 句级编辑标注、15K 审稿-回复-修改三元组）、REspGen 可控生成框架和 REspEval 20+ 指标评估套件…
tags:
  - "ACL 2026"
  - "对话系统"
  - "作者回复生成"
  - "同行评审"
  - "人在回路"
  - "可控文本生成"
  - "评估框架"
---

# Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review

**会议**: ACL 2026  
**arXiv**: [2602.11173](https://arxiv.org/abs/2602.11173)  
**代码**: [https://github.com/UKPLab/acl2026-respgen-respeval](https://github.com/UKPLab/acl2026-respgen-respeval)  
**领域**: 对话/科学文档处理  
**关键词**: 作者回复生成, 同行评审, 人在回路, 可控文本生成, 评估框架

## 一句话总结

本文将学术论文作者回复（rebuttal）生成重新定义为"作者在回路"任务，提出 Re3Align 数据集（3.4K 论文、440K 句级编辑标注、15K 审稿-回复-修改三元组）、REspGen 可控生成框架和 REspEval 20+ 指标评估套件，在 5 个 SOTA LLM 上系统验证了作者输入、可控性和评估引导精修的效果。

## 研究背景与动机

**领域现状**：作者回复（rebuttal）写作是学术同行评审中的关键环节，需要大量作者精力。NLP 辅助自动生成作者回复（ARG）是新兴但未充分探索的研究方向。

**现有痛点**：(1) 现有 ARG 工作仅使用审稿意见作为输入，忽略了作者的领域专业知识、独有信息和回复策略——但实际中许多审稿关切只有作者才能回答（如具体实验设计、澄清定义等）；(2) 缺乏提供细粒度作者信号的数据集——现有数据集无句级编辑标注、无审稿-回复段落对齐、无修改映射；(3) 评估仅限于表面相似度指标（ROUGE/BLEU），缺乏对可控性、输入利用率、回复质量和话语结构的多维评估。

**核心矛盾**：作者回复写作本质上需要整合作者专属信号（修改计划、领域知识、回复策略），但现有 NLP 方法将其视为通用的"审稿→回复"文本生成问题，产出的回复缺乏具体细节和作者独有信息。

**本文目标**：(1) 形式化定义"作者在回路"ARG 范式；(2) 构建支撑该范式的大规模三元组数据集；(3) 提供支持灵活作者输入和多属性控制的生成框架；(4) 建立 20+ 指标的全面评估体系。

**切入角度**：利用论文修改版本作为作者信号的代理——会议场景中回复描述计划的修改，修改版论文中的实际编辑可回溯代理作者的意图和专业知识。

**核心 idea**：将论文修改中的句级编辑作为作者专属信息的代理，构建审稿意见-作者回复-论文编辑三元组对齐数据集，使 ARG 模型可以利用作者的真实修改意图来生成高质量回复。

## 方法详解

### 整体框架

本文围绕"作者在回路"范式搭了数据集、生成、评估三件套，闭环地把作者的真实修改意图注入回复生成。输入端是一篇论文的审稿-回复-修改版完整记录：Re3Align 先把它对齐成句级的"审稿意见–作者回复–论文编辑"三元组；REspGen 以审稿意见为主输入，可选接入作者编辑信号、v1 论文段落检索、回复计划和长度约束，产出可控回复，并支持把评估反馈拿回来做迭代精修；REspEval 则在话语、可控性、输入利用、回复质量四个维度用 20+ 指标量化这份回复到底答没答到点子上、用没用上作者信息。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    IN["论文记录<br/>审稿 + 回复 + v1→v2 修改"]
    subgraph DATA["Re3Align 三元组数据集构建"]
        direction TB
        A["引用匹配<br/>抽审稿-回复段落对"] --> B["修改分析<br/>句级编辑标注"] --> C["双向对齐<br/>审稿/回复 → 编辑"]
    end
    IN --> DATA
    DATA --> GEN
    subgraph GEN["REspGen 可控生成框架"]
        direction TB
        D["输入配置<br/>审稿意见 + 作者编辑 + v1 段落检索"] --> E["回复计划 + 长度约束<br/>立场/策略/篇幅可控"] --> F["生成可控回复"]
    end
    GEN --> EVAL["REspEval 多维评估<br/>话语/可控性/输入利用/质量 20+ 指标"]
    EVAL -->|反馈理由 + 改进建议迭代精修| D
    EVAL --> OUT["最终作者回复"]
```

### 关键设计

**1. Re3Align 三元组数据集构建：用论文修改作为作者信号的代理**

活跃地向作者要"你为什么这么回复"在伦理和工程上都不现实，但论文的修改版本本身就是作者意图的事后落地——回复里承诺的改动，在 v1→v2 的句级编辑中都能找到对应。基于这一观察，作者从 EMNLP24（679 篇）和 PeerJ（2,715 篇）收集完整论文记录，用三步流水线把它对齐成三元组：(a) 引用匹配算法抽出 16,071 对审稿-回复段落（人工验证 98% 准确率）；(b) SOTA 修改分析模型标注 439,798 个句级编辑（对齐 F1 > 90%，意图分类 84.3 F1）；(c) 双向对齐策略（审稿→编辑 + 回复→编辑，微调 LLM 分类器 >90% 准确率）生成 15,521 个句级三元组。这是首个同时带句级编辑标注、审稿-回复段落对齐和修改映射的大规模数据集。

**2. REspGen 可控生成框架：把作者的语气、策略、长度做成可拨的旋钮**

此前 ARG 工作只喂审稿意见、产出"通用回复"，作者实际写 rebuttal 时却要同时拿捏立场、策略和篇幅。REspGen 因此叠了三层控制：**回复计划控制**把审稿意见分成 Criticism/Question/Request 三类，每类挂 16 种回复动作标签（归为合作、防御、对冲、社交、其他 5 大立场），作者可逐条指定回复策略序列；**长度约束**设上界词数（实验取人类回复长度 $+50$）；**输入配置**让作者编辑以"编辑字符串"（粗糙想法）或"编辑字符串 + 段落上下文 + 章节标题"（精细定位）两种粒度接入，并额外支持基于检索-重排的 v1 论文段落检索。这套配置让同一份审稿意见可以被引导出语气、策略、信息量都不同的回复。

**3. REspEval 多维评估套件：用原子事实检验替代表面相似度**

ROUGE/BLEU 只比字面重合，看不出回复有没有真正回应关切、有没有用上作者信息、有没有守住计划。REspEval 因此在四个维度铺了 20+ 指标：**话语分析**给出 5 类立场比例（%Coop/%Defe/%Hed/%Soc/%Other）、ArgumentLoad 及转换流；**可控性**测长度遵守率（%met + median diff）和计划保真度（P/R/F1 + 基于 LCS 的 Order Fidelity）；**输入利用**用原子事实检验算生成事实精确度（GFP，生成事实中被输入支持的比例）和输入覆盖召回率（ICR，作者编辑事实在回复中出现的比例）；**回复质量**用基于评审准则的 GPT-5 给针对性（Targ）、具体性（Spec）、说服力（Conv）5 分制打分。12 位研究者、1,365 条人工判断验证了这套指标，一致性评分 > 4.17/5，Krippendorff $\alpha = 0.81\text{-}0.89$。

### 损失函数 / 训练策略

REspGen 是提示驱动的大语言模型，不涉及参数训练，输入配置和属性控制全靠提示模板实现。评估引导的迭代精修构成闭环：把 REspEval 返回的指标、理由和改进建议，连同原始输入和初始草稿一起回灌给 REspGen，生成改进版回复——这也是实验里 Targ 从 .85 升到 .94 的来源。

## 实验关键数据

### 主实验

**不同 LLM 和设置下的回复质量对比（选取 GPT-4o 和 DeepSeek）**

| 设置 | GFP %sup | ICR %sup | Targ | Spec | Conv |
|------|---------|---------|------|------|------|
| Human baseline | .458 | .200 | .788 | .575 | .575 |
| GPT-4o noAIx（无作者输入） | .443 | .033 | .842 | .508 | .554 |
| GPT-4o wAIx(S) | .689 | .668 | .826 | .638 | .654 |
| GPT-4o wAIx(+v1) | .781 | .432 | .847 | .721 | .717 |
| GPT-4o +Refine(planC) | .695 | — | .938 | .771 | .742 |
| DeepSeek noAIx | .412 | .046 | .779 | .433 | .496 |
| DeepSeek wAIx(+v1) | .738 | .452 | .861 | .692 | .700 |
| DeepSeek +Refine(planC) | .734 | — | .913 | .746 | .742 |

### 消融实验

**作者输入粒度对事实利用的递进影响（Phi-4 模型）**

| 设置 | GFP %sup ↑ | GFP %unsup ↓ | GFP %con | ICR %sup ↑ |
|------|-----------|-------------|----------|-----------|
| noAIx（无作者输入） | .362 | .542 | .096 | .300 |
| wAIx 编辑字符串 | .575 | .374 | .051 | .509 |
| +段落上下文 | .577 | .364 | .059 | .470 |
| +v1 检索 | .705 | .236 | .059 | .358 |

**长度和计划控制的交互效果（Llama-3.3）**

| 设置 | lenC %met | planC F1 | Targ | Conv |
|------|----------|---------|------|------|
| +lenC only | 1.00 | — | .771 | .638 |
| +lenC & planC | 1.00 | .619 | .850 | .638 |
| +planC only | — | .486 | .892 | .671 |

### 关键发现

- 作者输入显著提升事实精确度（GFP %sup 从 .36-.44 提升到 .58-.78），不支持事实比例大幅降低
- 评估引导精修有效提升针对性（Targ 从 .85 提升到 .94）和说服力，但可能降低事实精确度——揭示质量-事实性权衡
- 长度和计划控制的同时施加存在质量-可控性权衡——同时控制两个属性时质量略低于仅控制一个
- ICR 在加入更多上下文后反而下降，说明信息过载导致模型无法优先处理核心编辑内容
- 所有模型在无作者输入时生成大量无支撑事实（>50%），证实了"作者在回路"的必要性

## 亮点与洞察

- "作者在回路"范式的提出是对 ARG 任务的本质性重新定义——从通用生成变为人机协作
- 利用论文修改版本作为作者信号的代理是巧妙的方法论创新，规避了实时采集的伦理和实践障碍
- REspEval 中基于原子事实检验的 GFP/ICR 指标比 ROUGE 更有意义地衡量了回复对作者信息的利用
- Order Fidelity 指标基于 LCS 的设计既简洁又合理，可推广到其他序列控制评估场景
- Table 1 对比先前工作在数据/生成/评估三维的差距，清晰展示了贡献的系统性

## 局限与展望

- 代理信号（论文编辑）与实际作者意图之间存在固有差距——并非所有修改都对应审稿关切
- 仅在英语学术文本上验证，其他语言和领域未测试
- 评估引导精修可能导致过拟合 REspEval 指标而非真实质量提升
- 未来可探索交互式多轮精修、与实际作者的用户研究、以及更细粒度的作者控制接口

## 相关工作与启发

- **vs Jiu-Jitsu (2023)**: 仅有段级对齐，无句级编辑标注，无作者输入，评估仅限 ROUGE/BERTScore
- **vs ReviewMT (2024)**: 仅有文档级对齐，评估仅限 ROUGE/BLEU/METEOR
- **vs Re2 (2025)**: 仅有文档级对齐和基本相似度+质量指标，无可控性研究

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统定义"作者在回路"ARG 范式，数据集、框架和评估三位一体
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个 LLM、9 种设置、20+ 指标、12 人人工验证，极为全面
- 写作质量: ⭐⭐⭐⭐ 结构完整、技术细节充分，但信息密度极高导致阅读门槛较高
- 价值: ⭐⭐⭐⭐⭐ 对学术写作 NLP 辅助具有重要推动作用，数据集和工具的实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Auto-PRE: An Automatic and Cost-Efficient Peer-Review Framework for Language Generation Evaluation](../../AAAI2026/dialogue/auto-pre_an_automatic_and_cost-efficient_peer-review_framework_for_language_gene.md)
- [\[ACL 2026\] Discourse Coherence and Response-Guided Context Rewriting for Multi-Party Dialogue Generation](discourse_coherence_and_response-guided_context_rewriting_for_multi-party_dialog.md)
- [\[ACL 2026\] Codebook-Injected Dialogue Segmentation for Multi-Utterance Constructs Annotation: LLM-Assisted and Gold-Label-Free Evaluation](codebook-injected_dialogue_segmentation_for_multi-utterance_constructs_annotatio.md)
- [\[ICLR 2026\] AQuA: Toward Strategic Response Generation for Ambiguous Visual Questions](../../ICLR2026/dialogue/aqua_toward_strategic_response_generation_for_ambiguous_visual_questions.md)
- [\[ACL 2025\] ReflectDiffu: Reflect between Emotion-intent Contagion and Mimicry for Empathetic Response Generation via a RL-Diffusion Framework](../../ACL2025/dialogue/reflectdiffu_empathetic_response.md)

</div>

<!-- RELATED:END -->
