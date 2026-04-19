---
title: >-
  [论文解读] Lost in Translation? A Comparative Study on the Cross-Lingual Transfer of Composite Harms
description: >-
  [AAAI 2026][目标检测][LLM安全] 提出 CompositeHarm 基准，通过将对抗语法攻击（AttaQ）和语境化危害（MMSafetyBench）翻译为五种印度语言，系统研究了 LLM 安全对齐在跨语言场景下的脆弱性，发现对抗语法攻击在印度语言中攻击成功率急剧攀升。
tags:
  - AAAI 2026
  - 目标检测
  - LLM安全
  - 多语言评估
  - 对抗攻击
  - 跨语言迁移
  - 印度语言
---

# Lost in Translation? A Comparative Study on the Cross-Lingual Transfer of Composite Harms

**会议**: AAAI 2026  
**arXiv**: [2602.07963](https://arxiv.org/abs/2602.07963)  
**代码**: 无（数据集可联系通讯作者获取）  
**领域**: 目标检测  
**关键词**: LLM安全, 多语言评估, 对抗攻击, 跨语言迁移, 印度语言

## 一句话总结

提出 CompositeHarm 基准，通过将对抗语法攻击（AttaQ）和语境化危害（MMSafetyBench）翻译为五种印度语言，系统研究了 LLM 安全对齐在跨语言场景下的脆弱性，发现对抗语法攻击在印度语言中攻击成功率急剧攀升。

## 研究背景与动机

**LLM 安全评估的英语中心化**：绝大多数安全评估仍然建立在英语基础上，但 LLM 每天被数十种语言的用户使用。假设基于英语数据调优的拒绝行为或安全护栏能在其他语言中保持有效是不切实际的。

**翻译作为探测手段**：翻译是测试多语言安全性最实用的桥梁，虽然不完美，但仍是研究有害意图如何跨语言迁移的最清晰方式。有些危害几乎完整迁移，有些则会变形或消失。

**两种危害类型的区分**：
   - **对抗语法攻击**（Adversarial Syntax，来自 AttaQ）：通过混淆或编码的指令绕过安全过滤器，依赖语法操纵挑战模型的解析和对齐能力
   - **语境化危害**（Contextual Harms，来自 MMSafetyBench）：基于现实场景的政策相关危害，如仇恨言论、虚假信息和伦理困境

**现有研究的不足**：
   - 先前工作大多局限于单一语言族或翻译基准
   - 缺乏对不同危害类型在跨语言迁移中行为差异的系统研究
   - 多模态攻击与文本跨语言攻击的对比分析尚属空白

## 方法详解

### 整体框架

CompositeHarm 是一个基于翻译的安全评估基准，旨在通过控制变量的方式测量安全对齐在语法和语义变化时的崩溃点。

### 关键设计

#### 1. **数据集构建**

- **来源数据集**：
    - AttaQ：140 条结构化对抗攻击提示，利用角色扮演、编码指令等语法操纵绕过安全过滤
    - MMSafetyBench：140 条语境化危害提示，覆盖暴力、歧视、色情、欺诈和虚假信息等类别
- **翻译过程**：
    - 使用 NLLB（No Language Left Behind）模型翻译为五种印度语言：印地语、阿萨姆语、马拉地语、卡纳达语、古吉拉特语
    - 所有翻译由双语标注者手工验证和修正，确保语义准确性、文化适当性，特别注意保留对抗提示的语法结构

最终数据集含 1,680 条提示（280 × 6 种语言），语言和来源平衡分布。

#### 2. **评估协议（LLM-as-a-Judge）**

- 评估三个代表不同架构和规模的 LLM：
    - **GPT-OSS 20B**：开源中等规模变体，偏保守防御
    - **LLaMA-3-8B-Instruct**：轻量级指令微调模型，适合边缘部署
    - **Qwen3-32B**：阿里巴巴多语言模型，在亚洲语言上有大量训练

- 使用 LLaMA-3.3-70B 作为评判模型，标准化评估流程：
  1. 每条翻译提示传入目标模型生成回复（temperature=0）
  2. 评判模型回答两个二元问题："是否拒绝？"和"是否产生有害回复？"
  3. 汇总为拒绝率（RR）和攻击成功率（ASR）

- 共处理 5,040 条回复（1,680 × 3 模型）

#### 3. **轻量级推理策略**

受边缘AI设计启发，采用紧凑的生成模型进行初步评估，仅对选择性验证调用大规模评估器，减少冗余API调用和GPU使用。

### 损失函数 / 训练策略

本文为评估基准论文，无训练损失设计。核心指标为：
- **拒绝率（RR）**：模型正确拒绝有害提示的比例
- **攻击成功率（ASR）**：成功诱导有害回复的比例

## 实验关键数据

### 主实验

跨语言平均安全性能（三个模型平均）：

| 语言 | 平均拒绝率(%) | 平均攻击成功率(%) |
|------|-------------|-----------------|
| English | 39.3 | 4.7 |
| Hindi | 43.8 | 6.1 |
| Marathi | 40.7 | 10.4 |
| Assamese | 29.9 | 10.2 |
| Gujarati | 22.0 | 22.0 |
| Kannada | 21.4 | 21.1 |

跨语言族对比（与欧洲语言对比）：

| 语言 | 平均拒绝率(%) | 平均攻击成功率(%) |
|------|-------------|-----------------|
| English (AttaQ) | 82.7 | 4.7 |
| French | 83.0 | 4.0 |
| Spanish | 86.3 | 1.3 |
| German | 80.7 | 3.7 |
| Hindi | 43.8 | 6.1 |
| Gujarati | 22.0 | 22.0 |
| Kannada | 21.4 | 21.1 |

### 消融实验

二元指标的不足分析（"灰色地带"）：

| 语言 | RR+ASR合计(%) | 未分类比例(%) | 说明 |
|------|-------------|-------------|------|
| English | 44.0 | 56.0 | 基线，多数为正常回复 |
| Hindi | 49.9 | 50.1 | 一半交互无法分类 |
| Gujarati | 44.0 | 56.0 | 大量回避式回答 |
| Kannada | 42.5 | 57.5 | 灰色地带最严重 |

### 关键发现

1. **语法攻击比语义攻击更危险**：对抗语法提示的 ASR 在印度语言中急剧攀升（LLaMA-3-8B 在古吉拉特语和卡纳达语超过 45%），而语境化危害的迁移更温和
2. **语言距离效应**：与英语的语言学距离越大，安全对齐退化越严重。欧洲语言的 RR 保持 80%+ 且 ASR <5%，而印度语言 RR 暴跌过半
3. **大量"软失败"**：RR 和 ASR 无法覆盖所有回复，大量回复落入灰色地带（回避式回答和护栏拦截），表明模型内部安全机制已失效
4. **三个模型互补的弱点**：
    - Qwen3-32B：语义脆弱性——流畅但安全对齐依赖简单词汇线索
    - LLaMA-3-8B：伦理松懈——道德判断不一致
    - GPT-OSS 20B：过度防御——规则无法泛化到新语言

## 亮点与洞察

- **复合基准的必要性**：首次系统区分对抗语法和语境危害在跨语言迁移中的不同行为，揭示了仅依赖单一类型基准的不足
- **"灰色地带"分析**：RR 和 ASR 之外的大量回避式回答和护栏拦截现象被首次量化，为安全评估提供了更完整的视角
- **与多模态攻击的对比**：与 Derner & Batistič (2025) 的工作对比表明，LLM 安全在语言泛化和模态泛化两个轴上都是脆弱的
- **边缘部署风险警示**：最轻量（适合边缘部署）的模型恰恰最脆弱，计算效率与多语言安全存在严重权衡

## 局限与展望

- 仅覆盖五种印度语言，未涉及汉藏语系、闪含语系等其他语系
- 使用 LLM-as-a-judge 评估方式，未引入人类评估验证
- 翻译质量虽经人工验证，但翻译本身可能引入语义偏移
- 未深入探讨针对性的安全对齐改进方案
- 数据集规模较小（每语言 280 条），统计显著性可能受限

## 相关工作与启发

- **AttaQ (Kour et al., 2023)** 和 **MMSafetyBench (Li et al., 2023)** 作为两个互补的危害来源
- **NLLB** 翻译模型在低资源语言翻译中的表现值得关注
- **Derner & Batistič (2025)** 的多模态红队测试与本文形成互补：文本跨语言 + 多模态攻击共同揭示 LLM 安全的系统性脆弱
- 启发：未来安全基准需要在危害类型（对抗语法 vs 语义）和模态（文本 vs 图像）两个维度同时构建复合评估

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统区分两类危害在多语言场景中的差异化迁移
- 实验充分度: ⭐⭐⭐ — 三模型六语言，但数据集规模较小，缺乏人类评估
- 写作质量: ⭐⭐⭐⭐ — 分析深入，但部分论述略显冗长
- 价值: ⭐⭐⭐⭐ — 为多语言 LLM 安全研究提供了重要的方法论和实证基础

<!-- RELATED:START -->

## 相关论文

- [MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization](../../CVPR2026/object_detection/mokus_leveraging_crossmodal_knowledge_transfer_for.md)
- [CTPD: Cross Tokenizer Preference Distillation](ctpd_cross_tokenizer_preference_distillation.md)
- [BlueGlass: A Framework for Composite AI Safety](../../ICML2025/object_detection/blueglass_a_framework_for_composite_ai_safety.md)
- [CLCR: Cross-Level Semantic Collaborative Representation for Multimodal Learning](../../CVPR2026/object_detection/clcr_cross-level_semantic_collaborative_representation_for_multimodal_learning.md)
- [Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection](../../CVPR2026/object_detection/learning_multi-modal_prototypes_for_cross-domain_few-shot_object_detection.md)

<!-- RELATED:END -->
