---
title: >-
  [论文解读] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation
description: >-
  [ACL 2026][医学图像][多智能体] 本文提出 MARCH，一个模拟放射科住院医-专科医-主治医层级协作流程的多智能体框架，通过三阶段（初始报告起草、检索增强修订、共识驱动定稿）生成 CT 报告，在 RadGenome-ChestCT 数据集上 CE-F1 达 0.399…
tags:
  - "ACL 2026"
  - "医学图像"
  - "多智能体"
  - "放射学报告生成"
  - "共识驱动"
  - "检索增强"
  - "3D CT"
---

# MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation

**会议**: ACL 2026  
**arXiv**: [2604.16175](https://arxiv.org/abs/2604.16175)  
**代码**: 无  
**领域**: 医学图像 / 报告生成  
**关键词**: 多智能体, 放射学报告生成, 共识驱动, 检索增强, 3D CT

## 一句话总结

本文提出 MARCH，一个模拟放射科住院医-专科医-主治医层级协作流程的多智能体框架，通过三阶段（初始报告起草、检索增强修订、共识驱动定稿）生成 CT 报告，在 RadGenome-ChestCT 数据集上 CE-F1 达 0.399，比最佳基线 Reg2RG 的 0.253 提升 57.7%。

## 研究背景与动机

**领域现状**：自动化放射学报告生成是医学 AI 的重要方向。现有视觉-语言模型（VLM）已在 2D 胸片报告上取得进展，但 3D 体积数据（如胸部 CT）的报告生成仍处于早期阶段。

**现有痛点**：(1) 端到端"黑箱"模型缺乏临床工作流中的迭代验证和交叉核查机制，容易产生临床幻觉；(2) 3D CT 数据中异常发现稀疏，单一模型难以可靠检测所有病理；(3) 单读者模式（single-reader）固有的认知偏差无法被纠正。

**核心矛盾**：临床实践中，放射科通过住院医-专科医-主治医的层级审核流程降低误诊率，但现有自动化系统是单智能体的，缺乏这种多层验证机制。

**本文目标**：设计一个模拟放射科临床层级结构的多智能体框架，实现可解释、可验证的 CT 报告生成。

**切入角度**：借鉴放射科的 readout session 制度——住院医初读、专科医复审、主治医终审——将不同职责分配给不同 AI 智能体。

**核心 idea**：用多智能体层级结构替代单一端到端模型，通过检索增强和多轮共识讨论显著提升临床准确性。

## 方法详解

### 整体框架

MARCH 由三个阶段组成：(1) 住院医智能体从 3D CT 扫描中生成初始报告草稿；(2) 检索智能体从临床数据库检索相关病例，专科医智能体据此修订报告；(3) 主治医智能体主持多轮共识讨论，多个专科医智能体迭代交换立场直至达成临床共识。输入为胸部 CT 体积数据，输出为最终放射学报告。

### 关键设计

1. **住院医智能体（Resident Agent）+ 多区域分割**:

    - 功能：从 3D CT 中提取特征并生成初始报告草稿
    - 核心思路：使用 SAT（Segment Anything with Text）模型将 CT 分割为 10 个解剖亚区域（如骨骼、乳腺等），再用冻结的双流 ViT3D（来自 RadFM 预训练）提取空间特征，最后通过 LoRA 微调的 LLaMA-2-Chat-7B 生成文本报告 $T = A_{res}(I; \theta_{res})$
    - 设计动机：3D 体积数据中异常发现往往局限于特定解剖区域且非常稀疏，全局编码容易遗漏。多区域分割强制模型关注局部解剖和病理实体，缓解了异常检测的稀疏性问题

2. **检索增强修订（Retrieval-Augmented Revision）**:

    - 功能：通过检索相似病例为报告修订提供循证依据
    - 核心思路：设计三种检索范式——(i) 图像到图像/图像到文本检索：用 3D 视觉编码器检索视觉相似的 CT 及对应报告；(ii) Logits 检索：用分类头预测 18 种临床异常的 logits 向量，检索诊断谱相似的报告。每种检索各取 top-3，拼接为结构化证据 $R = A_{ret}(I, D)$，由专科医智能体 $A_{fel}$ 融合证据修订初稿 $T' = A_{fel}(T, R)$
    - 设计动机：单独的生成模型可能遗漏或幻觉，检索增强提供了"第二意见"和循证基础，类比临床中查阅文献和参考病例的过程

3. **共识驱动定稿（Consensus-Driven Finalization）**:

    - 功能：通过多轮立场交换解决诊断分歧
    - 核心思路：主治医智能体 $A_{att}$ 首先聚合多个专科医的修订报告生成初始共识 $T^{(0)}$。在后续轮次中，每个专科医 $A_{fel,i}$ 审查当前共识并给出立场 $S_i^{(t)}$（同意/纠正/补充），主治医整合所有立场更新报告 $T^{(t+1)} = A_{att}(T^{(t)}, \{S_i^{(t)}\})$。迭代持续至达成稳定共识或达到最大轮数
    - 设计动机：模拟真实放射科的 readout session，当多位医生意见不一致时通过讨论而非简单投票来解决分歧，这种"魔鬼代言人"机制在临床中被证明能显著降低误诊率

### 损失函数 / 训练策略

住院医智能体使用 AdamW 优化器（lr=1e-5），训练 10 个 epoch。ViT3D 骨干冻结，LLaMA-2-Chat-7B 通过 LoRA 微调。专科医和主治医智能体使用 GPT-4.1/GPT-4o 作为 LLM 骨干（temperature=0）。

## 实验关键数据

### 主实验

| 方法 | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CE-Precision | CE-Recall | CE-F1 |
|------|--------|--------|--------|---------|-------------|-----------|-------|
| R2GenPT | 0.433 | 0.242 | 0.399 | 0.323 | 0.340 | 0.066 | 0.110 |
| MedVInT | 0.443 | 0.246 | 0.404 | 0.326 | 0.377 | 0.148 | 0.212 |
| M3D | 0.436 | 0.245 | 0.400 | 0.326 | 0.407 | 0.090 | 0.148 |
| RadFM | 0.442 | 0.237 | 0.399 | 0.315 | 0.382 | 0.131 | 0.195 |
| Reg2RG | 0.473 | 0.249 | 0.441 | 0.367 | 0.423 | 0.181 | 0.253 |
| **MARCH** | **0.482** | **0.257** | **0.456** | **0.383** | **0.495** | **0.335** | **0.399** |

### 消融实验

| 配置 | BLEU-1 | BLEU-4 | METEOR | CE-F1 |
|------|--------|--------|--------|-------|
| Resident-only | 0.469 | 0.246 | 0.435 | 0.219 |
| SR-SA（单轮单智能体） | 0.476 | 0.250 | 0.447 | 0.332 |
| SR-MA（单轮多智能体） | 0.475 | 0.251 | 0.454 | 0.352 |
| MR-MA（多轮多智能体） | 0.479 | 0.255 | 0.456 | 0.362 |
| **MARCH（完整）** | **0.482** | **0.257** | **0.456** | **0.399** |

### 关键发现

- CE-F1 从 Resident-only 的 0.219 提升到完整 MARCH 的 0.399，提升 82%，主要来自检索增强（+0.113）和共识机制（+0.037）
- 检索增强对临床效能贡献最大（SR-SA vs Resident-only: CE-F1 +0.113），说明循证修订是减少幻觉的关键
- 不同 LLM 骨干（GPT-4.1-mini/GPT-4.1/GPT-4o/GPT-5）性能差异很小（CE-F1 0.391-0.399），表明框架设计比 LLM 能力更重要
- MARCH 在低频异常（如 hiatal hernia、pericardial effusion）上的检测提升尤为显著

## 亮点与洞察

- 将放射科层级协作流程直接映射为多智能体架构是优雅的设计——不是随意分配角色，而是对应临床中已验证有效的误诊防范机制
- 三种互补的检索范式（视觉、文本、logits）覆盖了不同类型的相似性，这种多模态检索组合可迁移到其他需要循证的医学 AI 任务
- 共识机制使用"立场"（同意/纠正/补充）而非简单投票，保留了分歧的信息量

## 局限与展望

- 依赖 GPT-4 系列作为推理骨干，成本高且不可部署在医院内部，未验证开源 LLM 的可行性
- 缺乏长期记忆机制，无法利用患者历史影像对比或从既往诊断错误中学习
- 仅在 RadGenome-ChestCT 上评估，未验证对其他解剖部位（如脑部、腹部）的泛化性
- 共识轮数需要预设上限，最优轮数的确定缺乏自适应机制

## 相关工作与启发

- **vs Reg2RG**: Reg2RG 使用区域引导的检索增强但仍是单智能体，MARCH 在其基础上增加多智能体共识，CE-F1 从 0.253 提升到 0.399
- **vs RadFM**: RadFM 是通用 3D 医学基础模型，单模型端到端生成，缺乏验证纠错机制
- **vs MedAgent**: 一般医学多智能体系统主要用于诊断和推荐，MARCH 是首个针对 3D 报告生成的多智能体框架

## 评分

- 新颖性: ⭐⭐⭐⭐ 临床层级结构到多智能体的映射自然且有意义
- 实验充分度: ⭐⭐⭐⭐ 消融完整，包含 LLM 骨干对比和逐异常分析
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，临床背景交代充分
- 价值: ⭐⭐⭐⭐ 为高风险医学 AI 提供了可解释的协作范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] CT-FineBench: A Diagnostic Fidelity Benchmark for Fine-Grained Evaluation of CT Report Generation](ct-finebench_a_diagnostic_fidelity_benchmark_for_fine-grained_evaluation_of_ct_r.md)
- [\[ACL 2026\] RA-RRG: Multimodal Retrieval-Augmented Radiology Report Generation with Key Phrase Extraction](ra-rrg_multimodal_retrieval-augmented_radiology_report_generation_with_key_phras.md)
- [\[ACL 2025\] Automated Structured Radiology Report Generation](../../ACL2025/medical_imaging/automated_structured_radiology_report_generation.md)
- [\[ACL 2025\] Online Iterative Self-Alignment for Radiology Report Generation](../../ACL2025/medical_imaging/oisa_radiology_report_gen.md)
- [\[ACL 2026\] Beyond the Individual: Virtualizing Multi-Disciplinary Reasoning for Clinical Intake via Collaborative Agents](beyond_the_individual_virtualizing_multi-disciplinary_reasoning_for_clinical_int.md)

</div>

<!-- RELATED:END -->
