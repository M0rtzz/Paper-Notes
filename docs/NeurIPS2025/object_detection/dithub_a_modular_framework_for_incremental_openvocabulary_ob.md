---
title: >-
  [论文解读] DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection
description: >-
  [NeurIPS 2025][目标检测][open-vocabulary detection] 提出 DitHub，借鉴版本控制系统（Git）思想构建开放词汇目标检测的模块化适配框架——将不同领域的高效适配模块（LoRA）作为"分支"管理，支持按需获取（fetch）和合并（merge），在 ODinW-13 上达到 SOTA，首次系统性研究目标检测中适配模块的组合特性。
tags:
  - NeurIPS 2025
  - 目标检测
  - open-vocabulary detection
  - incremental learning
  - modular deep learning
  - LoRA
  - adaptation modules
  - version control
---

# DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection

**会议**: NeurIPS 2025  
**arXiv**: [2503.09271](https://arxiv.org/abs/2503.09271)  
**代码**: [https://aimagelab.github.io/DitHub/](https://aimagelab.github.io/DitHub/)  
**领域**: 目标检测  
**关键词**: open-vocabulary detection, incremental learning, modular deep learning, LoRA, model merging, version control

## 一句话总结

DitHub 将开放词汇目标检测的增量适配问题重新构造为"版本控制"问题——为每个类别训练独立的 LoRA 专家模块，通过 branch（分支）、fetch（检索）、merge（合并）三个原语管理不断扩展的模块库，在 ODinW-13 全量数据上以 62.19 mAP 超越 ZiRa 4.21 个点，同时保持 47.01 的零样本 COCO 性能。

## 研究背景与动机

**领域现状**：开放词汇目标检测器（如 Grounding DINO）通过文本提示可泛化到任意类别，但在面对稀有类或需要跨多个专业领域持续适配时，仍需要高效的增量学习方案。ZiRa 等方法已开始探索增量视觉-语言目标检测（IVLOD），在 ODinW-13 上取得了不错的结果。

**现有痛点**：现有增量适配方法采用"单体式"策略——所有新知识压缩到单一权重集合中。这导致：（1）难以对特定类别进行选择性更新而不影响其他类别；（2）稀有类别的知识容易在统一权重中被稀释；（3）同一类别在不同域（如 RGB vs 热成像）出现时无法优雅地更新对应知识。

**核心矛盾**：增量检测需要同时实现"类别专精化"和"跨类别/跨域组合"，但单体权重架构天然地将所有类别知识耦合在一起，选择性更新和组合极度困难。

**本文目标**：如何在开放词汇检测中增量适配新领域/新类别，同时支持：（1）对已学类别的选择性更新；（2）跨域模块的灵活组合；（3）零样本能力的有效保持。

**切入角度**：从模块化深度学习（Modular Deep Learning）出发，借鉴版本控制系统（Git）的理念，将每个类别的知识封装为独立的 LoRA 模块，像管理代码分支一样管理检测知识。

**核心 idea**：为每个类别维护独立的 LoRA A 矩阵作为"专家分支"，共享 B 矩阵保证内存效率，通过 warmup→branch→fetch→merge 的流程实现可扩展的增量检测。

## 方法详解

### 整体框架

DitHub 基于预训练的 Grounding DINO，冻结主干网络，仅在编码器上插入 LoRA 模块进行适配。核心思想是将 LoRA 的低秩分解 ΔW = BA 拆分为两部分角色：A 矩阵编码类别专属知识（每类一个），B 矩阵编码通用知识（全局共享）。整个框架通过三个 Git 风格的原语运作：

- **Branch**：新任务到来时，先进行 warmup（类别无关预热），然后为任务中每个类别分支出独立的 A 矩阵
- **Fetch**：若某类别在之前任务中已出现，从模块库中检索已有的类别专家模块
- **Merge**：将检索到的旧模块与当前 warmup 模块加权合并，作为新一轮专精化的初始化

推理时，可以单独激活某个类别专家模块进行精细检测，也可以将多个类别模块平均合并后一次前向传播完成多类检测。

### 关键设计

1. **Warmup-then-Specialization 两阶段训练**
    - 功能：将 LoRA 训练解耦为类别无关的预热阶段和类别专属的专精化阶段
    - 核心思路：每个新任务开始时，先训练一个共享的 warmup 矩阵 $A_{wu}$，为所有类别提供鲁棒的公共初始化；然后分支为 $|C_t|$ 个独立专家，每个专家通过随机策略只在包含对应类别的图像上更新
    - 设计动机：共同的初始化点带来更好的线性模式连通性（linear mode connectivity），这是后续模块可组合性的关键前提；同时 warmup 为稀有类提供良好起点，避免数据不足导致的训练不充分

2. **A 矩阵专属 + B 矩阵共享的非对称设计**
    - 功能：在保持类别专精化的同时将内存开销减半
    - 核心思路：A 矩阵（$r \times k$）每类独立存储，编码类别特有检测知识；B 矩阵（$d \times r$）全局共享，编码通用检测能力。任务结束时 B 通过加权合并更新：$B_t = (1-\lambda_B) B_{t-1} + \lambda_B B^{opt}$
    - 设计动机：独立 (A,B) 对的内存随类别数线性增长，共享 B 直接减半；实验验证 rank=2 时 DitHub 与 ZiRa 内存相当但高出 +2.28 mAP，rank=1（A 退化为向量）仍有竞争力

3. **Fetch-Merge 类别重现处理机制**
    - 功能：处理同一类别在不同任务/域中重复出现的场景
    - 核心思路：若类别 $c$ 已有存储模块 $A_c^{old}$，则用加权合并生成新初始化：$A_c^{cur} = (1-\lambda_A) A_c^{old} + \lambda_A A_{wu}$，其中 $\lambda_A$ 设较低值（0.1~0.3），优先保留已有知识
    - 设计动机：直接从 warmup 重新训练会丢失之前积累的类别知识；直接用旧模块则无法融入新任务的域信息。加权合并兼顾历史知识保持和新域适应，在 ODinW-O 上特别有效（+4.75 mAP）

### 损失函数 / 训练策略

- 使用 Grounding DINO 的标准检测损失：Focal Loss（分类）+ L1 Loss + GIoU Loss（定位）
- Warmup 和 specialization 阶段各分配相同 epoch 数
- 专精化阶段的随机训练策略：每张图像随机选择一个存在的类别，只更新对应的 A 矩阵
- $\lambda_A$：ODinW-13 设 0.3，ODinW-O 设 0.1（重叠类更多，需更多保留历史知识）
- $\lambda_B$：固定 0.7（偏向最新任务的 $B^{opt}$，因为它已隐式包含早期任务知识）
- 仅适配 Grounding DINO 的编码器部分，LoRA rank 默认 r=16

## 实验关键数据

### 主实验

**ODinW-13 全量数据（Full-shot）**：

| 方法 | ZCOCO | Avg mAP | 相对 ZiRa |
|------|-------|---------|-----------|
| Grounding DINO (zero-shot) | 47.41 | 46.80 | — |
| TFA | 30.97 | 47.93 | -10.05 |
| AT | 42.30 | 51.14 | -6.84 |
| OW-DETR | 31.22 | 55.58 | -2.40 |
| CL-DETR | 32.15 | 57.26 | -0.72 |
| iDETR | 37.32 | 58.71 | +0.73 |
| ZiRa | 46.26 | 57.98 | — |
| **DitHub** | **47.01** | **62.19** | **+4.21** |

**ODinW-13 Few-shot**：

| 方法 | 1-shot Avg | 5-shot Avg | 10-shot Avg |
|------|-----------|-----------|------------|
| ZiRa | 48.56 | 51.77 | 53.20 |
| DitHub | 49.19 | **52.85** | **54.43** |
| 差值 | +0.63 | +1.08 | +1.23 |

**ODinW-O（类别重现场景）**：

| 方法 | ZCOCO | Avg mAP |
|------|-------|---------|
| Grounding DINO | 47.41 | 53.15 |
| ZiRa | 44.43 | 57.63 |
| **DitHub** | **46.51** | **62.38 (+4.75)** |

### 消融实验

**组件消融（ODinW-13 Full-shot）**：

| 配置 | Avg mAP | ZCOCO | 说明 |
|------|---------|-------|------|
| Base（无 warmup 无合并） | ~56 | ~48 | 完全灾难性遗忘 |
| +Warmup | ~59 | ~47.5 | warmup 显著提升 |
| +Warmup +B 合并 | ~59.5 | ~47.3 | B 合并小幅提升 |
| DitHub（+A 合并） | 62.19 | 47.01 | A 合并是最关键组件 |

**专精化 vs 非专精化（EnE）**：

| 方法 | ZCOCO | Avg mAP |
|------|-------|---------|
| EnE（随机分配，无类别专精） | 46.86 | 60.96 |
| DitHub（类别专精化） | 47.01 | 62.19 (+1.23) |

**LoRA Rank 消融**：

| Rank | 内存 (MB) | Avg mAP |
|------|----------|---------|
| r=1 | ~9 | 57.04 |
| r=2 | ~18 | 60.26 |
| ZiRa | ~18 | 57.98 |
| r=8 | ~74 | 61.93 |
| r=16 | ~147 | 62.19 |

### 关键发现

- DitHub 在全量设置下 9/13 个任务取得最优，平均超越 ZiRa +4.21 mAP，且零样本保持更优（+0.75）
- 随任务推进，DitHub 的性能退化速率低于 ZiRa，模块化设计的抗遗忘优势随任务数增多而放大
- 在类别重现场景（ODinW-O）优势更大（+4.75 mAP），验证了 fetch-merge 机制的有效性
- rank=2 的 DitHub 与 ZiRa 内存相当但高出 +2.28 mAP，展示极佳的性能-内存权衡
- 类别专精化（vs EnE）带来 +1.23 mAP 提升，验证了按类别分支训练而非随机分配的必要性
- DitHub 支持免训练遗忘（unlearning）：减去某类的 A 模块即可移除对该类的检测能力

## 亮点与洞察

- **Git 版本控制的类比极为精准**：branch/fetch/merge 三个原语完美映射到增量检测的"新建/检索/融合"操作，概念清晰且易于扩展
- **warmup→specialization 的解耦设计**启发性强：共同初始化点保证了模块间的可组合性（linear mode connectivity），这一洞察建立了模块化检测与模型合并理论的桥梁
- **A/B 矩阵的非对称角色分配**是关键工程创新：A 编码类别知识（需独立），B 编码通用知识（可共享），自然地将 LoRA 的数学结构与模块化需求对齐
- 首次在目标检测中系统研究 LoRA 模块的组合特性，填补了从 NLP（LoRAHub）到视觉检测的研究空白
- 免训练遗忘能力（减去模块权重）具有实际合规价值（隐私法规要求选择性移除特定类别检测）

## 局限与展望

- **仅验证 Grounding DINO**：未测试 YOLO-World、OWLv2 等其他开放词汇检测器，框架通用性待验证
- **推理时模块选择需手动指定**：缺乏自动化的路由机制，无法根据输入自动决定激活哪些专家模块
- **多类合并策略较简单**：目前仅用简单平均合并多个类别模块，未探索更复杂的自适应加权或注意力路由
- **ODinW 规模有限**：最大仅 13/35 个子数据集，大规模场景（数百个类别/域）下模块库管理的可扩展性未知
- **λ 超参数需要按数据集调整**：$\lambda_A$ 在 ODinW-13 和 ODinW-O 上最优值不同（0.3 vs 0.1），缺乏自适应机制

## 相关工作与启发

- **vs ZiRa**（NeurIPS 2024）：ZiRa 使用可重参数化的侧分支做单体适配；DitHub 用模块化 LoRA 实现类别级粒度管理，在相同内存下性能更优
- **vs LoRAHub**（NLP）：LoRAHub 在 LLM 中验证了 LoRA 的可组合性；DitHub 首次将这一思路引入目标检测，并发现检测任务因类别非均匀分布而有不同的合并特性
- **vs CL-DETR/iDETR**：传统增量检测方法通过架构修改或正则化防遗忘；DitHub 通过模块化设计从根本上避免类间干扰
- **启发**：模块化适配思想可直接迁移到分割（为每个域训练 SAM LoRA）、个性化生成（文本到图像的概念定制）等场景；与 MoE 架构天然契合，未来可考虑训练路由器自动选择模块

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 理由：Git 版本控制到检测模块管理的类比新颖且实用，但 LoRA 组合在 NLP 中已有先例
- **实验充分度**: ⭐⭐⭐⭐⭐ — 理由：ODinW-13 + ODinW-O 双基准，full/1/5/10-shot 全覆盖，组件消融、rank 消融、λ 敏感性、EnE 对照、遗忘能力展示面面俱到
- **写作质量**: ⭐⭐⭐⭐ — 理由：Git 类比使框架直觉易懂，算法伪代码清晰；但符号较多，初读需要适应
- **价值**: ⭐⭐⭐⭐ — 理由：模块化增量检测是真实部署场景的核心需求，框架设计通用性强，+4.21 mAP 的提升幅度实质性
---
title: >-
  [论文解读] DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection
description: >-
  [NeurIPS 2025][目标检测][open-vocabulary detection] 提出 DitHub，借鉴版本控制系统（Git）思想构建开放词汇目标检测的模块化适配框架——将不同领域的高效适配模块（LoRA）作为"分支"管理，支持按需获取（fetch）和合并（merge），在 ODinW-13 上达到 SOTA，首次系统性研究目标检测中适配模块的组合特性。
tags:
  - NeurIPS 2025
  - 目标检测
  - open-vocabulary detection
  - incremental learning
  - modular deep learning
  - LoRA
  - adaptation modules
  - version control
---

# DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection

**会议**: NeurIPS 2025  
**arXiv**: [2503.09271](https://arxiv.org/abs/2503.09271)  
**代码**: [https://aimagelab.github.io/DitHub/](https://aimagelab.github.io/DitHub/)  
**领域**: 目标检测  
**关键词**: open-vocabulary detection, incremental learning, modular deep learning, LoRA, adaptation modules, version control  

## 一句话总结
提出 DitHub，借鉴版本控制系统（Git）思想构建开放词汇目标检测的模块化适配框架——将不同领域的高效适配模块（LoRA）作为"分支"管理，支持按需获取（fetch）和合并（merge），在 ODinW-13 上达到 SOTA，首次系统性研究目标检测中适配模块的组合特性。

## 背景与动机
开放词汇目标检测器（如 Grounding DINO）可通过文本提示泛化到任意类别，但面对稀有类或需要在多个专业领域上工作时仍需适配。现有增量学习方法：
- **单体式适配**（single set of weights）：在整个模型上微调，不同领域间会遗忘
- **LoRA 微调**：高效但不同领域的 LoRA 模块相互独立，未研究如何组合多个领域的知识

关键问题：能否像 Git 管理代码一样管理 LoRA 适配模块——各领域独立开发（branch），需要时合并（merge）？

## 核心问题
如何在开放词汇检测中**增量地**适配新领域/新类别，同时保持对已学领域的能力、支持跨领域组合，且不需要存储所有历史训练数据？

## 方法详解

### 整体框架
DitHub 维护一个适配模块库，每个模块是一个轻量级 LoRA。三个核心操作类比 Git：
- **Branch**: 为新领域创建新的 LoRA 适配模块并独立训练
- **Fetch**: 需要在新领域检测时，从库中检索相关模块
- **Merge**: 将多个模块的权重组合，实现跨领域知识融合

### 关键设计
1. **模块化适配**: 每个领域/任务对应一个独立的 LoRA 模块，训练时冻结基础模型（Grounding DINO），仅更新 LoRA 参数。低参数量（每个模块 ~0.5M 参数），存储高效。

2. **模块组合策略**: 系统性研究了多种 LoRA 合并方法在检测任务中的效果：

    - 简单平均（averaging）
    - Task Arithmetic（任务算术加权）
    - TIES-Merging（修剪+符号投票+合并）
    - 基于注意力/相似度的自适应加权

3. **增量学习协议**: 定义了类别重现（class reappearance）的评估场景——同一类别在不同增量阶段重复出现，更贴近真实世界。提出 ODinW-O 新基准来评估此能力。

### 损失函数 / 训练策略
基于 Grounding DINO 的标准检测损失（Focal Loss + L1 + GIoU），每个领域独立训练 LoRA，推理时按需合并。

## 实验关键数据

| 方法 | ODinW-13 AP | 类型 |
|------|-----------|------|
| Grounding DINO (zero-shot) | ~55 | 基线 |
| Full fine-tuning | ~63 | 单体 |
| CL 方法 (EWC, LwF) | ~58-60 | 增量 |
| **DitHub** | **SOTA** | 模块化 |

首次系统性证明：(1) LoRA 模块在检测任务中具有可组合性，(2) 模块化方法在增量开放词汇检测中优于传统持续学习方法。

### 消融实验要点
- 不同合并策略的效果比较：Task Arithmetic 因检测任务的特殊性（类别非均匀分布）表现不同于 LLM 任务
- 模块数量增长对合并质量的影响：适中数量（~5-8）效果最优
- 类别重现场景下 DitHub 的优势尤为明显——同一类别的不同领域 LoRA 能互补

## 亮点
- **Git 版本控制的类比**非常直觉——分支/获取/合并的概念清晰易懂
- 首次在目标检测领域系统性研究适配模块的组合特性
- 提出 ODinW-O 基准，填补了类别重现评估的空白
- 模块化架构天然支持隐私保护——不同机构的数据可以独立训练 LoRA 再合并

## 局限与展望
- 仅基于 Grounding DINO，未测试其他开放词汇检测器（如 YOLO-World）
- 模块合并策略在某些领域组合下可能产生冲突
- 未探索自动化的模块选择机制（当前需手动指定要 fetch 的模块）
- ODinW 数据集规模相对有限，大规模场景验证不足

## 与相关工作的对比
- **vs 传统 CL（EWC、LwF）**: 传统方法通过正则化防遗忘但性能仍下降；DitHub 用独立模块完全避免遗忘
- **vs LoRAHub（NLP）**: LoRAHub 在 LLM 中验证 LoRA 组合，DitHub 首次将此思路引入目标检测
- **vs CQ-DINO（同批次笔记）**: CQ-DINO 解决大词汇量检测的梯度问题；DitHub 解决多领域增量适配问题，两者互补

## 启发与关联
- 模块化适配思想可迁移到分割（如为每个领域训练 SAM LoRA 再合并）
- 类比版本控制的框架设计可用于其他模型定制场景（如个性化 VLM）
- 可与 CQ-DINO 结合：CQ-DINO 解决单次大词汇量，DitHub 解决增量扩展词汇量

## 评分
- 新颖性: ⭐⭐⭐⭐ Git 类比新颖，但 LoRA 组合在 NLP 中已有探索
- 实验充分度: ⭐⭐⭐⭐ ODinW-13 + ODinW-O 两个基准，多种合并策略对比
- 写作质量: ⭐⭐⭐⭐ Git 类比使论文概念清晰易读
- 价值: ⭐⭐⭐⭐ 模块化增量检测是实际部署的真实需求

<!-- RELATED:START -->

## 相关论文

- [Open-Det: An Efficient Learning Framework for Open-Ended Detection](../../ICML2025/object_detection/open-det_an_efficient_learning_framework_for_open-ended_detection.md)
- [Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection](../../ICCV2025/object_detection/dynamicdino_finegrained_mixture_of_experts_tuning_for_realti.md)
- [ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection](../../CVPR2025/object_detection/abra_teleporting_fine-tuned_knowledge_across_domains_for_open-vocabulary_object_.md)
- [CQ-DINO: Mitigating Gradient Dilution via Category Queries for Vast Vocabulary Object Detection](cq-dino_mitigating_gradient_dilution_via_category_queries_for_vast_vocabulary_ob.md)
- [NoOVD: Novel Category Discovery and Embedding for Open-Vocabulary Object Detection](../../CVPR2026/object_detection/noovd_novel_category_discovery_and_embedding_for_open-vocabulary_object_detectio.md)

<!-- RELATED:END -->
