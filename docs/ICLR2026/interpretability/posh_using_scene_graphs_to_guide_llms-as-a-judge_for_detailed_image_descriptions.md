---
title: >-
  [论文解读] PoSh: Using Scene Graphs to Guide LLMs-as-a-Judge for Detailed Image Descriptions
description: >-
  [ICLR 2026][detailed image description] 提出PoSh评估指标，通过从生成描述和参考描述中提取场景图 $G(d) = \langle O(d), E(d), K(d) \rangle$ 作为结构化rubric，引导开源14B LLM（Qwen3-14B）进行QA式细粒度错误定位，在DOCENT艺术品基准和CapArena上以+0.05 Spearman ρ超越GPT-4o-as-Judge，且完全可复现。
tags:
  - ICLR 2026
  - detailed image description
  - scene graph
  - LLM-as-Judge
  - fine-grained evaluation
  - assistive text
---

# PoSh: Using Scene Graphs to Guide LLMs-as-a-Judge for Detailed Image Descriptions

**会议**: ICLR 2026  
**arXiv**: [2510.19060](https://arxiv.org/abs/2510.19060)  
**代码**: [GitHub](https://github.com/amith-ananthram/posh)  
**领域**: 多模态评估  
**关键词**: detailed image description, scene graph, LLM-as-Judge, fine-grained evaluation, assistive text

## 一句话总结
提出PoSh评估指标，通过从生成描述和参考描述中提取场景图 $G(d) = \langle O(d), E(d), K(d) \rangle$ 作为结构化rubric，引导开源14B LLM（Qwen3-14B）进行QA式细粒度错误定位，在DOCENT艺术品基准和CapArena上以+0.05 Spearman ρ超越GPT-4o-as-Judge，且完全可复现。

## 研究背景与动机

**领域现状**：VLM已能生成详细图像描述（100-300词），但评估方法严重滞后。CIDEr/SPICE设计用于短文本，LLM-as-Judge不可复现且产出粗粒度不可解释的分数。

**现有痛点**：
- 长描述中属性/关系误附着是核心错误（如"倒水的男人"被描为"中央的男人"），现有指标对此不敏感
- SPICE/CAPTURE虽用场景图但忽略对象附着（object attachment），容易误报高分
- 闭源LLM评估（GPT-4o）成本高且不可复现，开源LLM-as-Judge（LLaVA-Critic）不提供可解释的细粒度分数
- 缺乏含细粒度人工判断的评估基准——大多数详细描述基准无人工标注

**核心矛盾**：需要cheap、reliable、interpretable的评估方法，但cheap与reliable/interpretable通常矛盾。

**本文目标** 同时实现可解释性（细粒度错误定位到文本段）、与人类判断的高相关性、和完全开源可复现。

**切入角度**：场景图将描述的表面多样性降维为视觉组件（实体+属性+关系）→ 作为LLM-Judge的结构化checklist → 每个组件独立验证存在性 → 聚合为粗粒度分数。

**核心 idea**：用场景图结构化评估的"评什么"（实体、属性、关系），用LLM-QA灵活处理"怎么比"（表面形式差异）。

## 方法详解

### 整体框架

PoSh三步流程：
1. **场景图提取**：用依存句法分析(spaCy) + 共指消解(Maverick)从生成描述和参考描述中提取句级场景图，合并为完整场景图
2. **细粒度评分**：将场景图中每个组件转为模板化问题，用Qwen3-14B做QA验证其在对方文本中的存在性
3. **粗粒度聚合**：分别平均mistakes分（生成→参考）和omissions分（参考→生成）

### 关键设计

1. **保持附着关系的场景图提取**:
    - 功能：从描述文本提取结构化表示 $G(d) = \langle O(d), E(d), K(d) \rangle$，其中 $O$ 为实体集合，$E \subseteq O \times A$ 为属性边，$K \subseteq O \times R \times O$ 为关系边
    - 核心思路：句级依存句法分析 → 跨句共指消解合并实体 → 保留每个属性/关系到其宿主实体的附着链接 → 每个组件定位到原文span
    - 设计动机：SPICE忽略附着关系导致"把A的属性算到B头上"不被惩罚；PoSh通过保持附着链确保属性/关系检验时使用正确的实体标识符

2. **基于唯一标识符的三轮QA验证**:
    - 功能：为每个场景图组件生成模板化问题，用LLM判断其在对方文本中的存在性（1-5分）
    - 核心思路：处理同类实体碰撞（如多个"man"）需唯一标识符。三轮验证：(1) 顶层实体（"man"本身）→ (2) 部分-从属实体（"face of the man"）→ (3) 属性/关系（使用已确认存在的最简标识符）。标识符候选包括类名、表面形式、属性修饰、关系修饰，由LLM重写为自然表达
    - 设计动机：避免强制对齐两个场景图的组件——对方文本可能用完全不同的词指代同一对象（如参考用"trio"，生成分别提到三个人）

3. **可解释的粗粒度聚合**:
    - 功能：将细粒度的per-component分数聚合为mistakes、omissions、overall三个维度
    - 核心思路：$\text{Mistakes} = \text{mean}_{c \in O(\text{gen})}(\pi(c))$，$\text{Omissions} = \text{mean}_{c \in O(\text{ref})}(\rho(c))$，其中 $\pi(c) = \Psi(c_{\text{gen}}, \text{ref})$，$\rho(c) = \Psi(c_{\text{ref}}, \text{gen})$
    - 设计动机：粗粒度分数直接来自细粒度分数的平均——知道总分后可追溯到哪些实体的哪些属性出了问题，提供诊断能力

### 损失函数 / 训练策略

PoSh是推理时指标，无训练过程。QA评分器Ψ使用Qwen3-14B，存在性分数从token logits的加权平均提取（1-5分），实体存在性判定阈值2（在小型验证集上调优）。运行效率：单H100 GPU上400个样本15分钟（每个2秒），而DCScore因依赖GPT-4需2小时以上。

## 实验关键数据

### DOCENT基准 — 粗粒度指标对比

| 指标 | 参数量 | Mistakes ρ | Omissions ρ | Overall ρ | 可复现 |
|------|--------|-----------|------------|----------|--------|
| SPICE | - | 0.308 | 0.464 | 0.458 | ✓ |
| CAPTURE | - | 0.259 | 0.447 | 0.453 | ✓ |
| LLaVA-Critic | 72B | 0.412 | 0.509 | 0.546 | ✓ |
| DCScore | GPT-4o | **0.541** | 0.395 | 0.471 | ✗ |
| GPT-4o (ref+img) | - | 0.484 | 0.380 | 0.510 | ✗ |
| **PoSh** | **14B** | 0.519 | **0.581** | **0.599** | **✓** |

### 细粒度指标对比（DOCENT）

| 方法 | Mistakes F1 | Omissions F1 |
|------|------------|-------------|
| Random | 0.503 | 0.499 |
| 4GramEmbed | 0.483 | 0.641 |
| SGEmbed | 0.514 | 0.658 |
| **PoSh** | **0.580** | **0.680** |

### 关键发现
- PoSh在DOCENT上Overall acc=70.7%，超越GPT-4o (67.3%)和GPT-5 text-only (68.0%)，且完全开源可复现
- 在CapArena上PoSh对复杂场景（≥3人）的模型排名与人类的相关性优于72B的LLaVA-Critic（ρ=0.727 vs 0.686）
- 场景图子组件验证：元素提取F1=0.892，元素验证F1=0.852——高质量的结构化提取是PoSh成功的基础
- PoSh作为RL奖励函数（DAPO）优于SFT：omission改善+0.432，overall改善+0.135
- DOCENT排行榜显示：开源模型在mistakes上有竞争力，但在omissions上明显落后闭源模型——覆盖率是关键差距

## 亮点与洞察
- **场景图作为结构化rubric**：既利用了场景图的结构化降维能力（减少评估对象的表面形式多样性），又通过LLM-QA保持了灵活性（不强制对齐），两者互补
- **从细粒度到粗粒度的可解释性**：每个粗粒度分数都有对应的细粒度span-level错误支撑，这是现有指标（包括GPT-4o-as-Judge）不具备的
- **DOCENT基准的社会价值**：辅助文本生成对视觉障碍者的网络可及性至关重要，艺术品的复杂视觉场景（平均161个视觉组件）是当前VLM的真实挑战

## 局限与展望
- 依赖依存句法分析和共指消解的质量——非英语语言的工具成熟度可能不足
- 当前不加权各组件（实体/属性/关系同等重要），未来可引入任务特定权重
- DOCENT仅含100张图的生成评判，规模受限于人工标注成本（细粒度18分钟/样本）
- reference-based设计依赖参考描述的质量和覆盖率

## 相关工作与启发
- **vs SPICE**：SPICE也用场景图但忽略对象附着→给误附着细节打高分；PoSh保持附着链确保正确验证
- **vs DCScore**：DCScore用GPT-4提取factoids+验证，在mistakes上最强(ρ=0.541)，但提取覆盖率不足导致omissions弱(ρ=0.395)；PoSh用句法分析确保全覆盖
- **vs LLaVA-Critic**：72B VLM-as-Judge在CapArena上最优，但不提供可解释的细粒度分数；PoSh用14B达到接近性能且完全可解释

## 评分
- 新颖性: ⭐⭐⭐⭐ 场景图+LLM-QA的结合设计精巧，DOCENT基准填补空白
- 实验充分度: ⭐⭐⭐⭐⭐ DOCENT细粒度+粗粒度+CapArena跨域+RL奖励函数+子组件验证
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、社会影响有说服力、实验系统全面
- 价值: ⭐⭐⭐⭐ 为详细图像描述评估提供了可部署的开源工具，推动辅助文本生成进步

<!-- RELATED:START -->

## 相关论文

- [RADAR: Reasoning-Ability and Difficulty-Aware Routing for Reasoning LLMs](radar_reasoning-ability_and_difficulty-aware_routing_for_reasoning_llms.md)
- [On the Possible Detectability of Image-in-Image Steganography](../../CVPR2026/interpretability/on_the_possible_detectability_of_imageinimage_steg.md)
- [Enhancing Automated Interpretability with Output-Centric Feature Descriptions](../../ACL2025/interpretability/output_centric_interpretability.md)
- [Edit-As-Act: Goal-Regressive Planning for Open-Vocabulary 3D Indoor Scene Editing](../../CVPR2026/interpretability/edit-as-act_goal-regressive_planning_for_open-vocabulary_3d_indoor_scene_editing.md)
- [Recursive Think-Answer Process for LLMs and VLMs](../../CVPR2026/interpretability/recursive_think-answer_process_for_llms_and_vlms.md)

<!-- RELATED:END -->
