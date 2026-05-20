---
title: >-
  [论文解读] Data-Juicer Sandbox: A Feedback-Driven Suite for Multimodal Data-Model Co-development
description: >-
  [ICML 2025][视频生成][数据-模型协同开发] 提出 Data-Juicer Sandbox 沙箱套件，通过"探测-分析-精炼"(Probe-Analyze-Refine) 工作流，在低成本小规模实验中系统探索数据处理算子 (OP) 与模型性能的交互关系，将获得的数据配方迁移到大规模场景…
tags:
  - "ICML 2025"
  - "视频生成"
  - "数据-模型协同开发"
  - "多模态大模型"
  - "数据处理算子"
  - "沙箱实验"
  - "数据质量与多样性"
---

# Data-Juicer Sandbox: A Feedback-Driven Suite for Multimodal Data-Model Co-development

**会议**: ICML 2025  
**arXiv**: [2407.11784](https://arxiv.org/abs/2407.11784)  
**代码**: [modelscope/data-juicer](https://modelscope.github.io/data-juicer/en/main/docs/Sandbox.html)  
**领域**: 视频生成  
**关键词**: 数据-模型协同开发, 多模态大模型, 数据处理算子, 沙箱实验, 数据质量与多样性

## 一句话总结

提出 Data-Juicer Sandbox 沙箱套件，通过"探测-分析-精炼"(Probe-Analyze-Refine) 工作流，在低成本小规模实验中系统探索数据处理算子 (OP) 与模型性能的交互关系，将获得的数据配方迁移到大规模场景，在 VBench 排行榜取得第一名。

## 研究背景与动机

多模态大模型的优化长期面临"数据中心"与"模型中心"两条路径割裂的问题：

- **模型中心方法**在固定数据先验下优化架构和算法，忽略数据质量对模型的影响
- **数据中心方法**独立于模型训练上下文来处理数据集，缺乏模型反馈信号
- 两者缺乏协同，严重依赖启发式探索和单一视角的专家经验

在大模型时代，数据处理和模型训练的成本急剧增长，研究者通常被迫在"追求结果"和"深入探索"之间做选择。缺乏成本可控的平台来加速数据-模型协同开发，使得一个领域的改进难以直接反馈和增强另一个领域。

**核心问题**：如何在可控成本下，系统性地探索数据处理操作对模型性能的影响，并将小规模实验的洞察迁移到大规模生产环境？

## 方法详解

### 整体框架

Data-Juicer Sandbox 是一个反馈驱动的沙箱套件，采用**三层解耦架构**：

1. **顶层——工作流层**：将协同开发组织为有序的作业列表，分为四个阶段——探测数据/模型、精炼数据配方、执行数据/模型操作、评估。用户可灵活调整任务序列或复用内置工作流
2. **中层——行为层**：以 Hook 函数形式封装通用行为，如数据分析、模型训练触发、评估回调等
3. **底层——能力层**：以工厂类 (Factory Classes) 封装底层能力，包括 Data-Juicer 提供的 100+ 多模态数据分析/过滤/合成算子 (OP)，以及集成的 SOTA 开源模型训练与评估框架（Mini-Gemini、EasyAnimate、ModelScope、VBench、MMBench、TextVQA、MME 等）

所有组件通过配置文件管理，支持自定义编排，大幅降低认知负担。

### 关键设计：Probe-Analyze-Refine 工作流

这是论文的核心方法，分为四个递进阶段：

**阶段一：单算子数据池 (Single-Operator Data Pools)**

给定初始数据集 $\mathcal{D}$，对每个感兴趣的过滤算子 $\mathcal{OP}_i$：

1. 用该算子处理数据集：$\mathcal{P}_i = \mathcal{DJ}[\mathcal{OP}_i(\rho_i)](\mathcal{D})$
2. 按算子生成的统计量排序，将结果等分为三个数据池：$\mathcal{P}_{i,\text{low}}$、$\mathcal{P}_{i,\text{mid}}$、$\mathcal{P}_{i,\text{high}}$
3. 随机采样 $\mathcal{D}$ 作为对照组 $\mathcal{D}_{rand}$，保证所有 $3N+1$ 个数据池大小相同
4. 在各数据池上独立训练参考模型，使用一致的超参数、数据量和计算资源
5. 通过模型评估指标的反馈，挖掘洞察并识别 Top OP

**阶段二：多算子数据池 (Multi-Operators Data Pools)**

将多个 OP 顺序组合：$\mathcal{P}_S = (\mathcal{DJ}[\mathcal{OP}_i] \circ \mathcal{DJ}[\mathcal{OP}_j] \circ \cdots)(\mathcal{D})$

由于组合数随 OP 数量指数增长，提出两种实用组合策略：

- **策略一**：按单算子实验中的排名递减组合 Top OP
- **策略二**：基于 Pearson 相关系数聚类 OP，在各类别内组合 Top OP

**阶段三：金字塔形数据池 (Pyramid-shaped Data Pools)**

解决数据**质量与多样性的权衡**：更多 OP 组合 → 更高质量但更少数据量。

构造层次金字塔结构：$n_s$ 个 Top OP 组合成 $2^{n_s}-1$ 个数据池：

- **顶层**：所有 OP 组合 (如 $\mathcal{OP}_{1,2,3}$)，数据量最小但质量最高
- **中层**：两两组合 (如 $\mathcal{OP}_{1,2}$)，数据量稍大
- **底层**：单个 OP，数据量最大但平均质量较低

两种训练设置的对比：

1. **重复训练**：仅用顶层高质量数据池，以不同重复率训练
2. **非重复训练**：逐步加入低层数据池并去重，保持相同计算量

**阶段四：扩展与迁移**

所有数据池从 $\mathcal{D}$ 均匀采样且一致派生，使小规模实验的洞察可外推到更大规模场景。

### 损失函数 / 训练策略

本文不修改模型训练损失函数，而是**专注于数据侧的优化**。核心训练策略包括：

- **成本控制**：所有实验保持一致的计算开销，小池训练时间 $T_{pool}$ 相比全量训练 $T_{full}$ 缩减比例为 $r$，使总成本 $(1+mr) \times T_{full} \leq M \times T_{full}$
- **早停策略**：对不 promising 的实验试次进行提前终止
- **可迁移性保证**：通过 Hoeffding 不等式理论分析，证明小池实验与全量实验的性能偏差 $\epsilon$ 随 $r$ 增加呈指数级衰减

## 实验关键数据

覆盖 5 个模型、4 类任务、100+ 实验、70+ 评估指标、40+ 数据处理算子。

### 主实验

**单算子排名——Top-3 OP 的平均性能变化（相对基线 %）：**

| 任务 | 算子 | $\mathcal{P}_{low}$ | $\mathcal{P}_{mid}$ | $\mathcal{P}_{high}$ |
|------|------|------|------|------|
| I2T (图生文) | Image NSFW Filter | +7.13 | +18.44 | **+66.38** |
| I2T | Text Action Number | **+59.90** | +0.29 | -2.05 |
| T2V (文生视频) | Video Aesthetics Score | -0.98 | +0.13 | **+0.96** |
| T2V | Video NSFW Score | **+0.82** | -0.05 | -0.57 |
| ITP (预训练) | CLIP Image-Text Sim. | -32.57 | -6.39 | **+39.53** |
| IC (图像描述) | Text Length | **+0.76** | -3.13 | -11.36 |

### 消融实验

**数据质量 vs 多样性的权衡（金字塔实验）：**

| 配置 | 关键指标 | 说明 |
|------|------|------|
| 高质量数据 × 重复训练 | 早期优势明显 | 数据重复在 epoch 较少时效果好 |
| 低质量但更多数据 | 后期赶上甚至超越 | 多样性在训练充分时更重要 |
| 最优平衡点 | 取决于计算预算 | 计算预算有限偏质量，充裕偏多样性 |

### 关键发现

1. **输出模态决定关键 OP**：T2V 任务的 Top OP 全是视频相关算子；I2T 和 ITP 任务的 Top OP 多为文本/图文相关算子。启示：应将更多资源分配给与模型输出模态对应的数据处理
2. **不同 OP 的最优统计范围不同**：有些 OP 在 high 区间最好（如 NSFW, Image-Text Similarity），有些在 low 区间最好（如 Text Action Number），需要逐一探测而非凭直觉选择
3. **OP 组合效果非线性**：Top-2 组合通常优于单算子，但 Top-3 组合不一定进一步提升
4. **小规模洞察可迁移**：在小数据池上发现的最优配方，应用到大规模场景后仍保持性能优势
5. **VBench 排行榜第一**：从 EasyAnimate 小池实验获得的洞察，迁移到架构不同的 T2V-Turbo 模型后，超越 Gen-3、VEnhancer 等竞品
6. **类 Scaling Law 行为**：CLIP 预训练中小 FLOPS 模型找到的最优配方在 FLOPS 增加时仍保持优势；InternVL 从 1B 扩展到 26B 时配方优势稳定

## 亮点与洞察

- **方法论创新**：首次提出系统的数据-模型协同开发沙箱，将以往"拍脑袋"的数据清洗和模型调优转变为反馈驱动的科学实验流程
- **成本效益突出**：通过大量小规模实验（$r$ 远小于 0.01）取代少量大规模启发式迭代，大幅降低总体开发成本
- **金字塔设计精巧**：数据质量与多样性的权衡框架为数据混合策略提供了清晰的分析工具
- **跨模型跨架构迁移**：在 EasyAnimate 上发现的数据配方可直接用于架构不同的 T2V-Turbo 并取得 SOTA，证明配方的通用性
- **基础设施价值**：Data-Juicer Sandbox 作为开源中间件填补了多模态数据-模型协同开发的基础设施空白

## 局限与展望

1. **OP 探索空间有限**：目前仅充分探索了过滤类 OP，合成类（如数据增强、标注重写）和映射类 OP 的探索较少
2. **模型覆盖有限**：主要验证在 CLIP、LLaVA-like、DiT-based 模型上，对闭源模型的适用性未验证
3. **计算成本仍不低**：100+ 实验的总成本仍然可观，对资源受限的团队可能不易承受
4. **配方迁移的理论保证**：Hoeffding 不等式的分析较粗糙，未考虑数据分布偏移等实际问题
5. **自动化程度可提升**：OP 组合和超参搜索目前主要靠启发式策略，未来可结合 AutoML 或贝叶斯优化

## 相关工作与启发

- **Data-Juicer** (Chen et al., 2024)：本文的基础系统，提供了 100+ 多模态数据算子
- **DataComp** (Gadre et al., 2023)：CLIP 数据竞赛，强调数据筛选对预训练的重要性
- **Mini-Gemini / InternVL-2.0**：LLaVA 系列图文生成模型，本文的实验载体
- **EasyAnimate / T2V-Turbo**：基于 DiT 的视频生成模型，验证了配方跨架构迁移
- **VBench**：视频生成的综合评估基准，本文取得排行榜第一

**启发**：本文的"先小规模系统探索、再大规模迁移"的方法论可推广到其他资源密集型 AI 开发场景，例如 RLHF 数据配方优化、多任务学习的数据混合比例搜索等。

## 评分

| 维度 | 分数 | 说明 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 首个系统的数据-模型协同开发沙箱 |
| 技术深度 | ⭐⭐⭐⭐ | 工作流设计合理，金字塔分析有启发性 |
| 实验充分度 | ⭐⭐⭐⭐⭐ | 100+实验，4类任务，5个模型，VBench第一 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，但公式符号较多增加阅读负担 |
| 实用性 | ⭐⭐⭐⭐⭐ | 开源基础设施，配方可直接复用 |
| **综合** | **⭐⭐⭐⭐☆** | 工程导向的系统性工作，实验扎实，实用价值高 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] S²Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation](../../NeurIPS2025/video_generation/s2q-vdit_accurate_quantized_video_diffusion_transformer_with_salient_data_and_sp.md)
- [\[ICML 2026\] Exploring Data-Free LoRA Transferability for Video Diffusion Models](../../ICML2026/video_generation/exploring_data-free_lora_transferability_for_video_diffusion_models.md)
- [\[CVPR 2025\] Presto: Long Video Diffusion Generation with Segmented Cross-Attention and Content-Rich Video Data Curation](../../CVPR2025/video_generation/long_video_diffusion_generation_with_segmented_cross-attention_and_content-rich_.md)
- [\[ICML 2025\] How Far is Video Generation from World Model: A Physical Law Perspective](how_far_is_video_generation_from_world_model_a_physical_law_perspective.md)
- [\[ICCV 2025\] DIVE: Taming DINO for Subject-Driven Video Editing](../../ICCV2025/video_generation/dive_taming_dino_for_subject-driven_video_editing.md)

</div>

<!-- RELATED:END -->
