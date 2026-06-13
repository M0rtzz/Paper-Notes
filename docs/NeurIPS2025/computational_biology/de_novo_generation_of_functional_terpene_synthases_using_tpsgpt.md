---
title: >-
  [论文解读] De novo generation of functional terpene synthases using TpsGPT
description: >-
  [NeurIPS 2025][计算生物][萜烯合酶] TpsGPT 通过在 79K 萜烯合酶（TPS）序列上微调蒸馏版 ProtGPT2 Tiny（38.9M 参数），生成 28K 候选序列，经多阶段过滤（困惑度/pLDDT/EnzymeExplorer/CLEAN/InterPro/Foldseek）筛选出 7 条进化距离远（<60% 序列相似度）但结构保守的从头 TPS 序列，湿实验验证其中 2 条具有 TPS 酶活性——以不到 $200 GPU 成本实现功能酶从头设计。
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "萜烯合酶"
  - "蛋白质语言模型"
  - "ProtGPT2"
  - "从头设计"
  - "湿实验验证"
---

# De novo generation of functional terpene synthases using TpsGPT

**会议**: NeurIPS 2025  
**arXiv**: [2512.08772](https://arxiv.org/abs/2512.08772)  
**代码**: [https://github.com/colorfulcereal/TpsGPT](https://github.com/colorfulcereal/TpsGPT)  
**领域**: 蛋白质设计 / 酶工程  
**关键词**: 萜烯合酶, 蛋白质语言模型, ProtGPT2, 从头设计, 湿实验验证

## 一句话总结
TpsGPT 通过在 79K 萜烯合酶（TPS）序列上微调蒸馏版 ProtGPT2 Tiny（38.9M 参数），生成 28K 候选序列，经多阶段过滤（困惑度/pLDDT/EnzymeExplorer/CLEAN/InterPro/Foldseek）筛选出 7 条进化距离远（<60% 序列相似度）但结构保守的从头 TPS 序列，湿实验验证其中 2 条具有 TPS 酶活性——以不到 $200 GPU 成本实现功能酶从头设计。

## 研究背景与动机

**领域现状**：萜烯合酶（TPS）是催化萜烯骨架合成的关键酶家族，萜烯是最大的天然产物类别（>76,000 种），包括抗癌药 Taxol 等重要药物。工业上萜烯的化学合成成本高、步骤多，合成生物学利用 TPS 酶催化是更高效的路线。

**现有痛点**：(a) **定向进化**是主流酶工程手段但成本极高（数十万美元级）、速度慢，且生成的序列与天然蛋白高度相似，无法探索广阔的序列空间；(b) 计算方法如 HMMER 只能发现自然界已有的 TPS，无法设计新酶；(c) 结构导向设计（如 RFdiffusion）需要催化位点的详细知识，对 TPS 这种功能机理复杂的酶族不实用。

**核心矛盾**：TPS 是代表性不足的酶家族（仅 1125 条实验验证序列），但序列空间极其广阔。已有方法要么无法探索新序列（数据库挖掘），要么成本过高（定向进化），要么需要不可得的结构知识（结构设计）。

**本文目标** 仅从序列数据出发（无需结构知识），低成本地从头生成功能性 TPS 酶。

**切入角度**：蛋白质语言模型（PLM）已被证明可以通过微调在特定家族上生成有效蛋白质。但现有 PLM 微调方法通常需要大规模家族数据和控制标签。作者从 1125 条种子序列出发挖掘 79K 同源 TPS，在小模型（ProtGPT2 Tiny, 38.9M）上微调，降低成本。

**核心 idea**：精心构建酶族特异性数据集 + 蒸馏 PLM 微调 + 多阶段计算过滤 = 低成本功能酶从头设计。

## 方法详解

### 整体框架
**输入**：79K TPS 序列数据集（从 UniProt 挖掘）。**模型**：ProtGPT2 Tiny（38.9M 参数）微调。**输出**：28K 候选序列 → 多阶段过滤 → 7 条候选 → 湿实验验证 → 2 条确认活性。

### 关键设计

1. **数据集构建**：

    - 功能：从 1125 条实验验证的 TPS 种子序列扩展到 79K 训练集
    - 核心思路：(a) 用 HMMER hmmsearch + Pfam/SUPERFAMILY 数据库在 UniProt 和 BFD 大规模数据库中检索 TPS 同源序列；(b) 严格过滤：去除 300-1100 氨基酸范围外的序列、排除非 TPS 域更强匹配的序列、要求包含 TPS 催化基序（DDXXD, NSE/DTE, DXDD）、过滤掉 >80% 相似度的异戊烯基二磷酸合酶
    - 设计动机：1125 条太少无法直接训练 PLM，但通过计算挖掘可以扩展 70 倍。严格过滤确保训练数据质量——只保留高置信度的 TPS 序列

2. **数据划分策略**：

    - 功能：确保训练/验证集无数据泄露
    - 核心思路：使用 SpanSeq 在 30% 序列相似度阈值下将 79K 序列聚类为 6 个分区。5 个分区（~63K）用于训练，1 个分区（~16K）用于验证。确保训练和验证集之间最大序列相似度 ≤30%
    - 设计动机：传统随机划分可能导致高度相似的序列分别出现在训练和验证集（数据泄露）。SpanSeq 基于序列相似度的划分保证了评估的可靠性

3. **模型微调**：

    - 功能：在 TPS 数据上微调蒸馏版 ProtGPT2
    - 核心思路：ProtGPT2 Tiny（38.9M 参数，原版 738M 的蒸馏版），block size=512 tokens，batch size=64×8 梯度累积=有效 batch 512，学习率 1e-4，最大 4000 步。训练损失从 8.4 降至 4.94，验证损失从 8.0 降至 7.32
    - 设计动机：蒸馏版推理速度快 6 倍、困惑度与原版可比，使得大规模序列生成（28K 条）计算可行。单张 NVIDIA L4 即可完成微调

4. **多阶段计算过滤管线**：

    - 功能：从 28K 候选中筛选出最有可能具有功能的序列
    - 核心思路：三层过滤——
        - **序列过滤**：困惑度排序取 top 10%（2800 条）；与训练集最大序列相似度 ≤60%（保证进化距离）
        - **功能过滤**：EnzymeExplorer TPS 检测得分 ≥0.7；CLEAN 预测 EC 号属于萜烯生物合成通路；InterPro 域预测检测到 TPS 特异性域
        - **结构过滤**：ESMFold pLDDT ≥70（结构稳定性）；Foldseek TM-score 在 0.6-0.9 之间（与训练集结构相似但不完全相同）
    - 设计动机：每层过滤从不同维度验证——序列质量（困惑度）、功能可能性（酶分类）、结构可行性（折叠置信度），层层筛选将 28K→2800→77→7

5. **湿实验验证**：

    - 功能：在酿酒酵母中异源表达并检测酶活性
    - 核心思路：在工程化的 JWY501 酵母株（高产 GGPP 底物）中表达 7 条基因，LC-MS 检测产物。提取离子色谱（XIC）在 $\text{C}_{20}\text{H}_{36}\text{O}_2$ 质量处检测到 TpsGPT1 和 TpsGPT2 产生类双萜产物（如当归环酮）
    - 设计动机：计算验证再全面也不能替代实验——湿实验是功能性从头酶设计的最终证明

### 损失函数 / 训练策略
- 标准自回归语言模型损失（下一 token 预测）
- 超参数搜索：学习率 {1e-6, 1e-5, 1e-4, 1e-3}，选 1e-4（验证损失最优）
- 最大步数 {1200, 1875, 3000, 4000}，选 4000（收敛）
- 全部 GPU 成本 <$200（单张 NVIDIA L4）

## 实验关键数据

### 主实验 — 7 条候选 TPS 的计算验证

| 序列 ID | EnzymeExplorer ↑ | pLDDT ↑ | TM-Score | Max Seq ID | CLEAN EC | 实验活性 |
|---------|------------------|---------|----------|------------|----------|---------|
| **TpsGPT1** | 0.75 | 78 | 0.73 | 49.67% | 萜烯合酶 (4.2.3.75) | **✓** |
| **TpsGPT2** | 0.72 | 74 | 0.79 | 59.72% | 角鲨烯合酶 (2.5.1.21) | **✓** |
| TpsGPT3 | 0.73 | 74 | 0.84 | 60.00% | 环二烯醇合酶 | 待验 |
| TpsGPT4 | 0.73 | 70 | 0.65 | 60.08% | 角鲨烯合酶 | 待验 |
| TpsGPT5 | 0.78 | 80 | 0.72 | 59.75% | β-香树脂合酶 | 待验 |
| TpsGPT6 | 0.73 | 71 | 0.69 | 57.33% | 角鲨烯合酶 | 待验 |
| TpsGPT7 | 0.74 | 71 | 0.72 | 52.19% | 环木菠萝烷醇合酶 | 待验 |

### 过滤管线效率

| 过滤阶段 | 剩余序列数 | 淘汰率 |
|----------|-----------|--------|
| 原始生成 | 28,000 | — |
| 困惑度 Top 10% | 2,800 | 90% |
| pLDDT ≥70 | ~1,120 (40%) | 60% |
| EnzymeExplorer ≥0.7 | 77 | 93% |
| MaxID ≤60% + Foldseek 0.6-0.9 | 7 | 91% |
| 湿实验验证 | 2/7 有活性 | 29% |

### 关键发现
- pLDDT 分布：40% 的生成序列达到 ≥70 的结构置信度，说明 TpsGPT 学会了生成可折叠蛋白质
- 进化距离：7 条候选与训练集最大序列相似度在 49.67%~60.08%，确实是进化远端的新序列
- 结构保守：所有候选 TM-score 在 0.6-0.9，属于同一结构家族但不是简单复制
- TpsGPT1 最独特（49.67% maxID）且实验验证有活性，说明模型成功探索了远端序列空间
- 产物含氧（$\text{C}_{20}\text{H}_{36}\text{O}_2$），尚无法确认为经典 TPS 产物，需进一步催化机理研究

## 亮点与洞察
- **极致的低成本**：<$200 GPU + 38.9M 参数模型 = 功能酶从头设计，大幅降低了蛋白质设计的门槛。对比定向进化的数十万美元成本，这是数量级的改进
- **数据集构建是关键**：1125 种子→79K 高质量 TPS 数据集的精心构建（HMMER+多重过滤+催化基序检查）是整个管线的基础。数据工程的重要性不亚于模型选择
- **多维度过滤策略**：序列（困惑度+相似度）+功能（3 种分类器/域预测工具）+结构（pLDDT+TM-score）的组合过滤，从 28K 精准定位到 7 条，每一层都有明确的物理/生物学意义
- **可推广框架**：该 pipeline（数据挖掘→PLM 微调→生成→多阶段过滤→实验验证）可直接应用于其他代表性不足的酶家族（如溶菌酶、P450）

## 局限与展望
- 仅 2/7 实验验证有活性（29% 成功率），且产物含氧尚不确定是否为经典萜烯
- 无法条件化生成——不能指定要生成哪种亚类的 TPS（如单萜/倍半萜/二萜）
- 仅用序列信息，未利用 3D 结构信息（如催化口袋）指导生成
- ProtGPT2 Tiny 模型容量有限（38.9M），可能限制了序列空间的探索能力
- 过滤管线依赖多个外部工具（EnzymeExplorer, CLEAN, InterPro, ESMFold, Foldseek），任一工具的偏差会传递到最终结果
- 可探索条件生成（加控制标签）、结合结构约束（活性位点 motif 注入）、使用更大模型

## 相关工作与启发
- **vs RFdiffusion（结构导向设计）**：需要催化位点的详细结构知识，对 TPS 不实用；TpsGPT 仅需序列数据
- **vs ProGEN（大规模 PLM 微调）**：ProGEN 280M 参数，需要大数据集+控制标签；TpsGPT 用 38.9M 蒸馏模型+无控制标签，更轻量
- **vs HMMER 数据库挖掘**：只能发现自然界已有的序列；TpsGPT 生成进化远端（<60% 相似度）的新序列
- **启发**：PLM 微调对"小数据酶族"特别有价值——只需 ~1000 种子序列+数据挖掘扩展即可训练有效生成模型

## 评分
- 新颖性: ⭐⭐⭐ PLM 微调生成蛋白质并非首创，但针对 TPS 酶族的完整数据+生成+过滤+实验管线是新的
- 实验充分度: ⭐⭐⭐⭐ 计算验证全面（6 种验证指标），且有湿实验验证（2/7 有活性），但样本量小
- 写作质量: ⭐⭐⭐⭐ 管线描述清晰，过滤标准有据可依，结果展示直观
- 价值: ⭐⭐⭐⭐ 低成本功能酶从头设计对合成生物学有直接应用价值，管线可推广到其他酶族

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] One Small Step with Fingerprints, One Giant Leap for De Novo Molecule Generation from Mass Spectra](one_small_step_with_fingerprints_one_giant_leap_for_de_novo_molecule_generation_.md)
- [\[ICML 2025\] PepTune: De Novo Generation of Therapeutic Peptides with Multi-Objective-Guided Discrete Diffusion](../../ICML2025/computational_biology/peptune_de_novo_generation_of_therapeutic_peptides_with_multi-objective-guided_d.md)
- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[ICML 2025\] UniMoMo: Unified Generative Modeling of 3D Molecules for De Novo Binder Design](../../ICML2025/computational_biology/unimomo_unified_generative_modeling_of_3d_molecules_for_de_novo_binder_design.md)
- [\[ICML 2025\] Latent Imputation before Prediction: A New Computational Paradigm for De Novo Peptide Sequencing](../../ICML2025/computational_biology/latent_imputation_before_prediction_a_new_computational_paradigm_for_de_novo_pep.md)

</div>

<!-- RELATED:END -->
