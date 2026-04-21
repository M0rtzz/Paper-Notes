---
title: >-
  [论文解读] MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild
description: >-
  [多模态] 提出 MolParser，一个端到端的光学化学结构识别 (OCSR) 方法，通过扩展 SMILES 表示（E-SMILES）处理 Markush 结构、构建 700 万级大规模训练集 MolParser-7M，并利用主动学习引入真实文献数据，在 WildMol 基准上以 76.9% 准确率显著超越现有方法。
tags:
  - 多模态
---

# MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild

## 论文信息

- **会议**: ICCV 2025
- **arXiv**: 2411.11098
- **代码/数据**: HuggingFace 公开（MolParser-7M 数据集）
- **领域**: 多模态视觉语言模型 / 化学分子识别
- **关键词**: OCSR, molecular recognition, SMILES, end-to-end, active learning, Markush structure

## 一句话总结

提出 MolParser，一个端到端的光学化学结构识别 (OCSR) 方法，通过扩展 SMILES 表示（E-SMILES）处理 Markush 结构、构建 700 万级大规模训练集 MolParser-7M，并利用主动学习引入真实文献数据，在 WildMol 基准上以 76.9% 准确率显著超越现有方法。

## 研究背景与动机

化学文献和专利中大量关键信息以分子结构图的形式呈现，自动提取机器可读的分子结构（OCSR 任务）具有重要价值。现有方法面临三大挑战：

**表示局限**：标准 SMILES 无法表示专利文献中常见的 Markush 结构（包含 R-group 变量的分子家族）、连接点、抽象环、聚合物等

**数据稀缺**：最大公开数据集仅 30 万合成样本（MolGrapher-300k），且全为合成数据，与真实文献的风格差异显著

**In-the-wild 鲁棒性差**：真实专利/论文中的分子图像存在缩写、噪声、模糊、多样绘制风格等问题，现有方法表现不佳

## 方法详解

### 整体框架

MolParser 将 OCSR 视为图像描述（image captioning）任务，输入分子结构图像，输出 E-SMILES 字符串。模型由三部分组成：

- **图像编码器**：ImageNet 预训练的 Swin-Transformer（Tiny/Small/Base 三种规格）
- **特征压缩器**：类似 LLaVA 的两层 MLP 作为视觉-语言连接器
- **SMILES 解码器**：BART-Decoder，自回归生成 E-SMILES 序列

### 关键设计 1：扩展 SMILES (E-SMILES)

格式为 `SMILES<sep>EXTENSION`，其中：
- **SMILES 部分**：标准 RDKit 兼容 SMILES
- **EXTENSION 部分**：使用 XML-like 特殊标记描述特殊功能基团
    - `<a>...</a>`：Markush R-group 和缩写基团
    - `<r>...</r>`：不确定位置的环连接
    - `<c>...</c>`：抽象环
    - `<dum>`：连接点
- 功能基团描述格式：`[INDEX]:[GROUP_NAME]`

E-SMILES 兼容 RDKit 且对 LLM 友好，便于后续分析处理。

### 关键设计 2：MolParser-7M 数据集

**预训练数据（~770 万）**：

| 子集 | 占比 | 来源 |
|------|------|------|
| Markush-3M | 40% | PubChem 随机基团替换 |
| ChEMBL-2M | 27% | ChEMBL 数据库 |
| Polymer-1M | 14% | 随机生成聚合物 |
| PAH-600k | 8% | 随机稠环分子 |
| BMS-360k | 5% | 长碳链分子 |
| MolGrapher-300K | 4% | MolGrapher 论文数据 |
| Pauling-100k | 2% | Pauling 风格图像 |

**微调数据（~60 万）**：66% 人工标注真实数据 + 32% 筛选合成数据 + 1% 手写分子。

### 关键设计 3：主动学习数据引擎

1. 训练 YOLO11 检测模型 (MolDet) 定位 PDF 中的分子，从 122 万真实 PDF 提取 2000 万分子图像
2. 去重后保留 400 万张，进行 5 折交叉训练得到 5 个模型
3. 每张图像生成 5 个预测，计算 Tanimoto 相似度评分作为置信度
4. 选择置信度 0.6-0.9 的样本（有挑战且有价值）进行人工标注
5. 模型预测作为预标注，标注时间从 3 分钟/样本降至 30 秒/样本（节省 90% 人力）
6. 每 8 万条标注更新模型并重复循环，最终获得 40 万高质量标注

### 训练策略：课程学习

预训练阶段逐步增加难度：先用简单分子（token < 60）不做数据增强 → 逐步增加增强强度和分子复杂度 → 微调阶段使用真实数据。

## 实验关键数据

### 主实验：跨基准对比

| 方法 | USPTO | UoB | CLEF | JPO | ColoredBG | USPTO-10K | WildMol-10K |
|------|-------|-----|------|-----|-----------|-----------|-------------|
| OSRA 2.1 | 89.3 | 86.3 | 93.4 | 56.3 | 5.5 | 89.7 | 26.3 |
| MolGrapher | 91.5 | 94.9 | 90.5 | 67.5 | 7.5 | 93.3 | 45.5 |
| DECIMER 2.7 | 59.9 | 88.3 | 72.0 | 64.0 | 14.5 | 82.4 | 56.0 |
| MolScribe | 93.1 | 87.4 | 88.9 | 76.2 | 21.0 | 96.0 | 66.4 |
| **MolParser-Base** | **93.0** | **91.8** | **90.7** | **78.9** | **57.0** | **94.5** | **76.9** |

- 在最具挑战性的 WildMol-10K（真实专利分子）上，MolParser (76.9%) 大幅超越 MolScribe (66.4%) 和 MolGrapher (45.5%)
- ColoredBG 数据集上提升最为显著（57.0% vs 21.0%）

### 消融实验

| 训练数据 | 微调 | WildMol-10K ↑ |
|---------|------|---------------|
| MolGrapher-300k | - | 22.4 |
| MolParser-7M (pt) | - | 51.9 |
| MolParser-7M (pt+ft) | - | 75.9 |
| MolParser-7M (pt) | MolParser-7M (ft) | **76.9** |

| 数据增强 | 课程学习 | WildMol-10K ↑ |
|---------|---------|---------------|
| ✗ | ✗ | 40.1 |
| ✓ | ✗ | 69.5 |
| ✓ | ✓ | **76.9** |

**关键发现**：
- 训练数据规模至关重要：从 300k 扩展到 7M 将准确率从 22.4% 提升到 51.9%
- 真实数据微调贡献巨大：+25% 提升（51.9→76.9）
- 课程学习策略带来额外 7.4% 提升

### 速度-精度 Pareto 前沿

| 模型 | 吞吐量 (FPS) | WildMol-10K | WildMol-10K-M |
|------|-------------|-------------|---------------|
| MolParser-Tiny | 131.6 | 73.1 | 15.3 |
| MolParser-Small | 116.3 | 76.3 | 34.8 |
| MolParser-Base | 39.8 | 76.9 | 38.1 |
| MolGrapher | 2.2 | 45.5 | - |

MolParser-Tiny 速度是 MolGrapher 的 60 倍，准确率高出 27.6%。

### 附加发现：分子性质预测

MolParser 训练后的 Swin-T 视觉编码器可作为分子指纹提取器，在 MoleculeNet 基准上达到与 2D/3D 图神经网络方法相当的性能（平均 ROC-AUC 73.7 vs 最佳 74.5），表明 OCSR 训练学到了化学语义特征。

## 亮点与洞察

1. **E-SMILES 设计实用且优雅**：保持 RDKit 兼容性的同时支持复杂 Markush 结构，是对 SMILES 的重要工程扩展
2. **主动学习数据引擎**：置信度 0.6-0.9 的样本选择策略非常巧妙——过低质量差，过高已会预测
3. **数据规模 vs 模型规模**：实验证明数据规模和真实数据的重要性远大于模型参数量
4. **意外发现**：OCSR 预训练的视觉编码器保留了丰富的化学语义信息，可用于分子性质预测

## 局限性

- 分子手性（chirality）识别尚未充分利用
- Markush 结构识别准确率仍较低（38.1%），需要更多标注数据
- 过大的端到端模型（如 Mini-InternVL, 2.2B 参数）反而训练更困难，表现不如较小模型
- 回归到 E-SMILES 字符串的方式对超长分子可能不够鲁棒

## 相关工作与启发

- 端到端方法 vs 图重建方法：端到端方法速度快但需要大量训练数据，MolParser 通过数据引擎和 7M 数据集解决了这一限制
- 与 LLaVA 架构的关系：MolParser 采用了类似的视觉-语言连接器设计
- 化学反应解析的扩展应用：结合 GPT-4o 进行反应方程式识别，展示了 MolParser 作为基础组件的更广泛价值

## 评分

⭐⭐⭐⭐ — 系统性很强的工作，从表示（E-SMILES）到数据（7M）到模型（端到端）到应用（反应解析+分子指纹）形成完整闭环。主动学习数据引擎是核心创新。Markush 识别仍有较大改进空间。

<!-- RELATED:START -->

## 相关论文

- [WebDS: An End-to-End Benchmark for Web-based Data Science](../../ICLR2026/multimodal_vlm/webds_an_end-to-end_benchmark_for_web-based_data_science.md)
- [Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving](hints_of_prompt_enhancing_visual_representation_for_multimodal_llms_in_autonomou.md)
- [Training-Free Personalization via Retrieval and Reasoning on Fingerprints](training-free_personalization_via_retrieval_and_reasoning_on_fingerprints.md)
- [SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)
- [Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models](physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la.md)

<!-- RELATED:END -->
