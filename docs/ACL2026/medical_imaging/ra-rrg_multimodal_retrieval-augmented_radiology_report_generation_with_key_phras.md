---
title: >-
  [论文解读] RA-RRG: Multimodal Retrieval-Augmented Radiology Report Generation with Key Phrase Extraction
description: >-
  [ACL 2026][医学图像][放射报告生成] 提出 RA-RRG 框架，通过 LLM 从放射报告中提取临床关键短语并构建检索库，给定胸部 X 光影像后检索相关短语并输入 LLM 生成报告，无需 LLM 微调即可有效抑制幻觉，仅需 18 GPU 小时训练，在 CheXbert 指标上达到 SOTA。
tags:
  - "ACL 2026"
  - "医学图像"
  - "放射报告生成"
  - "检索增强生成"
  - "关键短语提取"
  - "幻觉抑制"
  - "多视图"
---

# RA-RRG: Multimodal Retrieval-Augmented Radiology Report Generation with Key Phrase Extraction

**会议**: ACL 2026  
**arXiv**: [2504.07415](https://arxiv.org/abs/2504.07415)  
**代码**: [GitHub](https://github.com/deepnoid-ai/RA-RRG)  
**领域**: 医学影像 / 放射报告生成  
**关键词**: 放射报告生成, 检索增强生成, 关键短语提取, 幻觉抑制, 多视图

## 一句话总结
提出 RA-RRG 框架，通过 LLM 从放射报告中提取临床关键短语并构建检索库，给定胸部 X 光影像后检索相关短语并输入 LLM 生成报告，无需 LLM 微调即可有效抑制幻觉，仅需 18 GPU 小时训练，在 CheXbert 指标上达到 SOTA。

## 研究背景与动机

**领域现状**：自动化放射报告生成（RRG）是减轻放射科医生工作负担的重要方向。多模态 LLM（如 LLaVA-Rad、MAIRA）已展示了从胸部 X 光直接生成报告的能力，但需要大量计算资源和大规模微调数据。

**现有痛点**：(1) MLLM 方法训练成本高（>200 GPU 小时），限制了临床部署；(2) 检索增强方法（如 CXR-RePaiR）检索的是完整句子或报告，但放射报告中多个发现常共现于同一句子中，朴素检索可能引入与当前影像无关甚至矛盾的信息；(3) 报告中常包含与先前检查的比较性陈述（如"unchanged"、"improved"），在单图像设置下这些构成"比较性幻觉"。

**核心矛盾**：检索增强方法需要足够细粒度的检索单元来避免共现信息污染，但过细的分割又会丢失临床上下文。需要在粒度和信息完整性之间找到平衡。

**本文目标**：设计一种无需 LLM 微调的检索增强 RRG 框架，能够检索细粒度的、无幻觉的临床关键短语，并生成准确的放射报告。

**切入角度**：利用 RadGraph 提取报告的知识图谱结构，再用 LLM 将其精炼为最小临床有意义的短语，同时显式排除比较性表述。

**核心 idea**：用 LLM 精炼 RadGraph 输出为无幻觉关键短语 → 训练多模态检索器匹配影像与短语 → 用 LLM 将检索到的短语扩展为连贯报告，全程不微调 LLM。

## 方法详解

### 整体框架
RA-RRG 分为三个阶段：(1) 关键短语提取——用 RadGraph 解析报告结构后，LLM（Llama 70B）将其精炼为去除比较性幻觉的关键短语；(2) 多模态检索器训练——使用双视觉编码器（XrayDINOv2 + XrayCLIP）提取视觉特征，DETR 解码器输出语义嵌入，与 MPNet 文本嵌入对齐；(3) 报告生成——将检索到的短语输入 GPT-4o 生成连贯报告，无需 LLM 微调。

### 关键设计

1. **LLM 辅助关键短语提取**:

    - 功能：将放射报告分解为最小临床有意义的短语，同时去除幻觉诱导内容
    - 核心思路：首先用 RadGraph 提取报告的 FINDINGS 部分的实体和关系，构建 RadGraph 短语；然后用 Llama 70B 将 RadGraph 输出和原始报告联合输入，精炼为关键短语，同时排除比较性陈述（如 unchanged、improved）。训练集平均每张图像关联 7.16 个关键短语，共 243,064 个唯一短语
    - 设计动机：纯 RadGraph 输出可能产生碎片化的图结构且不处理比较性幻觉；纯 LLM 处理原始文本可能遗漏领域特定临床细节。两者联合输入互补

2. **双编码器 + DETR 解码器的多模态检索器**:

    - 功能：从影像中预测语义嵌入，匹配关键短语向量数据库
    - 核心思路：视觉侧融合 XrayDINOv2（自监督特征）和 XrayCLIP（视觉-语言对齐特征），通过通道拼接获得互补视觉表示。DETR 解码器并行解码 $N=50$ 个查询嵌入，每个嵌入通过选择分类器判断是否激活，语义嵌入通过三层 FFN 生成。文本侧使用冻结的 MPNet 编码关键短语，加入 NEFTune 风格噪声防止过拟合。训练使用匈牙利匹配 + 短语匹配损失 + 批内语义对比损失
    - 设计动机：单一视觉编码器无法同时捕捉自监督的细粒度特征和跨模态对齐特征；DETR 式集合预测天然适合"一图对多短语"的检索场景

3. **零训练 LLM 报告生成**:

    - 功能：将检索到的短语列表整合为连贯的放射报告
    - 核心思路：将检索到的关键短语和任务指令一起输入 GPT-4o，生成完整报告。由于短语已经过幻觉过滤，LLM 只需执行语言组织任务而非临床判断。同一框架自然扩展到多视图（正位+侧位）：分别从每张图像检索短语后合并输入
    - 设计动机：避免 LLM 微调的高成本，同时利用 LLM 强大的语言生成能力将碎片化短语组织为连贯文本

### 损失函数 / 训练策略
总损失 $\mathcal{L} = \sum_b \mathcal{L}_{PM}(y^b, \hat{y}^b) + \lambda \mathcal{L}_{SC}(E)$，其中短语匹配损失 $\mathcal{L}_{PM}$ 使用匈牙利算法分配 + distribution-balanced 分类损失 + 余弦相似度损失。批内语义对比损失 $\mathcal{L}_{SC}$ 采用 CLIP 风格的对称交叉熵，使用软目标避免惩罚语义相近的非匹配对。$\lambda = 0.1$。视觉和文本编码器参数冻结，仅训练 DETR 解码器。

## 实验关键数据

### 主实验
MIMIC-CXR 单视图 RRG（FINDINGS section）：

| 类型 | 模型 | CheXbert micro-F1 | RadGraph F1 | ROUGE-L |
|------|------|--------------------|-------------|---------|
| 生成 | LLaVA-Rad | 57.3 | - | 30.6 |
| 生成 | M4CXR | 58.1 | 21.7 | 28.4 |
| 检索 | MCA-RG | - | - | 30.0 |
| **检索** | **RA-RRG** | **62.3** | **24.3** | **30.7** |

### 消融实验

| 配置 | CheXbert micro-F1 | RadGraph F1 |
|------|--------------------|-------------|
| 仅 RadGraph 短语 | 59.1 | 22.8 |
| LLM 关键短语 (无比较性过滤) | 60.5 | 23.4 |
| LLM 关键短语 (含比较性过滤) | **62.3** | **24.3** |
| 单编码器 (仅 CLIP) | 58.7 | 22.1 |
| 双编码器 (CLIP + DINOv2) | **62.3** | **24.3** |

### 关键发现
- 比较性幻觉过滤贡献显著（micro-F1: 60.5 → 62.3），证明排除"unchanged/improved"等表述的必要性
- 双编码器融合比单编码器提升 3.6% micro-F1，DINOv2 和 CLIP 特征互补
- RA-RRG 仅需 18 GPU 小时训练（vs MLLM >200 GPU 小时），在 CheXbert 指标上超越所有 MLLM
- 框架自然扩展到多视图 RRG，多视图结果进一步提升

## 亮点与洞察
- 关键短语作为检索单元的设计在粒度上找到了很好的平衡——比句子更细避免共现污染，比实体更粗保留临床语境。这个设计可以推广到任何需要细粒度检索的领域
- LLM 在两个阶段分别承担不同角色：提取阶段做知识精炼（Llama 70B），生成阶段做语言组织（GPT-4o），两个阶段都不需要微调，最大化了 LLM 的即用价值
- 比较性幻觉的显式定义和处理是很有实际价值的贡献——这类幻觉在放射学中普遍存在但被此前的方法忽视

## 局限与展望
- 依赖商业 API（GPT-4o）进行报告生成，成本和隐私问题限制临床部署
- RadGraph 本身可能在复杂报告上产生不完整的图结构
- 关键短语检索的召回率受限于训练集的短语覆盖——罕见发现可能无匹配短语
- 未来可以用开源 LLM 替代 GPT-4o，或将检索器与小型生成模型端到端训练

## 相关工作与启发
- **vs CXR-RePaiR**: 检索完整报告/句子，存在共现信息污染；RA-RRG 检索最小临床短语，更精确
- **vs MAIRA-1/LLaVA-Rad**: 这些 MLLM 需要大规模微调，RA-RRG 通过检索+冻结 LLM 实现更低成本

## 评分
- 新颖性: ⭐⭐⭐⭐ 关键短语提取 + 双编码器检索 + 零训练 LLM 生成的组合有创新
- 实验充分度: ⭐⭐⭐⭐ 在两个数据集上全面评估，消融充分，包含幻觉分析
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，架构图直观
- 价值: ⭐⭐⭐⭐ 为资源受限场景下的放射报告生成提供了实用方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)
- [\[ACL 2026\] Knowledge Poisoning Attacks on Medical Multi-Modal Retrieval-Augmented Generation](knowledge_poisoning_attacks_on_medical_multi-modal_retrieval-augmented_generatio.md)
- [\[ACL 2026\] HeteroRAG: A Heterogeneous Retrieval-Augmented Generation Framework for Medical Vision Language Tasks](heterorag_a_heterogeneous_retrieval-augmented_generation_framework_for_medical_v.md)
- [\[ACL 2025\] Automated Structured Radiology Report Generation](../../ACL2025/medical_imaging/automated_structured_radiology_report_generation.md)
- [\[CVPR 2026\] OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation](../../CVPR2026/medical_imaging/orapo_oracle-educated_reinforcement_learning_for_data-efficient_and_factual_radi.md)

</div>

<!-- RELATED:END -->
