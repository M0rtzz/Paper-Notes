---
title: >-
  [论文解读] CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records
description: >-
  [AAAI 2026][医学图像][LLM临床决策] 提出 CliCARE 框架，将非结构化的纵向癌症电子病历（EHR）转化为时序知识图谱（TKG），并与临床指南知识图谱对齐融合，为 LLM 提供循证依据的临床决策支持，同时设计了与专家评估高度相关的 LLM-as-a-Judge 评估协议。
tags:
  - "AAAI 2026"
  - "医学图像"
  - "LLM临床决策"
  - "电子健康记录"
  - "时序知识图谱"
  - "临床指南对齐"
  - "RAG"
---

# CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records

**会议**: AAAI 2026  
**arXiv**: [2507.22533](https://arxiv.org/abs/2507.22533)  
**代码**: [有](https://github.com/sakurakawa1/CliCARE)  
**领域**: 医学图像  
**关键词**: LLM临床决策, 电子健康记录, 时序知识图谱, 临床指南对齐, RAG

## 一句话总结

提出 CliCARE 框架，将非结构化的纵向癌症电子病历（EHR）转化为时序知识图谱（TKG），并与临床指南知识图谱对齐融合，为 LLM 提供循证依据的临床决策支持，同时设计了与专家评估高度相关的 LLM-as-a-Judge 评估协议。

## 研究背景与动机

LLM 在临床决策支持中展现出辅助医生、减轻认知负担的潜力，但在肿瘤学等高风险领域的落地面临三大挑战：

**长上下文时序推理能力不足**：癌症 EHR 跨越数年、超过 20,000 tokens，甚至包含多语言条目。现有模型面临"迷失在中间"问题，无法有效处理碎片化的纵向记录

**临床幻觉风险**：标准 RAG 检索的碎片化文本无法捕获患者轨迹中的顺序依赖关系，也无法有效对接过程导向的临床指南（CPG）——事实错误的推荐可能危及患者安全

**评估不可靠**：传统指标（ROUGE、BLEU）专注于词汇重叠，无法衡量临床有效性、事实准确性和安全性。LLM-as-a-Judge 方法又存在位置偏差、冗余偏好等系统性偏差

核心差距：**现有研究将长上下文处理、知识锚定和可靠评估作为独立问题解决，缺乏一个统一框架同时应对这三个挑战。**

## 方法详解

### 整体框架

CliCARE 包含三个核心阶段：

1. **EHR→TKG 转化**：将非结构化病历转为患者中心的时序知识图谱
2. **轨迹-指南对齐**：将真实患者轨迹与规范性临床指南图谱深度对齐
3. **专家验证的 LLM 评估**：确保评估结果与临床专家判断高度相关

### 关键设计

#### 1. EHR-to-TKG 转化

**事件抽取管线** $E_p = f_{pipeline}(D_p)$：

- **历史记录压缩**：将患者完整病历按时间排序，使用 Longformer（预训练于临床文本）对历史记录 $D_p^{hist}$ 进行抽取式摘要，得到既往病史 $S_p^{hist}$
- **最近记录保留**：最近一次临床笔记 $d_{\tau_n}$ 作为现病史，保持完整
- **信息抽取**：使用 BERT 从合并文本中提取关键临床事实（诊断确认、分期更新、治疗方案、生物标志物趋势、影像评估等）

**TKG 实例化** $G_t = (E_t, R_t, T)$：

- 构建通用静态生物医学知识图谱 $G_B$，包含标准化医学概念和关系
- 实体链接函数 $\phi: \mathcal{E}_p \rightarrow \mathcal{E}_B$ 将临床实体映射到标准本体条目
- 每个实体表示为 $e = (e_B, \tau, A)$：标准实体 + 时间戳 + 事件属性
- 采用层级时间粒度：宏观临床事件精确时间戳，事件内部用相对时序关系

#### 2. 轨迹-指南对齐

**知识形式化**：

- 从 TKG 提取患者时序轨迹 $Tr_p = \langle e_1, e_2, \ldots, e_m \rangle$
- 从指南知识图谱 $G_g$ 枚举所有规范治疗路径 $\{Pa_k\}_{k=1}^K$

**相似度匹配**：使用生物医学 BERT 计算语义相似度：

$$\text{Score}(Tr_p, Pa_k) = \sum_{j=1}^l \max_{e_i \in Tr_p} \text{cos\_sim}(f_{BERT}(\text{desc}(s_j)), f_{BERT}(\text{desc}(e_i)))$$

**LLM 重排序**：将 Top-N 候选路径和匹配分数作为上下文，让 LLM 以临床推理者身份进行零样本重排序，弥补纯算法匹配无法捕获的临床逻辑。

**对齐扩展**（Bootstrap 启发）：

- 以 LLM 重排后的最优对齐作为种子集 $A'$
- 对每个未对齐事件，寻找与已有高置信对齐最一致的指南节点：

$$\hat{s} = \arg\max_{s_v \in Pa'_1} \sum_{(e_i, s_j) \in A'} \text{sim}((e_u, s_v), (e_i, s_j))$$

- 迭代扩展对齐集合，利用已建立的强关联推断新关联

#### 3. 专家验证的 LLM-as-a-Judge 评估

**评估维度**（与资深肿瘤科医生共同设计）：

- 事实准确性（Factual Accuracy）
- 完整性与全面性（Completeness & Thoroughness）
- 临床合理性（Clinical Soundness）
- 可操作性与相关性（Actionability & Relevance）

**偏差缓解**：

- 评判集成：GPT-4.1 + Claude 4.0 Sonnet + Gemini 2.5 Pro 三模型取平均
- 随机打乱评估项目顺序消除位置偏差
- 验证结果：LLM 评判与 3 位肿瘤科医生的 Spearman 秩相关系数 $\rho \approx 0.7$

### 损失函数 / 训练策略

CliCARE 的训练阶段主要涉及微调专家模型：

- 2000 样本分为 1800 训练 + 200 测试，10% 训练数据用于验证
- Batch size=1，最大上下文 20,000 tokens，学习率 5e-5 + cosine scheduler
- BF16 混合精度训练，最大输出 4,096 tokens，3 个 epoch
- 硬件：4× NVIDIA A800 GPU

框架同时支持生成式大模型（Gemini 2.5 Pro、GPT-4.1 等）和微调的专家模型（Qwen-3-8B 等），TKG+指南对齐的结构化输入对两者均有显著提升。

## 实验关键数据

### 主实验

在两个大规模临床数据集上对比多种 RAG 基线：

**CancerEHR 数据集（2000 例中国癌症患者，跨越 20+ 年）：**

| 方法 | 临床摘要↑ | 临床建议↑ |
|------|-----------|-----------|
| StandardRAG (Qwen-3-8B) | 1.485 | 1.527 |
| BriefContext (Gemini 2.5 Pro) | 4.527 | 4.468 |
| MedRAG* (Gemini 2.5 Pro) | 4.470 | 4.576 |
| **CliCARE (Qwen-3-8B)** | **3.173** | **3.215** |
| **CliCARE (Gemini 2.5 Pro)** | **4.976** | **4.965** |

**CliCARE + Gemini 2.5 Pro 在 CancerEHR 上接近满分（5 分制评分 4.976），大幅超越所有基线。**

**框架提升幅度（vs StandardRAG）：**

| 模型 | CancerEHR 摘要提升 | MIMIC-Cancer 摘要提升 |
|------|--------------------|-----------------------|
| Qwen-3-8B | +1.688 | +0.100 |
| Deepseek-R1 | +2.279 | +0.393 |
| Gemini 2.5 Pro | +2.241 | +0.835 |

### 消融实验

| 设置 | CancerEHR 摘要 | CancerEHR 建议 |
|------|----------------|----------------|
| CliCARE (Qwen) | 3.173 | 3.215 |
| w/o 扩展 | 3.012 (-) | 3.035 (-) |
| w/o 重排序 | 2.857 (-) | 2.866 (-) |
| w/o TKG 压缩 | 1.485 (-) | 1.527 (-) |
| CliCARE (Gemini) | 4.976 | 4.965 |
| w/o TKG 压缩 | 2.735 (-) | 2.818 (-) |

TKG 压缩是最关键组件，移除后性能大幅下降（Gemini 从 4.976→2.735）。

### 关键发现

- **结构化知识对复杂 EHR 至关重要**：在长记录的 CancerEHR 上提升最大，证明即使强大模型也需要结构化表示进行有效推理
- **大模型从长记录中获益更多**：Gemini 2.5 Pro 在最长记录段（66-100%）上得分最高，CliCARE 能帮助高级模型利用更丰富的上下文
- **小模型在长记录上退化**：Qwen-3-8B 在最长记录段上性能下降，但仍远优于 StandardRAG
- **简单数据集上 TKG 压缩可能适得其反**：在较简单的 MIMIC-Cancer 上移除 TKG 压缩反而轻微提升（Qwen 从 2.575→2.475），说明压缩过于激进时可能丢失信息

## 亮点与洞察

1. **端到端的知识锚定流水线**：从非结构化 EHR → TKG → 指南对齐 → LLM 生成，形成完整闭环，每一步都有理论支撑
2. **Bootstrap 对齐扩展**：利用高置信种子推断新对齐的思路优雅，类似图上的标签传播
3. **评估方法论贡献**：LLM-as-a-Judge 与专家评估的 $\rho \approx 0.7$ 相关性验证，为临床 NLP 评估提供了可信的度量方案
4. **模型无关性**：框架同时提升了 7B-级和顶级商业模型的性能，证明结构化知识表示比单纯扩大模型更有效

## 局限与展望

- **领域覆盖有限**：仅在肿瘤学数据上验证，需推广到心血管、神经科等其他临床领域
- **指南图谱构建成本**：需要领域专家参与构建临床指南知识图谱，扩展到新疾病时有人工成本
- **隐私考量**：框架依赖完整 EHR 数据处理，部署时需严格遵守数据隐私法规
- **评估仍有局限**：$\rho \approx 0.7$ 虽然较高但非完美，某些边界案例的评判可能与专家不一致
- **纪念碑对齐固定阈值**：BERT 语义相似度阈值固定为 0.7，可能需要针对不同指南自适应调整

## 相关工作与启发

- **ColaCare**：多智能体 EHR 建模架构，代表了 EHR 领域的最新趋势
- **MedRAG / KG2RAG / GNN-RAG**：图增强 RAG 方法，CliCARE 通过 TKG 对齐超越了这些方法
- **Longformer**：用于处理长文档的高效 Transformer，CliCARE 用其压缩历史病历
- 知识图谱 + LLM 的融合范式在其他高风险决策领域（法律、金融）同样有潜力

## 评分

- **创新性**: ★★★★★ — TKG 构建 + 指南对齐 + 评估协议三重创新
- **实验充分度**: ★★★★★ — 双数据集（中英文）、多基线、多模型、详细消融和长度分析
- **写作质量**: ★★★★★ — 问题定义清晰，框架设计逻辑严密
- **实用性**: ★★★★☆ — 有代码开源，但指南图谱构建和大规模 EHR 处理仍有工程挑战

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Foundation Models for Clinical Records at Health System Scale](../../ICML2025/medical_imaging/foundation_models_for_clinical_records_at_health_system_scale.md)
- [\[AAAI 2026\] Personalization of Large Foundation Models for Health Interventions](personalization_of_large_foundation_models_for_health_interventions.md)
- [\[AAAI 2026\] G2L: From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Efficient Fine-Tuning](g2lfrom_giga-scale_to_cancer-specific_large-scale_pathology_foundation_models_vi.md)
- [\[AAAI 2026\] Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment](unleashing_the_potential_of_large_language_models_for_text-to-image_generation_t.md)
- [\[AAAI 2026\] Do Large Language Models Think Like the Brain? Sentence-Level Evidences from Layer-Wise Embeddings and fMRI](do_large_language_models_think_like_the_brain_sentence-level_evidences_from_laye.md)

</div>

<!-- RELATED:END -->
