---
title: >-
  [论文解读] Losses that Cook: Topological Optimal Transport for Structured Recipe Generation
description: >-
  [ACL 2026][文本生成][食谱生成] 提出一种基于 Sinkhorn 散度的拓扑损失函数，将食材列表表示为嵌入空间中的点云，最小化预测与真实食材之间的几何差异，显著提升结构化食谱生成中食材召回率和数量精度，在人类评估中 62% 的情况被偏好。
tags:
  - "ACL 2026"
  - "文本生成"
  - "食谱生成"
  - "拓扑损失"
  - "最优传输"
  - "结构化文本生成"
  - "复合损失函数"
---

# Losses that Cook: Topological Optimal Transport for Structured Recipe Generation

**会议**: ACL 2026  
**arXiv**: [2601.02531](https://arxiv.org/abs/2601.02531)  
**代码**: [GitHub](https://github.com/DarthReca/losses-cook)  
**领域**: 文本生成  
**关键词**: 食谱生成, 拓扑损失, 最优传输, 结构化文本生成, 复合损失函数

## 一句话总结
提出一种基于 Sinkhorn 散度的拓扑损失函数，将食材列表表示为嵌入空间中的点云，最小化预测与真实食材之间的几何差异，显著提升结构化食谱生成中食材召回率和数量精度，在人类评估中 62% 的情况被偏好。

## 研究背景与动机

**领域现状**：食谱生成不仅需要流畅的文本，还需要精确的食材、用量、时间、温度以及步骤的程序一致性。目前主流方法基于交叉熵（CE）损失对语言模型进行微调。

**现有痛点**：CE 将所有 token 视为同等重要，但食谱中存在强烈的不对称性——高影响 token（食材、用量、时间、温度、核心动作）和低影响 token（连接词）之间差异巨大。这导致常见的失败模式：食材召回率低、用量不准确、步骤虽语法正确但程序上不可执行。

**核心矛盾**：token 级别的训练目标无法捕捉食材集合的整体结构性质——漏掉一种关键食材（如意大利面里的鸡蛋）或温度翻倍，即使文本流畅，整个食谱也无法使用。

**本文目标**：设计能直接优化食材集合完整性和数值准确性的损失函数，同时保持文本流畅度。

**切入角度**：从最优传输理论出发，将食材列表视为嵌入空间中的点云，利用几何距离来衡量预测食材与真实食材的匹配程度。

**核心 idea**：用 Sinkhorn 散度最小化预测和参考食材点云之间的传输距离，将食材级别的结构约束显式编码到训练损失中。

## 方法详解

### 整体框架
输入为自然语言提示（如"生成一份意大利面 Carbonara 的食谱"），输出为结构化 JSON，包含食材列表和步骤指令列表。基于 Qwen3-4B 模型，使用 LoRA 进行微调，核心创新在于设计复合损失函数替代单一 CE。

### 关键设计

**1. 拓扑损失（Topological Loss）：把食材集合的几何结构显式塞进训练目标**

CE 对所有 token 替换一视同仁，把"salt"错成"pepper"和错成"egg"罚得一样重，可前者在嵌入空间里其实近得多。拓扑损失的做法是把整张食材列表看成嵌入空间里的一团点云：对预测序列中食材部分的 token，先用 softmax 把 logits 转成概率分布，再算加权嵌入 $emb_{soft} = P \cdot E$（$E$ 为词嵌入矩阵）构建预测点云；对真实序列则直接做嵌入查找构建参考点云。两团点云的几何不相似度用 Sinkhorn 散度度量，$\mathcal{L}_{Topo} = \mathcal{S}_\epsilon(PC_{pred}, PC_{target})$。因为惩罚跟随嵌入距离走，模型被引导去对齐"语义邻近"的食材集合，而不是逐 token 死磕——漏一种关键食材或换成八竿子打不着的食材，几何距离会立刻变大。

**2. Dice 损失：从集合重叠的角度盯住关键 token 的覆盖率**

CE 是 token 级目标，对"这一份食谱里该出现的 token 集合到底覆盖了多少"并不敏感。Dice 损失改用可微分的 Dice 系数衡量预测 token 集合与参考 token 集合的重叠程度，直接鼓励模型把该有的关键 token 都生成出来。相比 CE 和 Focal Loss，它在关键 token 的覆盖上更有针对性，尤其在时间和温度这类数值精度指标上表现突出。

**3. 复合损失策略（Mixed Loss）：让两个自定义损失各补各的短板**

实验里 Topo 擅长食材召回、Dice 擅长时间温度精度，各有所长却都不全面，于是作者把它们和 CE 加权混合：$L = 0.6 L_{CE} + 0.2 L_{Dice} + 0.2 L_{Topo}$。CE 保住语言流畅性这条底线，Dice 提升数值精度，拓扑损失加强食材结构一致性，三者叠加拿到互补增益——混合后的 QP 和 TiP 都超过任一单独损失。

### 损失函数 / 训练策略
所有复合损失均以 $L = 0.6 L_{CE} + 0.4 L_{custom}$ 的形式与 CE 结合。训练基于 RECIPE-NLG 数据集的 5000 条子集（意面、米饭、三明治），并用 235 条人工策划的烹饪问题进行数据增强，覆盖食材识别、替代、缩放、数量推理等方面。

## 实验关键数据

### 主实验

| 模型 | R1↑ | BS↑ | AP↑ | QP↑ | IR↑ | TeP↑ | TiP↑ | AD↓ | SD↓ |
|------|-----|-----|-----|-----|-----|------|------|-----|-----|
| Gemini 2.0 (No-FT) | 15.08 | 88.50 | 43.80 | 44.51 | 37.47 | 76.88 | 36.92 | 36.21 | 48.60 |
| Qwen3-4B (CE) | 27.30 | 88.78 | 45.09 | 50.94 | 35.98 | 61.93 | 52.09 | 37.83 | 39.48 |
| Qwen3-4B (Topo) | 30.40 | 90.97 | 59.68 | 63.93 | 48.59 | 65.59 | 55.55 | 30.49 | 34.09 |
| Qwen3-4B (Topo+Dice) | **31.90** | **90.99** | 57.59 | **65.09** | 47.09 | 67.89 | **61.95** | **30.49** | **34.09** |

### 消融实验

| 配置 | IR↑ | QP↑ | 说明 |
|------|-----|-----|------|
| CE only | 35.98 | 50.94 | 基线 |
| CE + Focal | 43.09 | 54.94 | 小幅提升，但不如其他损失 |
| CE + Dice | 44.90 | 57.44 | 数值精度较好 |
| CE + Topo | 48.59 | 63.93 | 食材召回最佳 |
| CE + Topo + Dice | 47.09 | 65.09 | 综合最优 |

### 关键发现
- 拓扑损失在食材召回率上提升最大（+12.6% vs CE），证明嵌入空间点云对齐有效
- Dice 损失在温度精度上最强（74.58% vs CE 的 61.93%），擅长数值约束
- 混合 Topo+Dice 在 QP 和 TiP 上产生协同增益，超过单独使用任一损失
- 人类评估中 Topo+Dice 在总体质量上以 62% vs 11% 大幅胜出 CE，生成错误减少 67.5%

## 亮点与洞察
- 将食材列表建模为点云并用最优传输对齐是非常巧妙的思路，将集合匹配问题转化为几何问题，天然支持部分匹配和语义邻近性。这种思路可以迁移到任何需要集合级别匹配的生成任务（如实体列表生成、关键词提取等）
- Soft embedding 的设计使得拓扑损失可微分，可以端到端训练，无需额外的解码步骤
- 复合损失中不同组件的互补性（结构 vs 数值）提供了一个很好的损失函数设计范式

## 局限与展望
- 训练数据仅覆盖意面、米饭、三明治三类，泛化到其他菜系未经验证
- 数据增强仅 235 条人工问题，规模较小
- 评估指标依赖自动化提取流程，对非标准格式或罕见烹饪术语可能有噪声
- 拓扑损失依赖嵌入空间的几何性质，增加了计算开销
- 未来可扩展到更广泛的菜系、考虑过敏原和营养约束

## 相关工作与启发
- **vs CE-only 微调**: CE 将所有 token 等权，本文证明了针对关键 token 的损失设计能显著提升结构化输出质量
- **vs Focal Loss**: Focal 重新加权困难样本但无法捕获集合级别结构，在食谱特定指标上不如 Dice 和 Topo
- **vs 约束解码方法**: 本文从训练端解决问题，不增加推理复杂度

## 评分
- 新颖性: ⭐⭐⭐⭐ 将最优传输引入食谱生成的损失函数设计，思路新颖但应用场景较窄
- 实验充分度: ⭐⭐⭐⭐ 包含多模型对比、消融、人类评估，但数据规模和领域覆盖有限
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，方法阐述到位
- 价值: ⭐⭐⭐ 技术思路有启发性，但应用场景偏窄，需验证更广泛的可迁移性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport](../../ACL2025/nlp_generation/doc_level_mbr_optimal_transport.md)
- [\[AAAI 2026\] Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text](../../AAAI2026/nlp_generation/structured_language_generation_model_loss_calibration_and_formatted_decoding_for.md)
- [\[ACL 2026\] FACTS: Table Summarization via Offline Template Generation with Agentic Workflows](facts_table_summarization_via_offline_template_generation_with_agentic_workflows.md)
- [\[ACL 2026\] Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation](planning_beyond_text_graph-based_reasoning_for_complex_narrative_generation.md)
- [\[ACL 2026\] Children's English Reading Story Generation via Supervised Fine-Tuning of Compact LLMs with Controllable Difficulty and Safety](childrens_english_reading_story_generation_via_supervised_fine-tuning_of_compact.md)

</div>

<!-- RELATED:END -->
