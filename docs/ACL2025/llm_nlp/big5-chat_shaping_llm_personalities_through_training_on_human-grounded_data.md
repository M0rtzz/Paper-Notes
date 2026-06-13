---
title: >-
  [论文解读] Big5-Chat: Shaping LLM Personalities Through Training on Human-Grounded Data
description: >-
  [ACL 2025][LLM 其他][人格特质] 提出了 Big5-Chat 数据集（10万条对话），通过 SFT 和 DPO 训练方法将真实人类大五人格特质嵌入 LLM，效果显著优于基于提示的方法，且发现高尽责性/宜人性、低外向性/神经质的人格配置能提升模型推理能力。 将逼真的人格特质嵌入 LLM 对提升对话代理、教育工具…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "人格特质"
  - "大五人格"
  - "SFT"
  - "DPO"
  - "心理语言学"
---

# Big5-Chat: Shaping LLM Personalities Through Training on Human-Grounded Data

**会议**: ACL 2025  
**arXiv**: [2410.16491](https://arxiv.org/abs/2410.16491)  
**代码**: [github](https://github.com/wenkai-li/Big5-Chat.git)  
**领域**: LLM/NLP  
**关键词**: 人格特质, 大五人格, SFT, DPO, 心理语言学

## 一句话总结

提出了 Big5-Chat 数据集（10万条对话），通过 SFT 和 DPO 训练方法将真实人类大五人格特质嵌入 LLM，效果显著优于基于提示的方法，且发现高尽责性/宜人性、低外向性/神经质的人格配置能提升模型推理能力。

## 研究背景与动机

将逼真的人格特质嵌入 LLM 对提升对话代理、教育工具和心理健康平台的真实性至关重要。然而，现有方法主要依赖**提示词**来诱导人格（如"你是派对的灵魂"），存在三个核心问题：

**缺乏心理语言学深度**：提示诱导的人格仅反映表面特征，无法捕捉人类在语言中表达人格的微妙方式

**评估效度问题**：用心理问卷描述来诱导人格，再用同一套问卷来评估，存在严重的循环论证

**推理模式偏差**：基于角色的人格提示会过度约束 LLM 的行为，导致推理能力退化

根本原因在于缺乏大规模、基于真实人类行为的人格标注数据集，限制了训练方法的探索。

## 方法详解

### 整体框架

PsychSteer 方法包含两个阶段：(1) 训练专家生成器模型，(2) 利用 DExperts 框架生成 Big5-Chat 数据集，然后进行 SFT/DPO 对齐。

### 关键设计

1. **DExperts 框架**：在解码时通过专家生成器引导基础模型输出。在每个时间步 t，将基础模型的 logits 与专家生成器的 logits 加权组合：$z_t^{combined} = z_t^{base} + \gamma z_t^{expert}$，其中 $\gamma$ 控制人格特质的影响强度。

2. **专家生成器模型**：基于 PsychGenerator 数据集（84.6万条 Facebook 帖子，标注了大五人格分数），对 LLaMA-3-8B-Instruct 进行微调，训练 5 个独立的专家生成器，分别对应开放性、尽责性、外向性、宜人性和神经质。将浮点数标签转换为"高/低"二值标签。

3. **Big5-Chat 数据集构建**：从 SODA 社交对话数据集随机采样 1 万个场景，利用 PsychSteer 框架为每个场景生成配对对话（高/低人格水平），共产生 10 万条单轮对话（每个特质 2 万条，高低各半）。

4. **SFT 与 DPO 训练**：使用 LoRA 进行高效微调。DPO 训练时，负样本来自同一特质但相反水平的响应（如目标为高开放性，正样本为高开放性回复，负样本为低开放性回复）。

### 损失函数 / 训练策略

- SFT 使用标准交叉熵损失
- DPO 使用偏好优化损失，利用正负对比样本学习人格偏好
- 使用 LoRA 适配器进行参数高效微调
- 数据集质量验证：训练 RoBERTa-Large 分类器（5个回归头，MSE 损失），在测试集上达到 93.8% 准确率

## 实验关键数据

### 主实验

**人格评估（BFI 测试，LLaMA-3-70B-Instruct）**：

| 方法 | 高特质平均 ↑ | 低特质平均 ↓ | 特点 |
|------|------------|------------|------|
| Direct | 3.8 | 3.8 | 无人格诱导 |
| Prompt-Inst | 5.0 | 1.6 | 指令提示 |
| SFT | 5.0 | 1.2 | 训练方法 |
| DPO | 5.0 | 1.4 | 训练方法 |

**推理评估（LLaMA-3-70B-Instruct，SFT 平均）**：

| 推理领域 | Direct | SFT 平均 | 最佳特质 |
|---------|--------|---------|---------|
| 社会推理 | 46.6 | 50.0 | 高开放性 50.3 |
| 数学推理 | 59.8 | 63.6 | 高宜人性 65.0 |
| 幻觉检测 | 58.6 | 54.4 | 高尽责性 55.6 |
| 常识推理 | 53.7 | 79.4 | 高开放性 79.5 |
| 通用推理 | 54.0 | 53.2 | 高尽责性 53.7 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 数据集质量 - 专家生成器 | 80.4% 准确率 | 分类器评估人格特质准确性 |
| 数据集质量 - GPT-4o-mini 基线 | 59.2% 准确率 | Post-Completion 基线，低21.2% |
| SFT vs DPO（人格诱导） | 无显著差异 | 两者均优于提示方法 |
| Prompt-Demo（示范提示） | 低于 SFT/DPO | 上下文学习不足以深度嵌入人格 |

### 关键发现

1. **训练方法显著优于提示方法**：SFT 和 DPO 在 BFI 和 IPIP-NEO 人格测试中均大幅超越指令提示和示范提示
2. **人格与推理的关联**：高尽责性和高宜人性模型在推理任务中表现最好，低外向性和低神经质也有利于推理，这与心理学中人类大五人格与认知能力的关系一致
3. **SFT 产生更真实的特质内相关性**：训练模型的特质间相关模式更接近人类真实数据分布
4. **常识推理大幅提升**：SFT 在常识推理上从 53.7% 提升至约 79%，提升显著

## 亮点与洞察

- **人格-认知桥梁**：首次系统证明 LLM 中通过训练嵌入人格特质后，推理能力的变化模式与心理学研究中人类的表现模式一致，说明训练方法确实捕捉到了深层心理语言学特征
- **数据集构建方法创新**：巧妙地将领域特定的人格标注数据（Facebook 帖子）与通用社交场景（SODA）通过 DExperts 框架结合，解决了人格数据稀缺问题
- **双重评估框架**：不仅评估人格诱导效果（BFI、IPIP-NEO），还评估对推理能力的影响，提供了更全面的视角

## 局限与展望

1. **单一语言**：仅在英文上实验，人格在不同文化/语言中的表达可能不同
2. **二值化标签**：将人格分数简化为高/低两类，丧失了连续人格谱系的细粒度
3. **基础模型限制**：仅在 LLaMA-3 系列上验证，未在其他架构（如 Mistral、Qwen）上测试
4. **对话深度有限**：Big5-Chat 为单轮对话，多轮对话中人格一致性的表现未知
5. **推理能力变化的因果机制不清**：虽然发现了人格与推理的关联，但未深入解释为什么特定人格配置能提升推理

## 相关工作与启发

- **PsychGenerator (Vu et al., 2024)**：84.6万 Facebook 帖子的人格标注数据集，本文的核心数据来源
- **SODA (Kim et al., 2023)**：GPT-3.5 生成的多样化社交对话数据集
- **DExperts (Liu et al., 2021)**：解码时引导的可控文本生成框架
- 启发：训练方法在嵌入复杂心理特征（如情感、认知风格）方面可能比提示更有前景

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统地用训练方法嵌入人格并评估其对推理的影响，但DExperts框架本身非新
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖两种人格测试、五大推理领域、两种模型规模，消融和分析全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述到位，数据展示丰富
- 价值: ⭐⭐⭐⭐ 对 AI 拟人化、角色扮演和心理学交叉研究有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Understanding Silent Data Corruption in LLM Training](understanding_silent_data_corruption_in_llm_training.md)
- [\[ACL 2025\] Towards Geo-Culturally Grounded LLM Generations](geocultural_grounded_llm.md)
- [\[ACL 2025\] A Survey on Efficient Large Language Model Training: From Data-centric Perspectives](a_survey_on_efficient_large_language.md)
- [\[ACL 2025\] Game Development as Human-LLM Interaction](game_development_as_human-llm_interaction.md)
- [\[ACL 2025\] MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion](mathfusion_instruction_fusion.md)

</div>

<!-- RELATED:END -->
