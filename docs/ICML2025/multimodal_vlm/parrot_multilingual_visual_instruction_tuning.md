---
title: >-
  [论文解读] Parrot: Multilingual Visual Instruction Tuning
description: >-
  [ICML2025][多模态VLM][多语言] 提出 Parrot，通过文本引导的跨注意力机制和 MoE 模块将英语偏置的视觉特征转换为语言特定表示，以极少量多语言数据（每种语言约 10K 样本）显著提升 MLLM 的多语言能力。
tags:
  - "ICML2025"
  - "多模态VLM"
  - "多语言"
  - "多模态大模型"
  - "Mixture-of-Experts"
  - "视觉指令微调"
  - "语言对齐"
---

# Parrot: Multilingual Visual Instruction Tuning

**会议**: ICML2025  
**arXiv**: [2406.02539](https://arxiv.org/abs/2406.02539)  
**代码**: [AIDC-AI/Parrot](https://github.com/AIDC-AI/Parrot)  
**领域**: 多模态VLM  
**关键词**: 多语言, 多模态大模型, Mixture-of-Experts, 视觉指令微调, 语言对齐

## 一句话总结

提出 Parrot，通过文本引导的跨注意力机制和 MoE 模块将英语偏置的视觉特征转换为语言特定表示，以极少量多语言数据（每种语言约 10K 样本）显著提升 MLLM 的多语言能力。

## 研究背景与动机

当前多模态大模型（MLLM）的多模态对齐训练数据以英语为绝对主导，导致训练后模型丧失非英语语言的处理能力，作者将此现象称为 **多语言侵蚀（multilingual erosion）**。例如 LLaVA 在收到中文输入时仍倾向于用英语回复。

作者通过实验发现该问题的根源在于 **视觉 token 与非英语文本 token 之间的对齐失败**：

- 使用 OpenAI-CLIP 的模型在中文场景下表现混乱，而使用 Chinese-CLIP 的模型能够正确理解和生成中文
- t-SNE 可视化证实 Chinese-CLIP 的视觉特征与中文 prompt 在高维空间中更为接近

核心问题：**如何在非英语多模态数据极度匮乏的条件下，将英语偏置的视觉特征转换为语言特定的嵌入？**

## 方法详解

### 整体架构

Parrot 在标准 LLaVA 架构（Vision Encoder → Projector → LLM）的基础上，在 Projector 之后插入一个轻量级的 **多语言 MoE 模块**，由文本引导驱动视觉 token 的语言级对齐。

### 跨模态交叉注意力

首先，利用视觉特征的 [CLS] token $\mathbf{H}_v^{\text{cls}}$ 与文本嵌入 $\mathbf{H}_t$ 计算交叉注意力，融合视觉和文本信息：

$$\mathbf{H}_v' = \text{Softmax}\left(\frac{\mathbf{H}_v^{\text{cls}} \mathbf{H}_t^T}{\sqrt{C}}\right) \mathbf{H}_t$$

其中 $\mathbf{Q} = \mathbf{H}_v^{\text{cls}}$，$\mathbf{K} = \mathbf{V} = \mathbf{H}_t$。该步骤让视觉特征根据输入文本的语言信息动态调整。

### MoE 路由与语言专家

融合后的特征 $\mathbf{H}_v'$ 送入 MoE 路由器（线性层 + Softmax），生成专家激活概率：

$$\mathcal{P} = \text{Softmax}(\text{Linear}(\mathbf{H}_v'))$$

每个专家是一个双层 MLP（含 SiLU 激活），负责将英语偏置的视觉嵌入转换为特定语言的表示。专家数量设为 6，对应 6 种目标语言。加权聚合所有激活专家的输出：

$$\text{MoE}(\mathbf{H}_v) = \sum_{i=1}^{k} \mathcal{P}[i] \cdot \mathcal{E}(\mathbf{H}_v)_i$$

### MoE 重加权

为稳定训练、减少视觉语义信息的方差，最终的视觉嵌入采用残差连接：

$$\mathbf{G}_v = \mathbf{H}_v + \alpha \cdot \text{MoE}(\mathbf{H}_v)$$

$\alpha$ 为权衡参数，确保模型在多语言增强的同时不损失原始视觉语义。

### 两阶段训练

| 阶段 | 冻结 | 训练 | 数据 | 说明 |
|------|------|------|------|------|
| Stage 1: 模态对齐 | Vision Encoder + LLM | Projector | 英语图文对 | 不经过 MoE，纯对齐 |
| Stage 2: 多语言指令微调 | Vision Encoder | Projector + MoE + LLM | 英语 + 多语言数据 | MoE 随机初始化，文本引导优化 |

多语言数据获取：从 ShareGPT4V 数据集中随机抽取不重复的子集，用 GPT-4 翻译 + 人工校准，每种语言获得约 10K 图文对。

## 实验关键数据

### MMMB 基准（新提出，6 语言 × 15 类别 × 12K 题目）

| 模型 | LLM | en | zh | pt | ar | tr | ru |
|------|-----|----|----|----|----|----|----|
| LLaVA-1.5 | Vicuna-7B | 67.1 | 58.8 | 59.8 | 43.5 | 46.4 | 59.1 |
| LLaVA-NeXT | LLaMA3-8B | 70.9 | 64.3 | 63.2 | 48.3 | 48.0 | 66.4 |
| Qwen2-VL | Qwen2-7B | 80.5 | 80.2 | 78.1 | 74.1 | 71.7 | 79.3 |
| LLaVA-OneVision | Qwen2-7B | 79.0 | 78.2 | 75.9 | 73.4 | 67.8 | 76.4 |
| **Parrot** | **Qwen2-7B** | **80.1** | **80.0** | **79.6** | **76.6** | **75.0** | **79.9** |

- Parrot-Qwen2-7B 在 MMMB 的 pt / ar / tr 三种语言上取得 SOTA，en / zh 紧随其后
- 在 MMBench 多语言版本上同样在 4 种语言上 SOTA

### 多模态通用任务

Parrot 在 MME、MMStar、ScienceQA、RealWorldQA、SEED-Bench 等通用多模态基准上也保持了竞争力，说明多语言增强未损害模型的整体多模态能力。

### 关键消融

1. **多语言数据的作用**：加入多语言数据后所有语言性能提升，但单纯加数据对 LLaVA 提升有限，证明性能增益主要来自 Parrot 架构设计
2. **MoE 模块的作用**：去掉 MoE 后性能显著下降，验证语言级专家路由的有效性
3. **翻译 baseline 对比**：使用 Google Translation API 的"翻译→处理→回译"方案出现跷跷板效应（中文提升但俄语、葡语下降），说明简单翻译无法替代语言级对齐
4. **Scaling law**：扩大多语言数据量（至与中文数据量 70K 持平）后，pt +3.0、ar +5.2，模型规模扩大同样有效

### 训练效率

- 16×A100 GPU 上仅需 **21 小时** 完成全部训练
- 使用的多语言数据不到其他多语言 MLLM 的 1%

## 亮点与洞察

1. **多语言侵蚀现象的清晰诊断**：通过 OpenAI-CLIP vs Chinese-CLIP 对比实验和 t-SNE 可视化，精准定位了问题根源在于视觉 token 的语言偏置
2. **数据效率极高**：每种语言仅用 ~10K 样本即可实现显著的多语言提升，适合低资源场景
3. **模块化设计**：MoE 模块可即插即用，不改变主干架构，易于迁移到其他 MLLM
4. **新 benchmark MMMB**：6 语言 × 15 类别 × 12K 题目，采用 Yes/No 循环验证策略减少随机猜测影响，设计严谨

## 局限与展望

1. **语言覆盖有限**：仅涵盖 6 种语言，缺乏日语、韩语、印地语等重要语种的验证
2. **专家数量与语言数硬绑定**：6 个专家对应 6 种语言，扩展到更多语言时的 MoE 规模增长问题未讨论
3. **视觉编码器固定为 CLIP ViT-L/14**：未探索更强的视觉编码器（如 SigLIP、InternViT）对方法的影响
4. **仅验证了 7B 量级模型**：更大规模 LLM（如 70B）下的效果未知
5. **MMMB 基准质量依赖翻译**：虽有人工校准流程，但翻译质量在低资源语言上仍可能有偏差
6. **MoE 路由的可解释性不足**：虽然可视化了专家分布，但未深入分析不同语言激活模式的差异原因

## 相关工作与启发

- **LLaVA 系列**：Parrot 直接构建在 LLaVA 架构上，是其多语言扩展
- **MoE 用于多语言**：借鉴了 NLP 中 MoE 处理多语言任务的思路，但首次用于视觉-语言对齐层面
- **启发**：该方法可推广到其他需要跨域对齐的场景（如跨模态、跨风格），本质上是用已有的文本信号引导特征空间的重新分布

## 评分

- 新颖性: ⭐⭐⭐⭐ — 多语言侵蚀的诊断和文本引导 MoE 对齐的设计都有新意
- 实验充分度: ⭐⭐⭐⭐ — 多语言和通用基准双覆盖，消融全面，但语言种类和模型规模覆盖有限
- 写作质量: ⭐⭐⭐⭐ — 动机清晰、分析深入、图表说服力强
- 价值: ⭐⭐⭐⭐ — 实用性强，低数据高效率路线对工业落地有价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](../../NeurIPS2025/multimodal_vlm/visual_instruction_bottleneck_tuning.md)
- [\[NeurIPS 2025\] Learning to Instruct for Visual Instruction Tuning](../../NeurIPS2025/multimodal_vlm/learning_to_instruct_for_visual_instruction_tuning.md)
- [\[ICCV 2025\] SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](../../ICCV2025/multimodal_vlm/smolora_exploring_and_defying_dual_catastrophic_forgetting_in_continual_visual_i.md)
- [\[ICML 2025\] Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning](dynamic_mixture_of_curriculum_lora_experts_for_continual_multimodal_instruction_.md)
- [\[ACL 2025\] MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale](../../ACL2025/multimodal_vlm/mammoth_vl_multimodal_reasoning.md)

</div>

<!-- RELATED:END -->
