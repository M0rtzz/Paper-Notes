---
title: >-
  [论文解读] MyVLM: Personalizing VLMs for User-Specific Queries
description: >-
  [ECCV 2024][多模态VLM][VLM个性化] MyVLM首次探索VLM个性化问题，通过外挂概念识别头检测用户特定概念（如"你的狗"），并在VLM中间特征空间学习概念嵌入引导语言模型在回答中自然融入该概念，仅需3-5张图像即可实现个性化caption和VQA。 1. 领域现状：VLM（如BLIP-2、LLaVA）具备…
tags:
  - "ECCV 2024"
  - "多模态VLM"
  - "VLM个性化"
  - "概念嵌入"
  - "BLIP-2"
  - "LLaVA"
  - "少样本学习"
---

# MyVLM: Personalizing VLMs for User-Specific Queries

**会议**: ECCV 2024  
**arXiv**: [2403.14599](https://arxiv.org/abs/2403.14599)  
**代码**: [https://snap-research.github.io/MyVLM/](https://snap-research.github.io/MyVLM/)  
**领域**: 多模态VLM  
**关键词**: VLM个性化, 概念嵌入, BLIP-2, LLaVA, 少样本学习

## 一句话总结

MyVLM首次探索VLM个性化问题，通过外挂概念识别头检测用户特定概念（如"你的狗"），并在VLM中间特征空间学习概念嵌入引导语言模型在回答中自然融入该概念，仅需3-5张图像即可实现个性化caption和VQA。

## 研究背景与动机

1. **领域现状**：VLM（如BLIP-2、LLaVA）具备强大的视觉理解和文本生成能力，能描述图像内容。但它们只有通用知识，不认识特定用户的宠物、朋友或个人物品。

2. **现有痛点**：(1) 对每个用户微调VLM计算开销巨大且可能导致灾难性遗忘；(2) 模型编辑技术只能修改特定query的回答，不能泛化到新图像；(3) VLM的视觉编码器特征空间不够表达性，无法有效区分语义相似的概念（如区分你的狗和别人的狗）。

3. **核心矛盾**：个性化要求模型认识新概念并在不同场景中泛化，但又不能修改VLM原始权重（否则破坏通用能力）。同时概念识别和概念通信是两个不同的挑战。

4. **本文要解决什么？** (1) 如何在不改变VLM权重的前提下教它认识用户特定概念？(2) 如何让语言模型在生成回答时自然地将概念标识符融入上下文？

5. **切入角度**：将问题分解为"识别"和"通信"两步——先用外挂头判断概念是否出现在图像中，再用可学习嵌入引导LLM在输出中提及该概念。

6. **核心idea一句话**：外挂概念识别头+可学习概念嵌入+注意力正则化，在不修改VLM权重的情况下实现用户特定概念的识别和融入。

## 方法详解

### 整体框架

MyVLM分两阶段：(1) 识别——外挂概念识别头（CLIP分类器或人脸识别网络）检测目标概念是否在图像中；(2) 通信——将学习到的概念嵌入向量追加到视觉编码器输出后，经Q-Former/线性投影层传入LLM，引导其在回答中使用概念标识符。全程冻结VLM所有原始权重。

### 关键设计

1. **外挂概念识别头**:
    - 做什么：判断用户特定概念是否出现在当前图像中
    - 核心思路：物体识别用CLIP嵌入空间上的线性分类器；人物识别用预训练人脸识别网络。每个概念一个独立头
    - 设计动机：VLM冻结的视觉编码器特征不够区分语义相似物体（如不同的狗），外挂专用头避免修改视觉编码器且可灵活扩展到新概念

2. **概念嵌入学习**:
    - 做什么：学习一个嵌入向量使LLM在生成时融入概念标识符
    - 核心思路：概念嵌入 $e_*$ 追加到视觉特征后输入Q-Former。用3-5张包含概念的图像+对应caption（含概念标识符S*），通过标准交叉熵损失优化：$e_* = \arg\min_e \sum_{i=1}^{N} \mathcal{L}_{CE}(t_i, o(I_i, e))$
    - 设计动机：在VLM中间特征空间学习嵌入，利用已有的视觉-文本桥梁（Q-Former/线性层）将概念信息传达到LLM，无需修改任何原始参数

3. **泛化提升机制**:
    - 做什么：防止概念嵌入破坏VLM的原始行为
    - 核心思路：(1) Key/Value归一化——概念嵌入的K/V投影常显著大于原始图像特征，需要归一化到平均范数：$\hat{k}_* = \frac{k_*}{\|k_*\|} \cdot n_k$；(2) 注意力正则化——防止Q-Former query token过度关注概念嵌入而忽视图像特征：$\mathcal{L}_{reg} = \|softmax(Q \cdot \hat{k}_*)\|_2^2$
    - 设计动机：直接追加嵌入会导致注意力被概念token主导，query不再关注原始图像信息，生成不自然的caption

### 损失函数 / 训练策略

总损失 = 交叉熵损失 + 注意力正则化。只优化概念嵌入向量（一个d维向量），3-5张图像即可收敛。概念标识符用DreamBooth策略——物体用罕见词，人物用简短名字。

## 实验关键数据

### 主实验

| 方法 | Recall↑ | CLIPScore↑ | Sentence Sim↑ |
|------|---------|------------|---------------|
| PALAVRA (baseline) | 68.2 | 25.1 | 0.42 |
| Textual Inv. (baseline) | 72.5 | 25.8 | 0.44 |
| MyVLM (BLIP-2) | **89.3** | **27.2** | **0.51** |
| MyVLM (LLaVA) | **91.7** | **27.6** | **0.53** |

### 消融实验

| 配置 | Recall | CLIPScore | 说明 |
|------|--------|-----------|------|
| w/o K/V归一化 | 82.1 | 25.9 | caption不自然 |
| w/o 注意力正则化 | 85.4 | 26.3 | 概念token主导注意力 |
| w/o 概念识别头 | 71.8 | 27.0 | 无概念图像也输出概念 |
| Full MyVLM | **91.7** | **27.6** | 完整模型 |

### 关键发现

- **概念识别头不可或缺**：没有识别头时模型在不含目标概念的图像中也会错误输出概念标识符
- 注意力可视化表明概念嵌入确实"注意到"了概念所在区域，学到了有意义的空间关联
- **MyVLM可零样本迁移到VQA和REC任务**：用caption训练的嵌入可直接用于个性化视觉问答和指代表达理解，验证了嵌入确实捕获了概念语义

## 亮点与洞察

- **问题定义的开创性**：首次正式定义VLM个性化任务，区分了概念识别和概念通信两个子问题，为后续研究建立了框架
- **K/V归一化+注意力正则化**：解决了类似文本反转中嵌入范数过大导致注意力失衡的通用问题，可迁移到其他需要在attention空间插入新token的场景
- **跨任务迁移能力**：caption训练的嵌入可零样本用于VQA和REC，证明嵌入捕获的是通用概念表示而非特定任务信号

## 局限性 / 可改进方向

- 每个概念需要独立的概念头和嵌入向量，不能高效处理大量概念
- 人物识别依赖人脸识别网络，在侧脸、遮挡等情况下可能失效
- 当前只支持单概念查询，多个概念同时出现时的交互关系未探索
- 只在BLIP-2和LLaVA上验证，需扩展到更多VLM架构

## 相关工作与启发

- **vs 文本反转/DreamBooth**: 这些方法在图像生成任务中学习概念，MyVLM将个性化思路迁移到VLM理解任务
- **vs PALAVRA**: PALAVRA在CLIP文本空间优化token embedding用于检索，MyVLM在VLM中间特征空间操作，能生成上下文化的描述
- **vs 模型编辑**: 模型编辑修改特定query的回答，MyVLM能泛化到新图像和新问题
- VLM个性化可以与RAG等方式结合，支持更大规模的用户知识库

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次定义VLM个性化任务，方法设计优雅
- 实验充分度: ⭐⭐⭐⭐ 自建数据集+多结构+多任务验证
- 写作质量: ⭐⭐⭐⭐⭐ 动机推导清晰，方法解释易懂
- 价值: ⭐⭐⭐⭐⭐ 开辟VLM个性化新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Chimera: Improving Generalist Model with Domain-Specific Experts](../../ICCV2025/multimodal_vlm/chimera_improving_generalist_model_with_domain-specific_experts.md)
- [\[ICML 2026\] SLQ: Bridging Modalities via Shared Latent Queries for Retrieval with Frozen MLLMs](../../ICML2026/multimodal_vlm/slq_bridging_modalities_via_shared_latent_queries_for_retrieval_with_frozen_mllm.md)
- [\[NeurIPS 2025\] To See or To Read: User Behavior Reasoning in Multimodal LLMs](../../NeurIPS2025/multimodal_vlm/to_see_or_to_read_user_behavior_reasoning_in_multimodal_llms.md)
- [\[ACL 2025\] MIRe: Enhancing Multimodal Queries Representation via Fusion-Free Modality Interaction](../../ACL2025/multimodal_vlm/mire_enhancing_multimodal_queries_representation_via_fusion-free_modality_intera.md)
- [\[ICML 2025\] LADA: Scalable Label-Specific CLIP Adapter for Continual Learning](../../ICML2025/multimodal_vlm/lada_scalable_label-specific_clip_adapter_for_continual_learning.md)

</div>

<!-- RELATED:END -->
