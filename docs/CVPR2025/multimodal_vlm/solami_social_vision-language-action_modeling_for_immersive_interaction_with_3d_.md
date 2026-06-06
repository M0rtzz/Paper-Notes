---
title: >-
  [论文解读] SOLAMI: Social Vision-Language-Action Modeling for Immersive Interaction with 3D Autonomous Characters
description: >-
  [CVPR 2025][多模态VLM][社交交互] 提出 SOLAMI，首个端到端的社交视觉-语言-动作 (VLA) 建模框架，通过将语音和动作离散化为 token 并基于 decoder-only LLM 统一建模，实现用户与 3D 虚拟角色通过语音和肢体语言的沉浸式实时交互…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "社交交互"
  - "VLA模型"
  - "动作生成"
  - "3D角色"
  - "VR沉浸"
---

# SOLAMI: Social Vision-Language-Action Modeling for Immersive Interaction with 3D Autonomous Characters

**会议**: CVPR 2025  
**arXiv**: [2412.00174](https://arxiv.org/abs/2412.00174)  
**代码**: [项目主页](https://solami-ai.github.io/)  
**领域**: 人体理解/多模态  
**关键词**: 社交交互, VLA模型, 动作生成, 3D角色, VR沉浸

## 一句话总结

提出 SOLAMI，首个端到端的社交视觉-语言-动作 (VLA) 建模框架，通过将语音和动作离散化为 token 并基于 decoder-only LLM 统一建模，实现用户与 3D 虚拟角色通过语音和肢体语言的沉浸式实时交互，同时构建了合成多模态社交交互数据集 SynMSI。

## 研究背景与动机

### 领域现状

**领域现状**：沉浸式角色交互的需求**：心理学研究表明社交交互中沉浸度越高体验越好，但现有角色智能体仅限于文本或语音交互，缺乏 3D 肢体语言。

### 现有痛点

**现有痛点**：模块化方案的延迟问题**：现有方法（如 LLM-Agent 框架）通过文本串联各子模块（动作理解→文本→动作生成），这种方式传递高层信息但丢失细微线索，且多子模块的串联引入严重延迟。

### 核心矛盾

**核心矛盾**：来自机器人学的启示**：LLM-Agent 适合高层规划任务，但对低层操控任务，端到端 VLA 模型表现更优——虚拟角色本质上是虚拟人形机器人，适合 VLA 建模。

### 解决思路

**解决思路**：数据稀缺的挑战**：综合性多模态交互数据（同时包含语音+肢体动作的对话）极其稀缺，采集成本禁止性高。

### 补充说明

**补充说明**：单一运动任务的不足**：已有运动相关的 LLM 工作专注于单一任务（如文本到动作、动作理解），无法根据角色设定生成基于上下文的响应性动作。

## 方法详解

### 整体框架

SOLAMI 是端到端 VLA 模型：用户语音和动作通过 tokenizer 离散化为 token → decoder-only LLM 根据用户输入 token + 角色设定预测角色响应的动作和语音 token → 各自解码器还原为语音和动作。训练分三个阶段：Tokenizer 训练 → 多任务预训练（模态对齐）→ 指令微调（多轮对话）。

### 关键设计

**设计一：多部位分离式动作 Tokenizer**
- **功能**：将 SMPL-X 人体动作离散化为可被 LLM 处理的 token 序列
- **核心思路**：使用三个独立的 VQ-VAE 分别对身体动作 $m^b$、手部动作 $m^h$ 和双人间相对变换 $m_t$ 进行编码：$\hat{m}_t^u = Q^u(m_t^u) = \arg\min_{z_i \in \mathbb{Z}_u} \|m_t^u - z_i\|_2$。身体和手部 VQ-VAE 用 1D 时间卷积生成序列 token，相对变换 VQ-VAE 用 MLP 生成单个 token。语音使用 SpeechTokenizer 分离语义和声学信息，只取语义 token 输入 LLM，解码时用 SoundStorm + 声音克隆。
- **设计动机**：身体、手部和相对变换的运动特性差异大，分开建模可获得更高重建精度；仅用语义 token 减少 LLM 推理成本；支持声音克隆确保角色语音一致性。

**设计二：三阶段渐进训练策略**
- **功能**：从模态对齐到多轮对话，渐进式构建角色行为系统
- **核心思路**：阶段 1 训练 Tokenizer 并冻结；阶段 2 多任务预训练，用 46K 动作-文本对做文本到动作/动作理解，用 410K 语音-文本对做 TTS/ASR，以 4:6 采样比平衡规模差异，实现动作-文本和语音-文本对齐；阶段 3 用 5.7K 多模态会话数据做指令微调，仅监督角色响应部分。
- **设计动机**：直接用多模态交互数据训练效果差（消融实验证明），预训练阶段的模态对齐是必要的；分阶段训练让模型先学会基本的模态映射再学习复杂的社交行为。

**设计三：SynMSI 合成数据管线**
- **功能**：低成本自动构建大规模多模态社交交互数据集
- **核心思路**：(1) 收集 5.3K 角色相关话题 → (2) GPT-4o 根据话题和角色设定生成文本剧本 → (3) 用文本嵌入从 46K 动作数据库中检索最匹配的动作 → (4) 根据检索到的动作修正语音脚本确保协调 → TTS/声音克隆生成角色语音。迭代重复生成多轮对话。最终获得 6.3K 多轮多模态对话项。
- **设计动机**：直接采集多模态交互数据成本极高；通过检索+合成的方式复用已有动作数据集，同时用 LLM 确保对话内容的多样性和角色一致性。

### 损失函数

Tokenizer 训练：$\mathcal{L}_m = \lambda_r \mathcal{L}_r + \lambda_e \mathcal{L}_e + \lambda_c \mathcal{L}_c + \lambda_v \mathcal{L}_v$（重建+嵌入+承诺+速度损失）。指令微调使用 next-token prediction 的交叉熵，仅监督角色响应的动作和语音 token：$\mathcal{L}_{\text{IT}} = -\sum_{r=1}^{R}\sum_{i=1}^{L_M^r} \log p_\Theta(\hat{m}_i^r | \text{context}) - \sum_{r=1}^{R}\sum_{i=1}^{L_S^r} \log p_\Theta(\hat{s}_i^r | \text{context})$。

## 实验关键数据

### 主实验：社交交互质量

| 方法 | Motion FID ↓ | Speech Quality ↑ | Latency (s) ↓ | User Preference ↑ |
|------|------------|-----------------|--------------|-------------------|
| LLM-Agent (模块化) | 较差 | 中等 | 高延迟 | 低 |
| SOLAMI (端到端) | **更优** | **更优** | **更低** | **更高** |

### 消融实验：训练策略

| 配置 | 效果 |
|------|------|
| 无预训练直接微调 | 性能显著下降 |
| 仅动作预训练 | 缺乏语音对齐 |
| 完整三阶段 | 最佳 |
| Full fine-tuning vs LoRA | Full 略优但成本高 |

### 关键发现

1. 端到端 VLA 模型在语音和动作响应的精确性、自然度和延迟方面均优于模块化 LLM-Agent 方案
2. 多任务预训练阶段对最终性能至关重要——跳过预训练直接微调导致性能大幅下降
3. 合成数据管线有效缓解了多模态交互数据稀缺问题
4. 用户研究表明 SOLAMI 的交互体验显著优于基线方法

## 亮点与洞察

- **首个 3D 角色社交 VLA 模型**：将机器人学中 VLA 的思路迁移到虚拟角色交互，开拓了新方向
- **动作即语言**的建模方式：将 SMPL-X 动作和语音统一为 token 序列，让 LLM 成为统一的行为推理引擎
- **数据合成管线**实用价值高：通过检索+LLM 生成低成本构造了虽然合成但结构完整的多模态交互数据
- **VR 界面**的设计使得方法可以被端到端评估，弥合了虚拟角色研究与实际体验之间的鸿沟

## 局限与展望

- SynMSI 数据仍为合成数据，与真实人际交互存在分布差异
- 动作检索的上限受限于现有动作数据库的覆盖度
- 面部表情建模尚未纳入（仅涵盖身体和手部动作）
- 多人同时交互场景尚未支持

## 相关工作与启发

- SOLAMI 的端到端 VLA 思路可扩展到更多虚拟人应用（游戏 NPC、虚拟教师等）
- 动作 tokenizer 的分部位设计可为其他动作生成/理解任务提供参考
- 数据合成管线的"检索+修正"模式可推广到其他缺乏配对数据的跨模态任务

## 评分

⭐⭐⭐⭐ — 开创性地将 VLA 框架引入 3D 虚拟角色社交交互，问题定义清晰、系统设计完整、实验充分。SynMSI 数据合成管线和 VR 评估界面都具有独立的贡献价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging](../../NeurIPS2025/multimodal_vlm/better_tokens_for_better_3d_advancing_vision-language_modeling_in_3d_medical_ima.md)
- [\[CVPR 2025\] LayoutVLM: Differentiable Optimization of 3D Layout via Vision-Language Models](layoutvlm_differentiable_optimization_of_3d_layout_via_vision-language_models.md)
- [\[ICLR 2026\] Procedural Mistake Detection via Action Effect Modeling](../../ICLR2026/multimodal_vlm/procedural_mistake_detection_via_action_effect_modeling.md)
- [\[CVPR 2025\] Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Model](generalized_few-shot_3d_point_cloud_segmentation_with_vision-language_model.md)
- [\[CVPR 2025\] RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics](robospatial_teaching_spatial_understanding_to_2d_and_3d_vision-language_models_f.md)

</div>

<!-- RELATED:END -->
