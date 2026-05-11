---
title: >-
  [论文解读] DreamShot: Personalized Storyboard Synthesis with Video Diffusion Prior
description: >-
  [CVPR 2026][故事板生成] 提出 DreamShot，利用视频扩散模型的时空先验来生成人物一致、场景连贯的多镜头故事板，通过 Role-Attention Consistency Loss 解决多角色混淆问题，统一支持文本到镜头、参考到镜头和镜头到镜头三种模式。
tags:
  - CVPR 2026
  - 故事板生成
  - 视频扩散模型
  - 角色一致性
  - 多角色参考
  - 注意力约束
---

# DreamShot: Personalized Storyboard Synthesis with Video Diffusion Prior

**会议**: CVPR 2026  
**arXiv**: [2604.17195](https://arxiv.org/abs/2604.17195)  
**代码**: [https://ll3rd.github.io/DreamShot/](https://ll3rd.github.io/DreamShot/)  
**领域**: 视频生成  
**关键词**: 故事板生成, 视频扩散模型, 角色一致性, 多角色参考, 注意力约束

## 一句话总结

提出 DreamShot，利用视频扩散模型的时空先验来生成人物一致、场景连贯的多镜头故事板，通过 Role-Attention Consistency Loss 解决多角色混淆问题，统一支持文本到镜头、参考到镜头和镜头到镜头三种模式。

## 研究背景与动机

**领域现状**：故事板生成旨在为电影叙事生成连贯的关键镜头序列。当前方法主要分两类：基于图像扩散模型的方法（如 StoryDiffusion、AnyStory、StoryMaker）通过 IP-Adapter 或 ControlNet 保持角色一致性；基于视频模型的方法（如 StoryAnchors）利用时序一致性但仅支持文本或前帧条件。

**现有痛点**：图像模型天然倾向多样性而非时序稳定性，跨镜头角色一致性差，在多角色场景下出现严重的角色混淆（不同角色的面部、服装特征错误融合）。视频模型虽有更好的一致性但密集帧生成计算开销大，且缺乏细粒度的个性化控制。

**核心矛盾**：图像模型有灵活性但缺一致性，视频模型有一致性但缺效率——存在根本性的 trade-off。

**本文目标**：结合视频模型的时空一致性先验与图像级生成的效率和可控性，实现高质量个性化故事板。

**切入角度**：视频 VAE（如 Wan-VAE）将连续帧压缩到潜空间时保持因果时序结构，如果把每个故事板镜头重复 T 帧再编码，就能将独立的静态镜头转化为一个连贯的时序潜空间序列。

**核心 idea**：在视频扩散模型（DiT）的框架下，将角色参考图像作为时序前置锚点、故事板镜头作为后续时序段，利用 3D RoPE 位置编码自然传播角色身份信息，同时通过 RACL 约束跨角色注意力防止混淆。

## 方法详解

### 整体框架

DreamShot 建立在视频扩散模型（Video-VAE + DiT）之上。输入包括 K 个角色参考图像和 S 个镜头的文本脚本。每个参考图像单独编码为潜向量，每个故事板镜头重复 T 帧后由 VAE 编码。参考 token 和镜头 token 拼接后送入 DiT，其中自注意力在所有 token 上联合计算，交叉注意力则按镜头分别与对应文本对齐。支持三种模式：Reference-to-Shot（参考图生成）、Text-to-Shot（纯文本生成）、Shot-to-Shot（续写延续）。

### 关键设计

1. **基于视频 VAE 的镜头时序对齐**:

    - 功能：将独立的故事板镜头转化为视频 VAE 可处理的连贯时序序列
    - 核心思路：将每个镜头重复 T 帧（除第一个镜头外），编码后得到 $z_{shot} \in \mathbb{R}^{s \times d \times h \times w}$。参考图像编码为 $z_{ref}$，然后拼接 $z_t = [z_{ref}, z_{shot}]$，利用视频模型的 3D RoPE 编码时序和空间位置
    - 设计动机：参考图像放在序列前端、镜头按叙事顺序排列，让 DiT 自然地将角色身份沿时间轴向前传播。这个简单但关键的设计使 DiT 天然具备了图像模型所缺乏的跨镜头一致性传播能力

2. **Role-Attention Consistency Loss (RACL)**:

    - 功能：在多角色场景中防止跨角色特征混淆
    - 核心思路：首先通过显著性检测获取参考图的角色 mask，通过 grounding segmentation 获取故事板中的角色 mask，用 ArcFace 和 VLM 建立一对一对应关系。然后在 DiT 的自注意力中，计算参考角色 $r_k$ 与故事板角色 $s_k$ 之间的注意力图 $A_{r_k-s_k}$，用对应的 mask 作为监督约束，要求注意力集中在匹配角色区域
    - 设计动机：现有方法的角色混淆根源在于不同角色的特征在注意力计算中被错误地融合。RACL 显式约束每个角色只关注自己的对应区域，从训练层面消除混淆

3. **混合模式训练与生成**:

    - 功能：统一支持多种故事板生成场景
    - 核心思路：Reference-to-Shot 模式中只对镜头 token 加噪（参考图保持干净）；Text-to-Shot 中对所有镜头加噪；Shot-to-Shot 中前序镜头作为干净条件引导后续生成。使用 Flow Matching 目标训练
    - 设计动机：真实的故事板制作包含从头创作和续写延续两种需求，统一框架避免了为不同场景训练不同模型

### 损失函数 / 训练策略

主损失为 Flow Matching 目标 $\mathcal{L}_{diff}$，RACL 作为辅助损失约束角色注意力一致性。数据集由真实和合成视频中提取的时序连贯镜头序列构建，每个序列配有代表性参考帧和镜头级标注。

## 实验关键数据

### 主实验

论文强调了与图像模型方法的定性和定量对比，展示了在角色一致性、场景连贯性和生成效率方面的优势。DreamShot 在多角色场景中避免了角色混淆问题，而 StoryDiffusion、AnyStory 等图像模型方法频繁出现角色特征错位。

| 对比维度 | DreamShot | 图像模型方法 |
|---------|-----------|------------|
| 角色一致性 | 强（跨镜头身份稳定） | 弱（频繁角色混淆） |
| 场景连贯性 | 强（视频先验保证） | 弱（镜头间不一致） |
| 多角色支持 | 良好（RACL 约束） | 差（特征纠缠） |
| 生成效率 | 高（关键帧而非密集帧） | 中等 |

### 消融实验

| 配置 | 角色一致性指标 | 说明 |
|------|-------------|------|
| Full model | 最优 | RACL + 视频先验 |
| w/o RACL | 下降 | 多角色场景出现混淆 |
| 图像模型 backbone | 显著下降 | 缺乏时序一致性 |

### 关键发现

- 视频扩散先验对跨镜头一致性的贡献是决定性的，不是简单的图像模型"升级"能替代的
- RACL 在多角色（≥2）场景中的效果尤为显著，单角色场景下增益有限
- Shot-to-Shot 模式的续写质量高度依赖前序镜头的质量

## 亮点与洞察

- "用视频模型生成关键帧而非密集帧"的思路很巧妙——保留了视频先验的一致性优势，同时避免了大量冗余帧的计算浪费
- RACL 的设计直击多角色混淆的根本原因（注意力层面的特征纠缠），通过显式的 mask 监督约束注意力分布，思路清晰且有效
- 将参考图像放在 token 序列前端利用 3D RoPE 的时序编码来传播身份信息，这是对视频模型位置编码语义的创造性利用

## 局限与展望

- 依赖预训练视频模型（如 Wan2.1）的质量，受限于基础模型的生成能力
- RACL 需要角色 mask 检测和一对一匹配，在遮挡严重或角色外观相似时可能失效
- 当前评估主要基于定性比较，缺乏标准化的故事板生成 benchmark
- 未来可扩展至交互式编辑（修改特定镜头而保持其余不变）

## 相关工作与启发

- **vs StoryDiffusion/StoryAdapter**: 基于图像模型的跨帧注意力一致性方法，本质上受限于图像模型的帧独立性，本文通过视频先验从根本上解决了一致性问题
- **vs StoryAnchors**: 同样使用视频范式，但只支持文本/前帧条件，缺乏多角色参考控制

## 评分

- 新颖性: ⭐⭐⭐⭐ 视频先验驱动的故事板生成是新方向，RACL 设计巧妙
- 实验充分度: ⭐⭐⭐ 定性为主，缺乏标准化定量对比
- 写作质量: ⭐⭐⭐⭐ 动机清晰，框架描述完整
- 价值: ⭐⭐⭐⭐ 开辟了故事板生成的新范式，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Generative Neural Video Compression via Video Diffusion Prior](generative_neural_video_compression_via_video_diffusion_prior.md)
- [\[CVPR 2026\] MoVieDrive: Urban Scene Synthesis with Multi-Modal Multi-View Video Diffusion Transformer](moviedrive_urban_scene_synthesis_with_multi-modal_multi-view_video_diffusion_tra.md)
- [\[ICLR 2026\] JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization](../../ICLR2026/video_generation/javisdit_joint_audio-video_diffusion_transformer_with_hierarchical_spatio-tempor.md)
- [\[CVPR 2026\] NOVA: Sparse Control, Dense Synthesis for Pair-Free Video Editing](nova_sparse_control_dense_synthesis_for_pair-free_video_editing.md)
- [\[CVPR 2025\] StreetCrafter: Street View Synthesis with Controllable Video Diffusion Models](../../CVPR2025/video_generation/streetcrafter_street_view_synthesis_with_controllable_video_diffusion_models.md)

</div>

<!-- RELATED:END -->
