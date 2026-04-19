---
title: >-
  [论文解读] Personalized Generation In Large Model Era: A Survey
description: >-
   首篇跨模态个性化生成（PGen）系统综述，提出统一的用户中心视角将 NLP/CV/IR 社区的研究纳入同一框架，覆盖文本/图像/视频/音频/3D/跨模态六大模态。
tags:

---

# Personalized Generation In Large Model Era: A Survey

| 项目 | 内容 |
|------|------|
| **会议** | ACL 2025 (Findings) |
| **arXiv** | [2503.02614](https://arxiv.org/abs/2503.02614) |
| **代码** | 未公开 |
| **领域** | others |
| **关键词** | Personalized Generation, Survey, User Modeling, LLM, Diffusion Model, Multimodal |

## 一句话总结

首篇跨模态个性化生成（PGen）系统综述，提出统一的用户中心视角将 NLP/CV/IR 社区的研究纳入同一框架，覆盖文本/图像/视频/音频/3D/跨模态六大模态。

## 研究背景与动机

- **核心观察**：大模型时代，内容生成正从通用生成（one-size-fits-all）转向个性化生成（Personalized Generation, PGen），但各社区（NLP、CV、IR）的研究各自为政，缺乏统一视角。
- **已有综述的不足**：现有综述要么以模型为中心（如专门讨论 LLM/扩散模型的个性化），要么以任务为中心（如对话生成、角色扮演），没有跨社区的全景式综述。
- **本文定位**：首次提出模态无关的统一框架，跨越 NLP/CV/IR 社区边界进行系统化梳理。

## 方法详解

### 整体框架——统一用户中心视角

PGen 依赖两类用户输入：（1）**个性化上下文**：包含用户偏好的历史数据；（2）**多模态指令**：文本提示、语音命令等明确表达内容需求的信号。生成模型从个性化上下文中学习偏好，按照指令生成定制化内容。

### 关键设计——五大个性化上下文维度

| 上下文类型 | 说明 | 常见任务 |
|-----------|------|---------|
| **用户画像** | 年龄、性别、职业、地点等 | 对话系统、电商产品图 |
| **用户文档** | 评论、邮件、社交帖子 | 写作助手、个性化推荐 |
| **用户行为** | 搜索、点击、购买等交互 | 推荐系统、信息检索 |
| **个人面部/身体** | 面部结构、体型、表情、动作 | 肖像生成、虚拟试穿 |
| **个性化主体** | 宠物、私人物品等用户特定概念 | Subject-driven 生成 |

### 三大核心目标

1. **高质量**：生成内容的连贯性、相关性和美感
2. **指令对齐**：准确遵循用户的多模态指令
3. **个性化**：与用户偏好和个性化上下文一致

### PGen 工作流程

**用户建模阶段**：
- 表示学习：编码为稠密嵌入或离散文本表示
- 提示工程：设计任务特定提示组织用户信息
- RAG：过滤无关信息并整合外部相关数据

**生成建模阶段**：
- Step 1 - 基础模型选择：LLM / MLLM / Diffusion Model
- Step 2 - 引导机制：指令引导（ICL, instruction tuning）+ 结构引导（adapter, cross-attention）
- Step 3 - 优化策略：免调优（模型融合、多轮交互）/ 监督微调（Full or PEFT）/ 偏好优化（RLHF, DPO）

### 多级分类体系

综述按**模态→个性化上下文→任务**三层组织，覆盖 200+ 篇文献：

| 模态 | 代表性任务 | 代表性方法 |
|------|-----------|-----------|
| **文本** | 推荐、写作助手、对话、角色扮演 | LLM-Rec, REST-PG, PAED, CharacterLLM |
| **图像** | Subject-driven T2I, 人脸生成, 虚拟试穿 | DreamBooth, PhotoMaker, IDM-VTON |
| **视频** | Subject-driven T2V, Talking Head, 舞蹈生成 | AnimateDiff, EMO, AnimateAnyone |
| **3D** | Image-to-3D, 3D 人脸/人体 | MVDream, DreamBooth3D, DreamWaltz |
| **音频** | 音乐生成, 文本转语音 | UMP, DiffAVA |
| **跨模态** | 个性化字幕/评论, 对话 | MyVLM, Yo'LLaVA |

## 实验

本文为综述论文，不包含原创实验。主要贡献在于系统化的文献梳理和分类。

### 数据集汇总

| 模态 | 代表数据集 |
|------|-----------|
| 文本 | LaMP, LongLaMP, Amazon Reviews, MovieLens |
| 图像 | DreamBooth 数据集, VITON-HD, DeepFashion |
| 视频 | TikTok Dance, HDTF (Talking Head) |
| 3D | ShapeNet, THuman2.0 |
| 音频 | LibriSpeech, MusicNet |

### 评估指标总结

| 目标 | 指标 |
|------|------|
| 质量 | FID, IS, CLIP Score, BLEU, Perplexity |
| 指令对齐 | CLIP-T, BERT-Score |
| 个性化 | CLIP-I, DINO Score, Face-Sim, 用户研究 |

### 关键发现

- 文本模态的个性化研究最为成熟，图像模态次之，视频/3D/音频模态仍处于早期阶段
- 用户行为和用户文档是文本模态中最常用的个性化上下文，而 CV 领域更依赖个人面部/身体和个性化主体
- PEFT（尤其是 LoRA）已成为跨模态个性化微调的主流策略

## 亮点

- 首次将 NLP/CV/IR 社区的个性化生成研究纳入统一框架，填补了重要的综述空白
- 提出的模态无关工作流程（用户建模→生成建模）为不同社区的研究者提供了共同语言
- 多级分类体系清晰且可扩展，便于定位特定子领域的研究进展
- 展望部分讨论了可扩展性、偏好演化、隐私公平等五大开放挑战

## 局限性

- 作为综述论文，对每个子领域的讨论深度有限，部分新兴方向（如 3D 个性化）的文献覆盖可能不够全面
- 统一框架虽然提供了高层次抽象，但不同模态间的技术差异仍然很大，框架的指导意义有一定局限
- 未对比不同方法的性能数据，缺乏定量的方法对比分析
- 综述截止时间限制，可能遗漏了最新的工作

## 相关工作

- **模型中心综述**：Zhang et al. (2024) 专注 LLM 个性化；Zhang et al. (2024) 讨论扩散模型个性化
- **任务中心综述**：Chen et al. (2024) 讨论个性化对话；Tseng et al. (2024) 讨论角色扮演
- **基础模型综述**：Wu et al. (2024) 综述多模态大语言模型
- **推荐系统综述**：Ayemowa et al. (2024) 讨论生成式推荐

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 3 |
| 实验充分性 | N/A (综述) |
| 写作质量 | 4 |
| **总分** | **3.7** |

<!-- RELATED:START -->

## 相关论文

- [HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference](hata_trainable_and_hardware-efficient_hash-aware_top-k_attention_for_scalable_la.md)
- [CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model](coachme_sport_instruction.md)
- [Behavioural vs. Representational Systematicity in End-to-End Models: An Opinionated Survey](behavioural_vs_representational_systematicity_in_end-to-end_models_an_opinionate.md)
- [Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation](bone_soups_multi_objective_gen.md)
- [SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](../../CVPR2025/others/sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)

<!-- RELATED:END -->
