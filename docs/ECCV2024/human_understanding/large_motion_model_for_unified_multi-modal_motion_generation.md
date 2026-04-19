---
title: >-
  [论文解读] Large Motion Model for Unified Multi-Modal Motion Generation
description: >-
  [ECCV 2024][人体理解][运动生成] 提出 Large Motion Model (LMM)，首个以动作为中心的多模态统一动作生成基础模型，通过构建包含 10 个任务、16 个数据集、320K 序列的 MotionVerse 基准，设计支持身体部位感知的 ArtAttention 机制，以及结合随机帧率/掩码的预训练策略，实现跨任务的高质量动作生成。
tags:
  - ECCV 2024
  - 人体理解
  - 运动生成
  - 多模态
  - 扩散模型
  - 统一模型
  - large-scale
---

# Large Motion Model for Unified Multi-Modal Motion Generation

**会议**: ECCV 2024  
**arXiv**: [2404.01284](https://arxiv.org/abs/2404.01284)  
**代码**: [项目主页](https://mingyuan-zhang.github.io/projects/LMM.html)  
**领域**: 人体理解  
**关键词**: 运动生成, 多模态, 扩散模型, 统一模型, large-scale

## 一句话总结

提出 Large Motion Model (LMM)，首个以动作为中心的多模态统一动作生成基础模型，通过构建包含 10 个任务、16 个数据集、320K 序列的 MotionVerse 基准，设计支持身体部位感知的 ArtAttention 机制，以及结合随机帧率/掩码的预训练策略，实现跨任务的高质量动作生成。

## 研究背景与动机

**领域现状**: 人体动作生成是动画和视频制作的核心技术，涵盖 text-to-motion、music-to-dance、motion prediction 等多个子任务。每个子任务都已有针对性的 specialist model 取得了不错效果。

**现有痛点**: 这些专家模型各自为政，只在单一任务、单一数据集上训练，面临数据量有限、数据域单一的问题，导致模型能力受限、泛化性差。

**核心矛盾**: 构建统一动作模型虽然可以利用海量多源数据实现泛化，但面临三大障碍：(1) 不同数据集的动作格式不统一（关键点 vs 旋转表示）；(2) 不同任务的评估指标不同；(3) 跨任务的动作知识难以迁移（帧率、关键点数量、缺失部位各不相同）。

**本文目标** 如何将多模态、多任务的动作生成统一到一个通用模型中，并在各个任务上达到与专家模型可比甚至更优的性能。

**切入角度**: 从数据（统一表示）、架构（身体部位感知注意力）、训练策略（无监督预训练+有监督微调）三个层面系统性地解决以上挑战。

**核心 idea**: 将 16 个数据集的异构动作数据统一为身体部位分段的中间表示，设计 ArtAttention 机制处理缺失部位和多条件输入，通过随机帧率/掩码预训练充分利用大规模运动数据。

## 方法详解

### 整体框架

LMM 基于 Transformer-based Diffusion Model（扩散模型），整体流程分为两个阶段：

1. **无监督预训练阶段**: 不使用条件信号，仅利用运动序列本身，通过随机降采样和随机掩码策略增强模型对运动先验的学习能力。
2. **有监督微调阶段**: 引入多模态条件信号（文本、音乐、语音、视频），让模型学习条件与运动之间的对应关系。

在架构上，模型包含 Read-In Layer（数据集相关的输入编码器）→ ArtAttention 主干网络 → Read-Out Layer（数据集相关的输出解码器）。

### MotionVerse 数据集

为解决数据格式不统一问题，作者构建了 MotionVerse 基准：

- **规模**: 10 个任务、16 个数据集、320K 序列、~100M 帧
- **统一表示**: 采用类似 TOMATO 的中间格式，将运动表示分解为 10 个独立部位：全局朝向/轨迹、面部表情、头部、脊柱、左臂、右臂、左腿、右腿、左手、右手
- **缺失处理**: 允许部分身体部位缺失，并在元数据中标注
- **评测映射**: 训练 motion translator 将统一格式映射回各数据集特定格式，实现跨数据集评测
- **条件对齐**: 使用 ImageBind 将文本、语音、音乐、视频等多模态条件编码为统一特征空间

### 关键设计

1. **Read-In/Read-Out Layer（数据集自适应编解码层）**: 由于不同数据集的分布差异无法完全忽略，在输入输出端使用数据集相关的编码器/解码器。训练时有 10% 概率将数据集名替换为 "all"，使通用编解码器可用于实际应用。

2. **ArtAttention（关节注意力机制）**: 核心创新模块，分为两个分支：

    - **空间注意力（Body-part Attention）**: 对每一帧，在身体部位维度上使用注意力机制建模部位间关系。由于存在天然缺失部位和预训练中人为掩码的部位，不能使用固定系数，因此使用自注意力动态计算部位间贡献。
    - **时间注意力（Temporal Attention）**: 使用多头注意力，每个头对应一个身体部位。关键改进包括：
        - 使用 Mixture-of-Expert 从多模态条件特征中生成统一 Key 表示
        - 对运动特征 $\mathbf{K}_x$ 和条件特征 $\mathbf{K}_c$ 分别独立归一化（避免长条件序列稀释运动自相关性）
        - 引入 64 个可学习 token 作为无条件生成的占位符
        - 使用真实时间而非帧索引来支持不同帧率
    - 最终输出：$\mathbf{Y} = \mathbf{Y}_s + \mathbf{Y}_t$

3. **预训练策略（Random Downsampling + Random Masking）**:

    - **随机降采样**: 对序列进行随机下采样，重新计算速度项以匹配降采样率，使模型学习适应不同原始帧率的数据
    - **随机掩码**: 在原始缺失掩码 $\mathbf{M}_s$ 基础上，以一定概率额外掩码更多部位得到 $\mathbf{M}_t$，被掩码部位用可学习空 token 替换。计算 loss 时仅忽略 $\mathbf{M}_s$ 标记的部分，迫使模型利用可见部分推断被掩码部分
    - **关键作用**: 防止模型在扩散过程中仅依赖噪声序列自身恢复，迫使它在微调时更依赖条件信号

### 损失函数 / 训练策略

- **扩散模型标准损失**: 遵循标准的 DDPM 训练范式
- **预训练**: Adam 优化器，学习率 $2 \times 10^{-4}$，80K 迭代
- **微调**: 分两阶段，先 20K 步（lr=$2 \times 10^{-4}$），再 20K 步（lr=$2 \times 10^{-5}$）
- **Classifier-free guidance**: 微调时以 10% 概率随机掩码条件信号
- **四种模型变体**: LMM-Tiny (90M), LMM-Small (160M), LMM-Base (410M), LMM-Large (760M)
- 总 batch size 512，在最多 32 块 V100 GPU 上训练

## 实验关键数据

### 主实验

**Text-to-Motion (HumanML3D)**:

| 方法 | R-Precision Top1↑ | FID↓ | MM Dist↓ | Diversity↑ | MultiModality↑ |
|------|-------------------|------|----------|-----------|----------------|
| T2M-GPT | 0.491 | 0.116 | 3.118 | 9.761 | 1.856 |
| FineMoGen | 0.504 | 0.151 | 2.998 | 9.263 | 2.696 |
| MoMask | 0.521 | 0.045 | 2.958 | - | 1.241 |
| **LMM-Large** | **0.525** | **0.040** | **2.943** | **9.814** | 2.683 |

**Music-to-Dance (AIST++)**:

| 方法 | FID_k↓ | FID_g↓ | Div_k↑ | BAS↑ |
|------|--------|--------|--------|------|
| Bailando | 28.16 | 9.62 | 7.83 | 0.2332 |
| TM2D | 19.01 | 20.09 | 9.45 | 0.2049 |
| **LMM-Large** | 22.08 | 21.97 | **9.85** | 0.2249 |

### 消融实验

**预训练策略消融 (LMM-Base)**:

| 配置 | Downsample | Random Mask | Attention | HumanML3D Top1 | HumanML3D FID | AMASS 1000ms |
|------|-----------|-------------|-----------|---------------|---------------|-------------|
| 1 | ✗ | ✗ | ArtAttention | 0.031 | 32.814 | 89.3 |
| 3 | ✗ | ✓ | ArtAttention | 0.515 | 0.151 | 76.1 |
| 4 | ✓ | ✓ | SAMI | 0.400 | 1.866 | 80.9 |
| **5** | **✓** | **✓** | **ArtAttention** | **0.511** | **0.138** | **73.6** |

### 关键发现

1. **Random Masking 是必要组件**: 没有随机掩码时（实验1/2），模型几乎无法完成 text-to-motion 任务（FID > 30），因为强大的扩散模型可以直接从噪声恢复，不依赖条件信号。
2. **模型规模效应明显**: 从 Tiny→Large，R-Precision 从 0.496 提升至 0.525，FID 从 0.415 降至 0.040，展现清晰的 scaling law。
3. **长程预测优势**: 在 Motion Prediction 任务中，LMM-Large 在长时间步（880-1000ms）上显著优于专家模型，说明大规模预训练赋予了更强的运动先验。
4. **ArtAttention 优于 SAMI**: 在大模型场景下，ArtAttention 的独立归一化策略更适合处理多条件输入。
5. **泛化能力**: LMM-Large 在 3DPW（分布外数据集）上的优势更加显著，验证了大规模训练带来的泛化能力。

## 亮点与洞察

1. **"大模型"思路迁移到动作生成领域**: 首次系统性地将 LLM 领域的"大数据+统一表示+预训练微调"范式应用于人体动作生成，构建了清晰的 scaling 路径。
2. **部位分段表示设计精巧**: 将人体分为 10 个独立部位，既解决了不同数据集关键点不一致的问题，又天然支持缺失部位处理和部位级可控生成。
3. **独立归一化的条件注入**: 发现多条件下直接拼接会稀释运动自相关性，提出对运动特征和条件特征分别归一化，这个简单但有效的设计值得借鉴。
4. **Random Masking 的双重作用**: 既帮助模型吸收缺失部位数据的知识，又防止模型"作弊"不依赖条件信号。
5. **下游应用扩展**: 生成的运动序列可映射到 2D 平面，作为视频生成的引导信号，展示了与 video generation 生态的良好兼容性。

## 局限与展望

1. **中间表示局限**: 当前只能处理整个身体部位缺失的情况，无法细粒度处理单个关键点缺失。
2. **Motion Translator 引入噪声**: 统一表示到数据集特定表示的转换过程中引入额外误差，降低了动作质量。
3. **长序列生成**: 受显存限制，长序列需使用 zero-shot 方法生成，实用性受限。
4. **音乐-舞蹈相关性**: music-to-dance 数据占比偏低，导致 FID 指标未超越专家模型。
5. **更灵活的运动表示**: 需要探索更灵活的运动表示方法来减少 translator 带来的信息损失。

## 相关工作与启发

- **FineMoGen**: LMM 的直接基线，ArtAttention 由其 SAMI 模块升级而来
- **MDM / MoMask**: text-to-motion 领域的代表性方法，LMM-Large 在准确性上超越
- **ImageBind**: 用于将多模态条件对齐到统一特征空间
- **TOMATO**: 统一运动表示的参考格式
- **启发**: 该工作为后续构建更大规模的 motion foundation model 提供了数据处理的经验和训练策略的 baseline

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次系统构建动作生成领域的"大模型"，数据+架构+训练策略三方面都有创新
- **实验充分度**: ⭐⭐⭐⭐⭐ 10 个任务、9 个 benchmark，消融实验详尽，scaling 分析完整
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，motivtaion-solution 对应关系明确
- **价值**: ⭐⭐⭐⭐ 对动作生成社区有重要的基础设施价值，MotionVerse 数据集本身就是重大贡献

<!-- RELATED:START -->

## 相关论文

- [FreeMotion: A Unified Framework for Number-free Text-to-Motion Synthesis](freemotion_a_unified_framework_for_number-free_text-to-motion_synthesis.md)
- [Motion Mamba: Efficient and Long Sequence Motion Generation](motion_mamba_efficient_and_long_sequence_motion_generation.md)
- [HUMOS: Human Motion Model Conditioned on Body Shape](humos_human_motion_model_conditioned_on_body_shape.md)
- [GENMO: A GENeralist Model for Human MOtion](../../ICCV2025/human_understanding/genmo_a_generalist_model_for_human_motion.md)
- [GenM3: Generative Pretrained Multi-path Motion Model for Text Conditional Human Motion Generation](../../ICCV2025/human_understanding/genm3_generative_pretrained_multi-path_motion_model_for_text_conditional_human_m.md)

<!-- RELATED:END -->
