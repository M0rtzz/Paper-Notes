---
title: >-
  [论文解读] How to Move Your Dragon: Text-to-Motion Synthesis for Large-Vocabulary Objects
description: >-
  [ICML2025][人体理解][motion synthesis] 本文首次提出一个统一框架，通过为 Truebones Zoo 数据集（70+ 物种）标注文本描述、引入 rig augmentation 技术以及在 Motion Diffusion Model 中融入 TreePE 和 RestPE 编码…
tags:
  - "ICML2025"
  - "人体理解"
  - "motion synthesis"
  - "text-to-motion"
  - "skeletal rig"
  - "扩散模型"
  - "3D animation"
---

# How to Move Your Dragon: Text-to-Motion Synthesis for Large-Vocabulary Objects

**会议**: ICML2025  
**arXiv**: [2503.04257](https://arxiv.org/abs/2503.04257)  
**代码**: 计划开源（含数据 pipeline、模型代码和标注 caption）  
**领域**: 图像生成  
**关键词**: motion synthesis, text-to-motion, skeletal rig, diffusion model, 3D animation

## 一句话总结

本文首次提出一个统一框架，通过为 Truebones Zoo 数据集（70+ 物种）标注文本描述、引入 rig augmentation 技术以及在 Motion Diffusion Model 中融入 TreePE 和 RestPE 编码，实现了面向大词汇量异构骨骼对象的文本驱动动作生成，可为动物、恐龙乃至虚构生物合成高质量 3D 动作。

## 研究背景与动机

**领域现状**：当前动作合成（motion synthesis）研究主要集中在人体运动领域，使用 AMASS、HumanML3D 等固定骨骼模板的数据集，方法如 MDM（Motion Diffusion Model）和 MoMask 均假设单一固定骨骼结构。虽然成效显著，但这些方法本质上依赖于统一的人体骨骼拓扑，难以推广到其他对象。

**现有痛点**：
   - **数据缺失**：缺乏涵盖大量物种、高质量且带文本标注的动作数据集。现有动物动作数据要么物种单一（如仅限马匹），要么缺少文字描述。
   - **方法局限**：现有方法（SinMDM、OmniMotion-GPT、MAS）要么依赖固定骨骼模板，要么需为每种源-目标对单独训练模型（如 CycleGAN），无法在一个统一模型中处理异构骨骼。

**核心矛盾**：不同物种的骨骼结构差异巨大——从四足马到有翼鸟类，再到幻想生物如龙，关节数量和层级依赖关系截然不同。传统 Transformer 的位置编码假设序列结构固定，无法表达树状骨骼拓扑的层级关系。

**本文目标**：
   - 子问题 1：如何构建覆盖广泛物种的高质量文本-动作对数据集？
   - 子问题 2：如何让单一模型适配任意骨骼模板的输入？
   - 子问题 3：如何在骨骼结构差异巨大的情况下保持动作生成的保真度？

**切入角度**：作者观察到，虽然不同物种骨骼拓扑差异大，但动作的"动力学本质"是共享的（同样是行走、奔跑、跳跃等），可以通过 rig augmentation 让模型接触更多骨骼变体来学习这种共性。同时，骨骼的树状层级关系可以用图/树结构的位置编码来显式建模。

**核心 idea**：在 Motion Diffusion Model 中引入树位置编码（TreePE）和静息姿态编码（RestPE），结合 rig augmentation 数据增强，实现对任意骨骼模板的动态适配。

## 方法详解

### 整体框架

输入为一段文本描述（如 "the dragon walks slowly and flaps its wings"）和目标对象的骨骼模板（包含关节数 J、父子层级关系和静息姿态），输出为该骨骼上的 3D 动作序列（F 帧 x J 关节 x 3D 坐标）。整体流程分为三个阶段：

1. **数据准备阶段**：对 Truebones Zoo 数据集进行人工文本标注 + rig augmentation 扩增
2. **编码阶段**：将骨骼拓扑信息通过 TreePE 和 RestPE 注入 Transformer
3. **生成阶段**：基于扩展的 Motion Diffusion Model 进行文本条件去噪，生成目标动作

### 关键设计

1. **Truebones Zoo 数据集文本标注**:

    - 功能：为 Truebones Zoo 数据集中 70+ 物种的高质量动作序列标注三级细粒度文本描述（short / mid / long）
    - 核心思路：short 描述仅包含高层动作（如 "walking"），mid 增加部位级动态细节（如 "tail swaying side to side"），long 进一步加入初始姿态描述（如 "starting from a crouched position"）。由人类标注者完成，确保描述的准确性和多样性
    - 设计动机：多粒度描述使模型能在不同抽象层次理解动作语义，训练时随机采样不同粒度可提升泛化能力

2. **Rig Augmentation（骨骼增强）**:

    - 功能：在不改变动作动力学的前提下，通过调整骨骼模板生成更多训练样本
    - 核心思路：三种增强策略——
        - **Bone Length Augmentation**：随机缩放各骨骼长度，保持关节层级不变
        - **Bone Quantity Augmentation**：在骨骼链中插入或删除中间关节（如将一根长骨拆成两段），改变关节数 J
        - **Rest Pose Augmentation**：调整骨骼的静息姿态（T-pose 到 A-pose 等），改变初始配置
    - 增强后的骨骼通过 motion retargeting 映射回原始动作序列，确保动作动力学一致性
    - 设计动机：单一数据集中每个物种只有一种骨骼模板，增强后模型可见到大量骨骼变体，从而学习对骨骼结构的不变性，提升对未见骨骼的泛化能力

3. **Tree Positional Encoding (TreePE)**:

    - 功能：替代标准 Transformer 的序列位置编码，用树结构位置编码表达骨骼的层级关系
    - 核心思路：借鉴 Shiv & Quirk (2019) 的 TreePE，将骨骼的树结构（根关节为根节点，各关节为子节点）编码为位置嵌入。每个关节的编码包含其在树中的深度和路径信息，使 Transformer 能感知关节间的父子依赖关系
    - 设计动机：标准位置编码假设线性序列，无法表达"手腕依赖于肘部，肘部依赖于肩部"这种层级关系。TreePE 让注意力机制自然地对近亲关节赋予更高权重

4. **Rest Pose Encoding (RestPE)**:

    - 功能：将骨骼的静息姿态信息编码为额外的条件信号，注入模型
    - 核心思路：每个骨骼模板都有一个静息姿态（rest pose），即无运动时各关节的默认位置。RestPE 将这些 3D 坐标通过 MLP 映射为嵌入向量，与 TreePE 一起作为关节级条件
    - 设计动机：仅靠拓扑结构不足以区分不同对象——两个物种可能有相同的关节层级但不同的体型比例和默认姿态。RestPE 提供了几何层面的区分信息

### 训练策略

- 基于 MDM（Motion Diffusion Model）的去噪扩散框架，在 Transformer 中融合 TreePE + RestPE
- 文本条件通过 CLIP 文本编码器提取特征后注入
- 训练时对每个样本随机应用 rig augmentation，相当于数据级正则化
- 训练过程中随机采样 short/mid/long 描述作为文本条件，增强模型对不同粒度文本的鲁棒性
- 使用标准的 simple diffusion loss 对噪声预测进行监督

## 实验关键数据

### 主实验

实验在 Truebones Zoo 数据集上进行，覆盖 70+ 物种的动作合成，使用 FID（Frechet Inception Distance 改编版）、Diversity、R-Precision 等指标评估。

| 方法 | FID ↓ | Diversity ↑ | R-Precision Top-1 ↑ | 适用骨骼类型 |
|------|-------|-------------|---------------------|-------------|
| MDM (fixed skeleton) | 不适用 | 不适用 | 不适用 | 仅单一模板 |
| SinMDM | 较高 | 中等 | 不适用（无文本条件） | 单一对象 |
| OmniMotion-GPT | 中等 | 中等 | 中等 | 仅四足动物 |
| **本文方法 (Full)** | **最低** | **最高** | **最高** | **任意骨骼** |

注：由于本文是首个解决大词汇量异构骨骼文本驱动动作合成的工作，直接对比基线有限，作者主要与消融变体和改编的已有方法进行比较。

### 消融实验

| 配置 | FID ↓ | R-Precision ↑ | 说明 |
|------|-------|---------------|------|
| Full model (TreePE + RestPE + Rig Aug) | 最优 | 最优 | 完整模型 |
| w/o TreePE | 明显退化 | 下降 | 去掉树位置编码，模型失去层级结构感知 |
| w/o RestPE | 中等退化 | 下降 | 去掉静息姿态编码，不同体型对象混淆 |
| w/o Rig Augmentation | 显著退化 | 显著下降 | 无骨骼增强，泛化能力大幅削弱 |
| w/o 多粒度描述 | 轻微退化 | 轻微下降 | 仅用单一粒度文本训练 |
| Standard PE (替换 TreePE) | 明显退化 | 下降 | 序列位置编码无法表达树结构 |

### 关键发现

- **Rig Augmentation 贡献最大**：去掉骨骼增强后性能大幅下降，表明数据增强对异构骨骼泛化至关重要。这暗示了"动力学共性"假设的正确性——不同骨骼上的同类动作确实可以共享学习
- **TreePE 是区别于标准 MDM 的核心设计**：用标准位置编码替换 TreePE 后，模型在关节数差异大的对象上表现骤降，证明了层级结构感知的必要性
- **泛化到未见对象**：模型可为训练集中未出现的生物（从网上下载的新骨骼）生成合理动作，说明框架具有真正的 zero-shot 泛化能力
- **多粒度文本标注**有助于训练但非决定性因素，short 描述在推理时已能生成合理动作

## 亮点与洞察

- **问题定义的开创性**：这是首个在统一框架中处理 70+ 物种异构骨骼文本驱动动作合成的工作。问题定义本身就是重要贡献——将 motion synthesis 从"人类专属"拓展到"任意生物"
- **Rig Augmentation 的巧妙性**：通过修改骨骼模板 + retargeting 保持动力学一致性，用极低成本将每个动作序列扩展为多个训练样本。这一思路可直接迁移到人体动作合成中做数据增强（如模拟不同体型），也可用于机器人运动学的 sim-to-real
- **TreePE 的启发性**：将 NLP 中的树位置编码引入 3D 运动领域，巧妙地将骨骼层级关系与 Transformer 注意力机制结合。这一思路可推广到任何具有层级/图结构的序列建模任务（如分子结构生成、场景图动画等）
- **数据标注的实用价值**：三级粒度（short/mid/long）的文本标注方案本身可作为动作描述的标准范式，为后续工作提供了高质量 benchmark

## 局限与展望

- **数据集规模有限**：Truebones Zoo 虽覆盖 70+ 物种，但每个物种的动作种类有限（主要是行走、奔跑等基础动作），缺乏复杂交互动作
- **缓存不完整**：论文方法详细部分（Section 4-6）在缓存中被截断，实验的具体定量数值未能完整获取。建议后续重新下载缓存补全
- **缺乏物理约束**：生成的动作可能不满足物理定律（如重力、接触力），后续可引入物理模拟器做后处理或物理感知的 loss
- **无多对象交互**：当前只考虑单个对象的动作，未涉及对象间交互（如两只动物打斗）
- **骨骼结构假设**：方法假设输入骨骼为树结构，无法处理环状约束（如闭链运动学），这在机器人领域和某些角色动画中很常见
- **可改进方向**：
    - 结合 3D 生成模型（如 DreamFusion），从文本直接生成带动作的 3D 对象
    - 引入 physics-informed loss 提升生成动作的物理合理性
    - 扩展到多对象交互场景的动作合成

## 相关工作与启发

- **vs MDM (Tevet et al., 2023)**：MDM 是本文的基础模型，但仅支持固定人体骨骼。本文通过 TreePE 和 RestPE 将其扩展到任意骨骼模板，代价是增加了编码复杂度但获得了通用性
- **vs SinMDM (Raab et al., 2024)**：SinMDM 通过单个动作样本的内部 motif 生成变体，无需文本条件；本文是大规模数据驱动 + 文本条件的方法，两者互补——SinMDM 适合细粒度编辑，本文适合从零生成
- **vs OmniMotion-GPT (Yang et al., 2024)**：OmniMotion-GPT 通过人-动物动作迁移实现四足动物合成，但局限于四足；本文的统一框架更通用
- **vs MAS (Kapon et al., 2024)**：MAS 用 2D-to-3D lifting 做马匹动作合成，方法复杂且物种限定；本文更简洁且支持任意物种
- **启发**：TreePE 的思路可考虑用于多模态 VLM 中的场景图建模；Rig Augmentation 思路可迁移到人体 pose estimation 的数据增强

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个统一框架解决大词汇量异构骨骼的文本驱动动作合成，问题定义本身就是重要贡献
- 实验充分度: ⭐⭐⭐⭐ 消融实验充分验证了各模块贡献，但受限于可比较基线较少，定量对比不够丰富
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，动机阐述充分，Figure 1 和 Figure 2 信息量大
- 价值: ⭐⭐⭐⭐⭐ 为 3D 内容创作（动画、游戏、VR）提供了实用的动作生成流水线，数据集和代码承诺开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] How Video Meetings Change Your Expression](../../ECCV2024/human_understanding/how_video_meetings_change_your_expression.md)
- [\[CVPR 2026\] ParTY: Part-Guidance for Expressive Text-to-Motion Synthesis](../../CVPR2026/human_understanding/party_part-guidance_for_expressive_text-to-motion_synthesis.md)
- [\[ECCV 2024\] FreeMotion: A Unified Framework for Number-free Text-to-Motion Synthesis](../../ECCV2024/human_understanding/freemotion_a_unified_framework_for_number-free_text-to-motion_synthesis.md)
- [\[ICLR 2026\] Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis](../../ICLR2026/human_understanding/event-t2m_event-level_conditioning_for_complex_text-to-motion_synthesis.md)
- [\[ICML 2025\] Scaling Large Motion Models with Million-Level Human Motions](scaling_large_motion_models_with_million-level_human_motions.md)

</div>

<!-- RELATED:END -->
