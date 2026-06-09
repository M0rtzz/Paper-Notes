---
title: >-
  [论文解读] PreLAR: World Model Pre-training with Learnable Action Representation
description: >-
  [ECCV 2024][预训练][世界模型预训练] 本文提出PreLAR，在无动作标签的视频上进行世界模型预训练时，通过从相邻帧编码隐式动作表示并设计动作-状态一致性损失来弥合无动作预训练与有动作微调之间的差距，显著提升了下游视觉控制任务的样本效率。
tags:
  - "ECCV 2024"
  - "预训练"
  - "世界模型预训练"
  - "可学习动作表示"
  - "无监督预训练"
  - "模型强化学习"
  - "样本效率"
---

# PreLAR: World Model Pre-training with Learnable Action Representation

**会议**: ECCV 2024  
**arXiv**: N/A  
**代码**: [https://github.com/zhanglixuan0720/PreLAR](https://github.com/zhanglixuan0720/PreLAR)  
**领域**: LLM预训练  
**关键词**: 世界模型预训练, 可学习动作表示, 无监督预训练, 模型强化学习, 样本效率

## 一句话总结
本文提出PreLAR，在无动作标签的视频上进行世界模型预训练时，通过从相邻帧编码隐式动作表示并设计动作-状态一致性损失来弥合无动作预训练与有动作微调之间的差距，显著提升了下游视觉控制任务的样本效率。

## 研究背景与动机

**领域现状**：基于模型的强化学习(Model-Based RL, MBRL)通过构建环境的世界模型来进行决策。世界模型学习环境动态规律，需要大量与真实环境的交互来训练。最近的方法如APV提出了从大规模无标签视频中无监督预训练世界模型，允许在较少交互次数下微调出好的世界模型。

**现有痛点**：现有的无监督预训练方法（如APV）仅将世界模型预训练为视频预测模型——即给定当前帧预测下一帧。然而，最终使用的世界模型是动作条件化的(action-conditional)——即给定当前状态和动作预测下一状态。预训练阶段没有动作条件，微调阶段需要动作条件，这个差距(gap)限制了预训练对世界模型能力提升的效果。

**核心矛盾**：无标签视频中没有动作信息，所以预训练的世界模型无法学习"动作如何影响状态变化"这一核心知识。当微调开始时，模型需要从头学习动作与状态变化的关系，使得预训练积累的视频预测能力难以充分迁移。

**本文目标** (1) 如何在无动作标签的视频上也能进行动作条件化的世界模型预训练；(2) 如何使预训练中的隐式动作表示与微调时的真实动作对齐。

**切入角度**：两个相邻帧之间的变化隐含了"动作"信息——即是什么造成了从当前帧到下一帧的变化。可以通过编码这种帧间变化来构造隐式动作表示，使预训练也能以动作条件化的方式进行。

**核心 idea**：从无动作视频中的相邻帧提取隐式动作表示来进行动作条件化的世界模型预训练，并通过动作-状态一致性损失确保隐式动作与真实动作在表示空间中对齐。

## 方法详解

### 整体框架
PreLAR在RSSM(Recurrent State Space Model)世界模型架构基础上进行扩展。预训练阶段：从无标签视频的相邻帧对 $(o_t, o_{t+1})$ 中编码出隐式动作表示 $\hat{a}_t$，然后以 $\hat{a}_t$ 为条件训练世界模型预测 $o_{t+1}$。微调阶段：将真实动作 $a_t$ 通过动作编码器映射到与隐式动作相同的表示空间，然后用于下游控制任务的世界模型训练。

### 关键设计

1. **隐式动作表示编码器(Implicit Action Encoder)**:

    - 功能：从两个相邻时间步的观测中提取隐式动作表示
    - 核心思路：设计一个编码器网络 $E_{ia}$，输入为两个相邻时间步的隐藏状态 $(h_t, h_{t+1})$，输出为隐式动作表示 $\hat{a}_t = E_{ia}(h_t, h_{t+1})$。隐藏状态通过世界模型的编码器从观测 $o_t$ 和 $o_{t+1}$ 获得。编码器使用MLP实现，将两个状态的拼接映射到低维的动作表示空间。这个隐式动作表示捕捉了"从 $o_t$ 到 $o_{t+1}$ 发生了什么变化"的信息
    - 设计动机：在无动作标签的情况下，帧间变化是最自然的"动作"代理。通过从状态变化中反推动作，使预训练也能像微调一样以动作为条件

2. **动作-状态一致性损失(Action-State Consistency Loss)**:

    - 功能：自监督优化隐式动作表示，使其更接近真实动作的语义
    - 核心思路：设计一个一致性约束，要求隐式动作表示能够被另一个解码器正确地映射回状态变化。具体地，给定 $\hat{a}_t$ 和当前状态 $h_t$，通过转换模型预测下一状态 $\hat{h}_{t+1}$，然后要求 $\hat{h}_{t+1}$ 与真实的 $h_{t+1}$ 一致。损失函数为 $\mathcal{L}_{asc} = \|h_{t+1} - T(h_t, \hat{a}_t)\|^2$。这相当于要求隐式动作表示包含了足够的信息来描述状态转移
    - 设计动机：如果不加约束，隐式动作编码器可能学到退化解（如忽略动作信息）。一致性损失强制要求隐式动作必须包含状态转移的关键信息，这与真实动作的本质——驱动状态变化——是一致的

3. **动作空间对齐微调(Action Space Alignment Fine-tuning)**:

    - 功能：在微调阶段将真实动作映射到预训练时建立的隐式动作表示空间
    - 核心思路：设计一个动作编码器 $E_a$，将真实的低维动作 $a_t$（如关节角度、力矩等）映射为与隐式动作相同维度的表示 $\tilde{a}_t = E_a(a_t)$。这样世界模型的转换模块可以直接使用 $\tilde{a}_t$ 进行状态预测，而无需重新学习动作-状态的映射关系。预训练的转换模型权重在微调初期被冻结或以小学习率更新
    - 设计动机：通过对齐动作空间，预训练中学到的"动作如何影响状态"的知识可以无缝迁移到微调阶段，避免了从头学习动作条件化的开销

### 损失函数 / 训练策略
预训练总损失为：$\mathcal{L} = \mathcal{L}_{recon} + \mathcal{L}_{kl} + \lambda \mathcal{L}_{asc}$，其中 $\mathcal{L}_{recon}$ 为重建损失（预测下一帧），$\mathcal{L}_{kl}$ 为KL散度正则化，$\mathcal{L}_{asc}$ 为动作-状态一致性损失，$\lambda$ 为权衡系数。微调时沿用Dreamer的标准训练流程。

## 实验关键数据

### 主实验

| Meta-world任务 | 指标 (成功率) | PreLAR | APV | Dreamer(scratch) |
|---------------|-------------|--------|-----|------------------|
| Drawer-Open | Success Rate | **0.95** | 0.82 | 0.65 |
| Button-Press | Success Rate | **0.91** | 0.79 | 0.58 |
| Window-Open | Success Rate | **0.88** | 0.74 | 0.52 |
| Hammer | Success Rate | **0.72** | 0.56 | 0.38 |
| 平均 (10任务) | Success Rate | **0.83** | 0.69 | 0.51 |

### 消融实验

| 配置 | 平均成功率 | 说明 |
|------|----------|------|
| Full PreLAR | **0.83** | 完整方法 |
| w/o 隐式动作 (APV风格) | 0.69 | 退化为无动作条件预训练 |
| w/o 一致性损失 | 0.75 | 隐式动作退化，掉8% |
| w/o 动作对齐 | 0.71 | 预训练知识迁移不充分 |
| 随机动作替代隐式动作 | 0.68 | 验证隐式动作有效性 |

### 关键发现
- 隐式动作表示的引入是性能提升的最大贡献因素，验证了在预训练中引入动作条件化的必要性
- 动作-状态一致性损失对防止表示退化至关重要，去掉后隐式动作编码器倾向于忽略帧间差异
- PreLAR在样本效率上显著优于APV和从头训练的Dreamer，尤其在交互次数有限（<10万步）时优势更明显
- 在更复杂的任务（如Hammer）中，PreLAR的优势更加显著，说明动作条件化预训练对复杂动态的学习更有帮助

## 亮点与洞察
- **从"无动作"视频中挖掘"隐式动作"**的思路非常优雅。帧间变化就是最自然的动作编码，这个insight既直觉又有效
- **一致性损失作为自监督信号**巧妙地确保了隐式动作的语义质量，不需要任何动作标签就能学到有意义的动作表示。这一技巧可以迁移到其他需要从观测中推断隐含变量的任务
- PreLAR证明了世界模型预训练中"结构对齐"（预训练和微调的条件形式一致）比"数据量"更重要

## 局限与展望
- 目前仅在Meta-world仿真环境中验证，未在真实机器人环境中测试
- 隐式动作表示的维度选择需要手动调节，可能需要与真实动作空间维度匹配
- 预训练视频的来源和质量对效果有影响，但论文对此讨论有限
- 可以尝试将隐式动作表示与语言指令结合，实现跨模态的世界模型预训练
- 多步预测（而非单步相邻帧）可能产生更有层次的动作表示

## 相关工作与启发
- **vs APV**: APV将世界模型预训练为纯视频预测，没有动作条件。PreLAR通过引入隐式动作弥合了预训练与微调的差距，是对APV的直接改进
- **vs Dreamer**: Dreamer从头开始训练世界模型，需要大量环境交互。PreLAR通过预训练显著提升了样本效率，可以看作Dreamer + 高效预训练
- **vs VPT (Video PreTraining)**: VPT在Minecraft中用反向动力学模型从视频中预测动作标签，思路与PreLAR相似但在策略模型而非世界模型层面进行

## 评分
- 新颖性: ⭐⭐⭐⭐ 隐式动作表示用于世界模型预训练的idea新颖且直觉
- 实验充分度: ⭐⭐⭐ 仅在Meta-world环境验证，场景范围有限
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法逻辑性强
- 价值: ⭐⭐⭐⭐ 对世界模型预训练领域有重要启发，指明了减小预训练-微调差距的方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MR-PLIP: Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation](../../CVPR2025/llm_pretraining/mr_plip_multi_resolution_pathology.md)
- [\[ECCV 2024\] Scaling Backwards: Minimal Synthetic Pre-training?](scaling_backwards_minimal_synthetic_pre-training.md)
- [\[ICML 2025\] Metadata Conditioning Accelerates Language Model Pre-training](../../ICML2025/llm_pretraining/metadata_conditioning_accelerates_language_model_pre-training.md)
- [\[ACL 2026\] KoCo: Conditioning Language Model Pre-training on Knowledge Coordinates](../../ACL2026/llm_pretraining/koco_conditioning_language_model_pre-training_on_knowledge_coordinates.md)
- [\[NeurIPS 2025\] Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](../../NeurIPS2025/llm_pretraining/nemotron-climb_clustering-based_iterative_data_mixture_bootstrapping_for_languag.md)

</div>

<!-- RELATED:END -->
