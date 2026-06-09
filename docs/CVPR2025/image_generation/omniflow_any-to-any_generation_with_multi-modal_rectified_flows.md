---
title: >-
  [论文解读] OmniFlow: Any-to-Any Generation with Multi-Modal Rectified Flows
description: >-
  [CVPR 2025][图像生成][多模态生成] OmniFlow 将 Stable Diffusion 3 的整流流框架扩展到多模态（文本+图像+音频）联合生成场景，通过模块化的 Omni-Transformer 架构和新颖的多模态引导机制…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "多模态生成"
  - "整流流"
  - "any-to-any"
  - "MMDiT"
  - "多模态引导"
---

# OmniFlow: Any-to-Any Generation with Multi-Modal Rectified Flows

**会议**: CVPR 2025  
**arXiv**: [2412.01169](https://arxiv.org/abs/2412.01169)  
**代码**: [https://github.com/jacklishufan/OmniFlows](https://github.com/jacklishufan/OmniFlows)  
**领域**: 扩散模型 / 多模态VLM  
**关键词**: 多模态生成, 整流流, any-to-any, MMDiT, 多模态引导

## 一句话总结
OmniFlow 将 Stable Diffusion 3 的整流流框架扩展到多模态（文本+图像+音频）联合生成场景，通过模块化的 Omni-Transformer 架构和新颖的多模态引导机制，在无需从头训练的情况下实现了优于 CoDi 和 UniDiffuser 等前代 any-to-any 模型的生成质量。

## 研究背景与动机

**领域现状**：当前生成模型在单任务（如文本到图像、文本到音频）上取得了卓越成果，但这些模型每个只能完成一种任务，且训练代价巨大。为实现 any-to-any 生成，CoDi 等先前工作通过拼接多个模态特定的编码器和解码器来实现，但这种设计极大地限制了跨模态的信息交互能力。

**现有痛点**：CoDi 在执行如音频+文本到图像（A+T→I）的任务时，只是简单地对音频和文本嵌入做加权平均，然后喂给图像生成器。这种做法的问题在于，任意多组不同的模态嵌入可能平均到同一个向量，导致信息丢失严重。而 UniDiffuser、Chameleon 等统一模型虽能更好地整合跨模态信息，但需要从头训练，消耗巨大计算资源。

**核心矛盾**：在 any-to-any 生成中，"模块化以复用预训练权重"和"早期融合以充分交互信息"之间存在天然矛盾。CoDi 选择了模块化但牺牲了交互，UniDiffuser 选择了融合但需要从头训练。

**本文目标**：(1) 构建一个统一的多模态生成框架，让不同模态的特征能在每一层直接交互；(2) 保持模块化设计以复用预训练模型权重，节省训练成本；(3) 提供灵活的多模态引导机制来控制不同模态间的对齐强度。

**切入角度**：SD3 的 MMDiT 架构本身就是一种"双流+联合注意力"的设计——文本流和图像流拥有独立的参数，但通过联合注意力交互。作者观察到这种设计天然具有模块化的优势，可以无缝扩展到更多模态。

**核心 idea**：将 SD3 的双流 MMDiT 扩展为多流 Omni-Transformer，每个模态拥有独立的投影和前馈层参数（可独立预训练/初始化），但通过联合注意力实现跨模态的深度交互。

## 方法详解

### 整体框架
OmniFlow 的输入为多模态数据（文本、图像、音频），首先通过模态特定的 VAE 编码到潜在空间，然后按照多模态整流流的前向过程添加随机噪声。三个正弦嵌入分别编码每个模态的时间步 $t_1, t_2, t_3$，通过 MLP 融合为统一的时间步嵌入 $y$。最终，噪声化的潜在表示和时间步嵌入通过 $N$ 个 Omni-Transformer 块处理，每个模态的最终隐状态经线性层输出速度场预测 $v$。

### 关键设计

1. **多模态整流流（Multi-Modal Rectified Flow）**:

    - 功能：将整流流从单模态扩展到多模态联合分布建模
    - 核心思路：考虑多模态数据的联合分布 $(x_1^0, x_2^0, ..., x_n^0) \sim \pi_{data}$ 和对应的独立高斯噪声 $(x_1^1, x_2^1, ..., x_n^1)$，定义每个模态的前向过程为 $x_i^{t_i} = (1-t_i)x_i^0 + t_i x_i^1$。通过在 $[0,1]^n$ 空间中的路径来编码不同的 any-to-any 任务。例如文本到图像（T→I）对应从 $(1,0,1)$ 到 $(0,0,1)$ 的路径。关键洞察：对于只有部分模态的数据（如纯文本-图像对），可以将缺失模态的时间步设为 1（即纯噪声）。
    - 设计动机：这种路径编码方式使得同一个网络可以用统一的训练目标学习所有 any-to-any 任务，且当只有两个模态时退化为标准的单模态整流流公式。

2. **多模态引导机制（Multi-Modal Guidance）**:

    - 功能：让用户灵活控制生成输出中不同模态之间的对齐强度
    - 核心思路：定义 $\delta_{ij} = v_\theta(x_i^t, x_j^0) - v_\theta(x_i^t)$ 表示输入模态 $j$ 对输出模态 $i$ 的影响。最终引导公式为 $\hat{v}_\theta = v_\theta + \sum_{j \neq i}(\alpha_{ij} - 1)\delta_{ij}$，其中 $\alpha_{ij}$ 可独立调节。例如在音频+图像到文本（A+I→T）任务中，可以通过调节 $\alpha_{im}$ 和 $\alpha_{au}$ 分别控制生成文本偏向图像描述还是音频描述。
    - 设计动机：相比 CoDi 的简单嵌入平均，这种细粒度引导机制让用户能精确控制每对输入-输出模态间的关系强度。

3. **模块化 Omni-Transformer 架构**:

    - 功能：在保持模块化（可独立预训练）的同时实现深度跨模态交互
    - 核心思路：每个 Omni-Transformer 块中，各模态输入通过独立的投影层得到各自的 $q, k, v$，然后拼接为 $Q = Concat(q_1, q_2, q_3)$，$K = Concat(k_1, k_2, k_3)$，$V = Concat(v_1, v_2, v_3)$，执行联合注意力。联合注意力的输出再经过模态独立的 FFN。关键的是，联合注意力本身不含任何可训练参数，只有投影层和 FFN 是模态特定且独立的。
    - 设计动机：这种"参数独立+注意力交互"的设计使得各模态的参数可以从单任务专家模型初始化（如从 SD3 初始化图像和文本模块），大幅降低训练成本。

### 损失函数 / 训练策略
训练分三阶段：(1) 用 SD3 初始化文本和图像模块（Model 1）；(2) 用文本-音频对训练独立的音频模块（Model 2，文本分支从 SD3 初始化）；(3) 合并 Model 1 和 Model 2（文本分支权重取平均），得到 Model 3，在所有 any-to-any 任务上联合微调 150k 步。训练使用 8 张 A6000 GPU，AdamW 优化器，学习率 5e-6。文本编码器使用 Flan-T5-L 替代 SD3 的 T5-XXL（4.7B→783M），搭配 QFormer 和 TinyLlama-1.1B 解码器构建文本 VAE。

## 实验关键数据

### 主实验

| 数据集/Benchmark | 指标 | OmniFlow | 之前最优 any-to-any | 提升 |
|--------|------|------|----------|------|
| MSCOCO-30K | FID↓ | 13.40 | 9.71 (UniDiff) | - |
| MSCOCO-30K | CLIP↑ | 31.54 | 30.93 (UniDiff) | +0.61 |
| GenEval | Score↑ | 0.62 | 0.43 (UniDiff) | +0.19 |
| AudioCaps | FAD↓ | 1.75 | 1.80 (CoDi) | -0.05 |
| AudioCaps | CLAP↑ | 0.183 | 0.053 (CoDi) | +0.130 |

### 消融实验（音频/文本生成 recipe 选择）

| 配置 | FAD↓（音频） | CLAP↑（文本） | 说明 |
|------|---------|---------|------|
| rf/lognorm | 1.79 | .254 | 最优配置，最终采用 |
| rf/uniform | 1.82 | .227 | 均匀时间步采样 |
| v/linear | 1.86 | .126 | v-prediction + 线性调度 |
| eps/linear | 2.08 | .141 | ε-prediction + 线性调度 |
| SEDD (离散) | - | .180 | 离散文本扩散，不如连续方案 |
| MDLM (离散) | - | .163 | 离散文本扩散，不如连续方案 |

### 关键发现
- rf/lognorm 在音频和文本生成上全面最优，证明 SD3 中图像生成的最佳配置在其他模态上同样有效
- 离散文本扩散（SEDD、MDLM）在多模态设置下不优于连续方案，这与单任务文本生成的结论不同
- 时间步偏移（shift=3.0）对音频和文本生成均有显著提升，表明 SD3 的分辨率自适应调度在其他模态同样适用
- 联合训练可互相增益：例如 T2I 数据帮助提升音频条件生成图像的 Aesthetic 分数（+1.22）

## 亮点与洞察
- **模块化≠弱交互**：OmniFlow 证明了"参数独立但注意力共享"可以同时获得模块化的训练效率和早期融合的生成质量，这种设计范式值得在其他多模态系统中推广
- **路径编码的优雅性**：用 $[0,1]^n$ 空间中的路径来统一编码所有 any-to-any 任务，数学上简洁且实践中有效，只需 30M 训练图片就达到了 7B Transfusion 的竞争水平
- **多模态引导**：$\alpha_{ij}$ 机制让用户能细粒度控制跨模态对齐，不仅在数值上有效，定性实验还发现它甚至能捕捉训练数据中音频描述和图像描述的微妙风格差异

## 局限与展望
- 文本生成任务上性能显著落后于专用模型（如 BLIP-2），CIDEr 仅 47.3 vs BLIP-2 的 145.8，这可能是因为训练数据中混杂了不同风格的文本
- 只处理文本、图像、音频三种模态，未涵盖视频、3D 等模态
- 合成三元组（text-image-audio triplets）的质量可能限制了联合生成效果——理论分析表明三元组数据对正确建模联合分布至关重要
- 训练数据量（30M 图片）相比大规模模型仍然较少，扩大数据规模可能进一步提升效果

## 相关工作与启发
- **vs CoDi**: CoDi 通过拼接独立编码器和解码器实现 any-to-any，跨模态交互靠嵌入加权平均，信息损失严重。OmniFlow 通过联合注意力实现逐层交互，在 T2I（CLIP +0.85）和 T2A（CLAP +0.13）上大幅超越
- **vs UniDiffuser**: UniDiffuser 需从头训练且只支持文本+图像。OmniFlow 模块化设计允许复用 SD3 权重，用 1/60 的训练数据达到竞争性能
- **vs Transfusion/Chameleon**: 这些 7B 规模模型需 3.5B 训练图片，OmniFlow 3.4B 参数用 30M 图片在 GenEval 上达到等同（0.62 vs 0.63）性能

## 评分
- 新颖性: ⭐⭐⭐⭐ 多模态整流流的路径编码和模块化设计有亮点，但整体思路是 SD3 架构的自然扩展
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖 T2I、T2A、A2T、I2T 等多种任务，多个消融实验和 recipe 探索非常系统
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，数学公式规范，但部分实验设置散落在附录中
- 价值: ⭐⭐⭐⭐ 为多模态联合生成提供了高效且开源的解决方案，recipe 探索对社区有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Symbolic Representation for Any-to-Any Generative Tasks](symbolic_representation_for_any-to-any_generative_tasks.md)
- [\[CVPR 2025\] SPAI: Any-Resolution AI-Generated Image Detection by Spectral Learning](any-resolution_ai-generated_image_detection_by_spectral_learning.md)
- [\[CVPR 2025\] K-LoRA: Unlocking Training-Free Fusion of Any Subject and Style LoRAs](k-lora_unlocking_training-free_fusion_of_any_subject_and_style_loras.md)
- [\[ECCV 2024\] ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs](../../ECCV2024/image_generation/ziplora_any_subject_in_any_style_by_effectively_merging_loras.md)
- [\[ACL 2025\] FlashAudio: Rectified Flows for Fast and High-Fidelity Text-to-Audio Generation](../../ACL2025/image_generation/flashaudio_rectified_flow_tta.md)

</div>

<!-- RELATED:END -->
