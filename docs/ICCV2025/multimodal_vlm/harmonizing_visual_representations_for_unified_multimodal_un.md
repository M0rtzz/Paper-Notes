---
title: >-
  [论文解读] Harmonizing Visual Representations for Unified Multimodal Understanding and Generation
description: >-
  [ICCV 2025][多模态VLM][MAR编码器] 发现掩码自回归（MAR）模型的编码器天然兼具生成所需的细粒度图像特征和理解所需的高层语义表示，据此提出Harmon——以共享MAR编码器统一图像生成与理解的自回归框架，通过三阶段渐进训练在GenEval上以0.76 Overall超越所有统一模型，同时理解能力匹配使用独立SigLIP编码器的Janus系列。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "MAR编码器"
  - "统一视觉表示"
  - "掩码自回归"
  - "图像生成与理解"
  - "三阶段训练"
---

# Harmonizing Visual Representations for Unified Multimodal Understanding and Generation

**会议**: ICCV 2025  
**arXiv**: [2503.21979](https://arxiv.org/abs/2503.21979)  
**代码**: [GitHub](https://github.com/wusize/Harmon)  
**领域**: 多模态 / 统一生成理解  
**关键词**: MAR编码器, 统一视觉表示, 掩码自回归, 图像生成与理解, 三阶段训练

## 一句话总结

发现掩码自回归（MAR）模型的编码器天然兼具生成所需的细粒度图像特征和理解所需的高层语义表示，据此提出Harmon——以共享MAR编码器统一图像生成与理解的自回归框架，通过三阶段渐进训练在GenEval上以0.76 Overall超越所有统一模型，同时理解能力匹配使用独立SigLIP编码器的Janus系列。

## 研究背景与动机

**领域现状**: 统一图像生成与理解已成为下一代多模态智能的关键方向。现有方案或将扩散模型与MLLM松散集合（弱交互），或通过VQ离散化/VAE编码统一视觉表示（紧耦合）。

**现有痛点**: (1) VQGAN和VAE编码器主要为像素级重建预训练，缺乏高层语义，在理解任务上表现远逊于CLIP/SigLIP编码器；(2) Janus等方法用独立编码器分别处理生成和理解，虽效果好但放弃了统一表示带来的跨任务协同潜力；(3) ViLA-U尝试在VQ上联合训练对比对齐和重建，但语义对齐与像素保真难以平衡。

**核心矛盾**: 理解需要粗粒度高层语义，生成需要细粒度像素特征——如何用同一个编码器同时满足两种异质需求？

**本文目标**: 找到一种天然兼顾生成与理解的视觉表示，构建真正共享编码器的统一框架。

**切入角度**: 掩码图像建模（MIM）通过"遮挡-重建"预训练学到丰富语义；MAR将MIM扩展到自回归生成——其编码器可能天然具备双重能力。

**核心 idea**: MAR编码器的"在生成中学理解"特性使其成为统一编码器的理想选择——生成能力是理解能力的副产品。

## 方法详解

### 整体框架

Harmon由三部分组成：MAR编码器 $f_{\text{enc}}$、LLM $f_{\text{LLM}}$（Qwen2.5）、MAR解码器 $f_{\text{dec}}$。**生成路径**：文本prompt → LLM → 与MAR编码器输出交互 → MAR解码器预测遮挡patch（掩码自回归，K=64步）。**理解路径**：全部image patch输入MAR编码器 → 编码器输出+文本嵌入 → LLM做next-token预测回答问题。两条路径共享同一个MAR编码器。三阶段训练渐进解锁能力。

### 关键设计

1. **共享MAR编码器用于双任务**:

    - 功能：用同一个MIM预训练的MAR编码器同时服务图像生成和理解
    - 核心思路：生成时编码器接收已见patch $\mathbf{X}_{\text{seen}}$ 和buffer嵌入 $\mathbf{X}_{\text{buffer}}$，输出 $\mathbf{Z}_{\text{enc}} = f_{\text{enc}}(\mathbf{X}_{\text{seen}}, \mathbf{X}_{\text{buffer}})$，经LLM交互后送入解码器预测遮挡patch；理解时编码器接收全部patch，输出视觉表示供LLM做文本生成
    - 设计动机：Linear probing实验显示MAR编码器的特征在ImageNet上准确率远超VQGAN/VAE，GradCAM++可视化显示其特征对视觉概念有精确响应——生成预训练已隐式学到语义表示

2. **三阶段渐进训练**:

    - 功能：分阶段解锁生成和理解能力，避免任务冲突
    - 核心思路：Stage I（视觉-语言对齐）：22M图文对训练MAR编码器/解码器，LLM冻结，256分辨率。Stage II（综合多模态训练）：解锁LLM，用25M QA数据+50M图文数据联合训练，256分辨率。Stage III（高质量微调）：高质量QA+筛选后的10M图像，分辨率提升到512
    - 设计动机：直接端到端训练会导致生成和理解互相干扰；分阶段从对齐→能力构建→质量提升，让编码器在每个阶段的两个任务上都被充分优化

3. **掩码自回归生成+扩散解码**:

    - 功能：以掩码比例余弦递减的方式逐步生成图像
    - 核心思路：从全遮挡 $m_0 = hw$ 开始，K步按余弦曲线 $m_k = hw \cdot \cos(\frac{k}{2K}\pi)$ 递减遮挡数。每步解码器用小MLP作为去噪器预测遮挡patch，损失为 $\mathcal{L} = \mathbb{E}_{\varepsilon,t}[\|\varepsilon - \varepsilon_\theta(x_t|t, x_{\text{mask}})\|^2]$。推理时使用CFG（权重3.0）增强文本控制
    - 设计动机：MAR的掩码自回归范式天然适配LLM的因果注意力，且K步推理比纯自回归（token by token）高效得多

### 损失函数 / 训练策略

生成：扩散损失（MSE噪声预测）+ 10%空caption的CFG训练。理解：交叉熵损失（仅计算答案token）。两种损失按数据比例混合（Stage III比例1:3:16）。训练总耗时：Harmon-1.5B用32×A100训练8天。

## 实验关键数据

### 主实验

图像理解（多模态QA基准）：

| 模型 | 编码器 | LLM规模 | POPE↑ | MME-P↑ | MMB↑ | SEED↑ | MMMU↑ |
|------|--------|:-------:|:-----:|:------:|:----:|:-----:|:-----:|
| Janus-Pro† | SigLIP | 1.5B | 86.2 | 1444 | 75.5 | 68.3 | 36.3 |
| Show-o | MAGVIT-v2 | 1.3B | 80.0 | 1097 | 51.6 | 54.4 | 26.7 |
| **Harmon-1.5B** | MAR-H | 1.5B | **87.6** | 1155 | 65.5 | 67.1 | **38.9** |

图像生成（GenEval基准）：

| 模型 | Single | Two | Count | Colors | Position | ColorAttr | Overall↑ |
|------|:------:|:---:|:-----:|:------:|:--------:|:---------:|:--------:|
| Janus-Pro-1.5B | 0.98 | 0.82 | 0.51 | 0.89 | 0.65 | 0.56 | 0.73 |
| SDXL | 0.98 | 0.74 | 0.39 | 0.85 | 0.15 | 0.23 | 0.55 |
| **Harmon-1.5B** | **0.99** | **0.86** | **0.66** | 0.85 | **0.74** | 0.48 | **0.76** |

视觉质量（MJHQ-30K FID↓）：

| 模型 | MJHQ FID↓ |
|------|:---------:|
| Janus-Pro-1.5B | 9.53 |
| Show-o | 15.18 |
| **Harmon-1.5B** | **5.15** |

### 消融实验

共享编码器的协同效应：

| 分析 | 结论 |
|------|------|
| 理解 vs 独立编码器 | POPE: Harmon 87.6 ≈ Janus-Pro 86.2, MMMU: Harmon 38.9 > Janus-Pro 36.3 |
| 生成 vs VQ方法 | GenEval: Harmon 0.76 >> Show-o 0.53, LWM 0.47 |
| 共享 vs 独立路径 | 共训理解数据→生成能力提升（论文Fig观察到的协同效应） |

### 关键发现

1. **MAR编码器确实兼具双重能力**: 共享编码器在理解上匹配专用SigLIP编码器
2. **大幅超越VQ/VAE统一方法**: 在所有理解基准上显著高于Show-o、LWM、Chameleon
3. **生成质量SOTA**: MJHQ FID 5.15远优于所有统一方法，GenEval 0.76超越Janus-Pro
4. **跨任务协同存在**: 理解数据共训可提升生成性能，验证了"统一>分离"的价值

## 亮点与洞察

- **"在生成中学理解"的insight**: MAR的MIM预训练让编码器同时学到像素保真和语义表示——这一发现本身就有独立价值
- **实验验证极为扎实**: Linear probing+GradCAM++为"MAR适合统一"提供了令人信服的前期证据
- **设计简洁**: 共享编码器+LLM+解码器，无额外分支或适配器
- **MJHQ FID 5.15**: 在统一模型中遥遥领先，证明生成质量不需要牺牲

## 局限与展望

- 训练成本较高（32×A100, 8天），小团队难以复现
- 图像分辨率限制在512×512，高分辨率生成未验证
- 理解能力仍落后于专用MLLM（如InternVL2、Qwen2-VL）
- 未探索视频理解/生成的统一
- Stage III数据比例（1:3:16）偏向生成，理解侧可能未被充分优化

## 相关工作与启发

- **vs Janus/Janus-Pro**: 用独立SigLIP编码器做理解——Harmon证明共享编码器也能匹配
- **vs Show-o/D-DiT**: 用VQGAN/VAE编码器——理解上大幅落后，Harmon的MAR编码器是关键差异点
- **vs ViLA-U**: 在VQ上联合训练语义对齐+重建——难以平衡；Harmon的MAR天然兼备两者
- **启发**: "生成是理解的充分条件"——费曼的"What I cannot create, I do not understand"在AI中得到验证

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ MAR编码器用于统一框架的insight新颖且得到充分验证
- 实验充分度: ⭐⭐⭐⭐⭐ 理解+生成双维度对比全面，消融和初步实验完整
- 写作质量: ⭐⭐⭐⭐⭐ 动机论证有力，实验展示清晰系统
- 价值: ⭐⭐⭐⭐⭐ 对多模态统一架构有范式性贡献，MAR作为统一编码器的发现会影响后续工作

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] UniTok: A Unified Tokenizer for Visual Generation and Understanding](../../NeurIPS2025/multimodal_vlm/unitok_a_unified_tokenizer_for_visual_generation_and_understanding.md)
- [\[ICCV 2025\] Unified Multimodal Understanding via Byte-Pair Visual Encoding](unified_multimodal_understanding_via_byte-pair_visual_encoding.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[AAAI 2026\] URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](../../AAAI2026/multimodal_vlm/urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)
- [\[ICCV 2025\] DWIM: Towards Tool-aware Visual Reasoning via Discrepancy-aware Workflow Generation & Instruct-Masking Tuning](dwim_towards_tool-aware_visual_reasoning_via_discrepancy-aware_workflow_generati.md)

</div>

<!-- RELATED:END -->
