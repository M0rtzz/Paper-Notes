---
title: >-
  [论文解读] LaTexBlend: Scaling Multi-concept Customized Generation with Latent Textual Blending
description: >-
  [CVPR 2025][图像生成][多概念定制生成] LaTexBlend 通过在文本编码器后的潜在文本空间（Latent Textual Space）中表示和融合多个定制概念，实现了高保真、高效率的多概念定制图像生成，微调复杂度线性增长且推理无额外开销。
tags:
  - CVPR 2025
  - 图像生成
  - 多概念定制生成
  - 文本到图像
  - 潜在文本空间
  - 概念融合
  - 去噪偏差
---

# LaTexBlend: Scaling Multi-concept Customized Generation with Latent Textual Blending

**会议**: CVPR 2025  
**arXiv**: [2503.06956](https://arxiv.org/abs/2503.06956)  
**代码**: 有 (Project Page)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 多概念定制生成, 文本到图像, 潜在文本空间, 概念融合, 去噪偏差

## 一句话总结
LaTexBlend 通过在文本编码器后的潜在文本空间（Latent Textual Space）中表示和融合多个定制概念，实现了高保真、高效率的多概念定制图像生成，微调复杂度线性增长且推理无额外开销。

## 研究背景与动机

1. **领域现状**：定制化文本到图像生成（Customized T2I）已从单概念发展到多概念生成，用户希望将多个个性化主体整合到同一场景中。现有方法主要通过多概念联合训练、数据增强或合并多个单概念扩散分支来实现。

2. **现有痛点**：现有方法要么计算效率低下（联合训练/数据增强导致微调复杂度指数增长），要么质量不佳（概念遗漏、概念混淆、概念失真）。例如 Custom Diffusion 和 MuDI 需要为每种概念组合重新训练，Mix-of-Show 需额外重训，OMG 推理开销大。

3. **核心矛盾**：多概念生成中的"去噪偏差"（denoising deviation）是质量退化的根本原因。定制概念只用 3~5 张参考图微调，这些图通常是单物体居中的，缺少上下文多样性，导致定制概念无法达到预训练模型中通用概念的泛化能力。多概念时因空间竞争和低共现概率，偏差更严重。

4. **本文目标** (1) 如何在不指数增加微调开销的前提下支持任意多概念组合？(2) 如何减轻多概念生成中的结构退化和布局混乱？

5. **切入角度**：作者观察到，在文本编码器 + 线性投影之后的"潜在文本空间"中，K/V 特征已经包含了足够的概念信息。如果将概念信息集中到这个空间的紧凑表示中，就可以在此空间中自由组合，避免在更早期的文本编码过程中产生干扰。

6. **核心 idea**：在潜在文本空间中独立学习每个概念的紧凑表示并存入概念库，推理时通过简单的特征替换实现多概念无缝融合。

## 方法详解

### 整体框架
LaTexBlend 分为两个阶段：(1) **单概念微调**：对每个概念独立进行微调，将概念信息浓缩到潜在文本特征 $\mathbf{h}_c$ 中，存入概念库；(2) **多概念推理**：从概念库中取出多个概念的潜在文本特征，在潜在文本空间中通过特征替换（Blend）操作融合到基础文本流中，生成多概念图像。整个过程不需要多概念联合训练，也不需要额外推理开销。

### 关键设计

1. **辅助基础文本编码流（Base Flow）**:

    - 功能：在微调时引导概念信息集中到目标 token 的潜在文本特征中
    - 核心思路：除了常规的带可学习参数的编码流 $\mathscr{F}_c$，额外引入一个冻结的基础流 $\mathscr{F}_b$，使用预训练的投影矩阵 $\{W_{k_o}, W_{v_o}\}$。微调时将 $\mathbf{h}_c$（概念 token 特征）替换到 $\mathbf{h}_b$（基础流输出）中对应位置，然后用替换后的特征做去噪重建。由于梯度只能通过 $\mathbf{h}_c$ 回传到可学习分支，概念信息被迫集中到这个紧凑表示中。
    - 设计动机：标准 Custom Diffusion 中，概念信息分散在所有 $M$ 个 token 特征上（因为 text encoder 中所有 token 都 attend 到 "V*"）。引入基础流后，只让概念相关 token 的特征 $\mathbf{h}_c$ 承载概念信息，实现信息压缩，使得 $\mathbf{h}_c$ 足以独立表示概念。

2. **提示模板池（Prompt Template Pool）实现位置不变性**:

    - 功能：确保紧凑概念表示 $\mathbf{h}_c$ 在不同位置插入时都能正常工作
    - 核心思路：创建含 7 个不同长度提示模板（如 "A {}.", "Photo of {}."）的模板池。微调时随机从池中抽取不同模板分别构建 $\mathscr{F}_b$ 和 $\mathscr{F}_c$ 的输入提示，使 $\mathbf{h}_c$ 的提取和插入位置不断变化，消除对位置编码的依赖。
    - 设计动机：推理时 $\mathbf{h}_c$ 可能被插入到提示中的任意位置，如果微调时位置固定，生成时会出现透视变形、布局扭曲等伪影。

3. **融合引导（Blending Guidance）**:

    - 功能：在多概念推理时修正交叉注意力图，减少概念遗漏和身份混淆
    - 核心思路：设计两个引导项——$g_1$ 最大化每个概念的标识符 token "V*" 和类别描述符 "\<noun\>" 的注意力图重叠（增强身份绑定）；$g_2$ 最小化不同概念 token 之间的注意力图重叠（防止身份混淆）。将引导项的梯度加到噪声估计中：$\hat{\epsilon}'_t = \hat{\epsilon}_t + \lambda \nabla_{z_t}(g_1 + g_2)$，引导采样过程朝期望方向偏移。
    - 设计动机：预训练 T2I 模型在处理相似主体或大量主体时可能产生错误的交叉注意力图。概念数量少（≤3）时直接融合就够用，但概念数增多后引导能显著提升质量。

### 损失函数 / 训练策略
- 单概念微调使用标准扩散模型重建损失（Eq. 6），但输入条件替换为 Blend 后的特征 $\mathcal{F}(\mathbf{h}_c)$
- 每个概念独立微调，微调完成后 $\mathbf{h}_c$ 存入概念库
- 多概念推理时不需要额外训练，直接从概念库取出并融合

## 实验关键数据

### 主实验

| 评估维度 | LaTexBlend | Mix-of-Show | Cones 2 | OMG | MuDI |
|----------|-----------|-------------|---------|-----|------|
| 概念对齐 (用户评分) | **4.33** | 2.26 | 1.92 | 2.84 | 3.24 |
| 提示对齐 (用户评分) | **4.16** | 2.55 | 3.26 | 3.66 | 2.83 |
| 整体质量 (用户评分) | **4.76** | 2.31 | 3.12 | 1.53 | 3.54 |

用户研究包含 20 组生成案例、25 位参与者，LaTexBlend 在所有三项指标上均获最高分。定量实验中 LaTexBlend 在 $S_{\text{CLIP}}^I$ 和 $S_{\text{DINO}}$ 上也显著优于所有基线。

### 消融实验

| 配置 | $S_{\text{CLIP}}^T$ ↑ | $S_{\text{CLIP}}^I$ ↑ | $S_{\text{DINO}}$ ↑ | 说明 |
|------|---------|---------|---------|------|
| Full model | 0.3684 | **0.8052** | **0.6564** | 完整模型 |
| w/o base flow | 0.3718 | 0.5861 | 0.4337 | 去掉基础流，概念保真度严重退化 |
| w/o prompt variety | 0.3539 | 0.7155 | 0.5648 | 固定模板，出现透视变形等伪影 |

### 关键发现
- **基础流是最关键的设计**：去掉后 $S_{\text{DINO}}$ 从 0.6564 暴跌至 0.4337（-34%），概念保真度完全崩溃
- LaTexBlend 的微调复杂度随概念数**线性增长**，推理时概念数增加**零额外开销**
- 融合引导在概念数 ≤3 时改进微弱，但概念数增多后效果显著
- 支持与 Layout-to-Image 模型结合，在布局条件下也展现更高的主体出现率和保真度

## 亮点与洞察
- **潜在文本空间融合的巧妙选择**：这个空间恰好位于信息充分但融合代价低的"甜点"位置——比 token embedding 层深、比 U-Net 内部浅，既有足够的概念信息又不会引入昂贵的分支合并
- **辅助流+梯度隔离的信息压缩策略**非常优雅：通过冻结基础流并切断其梯度路径，巧妙地将概念信息"压缩"到目标 token 的 K/V 特征中，实现了紧凑且可组合的概念表示
- **去噪偏差分析**提供了对定制生成质量退化的深刻理解，可以推广到其他涉及少样本微调的生成任务

## 局限与展望
- 受限于预训练 T2I 模型本身对复杂场景和长提示的处理能力，概念数量的进一步扩展受到底层模型的限制
- 概念库中的每个概念需要独立微调（尽管线性扩展），尚未探索 zero-shot 或 few-shot 概念注入
- 融合引导需要在推理时计算注意力图的梯度，会有一定的推理延迟
- 未在最新的 DiT 架构（如 Flux、SD3）上验证，潜在文本空间的结论是否能迁移到新架构待考察

## 相关工作与启发
- **vs Custom Diffusion**: Custom Diffusion 微调 token embedding + 投影矩阵实现定制，但多概念需要联合训练，复杂度指数增长。LaTexBlend 在其基础上引入信息压缩和独立微调策略，彻底解决了可扩展性问题
- **vs OMG**: OMG 也支持可扩展微调，但推理时需要为每个概念运行独立分支再合并，推理开销随概念数线性增长。LaTexBlend 在潜在文本空间融合，推理零额外开销
- **vs Mix-of-Show**: Mix-of-Show 合并多个 LoRA，但每种新的概念组合需要额外重训。LaTexBlend 建立概念库后任意组合免训练

## 评分
- 新颖性: ⭐⭐⭐⭐ 潜在文本空间融合的思路新颖，信息压缩策略设计优雅
- 实验充分度: ⭐⭐⭐⭐ 定性定量比较全面，有用户研究和计算成本分析，但缺少在更大规模概念库上的测试
- 写作质量: ⭐⭐⭐⭐ 动机分析清晰，去噪偏差的可视化分析直观有力
- 价值: ⭐⭐⭐⭐ 解决了多概念定制生成的核心痛点，实用价值高

<!-- RELATED:START -->

## 相关论文

- [DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation](diffsensei_bridging_multi-modal_llms_and_diffusion_models_for_customized_manga_g.md)
- [Diffusion Self-Distillation for Zero-Shot Customized Image Generation](diffusion_self-distillation_for_zero-shot_customized_image_generation.md)
- [MARBLE: Material Recomposition and Blending in CLIP-Space](marble_material_recomposition_and_blending_in_clip-space.md)
- [MVCustom: Multi-View Customized Diffusion via Geometric Latent Rendering and Completion](../../ICLR2026/image_generation/mvcustom_multi-view_customized_diffusion_via_geometric_latent_rendering_and_comp.md)
- [coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](codrawagents_a_multi-agent_dialogue_framework_for_compositional_image_generation.md)

<!-- RELATED:END -->
