---
title: >-
  [论文解读] BootComp: Controllable Human Image Generation with Personalized Multi-Garments
description: >-
  [CVPR 2025][图像生成][多服装人物生成] 本文提出 BootComp，通过训练分解网络从人物图像中提取产品视图服装图来构建大规模合成配对数据集，再训练双路径扩散模型实现以多件参考服装为条件的可控人物图像生成，在 MP-LPIPS 上比 SOTA 提升 30%。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "多服装人物生成"
  - "合成数据流水线"
  - "分解网络"
  - "扩散模型组合"
  - "虚拟试穿"
---

# BootComp: Controllable Human Image Generation with Personalized Multi-Garments

**会议**: CVPR 2025  
**arXiv**: [2411.16801](https://arxiv.org/abs/2411.16801)  
**代码**: [https://omnious.github.io/BootComp](https://omnious.github.io/BootComp)  
**领域**: 可控图像生成 / 虚拟试穿  
**关键词**: 多服装人物生成、合成数据流水线、分解网络、扩散模型组合、虚拟试穿

## 一句话总结
本文提出 BootComp，通过训练分解网络从人物图像中提取产品视图服装图来构建大规模合成配对数据集，再训练双路径扩散模型实现以多件参考服装为条件的可控人物图像生成，在 MP-LPIPS 上比 SOTA 提升 30%。

## 研究背景与动机

**领域现状**：基于 T2I 扩散模型的可控人物图像生成是时尚领域的关键应用——服饰推荐、模特图生成、虚拟试穿等。需要以多件参考服装图片为条件生成穿着这些服装的人物图像。

**现有痛点**：核心瓶颈在于训练数据获取——收集每个人物穿着的所有服装照片极其困难。(1) 从图像中分割服装会导致"复制粘贴"问题（生成结果与参考完全相同，不改变姿态）；(2) 从视频不同帧提取配对数据规模有限且质量低；(3) 训练数据中多数只有单件服装-人物配对，模型在推理时无法泛化到多件服装组合。

**核心矛盾**：需要大量高质量的"多件服装→人物"配对数据，但收集这种数据在实际中几乎不可能。

**本文目标**：设计数据生成流水线解决配对数据难题，并训练可控生成模型实现多服装组合的人物生成。

**切入角度**：训练一个"分解网络"将人物图像中的穿着服装映射到产品视图（product view）图像，从而可以从任何人物图像中提取所有服装的参考图，构建大规模合成配对数据。

**核心 idea**：两阶段框架——(1) 分解网络 + 质量过滤生成合成配对数据；(2) 双扩散模型组合模块（冻结生成器+可训练编码器）在合成数据上训练。

## 方法详解

### 整体框架
Stage 1：训练分解网络 $f_\phi$，从人物图像中提取单件服装的产品视图。利用它从 240K 人物图像中生成多服装配对数据集。经质量过滤后得到 54K 高质量配对。Stage 2：两个 SDXL 扩散网络——编码器处理多件服装图片提取特征，生成器（冻结）利用这些特征生成人物图像。

### 关键设计

1. **分解网络（Decomposition Module）**:

    - 功能：将人物图像中特定类别的穿着服装映射为产品视图图像
    - 核心思路：将此看作图像到图像翻译问题。以预训练 T2I 扩散模型初始化，输入为人物解析模型分割出的服装区域 $x^s = S(y, m)$，通过 extended self-attention 将分割服装的 key/value 拼接到生成路径中。文本提示为"A product photo of {category}"来激活T2I的先验知识。
    - 设计动机：用单服装-人物配对（容易收集）训练分解网络，然后对任意人物图像提取所有服装，实现数据量的放大。

2. **合成数据质量过滤**:

    - 功能：移除分解网络生成的低质量服装图像
    - 核心思路：计算生成的产品视图 $\tilde{x}$ 与分割区域 $x^s$ 之间的感知相似度（使用 DreamSim），低于阈值 $\tau=0.4$ 的配对被丢弃。最终 240K 中保留 54K。
    - 设计动机：分解网络在人物解析结果不准确时会生成质量差的服装图，低质量数据会严重影响组合模块的训练。

3. **组合模块（Composition Module）**:

    - 功能：以多件服装为条件生成人物图像
    - 核心思路：两个 SDXL 网络，编码器 $g_\theta$ 可训练、生成器 $g_{\theta^-}$ 冻结。每件服装 $\tilde{x}_i$ 通过编码器提取 hidden states，在生成器的 self-attention 层通过 key/value 拼接进行条件化：query 来自人物图像路径，key/value 拼接所有服装的特征。
    - 设计动机：冻结生成器使 BootComp 可以无缝对接其他适配模块（ControlNet、IP-Adapter），实现姿态控制、风格化等下游任务，无需额外微调。

### 损失函数 / 训练策略
标准扩散模型 $\epsilon$-prediction 损失。分解网络训练 140K iter（4 H100），组合模块训练 115K iter（8 H100）。推理使用 DDPM 50步 + CFG scale 2.0。

## 实验关键数据

### 主实验

| 方法 | MP-LPIPS↓ | DINO↑ | M-DINO↑ | FID↓ |
|------|-----------|-------|---------|------|
| MIP-Adapter | 0.276 | 0.308 | 0.025 | 59.99 |
| Parts2Whole | 0.267 | 0.362 | 0.036 | 28.39 |
| BootComp (本文) | **0.187** | **0.379** | **0.046** | **27.63** |

### 消融实验

| 配置 | MP-LPIPS↓ | FID↓ | 说明 |
|------|-----------|------|------|
| 分割配对数据训练 | 0.374 | 59.27 | 复制粘贴问题严重 |
| 合成配对数据训练 | 0.197 | 29.41 | 大幅改善 |
| + 54K过滤后数据 | 最佳 | 最佳 | 过滤提升质量 |

数据规模实验：5K→15K→30K→50K，FID从34.15持续降至25.88，证明可扩展性。

### 关键发现
- BootComp 在 MP-LPIPS 上比 Parts2Whole 提升 30%，在服装细节保持上优势巨大
- 合成数据 vs 分割数据：FID 从 59.27 降到 29.41，证明分解网络生成的产品视图质量远优于简单分割
- 冻结生成器的设计使得 BootComp 免费获得姿态控制、风格迁移、虚拟试穿等能力

## 亮点与洞察
- **数据流水线是核心贡献**：解决了多服装配对数据获取这个根本瓶颈，分解网络+质量过滤的方案可复用。
- **冻结生成器的设计哲学**：只训练编码器使系统获得极强的可组合性——换风格换控制方式都不需要重训。
- **应用范围广**：虚拟试穿、姿态控制、卡通化、个性化生成等多种时尚应用一个框架搞定。

## 局限与展望
- 分解网络依赖人物解析模型的质量，解析错误会传播到后续环节
- 分辨率限制在 512×384，更高分辨率下效果未验证
- 某些复杂服装组合（如叠穿、配饰）的处理能力有待提升

## 相关工作与启发
- **vs Parts2Whole**: 最直接的竞争者，本文在细节保持上优势明显
- **vs MIP-Adapter**: 通用多条件生成，未针对服装细节优化
- 数据生成流水线的思路可以迁移到其他需要配对数据的可控生成任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 分解网络构建合成配对数据的思路很巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 全面比较+消融+可扩展性+多应用展示
- 写作质量: ⭐⭐⭐⭐ 框架清晰，动机强
- 价值: ⭐⭐⭐⭐⭐ 对时尚AI产业直接可用，数据方案可复用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Learning Flow Fields in Attention for Controllable Person Image Generation](learning_flow_fields_in_attention_for_controllable_person_image_generation.md)
- [\[CVPR 2025\] ConceptGuard: Continual Personalized Text-to-Image Generation with Forgetting and Confusion Mitigation](conceptguard_continual_personalized_text-to-image_generation_with_forgetting_and.md)
- [\[CVPR 2025\] PatchDPO: Patch-level DPO for Finetuning-free Personalized Image Generation](patchdpo_patch-level_dpo_for_finetuning-free_personalized_image_generation.md)
- [\[CVPR 2025\] DreamCache: Finetuning-Free Lightweight Personalized Image Generation via Feature Caching](dreamcache_finetuning-free_lightweight_personalized_image_generation_via_feature.md)
- [\[NeurIPS 2025\] SceneDesigner: Controllable Multi-Object Image Generation with 9-DoF Pose Manipulation](../../NeurIPS2025/image_generation/scenedesigner_controllable_multi-object_image_generation_with_9-dof_pose_manipul.md)

</div>

<!-- RELATED:END -->
