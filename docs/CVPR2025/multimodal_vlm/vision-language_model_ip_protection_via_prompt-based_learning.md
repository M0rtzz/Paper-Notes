---
description: 提出IP-CLIP框架通过轻量级IP-Prompt和风格增强分支保护VLM知识产权，授权域性能零下降，非授权域性能显著退化。
tags: [CVPR2025, 知识产权保护, CLIP, 提示学习, 非可迁移性, 域授权, 特征银行]
---

# Vision-Language Model IP Protection via Prompt-based Learning

**会议**: CVPR 2025  
**arXiv**: [2503.02393](https://arxiv.org/abs/2503.02393)  
**代码**: https://github.com/LyWang12/IP-CLIP (有)  
**领域**: 多模态VLM  
**关键词**: 知识产权保护, CLIP, 提示学习, 域泛化限制, 非迁移性学习

## 一句话总结

提出IP-CLIP框架，通过轻量级IP-Prompt学习（域token+图像token）和风格增强分支，在冻结CLIP骨干上实现VLM的知识产权保护——让模型在授权域保持高准确率的同时故意降低在非授权域的性能，授权域准确率下降为0%。

## 研究背景与动机

训练VLM（如CLIP）需要海量数据、算力和人力投入，保护模型知识产权（IP）至关重要。IP保护不仅包括所有权验证（谁拥有模型），还需限制模型只能在授权数据域部署。一个特别隐蔽的威胁是：授权用户可以轻松将模型迁移到相似但未授权的领域，造成隐性知识产权侵犯。

现有IP保护方法面临两大挑战：(1) 像NTL和CUTI这样的方法需要从头训练或大规模微调，对VLM来说成本过高；(2) 这些方法仅依赖视觉骨干，缺乏充分的语义信息来区分授权域和非授权域。CLIP的prompt tuning方法（如CoOp、MaPLe）已证明可以高效适配下游任务而无需全参数微调，这启发了作者用类似的轻量级方法实现IP保护。

## 方法详解

### 整体框架

IP-CLIP基于冻结的CLIP骨干，学习IP-Prompt（域token区分授权/非授权域 + 图像token捕捉视觉分布）。训练时授权域和非授权域图像并行输入冻结视觉编码器，IP Projector从多尺度特征中提取token构建prompt送入冻结文本编码器。通过在授权域优化分类准确率、在非授权域故意破坏性能来实现IP保护。推理时仅需输入图像，IP Projector自动生成对应prompt。

### 关键设计

1. **IP-Prompt学习（域token + 图像token）**:
    - 功能：学习可区分域身份与类别的提示
    - 核心思路：从冻结视觉编码器的多层 $[f_v^{(1)}, \ldots, f_v^{(M)}]$ 中提取两类token。域token $T_a / T_u$ 使用batch-wise feature statistics（均值 $\mu$ 和方差 $\sigma$）编码域风格信息，通过IP Projector映射得到。图像token $[V_1, \ldots, V_L]$ 从多尺度特征中编码域无关的内容信息。最终prompt结构为 $Prompt_a = [T_a; V_1, \ldots, V_L; [CLS]]$
    - 设计动机：域token编码域风格（"哪个域"），图像token编码图像内容（"哪个类别"），两者解耦使模型可以同时做域判断和类别识别。仅训练IP Projector实现极低参数开销

2. **风格增强分支（STAM）**:
    - 功能：通过特征银行增强模型区分域的能力
    - 核心思路：利用CLIP的zero-shot能力，为授权域和非授权域各构建 $N$-way $K$-shot（$K=5$）的特征银行 $B_a, B_u \in \mathbb{R}^{N \times C}$。STAM模块以输入特征 $f_v^a$ 为query，分别与 $B_a$（self-enhanced）和 $B_u$（cross-domain）做attention：$s_v^a = \text{Conv}(\text{softmax}(\frac{QK_a^T}{\sqrt{d_k}})V_a) + f_v^a$。特征银行在训练前构建，训练中冻结
    - 设计动机：特征银行提供域级别的"原型记忆"，帮助模型基于域代表性特征做更鲁棒的判断

3. **新评估指标体系**:
    - 功能：全面评价IP保护效果
    - 核心思路：提出加权指标 $W_{ua} = A_a^{IP} \cdot [D_u - D_a]$，同时考虑授权域性能保持度和非授权域性能下降度。当 $D_a=0$（授权域无退化）且 $D_u$ 大时达到最优。所有权验证指标 $O_{ua} = A_u^{SL} \cdot [A_a^{Method} - A_u^{Method}]$ 衡量水印触发的性能差异
    - 设计动机：仅看 $D_u$ 不够——如果授权域性能也大幅下降就失去了保护意义

### 损失函数 / 训练策略

总损失：$\mathcal{L} = \mathcal{L}_a - \mathcal{L}_u + \mathcal{L}_{ai} - \mathcal{L}_{ui} - \mathcal{L}_{kl} - \lambda_1 \cdot \mathcal{L}_m + \lambda_2 \cdot \mathcal{L}_{en}$

各项含义：$\mathcal{L}_a / \mathcal{L}_u$ 为对比损失（最大化/最小化授权/非授权域图文对齐），$\mathcal{L}_{kl}$ 用KL散度拉开两域文本特征分布，$\mathcal{L}_m$ 用MSE最大化域token距离，$\mathcal{L}_{en}$ 用熵约束让非授权域文本特征趋于均匀分布。使用Adam优化器，学习率 $10^{-5}$，在RTX 3090上训练。

Target-free场景（无非授权域数据）：通过风格增强生成OOD数据替代非授权域。

## 实验关键数据

### 主实验（Target-Specified IP保护，加权指标 $W_{ua}$）

| 数据集 | IP-CLIP | CUTI† | NTL† | CUTI(CNN) | NTL(CNN) |
|--------|---------|-------|------|-----------|----------|
| Office-31 | **74.84** | 72.48 | 54.98 | 70.09 | 62.11 |
| Office-Home-65 | **55.10** | 50.29 | 32.76 | 39.73 | 33.75 |
| Mini-DomainNet | **54.68** | 50.75 | 41.59 | 27.97 | 25.95 |
| 平均 $D_a$↓ | **0.00** | 1.27 | 2.07 | 0.53 | 1.55 |

### 消融/扩展实验

| 场景 | 指标 | 最佳方法 | 说明 |
|------|------|---------|------|
| 所有权验证 $O_{ua}$ | 71.3% | IP-CLIP | 超CUTI† 5.6%，超NTL† 18.7% |
| Target-free | $W_{ua}$=53.46 | IP-CLIP | 无非授权域数据也有效 |
| Target-specified | $D_a$=0.00% | IP-CLIP | 授权域性能零损失 |
| CLIP vs CNN骨干 | 平均+10~20% | CLIP-based | 语义空间更丰富 |

### 关键发现

- CLIP骨干的IP保护能力一致优于CNN骨干，多模态特征空间提供了更丰富的语义信息进行域区分
- IP-CLIP在授权域的性能下降 $D_a$ 近乎为零（三个数据集平均0.00%），说明IP保护几乎不影响正常使用
- Target-free场景通过风格增强自动构建非授权域替代品，$W_{ua}$ 从74.84降至53.46但仍有效
- 所有权验证指标 $O_{ua}=71.3\%$，显著优于所有基线方法（p<0.05）

## 亮点与洞察

- **即插即用**：IP-Prompt作为轻量模块可集成到各种CLIP-based模型前端，不修改原始模型参数
- 利用多尺度风格统计量（$\mu$, $\sigma$）作为域表示的方法巧妙，将style transfer思想跨界应用到IP保护
- 三个新指标 $W_{ua}$, $D_a$, $O_{ua}$ 提供了更公平的评价标准，避免了仅看 $D_u$ 的偏颇
- 同一框架统一了IP使用授权和所有权验证两个子问题

## 局限与展望

- 目前仅验证了CLIP类判别式模型，未扩展到生成式VLM（如LLaVA、GPT-4V）
- 授权域和非授权域需要共享标签空间，限制了某些实际场景的适用性
- Target-free性能显著低于Target-specified（$W_{ua}$ 53.46 vs 74.84），保护力度有限
- 未讨论对抗性攻击（精细微调/知识蒸馏/模型剪枝）的鲁棒性

## 相关工作与启发

- NTL/CUTI开创了可用性授权方向，但IP-CLIP证明了prompt-based方法在VLM场景下更高效
- 与CoOp/MaPLe相比，IP-CLIP将prompt从"任务适配"扩展到"安全约束"，开辟了prompt tuning新用途
- H-NTL用因果模型解耦content/style的思路与IP-CLIP的域token设计殊途同归
- 对模型安全研究的启发：可以在prompt层面轻量化地实现复杂的安全策略

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将prompt learning引入VLM IP保护，视角独特
- 实验充分度: ⭐⭐⭐⭐ 三数据集+三场景+新指标+统计显著性检验
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述详尽，公式较密但可追踪
- 价值: ⭐⭐⭐⭐ 实用价值高但仅限分类场景，生成任务IP保护仍待探索

<!-- RELATED:START -->

## 相关论文

- [NLPrompt: Noise-Label Prompt Learning for Vision-Language Models](nlprompt_noise-label_prompt_learning_for_vision-language_models.md)
- [VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)
- [Visual and Semantic Prompt Collaboration for Generalized Zero-Shot Learning](visual_and_semantic_prompt_collaboration_for_generalized_zero-shot_learning.md)
- [Cropper: Vision-Language Model for Image Cropping through In-Context Learning](cropper_vision-language_model_for_image_cropping_through_in-context_learning.md)
- [Beyond Words: Augmenting Discriminative Richness via Diffusions in Unsupervised Prompt Learning](beyond_words_augmenting_discriminative_richness_via_diffusions_in_unsupervised_p.md)

<!-- RELATED:END -->
