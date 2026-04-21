---
title: >-
  [论文解读] PureCC: Pure Learning for Text-to-Image Concept Customization
description: >-
  [CVPR 2026][图像生成][概念定制] 提出 PureCC 方法，通过分离"目标概念隐式引导"和"原始条件预测"的解耦学习目标，配合冻结表示提取器+可训练流模型的双分支训练管线和自适应引导缩放 $\lambda^{\star}$，实现高保真概念定制的同时最小化对原始模型行为和能力的影响。
tags:
  - CVPR 2026
  - 图像生成
  - 概念定制
  - 扩散模型微调
  - 隐式引导
  - 模型保持
  - 自适应缩放
---

# PureCC: Pure Learning for Text-to-Image Concept Customization

**会议**: CVPR 2026  
**arXiv**: [2603.07561](https://arxiv.org/abs/2603.07561)  
**代码**: https://github.com/lzc-sg/PureCC  
**领域**: 图像生成  
**关键词**: 概念定制, 扩散模型微调, 隐式引导, 模型保持, 自适应缩放

## 一句话总结
提出 PureCC 方法，通过分离"目标概念隐式引导"和"原始条件预测"的解耦学习目标，配合冻结表示提取器+可训练流模型的双分支训练管线和自适应引导缩放 $\lambda^{\star}$，实现高保真概念定制的同时最小化对原始模型行为和能力的影响。

## 研究背景与动机

**领域现状**：概念定制（Concept Customization）使用 3-5 张参考图让 T2I 模型学习个性化概念（主体、风格等）。主流方法分为 Tuning-free（如 DreamO、UNO 编码参考图特征注入）和 Tuning-based（如 DreamBooth 全参微调、LoRA 低秩微调）。

**现有痛点**：现有方法聚焦于高保真和多概念定制，忽视了两个重要问题：
   - **原始行为破坏**：学习 [V] dog 后，非目标元素（背景、风格、光照）也被意外改变，因为有限参考图中的冗余信息与目标概念无法解耦
   - **原始能力退化**：微调后模型的文本跟随能力和图像质量下降，KL 散度可视化显示分布产生了明显漂移

**核心矛盾**：现有方法将定制集中的所有语言-视觉知识作为学习源，但参考图太少（3-5张），模型无法区分目标概念和冗余背景信息。且学习目标中缺乏对原始模型的显式考虑，导致学概念时原始分布漂移。

**切入角度**：从 Classifier-Free Guidance 的隐式引导形式获得启发——CFG 将条件生成视为"无条件预测 + 隐式条件引导"，类比地，概念定制可以视为"原始条件预测 + 隐式目标概念引导"。这种解耦形式天然支持在学习概念的同时保持原始模型。

**核心idea**：$v_t^{PureCC} = v_t^{original} + \lambda^{\star} \cdot v_t^{target}$，原始预测由可训练模型提供（保持原始能力），目标引导由冻结提取器提供（纯净概念表示），$\lambda^{\star}$ 通过投影误差自适应平衡。

## 方法详解

### 整体框架
基于 SD 3.5-M（flow-based 生成模型）。两阶段训练：(1) 训练表示提取器——用 LoRA + 层级可调概念嵌入在定制集上微调一个预训练流模型；(2) 纯净学习——冻结提取器提供目标概念引导，另一个可训练流模型提供原始预测，用 $\mathcal{L}_{PureCC}$ 联合优化。

### 关键设计

1. **表示提取器（Stage 1）**

    - 功能：增强模型对个性化概念的理解，提供纯净的目标概念表示
    - 核心思路：用 LoRA 微调预训练流模型 $v_t^{\theta_1}$，引入**层级可调概念嵌入** $\{\mathbf{Y}_{tar}^l\}_{l=1}^L$——在每个 Transformer 层用不同的可学习嵌入替换 [V] token，使得不同层可以捕捉目标概念的不同细节（纹理、形状等）
    - 训练损失：标准 CFM 损失 $\mathcal{L}_{CC}^{Rep}$
    - 设计动机：层级嵌入比统一嵌入能捕捉更丰富的概念细节

2. **解耦学习目标**

    - 功能：将概念定制的速度场分解为"原始"和"目标"两个独立分量
    - 核心公式：$v_t^{PureCC} = v_t^{\theta_2}(x_t | y_{base}) + \lambda^{\star} \cdot [v_t^{\theta_1}(x_t | y_{tar}) - v_t^{\theta_1}(x_t | \emptyset)]$
    - 其中 $v_t^{original} = v_t^{\theta_2}(x_t | y_{base})$ 使用 Base Text（不含 [V]）作为条件，代表原始模型的预测能力
    - $v_t^{target} = \mathbf{R}(y_{tar})$ 是冻结提取器在 Target Text 和空条件下预测差，代表纯净的目标概念表示偏差
    - 设计动机：Base Text 条件下的输出足以代表原始模型性能，通过加法组合保持原始能力

3. **自适应引导缩放 $\lambda^{\star}$**

    - 功能：动态平衡概念保真度与原始模型保持
    - 核心思路：将 $\lambda^{\star}$ 定义为可训练模型已学概念表示对冻结模型概念引导的投影系数：$\lambda^{\star} = \frac{\langle \mathbf{R}(y_{complete}, y_{base}), \mathbf{R}(y_{tar}) \rangle}{\|\mathbf{R}(y_{tar})\|^2}$
    - 直觉：训练早期，可训练模型尚未学到概念方向，$\lambda^{\star}$ 自动降低以避免污染原始模型；训练后期方向对齐后，$\lambda^{\star}$ 增大以强化概念学习
    - 闭式解，无需额外超参数调节

4. **双分支训练管线（Stage 2）**

    - 冻结分支：表示提取器 $v_t^{\theta_1}$，提供 $v_t^{target}$
    - 可训练分支：另一个预训练流模型 $v_t^{\theta_2}$，用联合损失 $\mathcal{L}_{PCC} = \mathcal{L}_{CC} + \eta \cdot \mathcal{L}_{PureCC}$ 训练
    - $\mathcal{L}_{PureCC}$ 约束完整预测向解耦目标对齐，$\mathcal{L}_{CC}$ 保持速度场的生成先验

### 训练策略
- 基础模型 SD 3.5-M，LoRA rank=4，学习率 1e-4
- 用 DreamBooth 数据集的 14 个概念 + 自建 16 个概念（含实例和风格）
- 评估基准 DreamBenchPCC（扩展 DreamBench + 12 个风格概念）

## 实验关键数据

### 主实验（DreamBenchPCC，Instance 概念）

| 方法 | ΔCLIP-T↑ | ΔHPSv2.1↑ | Seg-Cons↑ | CLIP-I↑ | DINO↑ |
|------|----------|-----------|-----------|---------|-------|
| DreamBooth | -4.81 | -2.17 | 18.38 | 0.63 | 0.62 |
| Mix-of-Show | -2.71 | -1.08 | 15.72 | 0.72 | 0.61 |
| CIFC | -1.93 | -1.62 | 13.23 | 0.78 | 0.65 |
| DreamO (free) | - | - | - | 0.71 | 0.67 |
| **PureCC** | **-0.31** | **+0.10** | **69.37** | **0.81** | **0.73** |

### 消融实验

| 策略 | ΔCLIP-T↑ | ΔHPSv2.1↑ | Seg-Cons↑ | CLIP-I↑ | DINO↑ |
|------|----------|-----------|-----------|---------|-------|
| $\mathcal{L}_{CC}$（基线） | -4.52 | -2.01 | 23.74 | 0.65 | 0.66 |
| Merged Training | -1.17 | -0.34 | - | - | - |
| **PureCC（完整）** | **-0.31** | **+0.10** | **69.37** | **0.81** | **0.73** |

### 关键发现
- **Seg-Cons 指标是最突出的优势**：PureCC 达到 69.37，远超次优 DreamBooth+EWC 的 26.37，说明原始行为保持极好
- **ΔCLIP-T 接近零**（-0.31 vs DreamBooth 的 -4.81），说明文本跟随能力几乎未受损
- **HPSv2.1 甚至正增长**（+0.10），表明定制后图像质量不降反升
- 概念保真度同时达到最优（CLIP-I 0.81, DINO 0.73），证明保持≠牺牲保真
- 多概念定制中有效避免了语义纠缠（如 [V1] man 和 [V2] sunglasses 的颜色污染）

## 亮点与洞察
- **解耦学习目标**的设计极为优雅——从 CFG 的形式推广到训练阶段，将概念定制问题重新表述为"原始预测 + 概念增量"
- **自适应 $\lambda^{\star}$** 的闭式解设计很精炼——投影系数自动反映学习进度，无需手调超参
- **层级可调概念嵌入**是对标准 Textual Inversion 的有效增强——不同 Transformer 层用不同嵌入，捕捉概念的多尺度特征
- 首次系统性定义并评估了概念定制的"行为保持"（Seg-Cons 指标），填补了评估体系空白

## 局限与展望
- 双分支管线需要维护两个流模型的前向传播，训练成本约为单分支的 2 倍
- 层级嵌入增加了参数量和训练复杂度，对于极少参考图（1-2张）的场景效果待验证
- 实验主要在 SD 3.5-M 上验证，其他架构（如基于 DiT 的 FLUX）的适配性未探究
- 自适应 $\lambda^{\star}$ 依赖两分支的表示对齐质量，若提取器训练不充分可能影响缩放精度
- 仅评估静态图像生成，视频定制场景下的时序一致性未讨论

## 相关工作与启发
- **vs DreamBooth**：DreamBooth 全参微调导致严重分布漂移（ΔCLIP-T -4.81），PureCC 解耦目标+双分支将其限制在 -0.31
- **vs CIFC**：CIFC 用交叉注意力特征约束模型保持，PureCC 直接在速度场空间分离概念和原始分量，更本质
- **vs DreamO/UNO**（Tuning-free）：它们概念保真度（DINO 0.67/0.62）不及 PureCC（0.73），且无法做多概念组合

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 解耦学习目标的推导从 CFG 自然延伸到训练阶段，并提出闭式自适应缩放，思路新颖且理论优美
- 实验充分度: ⭐⭐⭐⭐ 定量评估引入保持性指标、多概念和风格-实例组合评估，但仅限单一模型
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰，Fig.2 的 pipeline 图直观，但某些符号定义可更简洁
- 价值: ⭐⭐⭐⭐⭐ 首次系统解决概念定制中原始模型保持问题，对实际应用（如持续定制多概念而不退化）价值大

<!-- RELATED:START -->

## 相关论文

- [CoLoGen: Progressive Learning of Concept-Localization Duality for Unified Image Generation](cologen_progressive_learning_of_concept-localization_duality_for_unified_image_g.md)
- [Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [LumiCtrl: Learning Illuminant Prompts for Lighting Control in Personalized Text-to-Image Models](lumictrl_learning_illuminant_prompts_for_lighting_control_in_personalized_text-t.md)
- [Generating Multi-Image Synthetic Data for Text-to-Image Customization](../../ICCV2025/image_generation/generating_multi-image_synthetic_data_for_text-to-image_customization.md)
- [Intrinsic Concept Extraction Based on Compositional Interpretability](intrinsic_concept_extraction_based_on_compositional_interpretability.md)

<!-- RELATED:END -->
