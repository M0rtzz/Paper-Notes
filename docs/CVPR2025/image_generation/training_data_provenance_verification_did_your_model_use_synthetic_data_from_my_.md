---
title: >-
  [论文解读] Training Data Provenance Verification: Did Your Model Use Synthetic Data from My Generative Model for Training?
description: >-
  [CVPR 2025][图像生成][数据溯源] 提出 TrainProVe 方法，基于泛化误差上界理论，通过影子模型训练和假设检验来验证可疑模型是否使用了特定生成模型的合成数据进行训练，准确率超过 99%。
tags:
  - CVPR 2025
  - 图像生成
  - 数据溯源
  - 知识产权保护
  - 扩散模型
  - 假设检验
  - 合成数据
---

# Training Data Provenance Verification: Did Your Model Use Synthetic Data from My Generative Model for Training?

**会议**: CVPR 2025  
**arXiv**: [2503.09122](https://arxiv.org/abs/2503.09122)  
**代码**: [GitHub](https://github.com/xieyc99/TrainProVe)  
**领域**: image_generation  
**关键词**: 数据溯源, 知识产权保护, 扩散模型, 假设检验, 合成数据

## 一句话总结

提出 TrainProVe 方法，基于泛化误差上界理论，通过影子模型训练和假设检验来验证可疑模型是否使用了特定生成模型的合成数据进行训练，准确率超过 99%。

## 研究背景与动机

高质量开源文本到图像模型（如 Stable Diffusion）大幅降低了获取逼真图像的门槛，但同时带来了知识产权保护的挑战。嫌疑人可能未经授权使用这些模型生成的合成数据来训练自己的特定任务模型。

现有的保护手段主要分为三种场景：Case 1 直接窃取合成图像（水印可解决）；Case 2 知识蒸馏训练同任务生成模型（训练数据溯源可解决）；Case 3 用合成数据训练不同任务模型（如分类器），这是最具挑战的场景——防守方无法访问训练数据，也无法直接比较模型（任务不同）。

**核心问题**: 当嫌疑模型是黑盒且执行不同任务时，如何验证其训练数据是否来源于特定生成模型？这是一个重要但此前未被探索的问题。

目前没有任何方法能解决 Case 3，本文提出首个可行方案。

## 方法详解

### 整体框架

TrainProVe 包含三个阶段：(1) 使用防守方的生成模型 $G_d$ 生成影子数据集 $\mathcal{D}_{sdw}$ 和验证数据集 $\mathcal{D}_v$；(2) 用影子数据训练影子模型 $M_{sdw}$；(3) 对影子模型和嫌疑模型在验证集上的表现进行假设检验，判断嫌疑模型是否合法。整个过程仅需防守方自己的生成模型，无需访问嫌疑模型内部。

### 关键设计1: 基于泛化误差上界的理论基础

**功能**: 提供验证方法的理论支撑。

**核心思路**: 利用域适应中的泛化误差上界定理，对于两个执行相同任务的模型 $M$ 和 $\hat{M}$，其在目标域 $T$ 上的泛化误差差异受训练数据分布距离约束。定理证明：当 $\hat{M}$ 的训练数据来自与 $M$ 相同的生成模型 $G$ 时，$\sup|\Delta\epsilon_T|$ 的上界更小；当数据来自不同源时，上界更大。即 $\sup_{P_2 = P(\mathbf{x}|G,\mathcal{T}_2)}|\Delta\epsilon_T| \leq \sup_{P_2 \perp G}|\Delta\epsilon_T|$。

**设计动机**: 需要一个可靠的理论依据来区分"同源训练"和"异源训练"。泛化误差上界提供了这样的数学保证——同源模型在同源验证数据上的表现更相似。

### 关键设计2: 影子模型策略

**功能**: 构建可比较的基准模型。

**核心思路**: 防守方使用自己的生成模型 $G_d$，结合不同于嫌疑人可能使用的文本提示 $\mathcal{T}_{sdw}$（如 `"a {class}"`）生成影子数据集训练影子模型 $M_{sdw}$。验证数据集使用另一组提示 $\mathcal{T}_v$（如 `"a photo of a {class}"`）生成。关键在于影子数据和验证数据都来自同一生成模型但使用不同提示，确保分布有差异但保持同源相似性。

**设计动机**: 直接比较嫌疑模型和生成模型不可行（任务不同），因此需要一个"中间桥梁"——与嫌疑模型执行相同任务的影子模型，其训练数据确定来自 $G_d$。

### 关键设计3: 单侧 Grubbs 假设检验

**功能**: 统计验证嫌疑模型是否与影子模型同源。

**核心思路**: 将验证集分 batch 输入两个模型，计算各 batch 的准确率集合 $\mathcal{A}_{sdw}$ 和 $\mathcal{A}_{sus}$。使用单侧 Grubbs 检验判断 $\text{mean}(\mathcal{A}_{sus})$ 是否为 $\mathcal{A}_{sdw}$ 中的异常低值。若拒绝 $H_0$（即嫌疑模型准确率显著低于影子模型），则判定嫌疑模型合法（未使用 $G_d$ 的数据）；反之判定为侵权。

**设计动机**: 需要一种稳健的统计方法来判断"相似"vs"不相似"。Grubbs 检验特别适合小样本离群值检测，且为单侧检验（只关心准确率是否显著低于预期），避免双侧检验的误判。

### 损失函数

影子模型使用标准交叉熵损失训练。方法本身无需额外损失设计，核心在于统计检验而非优化过程。

## 实验关键数据

### 主实验结果 (不同数据集上的验证性能)

| 数据集 | 方法 | Avg Acc | Avg F1 | Avg AUC |
|--------|------|---------|--------|---------|
| CIFAR10 | Random | 0.500 | 0.286 | 0.500 |
| CIFAR10 | Han et al. | 0.768 | 0.579 | 0.731 |
| CIFAR10 | TrainProVe-Ent | 0.868 | 0.774 | 0.902 |
| CIFAR10 | **TrainProVe** | **0.997** | **0.992** | **0.998** |
| CIFAR100 | Han et al. | 0.730 | 0.548 | 0.703 |
| CIFAR100 | **TrainProVe** | **0.992** | **0.979** | **0.981** |
| ImageNet-100 | Han et al. | 0.671 | 0.425 | 0.645 |
| ImageNet-100 | **TrainProVe** | **0.786** | **0.495** | **0.769** |

### 消融实验 (影子模型架构的影响, CIFAR10)

| 影子模型架构 | Avg Acc | Avg F1 | Avg AUC |
|-------------|---------|--------|---------|
| ResNet18 | 0.997 | 0.992 | 0.998 |
| ConvNeXt-B | 0.995 | 0.989 | 0.997 |
| Swin-B | 0.997 | 0.992 | 0.998 |

### 关键发现

1. **CIFAR10 上近乎完美**: TrainProVe 在四种文本到图像模型（SD v1.4、LCM、PixArt-α、Stable Cascade）上平均准确率达 99.7%，远超最接近的方法 Han et al.（76.8%）。
2. **架构无关性**: 影子模型使用 CNN（ResNet18/ConvNeXt-B）或 Transformer（Swin-B）均保持高性能，说明方法的鲁棒性。
3. **黑盒设定可行**: 仅通过 API 获取嫌疑模型的预测标签即可完成验证，无需任何模型内部信息。
4. **单模型即可**: 与 Han et al. 需要多个生成模型不同，TrainProVe 仅需防守方自己的生成模型。

## 亮点与洞察

- **首创性强**: 第一个解决合成数据训练溯源问题（Case 3）的方法，填补了重要研究空白。
- **理论扎实**: 基于泛化误差上界的严格数学推导，而非经验性设计。
- **实用性高**: 黑盒设定、单模型需求、统计检验判决，符合实际版权争议的法律取证场景。

## 局限与展望

- **ImageNet-100 上性能下降**: 准确率降至 78.6%，说明在类别多、数据复杂的场景下性能有待提升。
- **仅验证分类模型**: 当前仅支持图像分类的嫌疑模型，未扩展到检测、分割等其他任务。
- **假设较强**: 要求防守方知道嫌疑模型的任务类别（如预测标签集 $\mathcal{C}$），实际场景中可能不完全已知。
- 未来可探索非分类任务的溯源、多源混合训练数据的判别。

## 相关工作与启发

- **水印方法**: 仅适用于直接窃取图像的 Case 1，无法解决间接使用问题。
- **域适应理论**: 泛化误差上界在域适应中广泛应用，本文创新性地将其用于数据溯源。
- **启发**: "同源模型泛化能力相似"这一洞察可推广到更多 IP 保护场景，如检测模型蒸馏。

## 评分

⭐⭐⭐⭐ — 问题定义新颖且实用，理论基础扎实，CIFAR 上效果优异。但 ImageNet-100 上的性能下降暴露了方法的可扩展性问题，且仅限于分类任务的限制降低了实用性。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Enhancing Vision-Language Compositional Understanding with Multimodal Synthetic Data (SPARCL)](enhancing_vision-language_compositional_understanding_with_multimodal_synthetic_.md)
- [\[AAAI 2026\] Difficulty Controlled Diffusion Model for Synthesizing Effective Training Data](../../AAAI2026/image_generation/difficulty_controlled_diffusion_model_for_synthesizing_effec.md)
- [\[NeurIPS 2025\] Large-Scale Training Data Attribution for Music Generative Models via Unlearning](../../NeurIPS2025/image_generation/large-scale_training_data_attribution_for_music_generative_models_via_unlearning.md)
- [\[CVPR 2025\] BooW-VTON: Boosting In-the-Wild Virtual Try-On via Mask-Free Pseudo Data Training](boow-vton_boosting_in-the-wild_virtual_try-on_via_mask-free_pseudo_data_training.md)
- [\[AAAI 2026\] Targeted Data Protection for Diffusion Model by Matching Training Trajectory](../../AAAI2026/image_generation/targeted_data_protection_for_diffusion_model_by_matching_training_trajectory.md)

<!-- RELATED:END -->
