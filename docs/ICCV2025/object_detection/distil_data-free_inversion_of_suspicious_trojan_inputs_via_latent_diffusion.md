---
title: >-
  [论文解读] DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion
description: >-
  [ICCV 2025][目标检测][后门攻击防御] DISTIL 提出一种无需干净数据的木马触发器反演方法，通过在预训练引导扩散模型的潜空间中搜索触发器模式（而非像素空间），并注入均匀噪声正则化，有效区分真实后门触发器和对抗扰动，在 BackdoorBench 上精度最高提升 7.1%。
tags:
  - ICCV 2025
  - 目标检测
  - 后门攻击防御
  - 触发器反演
  - 扩散模型
  - 零样本检测
  - 模型安全
---

# DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion

**会议**: ICCV 2025  
**arXiv**: [2507.22813](https://arxiv.org/abs/2507.22813)  
**代码**: [https://github.com/AdaptiveMotorControlLab/DISTIL](https://github.com/AdaptiveMotorControlLab/DISTIL)  
**领域**: 目标检测 / 模型安全  
**关键词**: 后门攻击防御, 触发器反演, 扩散模型, 零样本检测, 模型安全

## 一句话总结

DISTIL 提出一种无需干净数据的木马触发器反演方法，通过在预训练引导扩散模型的潜空间中搜索触发器模式（而非像素空间），并注入均匀噪声正则化，有效区分真实后门触发器和对抗扰动，在 BackdoorBench 上精度最高提升 7.1%。

## 研究背景与动机

深度神经网络面临木马（后门）攻击的严重威胁：攻击者在训练数据中植入带有特定触发器的毒化样本，使模型在正常输入上表现正常，但在包含触发器的输入上产生定向误分类。这对自动驾驶、目标检测等安全关键应用构成重大风险。

**触发器反演（RET）** 是主要的事后防御手段——通过逆向工程推测攻击者使用的触发器模式。然而，现有 RET 方法存在三个核心痛点：

**对抗扰动混淆**：在高维像素空间中优化触发器，搜索结果往往是对抗扰动而非真实触发器，导致清洁模型也被误判为木马模型（假阳性高）

**强先验假设**：许多方法假设触发器是小 patch、特定形状等，对动态/隐形触发器无效

**依赖干净数据**：大多数方法需要访问干净训练数据进行像素空间优化，限制了实际可用性

**核心洞察**：木马模型对特定 shortcut 模式具有比清洁模型**更强的可转移性**——因为木马网络被显式训练来将触发器与目标类关联。如果能提取出这些 shortcut，就能通过转移性差异区分木马模型和清洁模型。

**切入角度**：将搜索空间从像素空间转移到**预训练扩散模型的潜空间**，利用扩散模型的图像流形约束来避免退化为对抗扰动，同时在每一步注入均匀噪声破坏脆弱的对抗解。

## 方法详解

### 整体框架

DISTIL 的流程如下：
1. 从纯高斯噪声 $x_T \sim \mathcal{N}(0, I)$ 出发
2. 使用被测分类器的梯度引导扩散模型的逆过程
3. 每步注入均匀噪声正则化
4. 生成候选触发器模式
5. 通过转移性评分区分木马模型与清洁模型

### 关键设计

#### 1. 分类器引导的扩散反演

DISTIL 修改扩散模型逆过程的均值为：

$$\tilde{\mu}_\theta(x_t, t, y^{\text{tar}}, y^{\text{src}}) = \mu_\theta(x_t, t) + \Sigma_\theta(x_t, t) \nabla_{x_t} \log \frac{f(y^{\text{tar}} | x_t)}{f(y^{\text{src}} | x_t)} + \lambda_1 \cdot \eta_t$$

其中梯度项同时**增大目标类概率、降低源类概率**，驱动扩散模型生成能触发分类器从源类跳转到目标类的模式。

**设计动机**：预训练的引导扩散模型天然被训练来跟随梯度信号，因此可以忠实地追踪分类器提供的引导，在潜空间中揭示真实的 shortcut 模式。

#### 2. 均匀噪声注入正则化

在每个扩散步骤中，DISTIL 向分类器输入注入均匀噪声 $\eta_t \sim \mathcal{U}(0,1)$，强度由 $\lambda_1$ 控制。

**设计动机**：对抗扰动本质上是脆弱的——微小变化即可使其失效。注入随机噪声迫使优化过程找到**鲁棒的、可转移的 shortcut 模式**（即真实触发器），而非对噪声敏感的对抗解。这是区分 DISTIL 与传统像素空间 RET 方法的关键。

#### 3. 木马检测评分

对于生成的候选触发器 $\delta^{\text{tar}}_{\text{src}}$，定义转移性评分：

$$\text{Score}(\delta^{\text{tar}}_{\text{src}}; f) = \mathbb{E}_{x' \sim \mathcal{X}'^{\text{src}}} \left[ \text{softmax}(f(x' + \delta))_{y^{\text{tar}}} - \text{softmax}(f(x' + \delta))_{y^{\text{src}}} \right]$$

整体木马评分取所有 $(y^{\text{src}}, y^{\text{tar}})$ 对的最大值。木马模型的评分远高于清洁模型。

#### 4. Fast DISTIL

穷举所有 $(y^{\text{src}}, y^{\text{tar}})$ 对的复杂度为 $O(K^2)$。Fast DISTIL 通过选择特征空间中距离目标类最远的源类，将复杂度降至 $O(K)$：

$$y^{\text{src}} = \arg\min_{y \neq y^{\text{tar}}} \cos(\phi(y), \phi(y^{\text{tar}}))$$

直觉：如果触发器能让最不相似的源类跳转到目标类，它必然利用了极强的模型特定 shortcut。

#### 5. 目标检测扩展

DISTIL 通过在引导项中增加 bounding box 位移梯度（推动检测框向角落移动），扩展到目标检测模型的木马扫描。

### 损失函数 / 训练策略

- 扩散模型使用 GLIDE 预训练权重，冻结不训练
- 采样步数 T=50
- 触发器生成最多重复 5 次直到分类器置信度超过阈值 $\lambda_2 = 0.95$
- 缓解阶段使用触发器注入干净图像（保留正确标签）微调分类器

## 实验关键数据

### 主实验

**BackdoorBench 分类器扫描（CIFAR-10，扫描精度 %）**：

| 攻击方法 | NC | SmoothInv | BTI-DBF | TRODO | **DISTIL** |
|---------|-----|-----------|---------|-------|-----------|
| BadNets | 76.4 | 86.3 | 84.0 | 86.2 | **94.9** |
| Blended | 65.2 | 84.9 | 85.7 | 85.0 | **93.4** |
| InputAware | 58.1 | 69.7 | 79.2 | 71.7 | **93.2** |
| LIRA | 54.9 | 70.8 | 80.0 | 82.5 | **90.6** |
| WaNet | 63.7 | 68.9 | 86.8 | 80.0 | **84.4** |

DISTIL 在 BackdoorBench 上平均精度 **88.5%**，比最佳基线高约 **7.1%**。

### 消融实验

| 配置 | 组件变化 | Round 0 | Round 4 | Round 11 |
|-----|---------|---------|---------|---------|
| A | 无扩散模型（纯像素优化） | 74.5 | 60.5 | 57.4 |
| B | 无噪声注入 | 81.9 | 81.6 | 76.9 |
| C | 无超参调优 | 80.6 | 78.0 | 73.3 |
| D | Fast DISTIL（$O(K)$） | 78.0 | 82.3 | 75.6 |
| **G (完整)** | **DISTIL 默认配置** | **83.1** | **84.6** | **80.4** |
| H | DISTIL + 干净数据 | 84.5 | 86.0 | 83.9 |

关键发现：
- 去掉扩散模型（Setup A）性能大幅下降，验证潜空间搜索的关键作用
- 去掉噪声注入（Setup B）也有显著下滑，验证正则化的重要性
- 换用更轻量的扩散模型（Setup E/F）性能仅轻微下降，说明对骨干模型不敏感

### 关键发现

1. **目标检测扩展**：在 TrojAI 目标检测基准上，DISTIL 达到 **63.7%** 精度，比次优方法高 **9.4%**
2. **缓解效果**：微调后攻击成功率（ASR）最低降至 **5.3%**（Blended 攻击），同时保持较高分类精度
3. **目标类预测**：在 GTSRB 上目标类预测精度平均 **72.0%**，显著优于所有基线

## 亮点与洞察

- **潜空间 vs 像素空间的范式转变**：将触发器搜索从像素空间转移到扩散模型潜空间，是本文最核心的贡献，从根本上缓解了对抗扰动混淆问题
- **零样本能力**：无需干净训练数据即可进行木马扫描，极大提升了实际部署可行性
- **跨任务通用性**：同一框架无缝扩展至分类和目标检测，展现了良好的通用性
- **可解释性**：生成的触发器人眼可辨、直观可解释（不同于对抗噪声）

## 局限与展望

- Fast DISTIL 的源类选择假设距离最远=最具诊断力，对于 all-to-all 攻击可能不成立
- 依赖预训练扩散模型的质量——虽然实验表明对骨干不敏感，但极端场景需要验证
- 当攻击者使用对抗训练来植入木马时，像素空间方法可能反而获得错误模式导致假阳性，本文通过潜空间搜索缓解了但未完全解决
- 尚未处理 LLM 等其他模态的后门检测

## 相关工作与启发

- **Neural Cleanse (NC)** [Wang et al., 2019]：RET 方法的开创性工作，像素空间优化小 patch
- **SmoothInv** [Sun & Kolter, 2023]：通过随机平滑增强分类器鲁棒性后做像素空间反演
- **BTI-DBF** [Xu et al., 2024]：解耦良性和木马特征的双分支触发器反演
- **GLIDE** [Nichol et al., 2021]：本文使用的引导扩散模型骨干
- 启发：扩散模型的"图像流形约束"可以用于其他需要区分"真实模式"和"伪造噪声"的安全场景

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 将扩散模型引入后门防御是全新思路
- 技术深度：⭐⭐⭐⭐ — 噪声注入正则化设计精巧，理论动机清晰
- 实验充分度：⭐⭐⭐⭐⭐ — 覆盖 3 个基准、11 种攻击、12 个基线方法
- 实用性：⭐⭐⭐⭐ — 零样本特性使部署门槛大幅降低

<!-- RELATED:START -->

## 相关论文

- [Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion](diffusion_curriculum_synthetic-to-real_data_curriculum_via_image-guided_diffusio.md)
- [DiffDoctor: Diagnosing Image Diffusion Models Before Treating](diffdoctor_diagnosing_image_diffusion_models_before_treating.md)
- [EvRT-DETR: Latent Space Adaptation of Image Detectors for Event-based Vision](evrt-detr_latent_space_adaptation_of_image_detectors_for_event-based_vision.md)
- [Attention to Neural Plagiarism: Diffusion Models Can Plagiarize Your Copyrighted Images!](attention_to_neural_plagiarism_diffusion_models_can_plagiarize_your_copyrighted_.md)
- [SFUOD: Source-Free Unknown Object Detection](sfuod_source-free_unknown_object_detection.md)

<!-- RELATED:END -->
