---
title: >-
  [论文解读] Copyright Infringement Detection in Text-to-Image Diffusion Models via Differential Privacy
description: >-
  [AAAI2026][图像生成][copyright infringement detection] 从差分隐私（Differential Privacy）角度形式化版权侵权的定义，提出 D-Plus-Minus（DPM）框架，通过对扩散模型分别进行"学习"和"遗忘"两个方向的微调，测量条件敏感度差异来事后检测文本到图像模型中的版权侵权行为。
tags:
  - AAAI2026
  - 图像生成
  - copyright infringement detection
  - differential privacy
  - 扩散模型
  - machine unlearning
---

# Copyright Infringement Detection in Text-to-Image Diffusion Models via Differential Privacy

**会议**: AAAI2026  
**arXiv**: [2509.23022](https://arxiv.org/abs/2509.23022)  
**代码**: [项目页面](https://leo-xfm.github.io/pubs/dpm)  
**领域**: image_generation  
**关键词**: copyright infringement detection, differential privacy, diffusion models, machine unlearning, text-to-image generation

## 一句话总结

从差分隐私（Differential Privacy）角度形式化版权侵权的定义，提出 D-Plus-Minus（DPM）框架，通过对扩散模型分别进行"学习"和"遗忘"两个方向的微调，测量条件敏感度差异来事后检测文本到图像模型中的版权侵权行为。

## 背景与动机

- Stable Diffusion 等大规模视觉生成模型已被发现会**记忆并复现**训练数据中的版权内容，引发严重的法律和伦理问题
- 现有检测方法存在明显不足：
    - **CopyScope** 等模型级框架只能量化整体侵权程度，无法定位到具体侵权概念或样本，难以提供法律证据
    - **基于 prompt 查询**的方法依赖构造特定提示词来触发侵权输出，容易受到模型更新、采样随机性等因素影响，缺乏鲁棒性和理论可解释性
    - **DIAGNOSIS** 方法需要先对受保护数据集进行"涂层"处理，要求获取原始训练数据
- 随着全球各地出台 AI 监管框架（如欧盟 AI Act），迫切需要一种**无需原始训练数据、无需输入 prompt、具备理论保证**的事后版权检测方法

## 核心问题

如何在**白盒访问模型但无法获取训练数据和对应 prompt** 的实际场景下，判断特定图像或视觉概念是否被扩散模型记忆（即是否构成版权侵权），并提供可量化、可解释的检测结果？

## 方法详解

### 1. 理论基础：差分隐私视角的版权侵权形式化

论文将版权侵权重新解释为**条件差分隐私的违反**：

- **条件公开性（Conditional Publicity）**：当训练数据中包含或排除某个版权概念时，模型输出发生显著变化（ε > 200），表明模型严重依赖该概念
- **版权侵权定义**：若模型 G 在包含版权样本 x_c 的数据集 D 上训练后，对于与 x_c 语义相关的 prompt，其输出分布与排除 x_c 的数据集 D' 上训练的模型输出分布有显著差异，则认定为侵权
- **非侵权定义**：若输出分布对训练数据中是否包含 x 不敏感（分布不变），则认定为非侵权

### 2. 条件敏感度指标（Conditional Sensitivity）

类比差分隐私中的局部敏感度，定义条件敏感度 CS(M, x̂_i) 来量化模型查询函数 M 对特定训练样本 x̂_i 的依赖程度：

$$CS(M, \hat{x}_i) = \max_{D, D': D \triangle D' \leq \{\hat{x}_i\}} |M(D) - M(D')|$$

### 3. D-Plus-Minus（DPM）检测框架

整个检测流程包含以下步骤：

**(a) 预处理：概念提取与图像收集**

- 从目标图像 x̂_i 中提取核心概念
- 构建语义邻域集合 U(x̂_i)，收集与该概念语义相似的若干图像
- 指定通用 prompt（格式：`a photo of [V] [class]`）

**(b) 分支训练（Branch Training）**

同时进行两个方向的微调：

- **学习分支 G_{D+}**：鼓励模型记忆目标概念（I = +1）
- **遗忘分支 G_{D-}**：训练模型遗忘目标概念（I = -1）

训练目标为标准扩散模型损失乘以分支指示符 I。

**(c) 条件敏感度测量**

使用 CLIP 图像编码器作为查询函数，通过余弦相似度比较微调模型与原始模型在相同 prompt 下的输出差异。选取多个训练步的模型进行测量。

**(d) 统计分析：正交分布校准**

微调不可避免会影响模型对无关内容的输出。DPM 通过生成正交图像（与目标概念无关的内容）构建参考分布，用正交敏感度对条件敏感度进行归一化：

$$\hat{CS}(M, \hat{x}_i, D^*) = \frac{CS(M, x_i, D^*)}{  \overline{CS(M, X_{\text{orth}}, D^*)}  }$$

**(e) 分支合并与最终评分**

计算两个分支的对比敏感度差值 ΔĈS，经 min-max 归一化和 Sigmoid 映射，得到最终 DPM 分数（范围 [0, 1]）。分数越高，越可能存在版权侵权。

### 4. 版权侵权检测数据集（CIDD）

论文构建了 CIDD 数据集，包含：

- 三个高风险类别：**人脸**（Human Face）、**建筑**（Architecture）、**艺术绘画**（Arts Painting）
- 共 429 个概念、2,397 张图像
- 每个概念标注侵权/非侵权二元标签，并配对 3-6 张邻域图像
- 提出四级版权侵权层次分类：技术层→内容层→结构/风格层→语义层

## 实验关键数据

在四个模型上的检测结果（加权平均 AUC / SoftAcc）：

| 模型 | AUC | SoftAcc |
|------|-----|---------|
| SD1.4 | 0.858 | 0.764 |
| SDXL-1.0 | 0.817 | 0.752 |
| SANA-0.6B | 0.840 | 0.757 |
| FLUX.1 | 0.812 | 0.725 |

- 所有模型加权平均 AUC 均超过 **80%**，SoftAcc 超过 **72%**
- 建筑类别在多数模型上表现最优（SD1.4 除外），SDXL 上 AUC 达 0.9256，FLUX 上达 0.9500
- 消融实验表明：双分支合并优于单分支；多时间步测量提高并稳定了检测性能；图像退化对检测影响很小

## 亮点

1. **理论创新**：首次将差分隐私理论引入版权侵权检测，将侵权问题形式化为条件差分隐私的违反，提供了严格的数学基础
2. **双分支对比设计**：通过学习/遗忘两个对立方向的微调来模拟训练数据的"包含"和"排除"，巧妙绕过了无法直接获取原始训练数据的困难
3. **统计校准机制**：利用正交 prompt 分布消除微调引入的全局参数偏移，使不同样本间的敏感度分数具有可比性
4. **实用的检测设置**：白盒模型访问、无需原始训练数据或对应 prompt，贴近真实场景
5. **层次化侵权分类体系**：从技术层到语义层的四级分类为版权侵权研究提供了系统化框架

## 局限与展望

1. **计算开销较大**：每个待检测样本都需要进行两轮微调（学习 + 遗忘），在大规模检测场景下效率有限
2. **仅针对文本到图像扩散模型**：尚未扩展到 LLM 或 LVLM 的版权检测，论文也承认这是未来工作方向
3. **依赖 CLIP 模型能力**：检测性能受 CLIP 图像编码器的泛化能力限制，不同类别表现差异较大
4. **数据集规模有限**：CIDD 仅包含三个类别和 429 个概念，可能不足以覆盖所有实际版权侵权场景
5. **需要白盒访问**：在闭源商业模型上无法直接应用，限制了实际部署范围
6. **语义层侵权未覆盖**：分类体系中最高层"语义层"（如情节、主题）在 CIDD 中未纳入，也是最难检测的一类

## 与相关工作的对比

| 方法 | 需要训练数据 | 需要 prompt | 理论保证 | 可定位具体概念 | 检测粒度 |
|------|:---:|:---:|:---:|:---:|------|
| CopyScope | ✗ | ✓ | ✗ | ✗ | 模型级 |
| DIAGNOSIS | ✓（涂层） | ✓ | 部分 | ✓ | 数据集级 |
| Prompt 工程类方法 | ✗ | ✓ | ✗ | 部分 | 样本级 |
| **DPM（本文）** | **✗** | **✗** | **✓** | **✓** | **概念级** |

DPM 是唯一不需要原始训练数据、不需要对应 prompt、且具有差分隐私理论保证的检测框架。

## 启发与关联

- **差分隐私 × 版权保护**的交叉视角很有启发性，将隐私泄露量化思路迁移到版权侵权检测是一个优雅的类比
- 双分支学习/遗忘设计与 **machine unlearning** 领域紧密相关，可以考虑引入更先进的遗忘算法提升效率
- 正交 prompt 校准的思路类似于实验中的**对照组设计**，在其他需要消除混杂因素的检测任务中也可借鉴
- 与 **membership inference attack** 在方法论上有共通之处，但 DPM 提供了更强的理论框架和更实用的检测流程

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（差分隐私视角的版权侵权形式化 + 双分支对比检测，理论和方法均有原创性）
- 实验充分度: ⭐⭐⭐⭐（四个模型、三个类别、消融完整，但缺乏与现有方法的直接定量对比）
- 写作质量: ⭐⭐⭐⭐（理论部分推导清晰，符号系统完整，但部分定义与实际操作的对应关系可更直观）
- 价值: ⭐⭐⭐⭐⭐（AI 版权问题高度重要且时效性强，理论贡献和数据集构建均有实际价值）

<!-- RELATED:START -->

## 相关论文

- [TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models](../../CVPR2026/image_generation/tina_text-free_inversion_attack_for_unlearned_text-to-image_diffusion_models.md)
- [BlackMirror: Black-Box Backdoor Detection for Text-to-Image Models via Instruction-Response Deviation](../../CVPR2026/image_generation/blackmirror_black-box_backdoor_detection_for_text-to-image_models_via_instructio.md)
- [Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective](../../ICLR2026/image_generation/continual_unlearning_for_text-to-image_diffusion_models_a_regularization_perspec.md)
- [StableGuard: Towards Unified Copyright Protection and Tamper Localization in Latent Diffusion Models](../../NeurIPS2025/image_generation/stableguard_towards_unified_copyright_protection_and_tamper_localization_in_late.md)
- [Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](../../CVPR2026/image_generation/neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)

<!-- RELATED:END -->
