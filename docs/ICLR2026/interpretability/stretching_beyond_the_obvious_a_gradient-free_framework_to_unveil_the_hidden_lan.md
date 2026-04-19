---
title: >-
  [论文解读] Stretching Beyond the Obvious: A Gradient-Free Framework to Unveil the Hidden Landscape of Visual Invariance
description: >-
  [ICLR 2026][visual invariance] 提出 Stretch-and-Squeeze（SnS）算法，一个无梯度、模型无关的双目标优化框架，通过在不同处理层级"拉伸"表征同时"压缩"目标单元激活来系统性地探测视觉系统的不变性流形，揭示了标准与鲁棒 CNN 之间不变性可解释性的分层差异。
tags:
  - ICLR 2026
  - visual invariance
  - gradient-free optimization
  - adversarial examples
  - feature visualization
  - CNN interpretability
  - robust models
---

# Stretching Beyond the Obvious: A Gradient-Free Framework to Unveil the Hidden Landscape of Visual Invariance

**会议**: ICLR 2026  
**arXiv**: [2506.17040](https://arxiv.org/abs/2506.17040)  
**代码**: [GitHub](https://github.com/zoccolan-lab/SnS)  
**领域**: llm_efficiency  
**关键词**: visual invariance, gradient-free optimization, adversarial examples, feature visualization, CNN interpretability, robust models

## 一句话总结

提出 Stretch-and-Squeeze（SnS）算法，一个无梯度、模型无关的双目标优化框架，通过在不同处理层级"拉伸"表征同时"压缩"目标单元激活来系统性地探测视觉系统的不变性流形，揭示了标准与鲁棒 CNN 之间不变性可解释性的分层差异。

## 研究背景与动机

理解视觉处理单元（无论是生物神经元还是 CNN 单元）编码了哪些特征组合，是视觉科学和深度学习的核心问题。现有方法的局限：

**最大激励图像（MEI）**方法只能找到少量强激活刺激，无法揭示单元在哪些变换下保持不变——而不变性对泛化至关重要

**Metamers**（表征匹配）方法在给定层严格匹配表征，只探索目标图像附近的窄邻域

**预定义变换测试**（旋转、平移、缩放等）无法发现单元实际学习到的不变轴
4. 基于梯度的方法无法应用于"黑箱"系统（如生物神经元）

SnS 的核心创新在于：通过**最大化**某个表征层的刺激距离同时**保持**目标单元激活，系统性地采样不变性流形的边界。

## 方法详解

### 整体框架

SnS 由三个组件构成：
1. **生成模型 $\psi$**：预训练 DeepGenerator，将 4096 维向量映射为 RGB 图像
2. **测试网络 $\phi$**：被分析的视觉系统
3. **无梯度优化器**：CMA-ES（协方差矩阵自适应进化策略）

### 关键设计

**双目标优化**：给定参考刺激 $\mathbf{x}_{\text{ref}}$，在层 $\kappa$（拉伸层）和层 $\ell$（压缩层）定义对抗性目标：

$$\mathcal{L}_{\text{stretch}}^{\kappa} = -\|\mathbf{a}^{\kappa} - \mathbf{a}_{\text{ref}}^{\kappa}\|_2, \quad \mathcal{L}_{\text{squeeze}}^{\kappa} = +\|\mathbf{a}^{\kappa} - \mathbf{a}_{\text{ref}}^{\kappa}\|_2$$

**不变性搜索**通过 Pareto 最优化求解（公式 5）：

$$\Xi_{\text{inv}} = \arg\min_{\mathbf{x}} \left[\mathcal{L}_{\text{stretch}}^{\kappa}(\Gamma(\mathbf{x}, \phi^{\kappa}), \Gamma(\mathbf{x}^{\star}, \phi^{\kappa})), \; |a_u^{\ell} - a_{\text{ref}}^{\ell}|\right]$$

即在层 $\kappa$ 最大化与 MEI 的表征距离，同时在层 $\ell$ 保持目标单元激活不变。

**对抗攻击搜索**则反转目标（公式 6）：最小化层 $\kappa$ 的表征距离，同时最大化层 $\ell$ 的激活变化。

**分层不变性探测**：通过改变拉伸层 $\kappa$ 的选择，揭示不同抽象层级的不变性：
- $\kappa=0$（像素空间/low_level）：主要产生亮度和对比度变化
- $\kappa=$中间卷积层（mid_level）：主要产生纹理和颜色变化
- $\kappa=$深层卷积层（high_level）：主要产生视角和姿态变化

### 损失函数

使用 CMA-ES 优化器在 Pareto 前沿意义下求解双目标问题，按 Pareto 支配关系排序解集合。优化在生成模型的 4096 维潜在空间中进行，通过自然图像先验正则化搜索，避免产生噪声样图像。

## 实验关键数据

### 主实验

**SnS 有效性验证**（77 个 $L_2$-鲁棒 ResNet50 读出单元）：

| 指标 | 对抗图像 | 不变图像 |
|------|---------|---------|
| 激活下降（相对 MEI） | 111% ± 7% | 34% ± 11% |
| $L_2$ 像素距离 | 72 ± 12 | 271 ± 32 |
| 对比仿射变换 | — | 显著超越旋转/平移/缩放 |

SnS 发现的不变图像比仿射变换更"极端"（像素距离更大），同时对目标单元的激活影响更小。

**分层不变性的可区分性**：

使用 PCA + SVM 分类器区分不同拉伸层产生的不变图像：
- 标准 ResNet50：几十个主成分即可达到近完美分类
- 鲁棒 ResNet50：达到 80%+ 准确率

### 消融实验

**人类与观察者网络的可解释性评估**（12-AFC 分类任务）：

| 条件 | 鲁棒 ResNet50 不变图像 | 标准 ResNet50 不变图像 |
|------|----------------------|----------------------|
| 像素空间拉伸 | 人类可识别（最高） | 人类难识别（最低） |
| 中层拉伸 | 人类可识别（中等） | 人类可识别（中等） |
| 深层拉伸 | 人类可识别（最低） | 人类可识别（最高） |

**关键发现**：鲁棒网络和标准网络的可解释性趋势**完全相反**！

- 鲁棒网络：拉伸越深层，可解释性越低
- 标准网络：拉伸越深层，可解释性越高

### 关键发现

1. **$L_2$ 对抗训练未能增加高层不变性的可解释性**：虽然 MEI 和像素级不变图像的人类识别率很高，但深层不变图像的可解释性差距在逐步缩小
2. **$L_{\infty}$ 鲁棒化效果不同**：其不变图像在所有观察者网络上的可解释性保持稳定甚至随深度增加
3. **ViT 的不变性呈现不同模式**：中层和深层的不变图像非常相似且可解释性高，符合 ViT 学习更全局化特征的观点
4. **对表征子采样鲁棒**：即使只用中间层少量神经元（类比神经科学实验的稀疏记录），SnS 仍能产生有效的不变图像
5. **不变性流形的内在维度**：低层最低、中层最高、深层次之，与已知的 CNN 表征维度趋势一致

## 亮点与洞察

1. **统一框架**：不变性和对抗攻击在同一双目标优化中通过交换拉伸/压缩方向实现，概念上非常优雅
2. **无梯度 = 真正模型无关**：可以直接应用于生物神经元，这是其他方法无法做到的
3. **超越 Metamers**：SnS 推向不变性流形的边界，而 metamers 只在目标图像附近探索；两者互补
4. **分层不变性揭示视觉系统本质**：从亮度→纹理→姿态的渐进不变性，本质上是特征选择性和特征不变性的同一过程的两面
5. **$L_2$ vs $L_{\infty}$ 鲁棒化的差异**为理解对抗训练的本质提供了新视角

## 局限性

1. 依赖预训练生成模型的表达能力——生成模型的先验限制了可探索的图像空间
2. 进化算法的计算成本较高，每个单元需要大量迭代
3. 仅在 ResNet50/ResNet18/VGG16/ViT 上验证，更大规模模型（如 DINO、ViT-22B）未探索
4. 尚未在真实生物神经元上进行闭环验证
5. 生成的不变图像描述过于丰富复杂，难以用简短文本概括

## 相关工作与启发

- 与 XDREAM（Ponce et al., 2019）共享进化优化思路，但扩展到不变性探测
- 与 Feather et al. (2023) 的 metamers 工作互补：metamers 在近距离探索，SnS 在远距离探索
- 对抗训练的视觉系统（robustified CNNs）与人类视觉的对齐在 MEI 层面良好，但在深层不变性层面出现分歧——这为未来的对齐方法提供了新的优化目标
- 启发：可以用 SnS 产生的"好/坏"不变图像来改进训练数据，使网络学习更人类化的不变性

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — 首个系统性的无梯度不变性探测框架，双目标优化设计优雅
- **实验充分性**: ⭐⭐⭐⭐ — 覆盖标准/鲁棒 CNN、ViT、人类实验，分析全面
- **实用性**: ⭐⭐⭐⭐ — 对计算神经科学有直接价值，对 AI 可解释性研究启发性强
- **写作质量**: ⭐⭐⭐⭐ — 方法描述清晰，图表质量高
- **综合评分**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

## 相关论文

- [Hidden Breakthroughs in Language Model Training](hidden_breakthroughs_in_language_model_training.md)
- [Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer](towards_understanding_subliminal_learning_when_and_how_hidden_biases_transfer.md)
- [How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Understanding](how_do_transformers_learn_to_associate_tokens_gradient_leading_terms_bring_mecha.md)
- [Beyond Linear Probes: Dynamic Safety Monitoring for Language Models](beyond_linear_probes_dynamic_safety_monitoring_for_language_models.md)
- [Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts](exploring_interpretability_for_visual_prompt_tuning_with_cross-layer_concepts.md)

<!-- RELATED:END -->
