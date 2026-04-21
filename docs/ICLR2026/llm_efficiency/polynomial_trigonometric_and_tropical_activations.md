---
title: >-
  [论文解读] Polynomial, trigonometric, and tropical activations
description: >-
  [ICLR 2026][LLM效率][激活函数] 系统探索基于正交基（Hermite多项式、Fourier三角基）和热带化（tropicalization）的可学习激活函数族，通过方差保持初始化解决多项式激活的梯度爆炸/消失问题，在GPT-2和ConvNeXt上成功替代GELU实现有效训练。
tags:
  - ICLR 2026
  - LLM效率
  - 激活函数
  - Hermite多项式
  - Fourier三角基
  - Tropical多项式
  - 方差保持初始化
---

# Polynomial, trigonometric, and tropical activations

**会议**: ICLR 2026  
**arXiv**: [2502.01247](https://arxiv.org/abs/2502.01247)  
**代码**: [K-H-Ismail/torchortho](https://github.com/K-H-Ismail/torchortho)  
**领域**: LLM Efficiency / Neural Network Architecture  
**关键词**: 激活函数, Hermite多项式, Fourier三角基, Tropical多项式, 方差保持初始化

## 一句话总结
系统探索基于正交基（Hermite多项式、Fourier三角基）和热带化（tropicalization）的可学习激活函数族，通过方差保持初始化解决多项式激活的梯度爆炸/消失问题，在GPT-2和ConvNeXt上成功替代GELU实现有效训练。

## 研究背景与动机

激活函数是深度神经网络的核心组件，它引入了非线性使网络能够逼近复杂函数。自ReLU、GELU、SwiGLU等激活函数提出以来，绝大多数现代深度学习模型都采用固定形式的激活函数。然而，一个自然的问题是：**哪些函数可以用作深度神经网络的激活函数？** 能否使用更具表达力的函数族（如多项式、三角函数）作为可学习的激活函数？

**现有痛点**：多项式激活函数虽然理论上具有强大的逼近能力，但在实践中面临严重的梯度爆炸和激活值爆炸问题。即使是低阶多项式，在深度网络中也会因层层复合而导致数值不稳定。这一问题长期限制了多项式类激活函数的实际应用。

**核心矛盾**：多项式/三角函数族具有丰富的数学结构和强大的表达能力，但直接在深度网络中使用会导致训练不稳定；而标准激活函数（ReLU、GELU）虽然稳定，但函数形式固定、缺乏可学习的自适应能力。

**切入角度**：本文基于正交基的数学性质，设计了方差保持的初始化方案，使得基于正交基的可学习激活函数能够在深度网络中稳定训练，无需额外的截断（clamping）机制。

**核心idea**：通过选择合适的正交基（Hermite多项式基、Fourier三角基）和设计方差保持初始化，可以将可学习的多项式和三角函数用作深度网络的激活函数，同时保持训练稳定性。

## 方法详解

### 整体框架
本文提出了三类可学习激活函数族：
1. 输入为标量激活值 $x$，输出也是标量
2. 每个激活函数由一组可学习系数参数化
3. 激活函数可以直接替换网络中的GELU/ReLU等标准激活
4. 通过方差保持初始化确保前向传播的方差稳定性

### 关键设计

1. **Hermite多项式激活函数**：基于概率论中的Hermite正交多项式基构建可学习激活函数。Hermite多项式在高斯测度下正交，这使得当输入服从正态分布时，各基函数的贡献相互独立。激活函数形式为 $\sigma(x) = \sum_{k=0}^{d} c_k H_k(x)$，其中 $H_k$ 是第 $k$ 阶Hermite多项式，$c_k$ 是可学习系数。方差保持初始化要求 $\sum_k c_k^2 = 1$（利用Hermite多项式的正交性），确保输出方差等于输入方差。

2. **Fourier三角激活函数**：基于Fourier级数的三角基函数构建可学习激活函数。形式为正弦和余弦函数的线性组合 $\sigma(x) = a_0 + \sum_{k=1}^{d} (a_k \cos(kx) + b_k \sin(kx))$。Fourier基在周期函数空间中是完备的正交基，特别适合捕捉数据中的周期性结构。通过类似的方差保持初始化确保训练稳定性。

3. **Tropical多项式激活函数**：通过"热带化"操作将标准多项式转换为tropical多项式。在tropical代数中，加法被max操作替代，乘法被普通加法替代。因此tropical多项式本质上是一系列仿射函数的逐点最大值，形成分段线性函数。这可以看作是对ReLU的自然推广（ReLU就是最简单的tropical多项式 $\max(0, x)$）。此外还引入了tropical有理函数，进一步扩展表达能力到非凸函数。

4. **方差保持初始化**：这是本文的核心技术贡献。对于多项式激活，层层复合会导致激活值和梯度的方差呈指数增长或衰减。作者利用正交基的数学性质，推导出在标准正态输入假设下保持方差的初始化条件。这使得深度网络（如GPT-2的12层Transformer）能够稳定训练，无需截断或梯度裁剪等额外手段。

5. **Hermite插值迁移**：一个实用的贡献是展示了如何用Hermite插值将预训练模型中的标准激活函数（如GELU）转换为可学习激活。通过同时匹配函数值和导数值，新的可学习激活在初始化时与原始激活相近，使得微调过程更加稳定。这对于将可学习激活应用于预训练模型的微调场景非常有价值。

6. **网络的多项式解释**：一个理论洞察是，使用多项式激活的网络可以被解释为多元多项式映射。这为理解网络的函数逼近行为提供了新的视角，也为网络分析提供了代数几何的工具。

### 损失函数 / 训练策略
训练策略与标准模型训练一致：
- GPT-2语言建模：在OpenWebText上进行next-token prediction，使用交叉熵损失
- ConvNeXt图像分类：在ImageNet-1K上进行分类训练，使用标准分类损失
- 激活函数的系数作为额外的可学习参数参与端到端的梯度下降优化

## 实验关键数据

### 主实验

| 模型/任务 | 指标 | GELU基线 | Hermite | Fourier | Tropical | 说明 |
|-----------|------|----------|---------|---------|----------|------|
| GPT-2 / OpenWebText | Perplexity | 基线值 | 降低 | 降低 | 可比 | 可学习激活改善语言建模 |
| ConvNeXt-T / ImageNet | Top-1 Acc | 基线值 | 提升 | 提升 | 可比 | 视觉任务同样有效 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无方差保持初始化 | 训练崩溃 | 证明初始化是必要条件 |
| 不同多项式阶数 | 性能vs稳定性 | 阶数越高越有效但需更谨慎初始化 |
| Hermite插值微调 vs 从头训练 | 微调效率 | 插值初始化显著加速收敛 |

### 关键发现
- 通过方差保持初始化，多项式和三角激活函数可以成功训练GPT-2（12层Transformer）和ConvNeXt等深度模型
- 可学习激活在语言建模（perplexity）和图像分类（accuracy）上均匹配甚至超越固定的GELU激活
- Tropical多项式提供了ReLU到更复杂分段线性函数的自然推广
- Hermite插值使得将可学习激活引入预训练模型变得可行
- 方差保持初始化是成功的关键——没有它，多项式激活在几层之后就会数值溢出

## 亮点与洞察
- **理论优美**：将正交基理论与神经网络激活函数设计完美结合，数学推导严谨
- **实用性好**：提供了torchortho库，可直接替换PyTorch中的标准激活函数
- **覆盖全面**：同时探索了多项式（Hermite）、三角（Fourier）和tropical三个函数族，给出了统一的分析框架
- **迁移学习友好**：Hermite插值方法使得可学习激活可以无缝接入已有的预训练模型
- **理解深刻**：揭示了多项式激活网络作为多元多项式映射的代数结构，打开了利用代数几何分析神经网络的大门
- **效率潜力**：可学习激活可能通过自适应调整激活形式来提升大规模模型的训练效率

## 局限与展望
- 实验规模相对有限：仅在GPT-2（124M）和ConvNeXt-T上验证，未扩展到更大规模模型（如GPT-3、LLaMA等）
- 可学习激活引入了额外的参数（系数），在超大规模模型中可能增加内存开销
- Tropical激活虽然理论有趣，但实际性能提升相比Hermite和Fourier不够显著
- 未充分探索可学习激活在训练过程中的演化动态——激活形状如何变化？不同层是否学到不同形式？
- 方差保持初始化依赖于输入为高斯分布的假设，实际网络中间层的激活值分布可能偏离此假设
- 缺乏与KAN（Kolmogorov-Arnold Network）等同期的可学习激活方法的直接对比

## 相关工作与启发
- **KAN（Kolmogorov-Arnold Networks）**：另一种基于可学习激活函数的网络设计，使用B样条基函数
- **SwiGLU、GeGLU**：门控线性单元类激活，在LLM中广泛使用
- **Maxout Networks**：最早探索分段线性激活的工作，Tropical激活可看作其推广
- **Mish、Swish**：自动搜索得到的激活函数，与本文的可学习方法形成对比
- 启发：可学习激活函数可能在需要特定函数逼近性质的场景（如科学计算、物理信息网络）中特别有价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ （系统性探索全新的激活函数族）
- 实验充分度: ⭐⭐⭐ （规模有限但验证充分）
- 写作质量: ⭐⭐⭐⭐ （数学推导清晰）
- 价值: ⭐⭐⭐⭐ （为激活函数设计开辟新方向，有开源代码支持）

<!-- RELATED:START -->

## 相关论文

- [Rethinking Benign Relearning: Syntax as the Hidden Driver of Unlearning Failures](rethinking_benign_relearning_syntax_as_the_hidden_driver_of_the_safety_tax.md)
- [Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models](understanding_and_improving_length_generalization_in_hierarchical_sparse_attenti.md)
- [Expert Divergence Learning for MoE-based Language Models](expert_divergence_learning_for_moe-based_language_models.md)
- [Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling](semantic_parallelism_redefining_efficient_moe_inference_via_model-data_co-schedu.md)
- [LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding](lycheedecode_accelerating_long-context_llm_inference_via_hybrid-head_sparse_deco.md)

<!-- RELATED:END -->
