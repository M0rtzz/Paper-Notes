---
title: >-
  [论文解读] C3A: 基于循环卷积的参数高效微调
description: >-
  [ACL 2025][模型压缩][PEFT] 提出Circular Convolution Adaptation (C3A)，用循环卷积算子替代LoRA的低秩矩阵分解来构造增量权重$\Delta W$——循环矩阵的秩与可训练参数数量解耦，实现"少参数+高秩"适配；同时利用FFT加速前向/反向传播，在计算和显存效率上均可与LoRA媲美。在GLUE、常识推理、数学推理、代码生成等任务上持续优于LoRA及其变体。
tags:
  - ACL 2025
  - 模型压缩
  - PEFT
  - LoRA
  - 循环卷积
  - 循环矩阵
  - FFT
  - 高秩适配
  - 块循环矩阵
---

# C3A: 基于循环卷积的参数高效微调

**会议**: ACL 2025  
**arXiv**: [2407.19342](https://arxiv.org/abs/2407.19342)  
**代码**: [Hugging Face PEFT](https://github.com/huggingface/peft) (已集成)  
**领域**: 模型压缩 / 参数高效微调  
**关键词**: PEFT, LoRA, 循环卷积, 循环矩阵, FFT, 高秩适配, 块循环矩阵  

## 一句话总结
提出Circular Convolution Adaptation (C3A)，用循环卷积算子替代LoRA的低秩矩阵分解来构造增量权重$\Delta W$——循环矩阵的秩与可训练参数数量解耦，实现"少参数+高秩"适配；同时利用FFT加速前向/反向传播，在计算和显存效率上均可与LoRA媲美。在GLUE、常识推理、数学推理、代码生成等任务上持续优于LoRA及其变体。

## 核心问题与动机
1. **LoRA的低秩瓶颈**：LoRA将增量权重表示为$\Delta W = BA$，其中$B \in \mathbb{R}^{d_1 \times r}$，$A \in \mathbb{R}^{r \times d_2}$。秩$r$同时决定了可训练参数数$r(d_1+d_2)$和$\Delta W$的秩上界——小$r$虽然参数少，但表达能力受限于低秩约束
2. **现有高秩方法的效率问题**：VeRA通过随机投影矩阵实现高秩适配，参数少但需要存储巨大的随机矩阵$r_v(d_1+d_2)$，显存开销远超LoRA；BOFT用蝴蝶因子化正交矩阵，在LLaMA-8B上直接OOM
3. **性能与效率的两难**：现有PEFT方法要么参数少但秩低(LoRA)，要么秩高但计算/显存昂贵(VeRA/BOFT/DoRA)——缺乏真正平衡性能和效率的方案
4. **Dense线性层缺乏归纳偏置**：Transformer的线性层没有CNN那样的结构先验，在下游微调数据有限时优化困难——需要一种合适的归纳偏置作为正则化
5. **秩与参数量的耦合是根本障碍**：LoRA中$r$同时控制两者，导致无法独立调节适配容量和参数预算
6. **FFT的成熟生态提供了机会**：循环矩阵可被Fourier基对角化，GPU上的cuFFT已高度优化——循环卷积在理论和工程上都具备替代矩阵乘法的条件

## 关键方法

### 1. 循环卷积适配 (Circular Convolution Adaptation)
**核心思想**：用长度为$d$的卷积核$\Delta w$替代LoRA的两个低秩矩阵$A$和$B$。

增量输出通过循环卷积计算：$\Delta z = \Delta w \star x = \mathcal{C}(\Delta w) x$

其中$\mathcal{C}(\Delta w)$是由$\Delta w$构造的循环矩阵——第一行为$\Delta w$，后续每行循环右移一位。

**秩与参数解耦**：循环矩阵$\mathcal{C}(\Delta w)$的秩等于$d - \text{Deg}(\gcd(f(x), x^d - 1))$，理论上界为$d$（满秩）。关键在于：**只需$d$个参数就能构造秩最高为$d$的矩阵**，而LoRA需要$r(d_1+d_2)$个参数才能达到秩$r$。

**FFT加速**：利用循环矩阵可被Fourier基对角化的性质：
$$\Delta w \star x = \text{FFT}(\text{FFT}(\Delta w) \circ \text{iFFT}(x))$$
前向和反向传播都可用FFT实现，时间复杂度$O(d\log d)$——远优于矩阵乘法的$O(d^2)$。

反向传播同样优雅：利用循环卷积的交换律$\mathcal{C}(\Delta w)x = \mathcal{C}(x)\Delta w$，梯度计算也化为循环卷积→FFT。

### 2. 块循环卷积扩展 (Block-Circular Convolution)
解决基础循环卷积的两个限制：① 仅适用方阵 ② 参数量固定。

将$x$和$\Delta z$分成大小为$b$的块，每对块分配独立卷积核：
$$\Delta z_i = \sum_{j=1}^{d_2/b} \Delta w_{ij} \star x_j$$

等价于块循环矩阵$\mathcal{C}_{\text{blk}}(\Delta w)$，总参数量$d_1 d_2 / b$。

- $b$是$d_1$和$d_2$的公约数，控制参数量（类似LoRA中$r$的角色）
- 但**$b$仅控制参数量，不约束矩阵秩**——这是与LoRA的本质区别
- 通常取$b = \gcd(d_1, d_2)$达到最大压缩比

### 3. 复杂度分析

| 方法 | 时间复杂度 | 可训练参数 | 辅助存储 |
|------|-----------|-----------|---------|
| LoRA | $O(r(d_1+d_2))$ | $r(d_1+d_2)$ | 0 |
| VeRA | $O(r_v(d_1+d_2))$ | $r_v+d_1$ | $r_v(d_1+d_2)$ |
| C3A | $O(\frac{d_1+d_2}{p}\log b + \frac{d_1 d_2}{b})$ | $\frac{d_1 d_2}{b}$ | $pb$ |

C3A的辅助存储仅$pb$（FFT工作空间），远小于VeRA的随机矩阵。

## 实验结果

### 表1: GLUE基准 (RoBERTa-Large)

| 方法 | 参数量 | 显存 | CoLA | QNLI | RTE | Avg. |
|------|--------|------|------|------|-----|------|
| Full FT | 354M | 43.40G | 64.87 | 92.40 | 84.48 | 86.25 |
| LoRA (r=8) | 0.786M | 34.12G | 65.16 | 93.73 | 83.75 | 85.96 |
| VeRA (r=256) | 0.061M | 34.16G | 63.66 | 94.11 | 83.03 | 85.91 |
| BOFT | 0.442M | 34.98G | 64.72 | 93.89 | 82.82 | 86.08 |
| **C3A (b=1024/8)** | **0.393M** | **31.79G** | **67.18** | **94.26** | **84.62** | **86.96** |

C3A在更少参数和更低显存下取得最高平均分(86.96 vs 86.25全参微调)。

### 表2: LLaMA3-8B 指令微调 (常识推理 + 数学 + 代码)

| 方法 | 参数占比 | 显存 | 常识Avg. | 数学Avg. | 代码Avg. |
|------|---------|------|---------|---------|---------|
| LoRA (r=32) | 0.70% | 51.18G | 83.9 | 52.7 | 57.6 |
| VeRA (r=16384) | 0.04% | 66.03G | 82.5 | 51.4 | 56.2 |
| DoRA (r=32) | 0.71% | 63.37G | 85.3 | 53.6 | 58.4 |
| BOFT | - | OOM | - | - | - |
| **C3A (b=4096/32)** | **0.26%** | **56.08G** | **85.9** | **54.1** | **58.7** |

在LLaMA3-8B上，C3A用LoRA 37%的参数量取得全面领先：常识推理+2.0、数学+1.4、代码+1.1。BOFT在H800 80GB上OOM。

## 亮点 / 我学到了什么
- **循环矩阵实现"参数-秩解耦"是核心洞察**：LoRA的低秩约束是结构性的而非必要的——C3A说明PEFT不必在参数量和表达能力间做tradeoff
- **归纳偏置在数据有限时是优势而非劣势**：循环模式（circulant pattern）提供的结构先验在微调场景下反而有利于优化，与预训练阶段形成有趣对比
- **FFT使理论优势转为实际效率**：循环矩阵对角化→FFT→前向反向都是$O(d\log d)$——数学优雅且工程实用
- **已集成到HuggingFace PEFT**：说明实用性和通用性已得到社区验证
- **对初始化不敏感**：消融实验表明zero/Gaussian/Kaiming/Xavier初始化对C3A影响都在标准差范围内，远优于LoRA对$A$/$B$初始化的敏感性
- **合成实验可视化效果直观**：LoRA(r=1)无法分类8簇数据，同参数量的C3A完美分类——清晰展示表达能力差异

## 局限与展望
- **$b$必须是$d_1$和$d_2$的公约数**：对于某些Transformer架构（维度互素或维度特殊），可能无法找到合适的$b$值，限制了通用性
- **FFT在边缘设备上的支持不成熟**：cuFFT在NVIDIA GPU上高度优化，但在移动端/边缘芯片上可能缺乏高效实现
- **循环模式的归纳偏置是否对所有任务有益尚不明确**：在某些需要高度非结构化适配的任务上，循环约束可能反而成为瓶颈
- **未与更多最新方法对比**：如GaLore、LoRA+、rsLoRA等2024年新方法
- **块大小$b$的选择策略**：论文建议取$\gcd(d_1, d_2)$但未深入分析不同$b$值的trade-off规律
- **可探索方向**：与LoRA的正交组合（循环+低秩混合）、自适应$b$选择、扩展到Mamba等非Transformer架构

## 相关工作
- **LoRA系列**: LoRA (Hu et al., ICLR 2022), DoRA (Liu et al., 2024), LoRA+ (Hayou et al., 2024), rsLoRA
- **其他PEFT**: VeRA (Kopiczko et al., 2023), BOFT (Liu et al., 2023), BitFit (Zaken et al., 2021), (IA)³ (Liu et al., 2022)
- **Adapter/Prompt**: Prefix-Tuning (Li & Liang, 2021), Prompt Tuning (Brown et al., 2020), Visual Prompting (Chen et al., 2023)
- **循环卷积/矩阵**: CirCNN (Ding et al., MICRO 2017), 信号处理中的循环卷积 (Rabiner et al., 1978), AES加密 (Dworkin et al., 2001)
- **Transformer压缩**: 循环投影 (Cheng et al., ICCV 2015), 结构化剪枝, 知识蒸馏

## 评分

| 维度 | 分数 | 说明 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐⭐ | 循环卷积引入PEFT，参数-秩解耦是根本性创新 |
| 技术深度 | ⭐⭐⭐⭐☆ | 数学推导完整(对角化+FFT+反向传播)，块扩展自然 |
| 实验充分性 | ⭐⭐⭐⭐⭐ | GLUE+LLM指令微调+8常识+2数学+4代码+消融+初始化+缩放实验 |
| 写作质量 | ⭐⭐⭐⭐☆ | 结构清晰，复杂度分析全面，但部分符号冗余 |
| 实用价值 | ⭐⭐⭐⭐⭐ | 已集成HuggingFace PEFT，即插即用 |
| **总分** | **9.0/10** | PEFT领域里程碑式工作，真正解决了秩-参数耦合问题 |

<!-- RELATED:START -->

## 相关论文

- [TableLoRA: Low-rank Adaptation on Table Structure Understanding for Large Language Models](table_lora_structure_understanding.md)
- [Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis](quaff_quantized_peft.md)
- [FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Large Language Models](fedex_lora_federated_exact_aggregation.md)
- [Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching](flipping_kd_small_to_large.md)
- [BeamLoRA: Beam-Constraint Low-Rank Adaptation](beamlora_beam_constraint_lora.md)

<!-- RELATED:END -->
