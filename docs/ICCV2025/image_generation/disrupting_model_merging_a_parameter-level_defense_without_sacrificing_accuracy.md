---
title: >-
  [论文解读] Disrupting Model Merging: A Parameter-Level Defense Without Sacrificing Accuracy
description: >-
  [ICCV 2025][图像生成][模型合并防御] 提出 PaRaMS（Parameter Rearrangement & Random Multi-head Scaling），一种参数级主动防御方法，通过功能等价的参数变换将模型推离共享损失盆地，使得被保护模型在合并时性能严重退化，同时保持未合并时的原始性能。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "模型合并防御"
  - "知识产权保护"
  - "参数重排列"
  - "注意力头缩放"
  - "功能等价变换"
---

# Disrupting Model Merging: A Parameter-Level Defense Without Sacrificing Accuracy

**会议**: ICCV 2025  
**arXiv**: [2503.07661](https://arxiv.org/abs/2503.07661)  
**代码**: 无  
**领域**: 扩散模型/图像生成  
**关键词**: 模型合并防御, 知识产权保护, 参数重排列, 注意力头缩放, 功能等价变换

## 一句话总结

提出 PaRaMS（Parameter Rearrangement & Random Multi-head Scaling），一种参数级主动防御方法，通过功能等价的参数变换将模型推离共享损失盆地，使得被保护模型在合并时性能严重退化，同时保持未合并时的原始性能。

## 研究背景与动机

预训练-微调范式使得模型合并（Model Merging）成为一种低成本获取他人专业能力的途径。通过简单的参数线性组合（如 Task Arithmetic），一个"搭便车者"（free-rider）可以将开源的微调模型合并到自己的模型中，几乎零成本地继承原模型的专业能力。这带来了严重的知识产权（IP）问题：

**被动防御的局限性**：现有的水印和指纹技术只能在合并后检测侵权，无法阻止合并本身

**合并操作的隐蔽性**：与代码复制不同，模型合并混合了参数空间，难以追溯来源

**IP 侵权成本极低**：Hugging Face 上已有近 3 万个合并模型，合并已成为普遍做法

因此，核心研究问题是：**如何在不改变模型功能的前提下，主动使模型在被合并时性能退化？** 这需要同时满足两个条件：(1) 合并前功能等价；(2) 合并后性能大幅下降。

本文的关键洞察来自损失地形（Loss Landscape）分析：从同一预训练检查点微调的模型通常落在共享的低损失盆地中，这是模型合并成功的根本原因。如果通过功能等价变换将模型推到远离该盆地的另一个盆地，合并就会失败。

## 方法详解

### 整体框架

PaRaMS 由两个互补的参数变换模块组成，分别针对 Transformer 架构中的两个核心组件：MLP 层和 Attention 层。两个变换都保证功能等价性（函数输出完全不变），同时使参数在空间中远离原始位置，从而破坏合并。最终防御为两者的组合：$\eta = \eta_{\text{perm}} \circ \eta_{\text{scaling}}$。

### 关键设计

1. **MLP 参数重排列（Parameter Rearrangement）**:

    - 功能：通过置换矩阵对 MLP 层的隐藏层神经元进行重新排序
    - 核心思路：对于两层 MLP $\text{MLP}(X) = W_2 \sigma(W_1 X + b_1) + b_2$，引入置换矩阵 $P$，令：
    $W_1' = PW_1, \quad b_1' = Pb_1, \quad W_2' = W_2 P^T$
      由于 $\sigma$ 是逐元素激活函数，$P$ 和 $P^T$ 的效果相互抵消，输出完全不变。
    - 设计动机：选择使参数距预训练模型最远的置换矩阵：
    $\arg\max_{\eta_{\text{perm}}} \|\theta_{\text{pre}}^{\text{MLP}} - \eta_{\text{perm}}(\theta_{\text{def}}^{\text{MLP}})\|^2$
      该优化可被转化为线性分配问题（Linear Assignment Problem）高效求解。

2. **随机多头缩放（Random Multi-head Scaling）**:

    - 功能：对 Attention 模块的 Q/K/V 矩阵施加对角缩放变换
    - 核心思路：对每个注意力头 $i$，采样对角矩阵 $A_i, B_i$，对角元素从 $\mathcal{U}(s_{\min}, s_{\max})$ 采样，然后：
    $Q_i \leftarrow Q_i A_i, \quad K_i \leftarrow K_i A_i^{-1}$
    $V_i \leftarrow V_i B_i, \quad W_O[:,i] \leftarrow B_i^{-1} W_O[:,i]$
      利用 $QK^T = QA(KA^{-1})^T$ 的恒等式保证功能不变。
    - 设计动机：随机缩放使注意力层参数大幅偏移，与 MLP 重排列互补，共同覆盖 Transformer 的所有关键模块。

3. **Dropout 剪枝防御（应对自适应攻击）**:

    - 功能：针对可能的自适应绕过攻击（如搜索逆置换），提出基于 dropout 的剪枝增强鲁棒性
    - 核心思路：在重排列前对部分参数施加 dropout，使攻击者即使恢复了排列顺序也无法完全还原参数
    - 设计动机：考虑了敌手可能尝试的各种反制手段，确保防御在对抗性场景下仍然有效

### 损失函数 / 训练策略

PaRaMS 是一种**后处理**方法，不涉及重新训练。防御过程仅需：
- 对模型的每个 MLP 层求解最优置换矩阵（线性规划）
- 对每个注意力头采样随机缩放因子
- 一次性变换参数即可，计算开销极小

## 实验关键数据

### 主实验

图像分类任务（ViT-B-32, Task Arithmetic 合并, $\lambda=0.8$）：

| 设置 | MMP-准确率(%) | MMP+准确率(%) | 下降幅度 |
|------|-------------|-------------|---------|
| Cars/RESISC45 | 70.29/94.24 | 0.51/2.13 | >65% |
| EuroSAT/SVHN | 98.06/95.90 | 9.67/19.41 | >75% |
| GTSRB/DTD | 98.20/67.82 | 1.93/2.66 | >65% |
| MNIST/RESISC45 | 99.60/90.62 | 2.10/10.10 | >80% |

图像生成任务（Stable Diffusion 1.5, 文本-图像对齐度）：

| 模型 | UMP- | UMP+ | MMP- | MMP+ |
|------|------|------|------|------|
| Prompt1 | 0.3286 | 0.3306 | 0.3416 | **0.1277** |
| Prompt2 | 0.3335 | 0.3428 | 0.3386 | **0.0820** |

### 消融实验

不同合并方法下的防御效果（ViT-B-32 平均准确率）：

| 合并方法 | MMP- | MMP+ | 降幅 |
|---------|------|------|------|
| Task Arithmetic | ~75% | <10% | >65% |
| TIES-Merging | ~70% | <10% | >60% |
| Weight Average | ~65% | <15% | >50% |
| AdaMerging | ~75% | <10% | >65% |
| TA + DARE | ~70% | <10% | >60% |

文本分类任务（Llama2）：

| 数据集 | UMP-/UMP+ | MMP-(TA) | MMP+(TA) |
|--------|-----------|----------|----------|
| Emotion | 99.8 | 97.6 | **21.4** |
| Twitter | 99.7 | 95.3 | **35.8** |

### 关键发现

- PaRaMS 在**所有**合并方法（TA、TIES、WA、ADA 及其 DARE 变体）上都能有效破坏合并，MMP+ 准确率通常降至 10% 以下
- UMP- 和 UMP+ 性能**完全一致**，验证了功能等价性
- 防御在合并 2-7 个模型的场景下均有效
- 跨任务（分类、生成、NLP）和跨架构（ViT、SD、Llama2）均有效

## 亮点与洞察

- **首个主动防御方案**：从被动检测转向主动预防，填补了模型合并安全的重要空白
- **功能等价性保证**：数学上严格证明变换不改变模型输出，不需要任何近似
- **基于损失地形的理论分析**：从"共享盆地"的视角优雅地解释了为什么合并能成功以及如何破坏它
- **多模态验证**：在视觉、语言、生成任务上均验证了方法的广泛适用性

## 局限与展望

- 攻击者如果拥有大量计算资源（如进行知识蒸馏而非参数合并），防御可能失效
- 对于仅合并 LoRA 参数的 PEM-TA 场景，由于只能使用缩放变换，防御效果相对较弱
- 随机缩放的范围 $[s_{\min}, s_{\max}]$ 的选择对防御效果有影响，寻找最优范围是一个开放问题
- 未探讨多个防御模型之间合并的场景

## 相关工作与启发

- 与模型水印（IPR 保护）技术互补：水印检测侵权，PaRaMS 预防侵权
- 神经网络的置换不变性（permutation symmetry）是一个被长期研究的话题，本文巧妙地将其用于安全防御
- 启示：模型合并的成功依赖于参数空间中的"共享结构"，破坏这种结构即可破坏合并

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次提出主动防御模型合并的概念，问题定义和解决思路都很新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 跨任务、跨架构、跨合并方法的全面验证，还分析了自适应攻击
- 写作质量: ⭐⭐⭐⭐ 问题形式化清晰，威胁模型明确
- 价值: ⭐⭐⭐⭐ 在开源模型安全领域具有重要实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Efficient Input-Level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation](efficient_input-level_backdoor_defense_on_text-to-image_synthesis_via_neuron_act.md)
- [\[ICCV 2025\] Omegance: A Single Parameter for Various Granularities in Diffusion-Based Synthesis](omegance_a_single_parameter_for_various_granularities_in_diffusion-based_synthes.md)
- [\[ICCV 2025\] Towards Robust Defense against Customization via Protective Perturbation Resistant to Diffusion-based Purification](towards_robust_defense_against_customization_via_protective_perturbation_resista.md)
- [\[ICCV 2025\] DCT-Shield: A Robust Frequency Domain Defense against Malicious Image Editing](dct-shield_a_robust_frequency_domain_defense_against_malicious_image_editing.md)
- [\[CVPR 2025\] Efficient Personalization of Quantized Diffusion Model without Backpropagation (ZOODiP)](../../CVPR2025/image_generation/efficient_personalization_of_quantized_diffusion_model_without_backpropagation.md)

</div>

<!-- RELATED:END -->
