---
title: >-
  [论文解读] IterIS: Iterative Inference-Solving Alignment for LoRA Merging
description: >-
  [CVPR 2025][图像生成][LoRA合并] IterIS提出了一种迭代推理-求解的LoRA合并方法，通过直接提取统一适配器的输入特征（而非近似）来建立更准确的优化目标，配合正则化减少样本需求至先前方法的1-5%，并引入自适应权重平衡优化，在文本到图像扩散模型、视觉语言模型和大语言模型的LoRA合并中显著超越基线。
tags:
  - CVPR 2025
  - 图像生成
  - LoRA合并
  - 多任务模型
  - 迭代优化
  - 参数高效微调
  - 多概念定制
---

# IterIS: Iterative Inference-Solving Alignment for LoRA Merging

**会议**: CVPR 2025  
**arXiv**: [2411.15231](https://arxiv.org/abs/2411.15231)  
**代码**: https://github.com/HKUST-LongGroup/IterIS-merging  
**领域**: 扩散模型  
**关键词**: LoRA合并, 多任务模型, 迭代优化, 参数高效微调, 多概念定制

## 一句话总结
IterIS提出了一种迭代推理-求解的LoRA合并方法，通过直接提取统一适配器的输入特征（而非近似）来建立更准确的优化目标，配合正则化减少样本需求至先前方法的1-5%，并引入自适应权重平衡优化，在文本到图像扩散模型、视觉语言模型和大语言模型的LoRA合并中显著超越基线。

## 研究背景与动机

1. **领域现状**：LoRA是目前最流行的参数高效微调方法，常用于为不同下游任务分别训练任务特定的LoRA。当需要多任务能力时，LoRA合并可以在不访问训练数据的情况下将多个LoRA组合为一个统一适配器。

2. **现有痛点**：现有基于真实分布的LoRA合并方法（如RegMean、Mix-of-Show）存在三个关键限制：(1) **粗糙假设**：假设统一适配器的输入特征与各任务特定LoRA的输入特征相同，但随网络深度增加这个假设偏差越来越大；(2) **海量样本需求**：为确保内积矩阵可逆和增强鲁棒性，通常需要大量无标注样本；(3) **不平衡优化**：不同任务的特征量级差异导致优化目标中某些项主导解的方向。

3. **核心矛盾**：真实分布方法建立优化目标时用了一个"偷懒"的近似——用各LoRA的输入特征代替统一适配器的输入特征。在浅层这个近似尚可，但随深度增加cumulative error导致性能严重退化。

4. **本文目标** (1) 消除粗糙假设——使用统一适配器的真实输入特征；(2) 减少样本需求；(3) 平衡多任务优化。

5. **切入角度**：深度学习模型是有向无环图(DAG)结构——如果从浅层到深层逐层合并，则前面层已合并后，后面层的统一适配器输入特征可以通过推理直接获得。

6. **核心 idea**：迭代地"推理获取统一适配器真实输入特征→建立准确优化目标→求解更优统一适配器"，逐步逼近最优解。

## 方法详解

### 整体框架
输入为N个任务特定LoRA-tuned模型和少量无标注样本。输出为一个统一模型（预训练模型+合并后的适配器）。工作流程：初始化（用各LoRA输入特征作为 $\tilde{X}_{nj}$ 的初始值）→ 迭代（逐层求解闭式解更新适配器 → 推理获取新的输入特征 → 更新优化目标）→ 收敛输出。

### 关键设计

1. **迭代推理-求解框架**:

    - 功能：逐步消除粗糙假设带来的近似误差
    - 核心思路：原始方法的优化目标为 $W^* = \arg\min_W \sum_i \|W_i^T X_i - W^T X_i\|_F^2$，其中右侧的 $X_i$ 应该是统一适配器的输入特征 $\tilde{X}_i$，但被近似为各LoRA的输入特征 $X_i$。IterIS的优化改为 $W^* = \arg\min_W \sum_i \lambda_i \|W_i^T X_i - W^T \tilde{X}_i\|_F^2$，闭式解为 $W^* = (\sum_i \lambda_i \tilde{X}_i \tilde{X}_i^T)^{-1}(\sum_i \lambda_i \tilde{X}_i X_i^T W_i)$。初始时 $\tilde{X}_i = X_i$，每轮迭代后用当前统一模型推理更新 $\tilde{X}_i$
    - 设计动机：利用DAG结构的特性，前面层合并后自然改变了后面层的输入分布。通过迭代逐步将近似的输入特征替换为真实的输入特征，理论上J-1次迭代（J层encoder）即可收敛

2. **高效正则化以减少样本需求**:

    - 功能：在仅需1-5%样本的情况下保持算法鲁棒性
    - 核心思路：对闭式解中的内积矩阵添加正则项：$\tilde{X}_i\tilde{X}_i^T + \alpha\|\tilde{X}_i\tilde{X}_i^T\|_F \cdot I$ 和 $\tilde{X}_i X_i^T + \alpha\|\tilde{X}_i X_i^T\|_F \cdot I$。正则化参数 $\alpha$ 通常设为 $10^{-4}$ 或更小。Frobenius范数缩放的单位矩阵使正则化强度自适应于矩阵的整体量级
    - 设计动机：原始方法需要大量样本来确保内积矩阵的数值稳定性和可逆性。正则化项使矩阵即使在少样本情况下也能保持良好条件数，同时避免过拟合

3. **自适应权重平衡**:

    - 功能：缓解不同任务间优化目标的不平衡问题
    - 核心思路：定义自适应权重 $\lambda_i = \|W_i\|_F^2 \cdot \|W_i^T X_i\|_F^{-2}$。直觉上，当某个任务的输出特征量级 $\|W_i^T X_i\|_F$ 较大时，反比降低其权重，防止该任务主导优化目标
    - 设计动机：均匀权重下，特征量级大的任务会主导解的方向，导致其他任务性能退化。自适应权重基于LoRA权重和输出特征的量级比例来平衡各项贡献

### 损失函数 / 训练策略
IterIS是无训练方法，核心是闭式解的逐层、逐迭代更新。每次迭代包括对所有样本的一次前向推理和J个层的闭式解计算。实验中迭代次数上限设为20。对于文本到图像扩散模型，使用50个输入样本用于推理。整个过程无需标注数据和梯度计算。

## 实验关键数据

### 主实验：多概念定制生成（7组概念对，Stable Diffusion v1.5）

| 指标 | Custom Diffusion | Textual Inversion | Linear Merging | IterIS |
|------|-----------------|-------------------|----------------|--------|
| Image-align1↑ | 0.6881 | 0.6569 | 0.6811 | **0.6889** |
| Image-align2↑ | 0.7091 | 0.6887 | 0.6963 | **0.7124** |
| Text-align↑ | 0.6509 | 0.6363 | 0.6547 | **0.6800** |

### 多风格caption生成（BLIP + SentiCap正/负情感）

| 方法 | Acc(pos,neg)↑ | CIDEr↑ | B-4↑ |
|------|--------------|--------|------|
| Linear merging | (0.522, 0.557) | 0.752 | 0.142 |
| RegMean | (0.624, 0.692) | 0.790 | 0.150 |
| **IterIS** | **(0.831, 0.781)** | **0.794** | **0.152** |

### 消融实验
- 粗糙假设分析：encoder深层的Frobenius距离显示RegMean的近似偏差随深度急剧增大，IterIS完全消除偏差
- 样本需求：IterIS仅需1-5%的无标注样本（RegMean相比），运行时间也相应减少
- 权重平衡：RegMean的优化项 $T_1$ 和 $T_2$ 量级悬殊，IterIS通过自适应权重使其比例均衡

### 关键发现
- 迭代通常在少于20步内收敛，归功于DAG结构的性质
- IterIS在多概念定制中甚至略超Custom Diffusion（后者使用梯度训练+数据正则化）
- 在NLP多任务合并中，IterIS同样显著超越RegMean等基线，证明方法的通用性
- 正则化项使得50个样本就足够（RegMean需要上千），大幅降低实际使用门槛

## 亮点与洞察
- **推理-求解的迭代范式**：这是一个非常干净的思路——不满足于近似，直接通过推理获取真实的输入特征来建立优化目标。利用了DAG结构的层次性，前面层解好了后面层的输入自然就准了
- **极少样本需求**：仅需1-5%的无标注样本，通过自适应正则化实现。这使得LoRA合并在数据受限场景（隐私保护、知识产权）中更加实用
- **通用性强**：同一个方法框架适用于扩散模型、VLM和LLM三个领域的LoRA合并，体现了方法论层面的清晰性

## 局限与展望
- 迭代过程中每轮需要对所有样本做一次前向推理，对于非常大的模型可能仍有计算开销
- 闭式解假设目标函数是二次型，可能对某些非线性层的LoRA合并不够精确
- 实验中迭代上限设为20以防过拟合，但最优迭代数可能因模型而异
- 自适应权重的设计基于经验，缺乏理论最优性保证
- 可改进方向：扩展到LoRA之外的PEFT方法合并（如adapter、prompt tuning），或结合gradient-based fine-tuning做混合优化

## 相关工作与启发
- **vs RegMean**: RegMean使用相同的优化框架但假设 $\tilde{X}_i = X_i$，IterIS通过迭代推理消除这个近似。在多风格caption中，IterIS的正/负情感准确率(0.831, 0.781)大幅超越RegMean的(0.624, 0.692)
- **vs Mix-of-Show**: Mix-of-Show在扩散模型上使用LBFGS求解类似优化，但同样有粗糙假设问题且需要梯度计算。IterIS用闭式解更高效
- **vs Linear Merging/Task Arithmetic**: 这些方法假设输入特征各向同性分布，过度简化导致性能差。IterIS在所有基线中表现最好

## 评分
- 新颖性: ⭐⭐⭐⭐ 迭代推理-求解范式简洁优雅，对粗糙假设的分析深入且解法自然
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖扩散模型、VLM、LLM三个领域，消融分析详细
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰（三个limitation图示直观），方法推导严谨
- 价值: ⭐⭐⭐⭐ LoRA合并是实用场景的真需求，IterIS提供了显著更好的解决方案

<!-- RELATED:START -->

## 相关论文

- [Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing](improving_diffusion_inverse_problem_solving_with_decoupled_noise_annealing.md)
- [Color Alignment in Diffusion](color_alignment_in_diffusion.md)
- [K-LoRA: Unlocking Training-Free Fusion of Any Subject and Style LoRAs](k-lora_unlocking_training-free_fusion_of_any_subject_and_style_loras.md)
- [Flat-LoRA: Low-Rank Adaptation over a Flat Loss Landscape](../../ICML2025/image_generation/flat-lora_low-rank_adaptation_over_a_flat_loss_landscape.md)
- [Unsupervised Conformal Inference: Bootstrapping and Alignment to Control LLM Uncertainty](../../ICLR2026/image_generation/unsupervised_conformal_inference_bootstrapping_and_alignment_to_control_llm_unce.md)

<!-- RELATED:END -->
