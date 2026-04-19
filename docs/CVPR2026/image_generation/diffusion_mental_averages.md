---
title: >-
  [论文解读] Diffusion Mental Averages
description: >-
  [CVPR 2026][图像生成][扩散模型] 提出 Diffusion Mental Averages (DMA)，通过在扩散模型的语义空间中对齐多个去噪轨迹，从预训练扩散模型中提取概念的"心理平均"原型图像——首次实现一致、逼真的概念平均可视化。
tags:
  - CVPR 2026
  - 图像生成
  - 扩散模型
  - 概念原型
  - 轨迹对齐
  - 语义平均
  - 模型偏见分析
---

# Diffusion Mental Averages

**会议**: CVPR 2026  
**arXiv**: [2603.29239](https://arxiv.org/abs/2603.29239)  
**代码**: [项目页面](https://diffusion-mental-averages.github.io)  
**领域**: Image Generation / Diffusion Models  
**关键词**: 扩散模型, 概念原型, 轨迹对齐, 语义平均, 模型偏见分析

## 一句话总结

提出 Diffusion Mental Averages (DMA)，通过在扩散模型的语义空间中对齐多个去噪轨迹，从预训练扩散模型中提取概念的"心理平均"原型图像——首次实现一致、逼真的概念平均可视化。

## 研究背景与动机

当人们想象"鸟"时，脑海中会浮现一只典型的小鸟（如麻雀），而非鸵鸟或珍稀物种。这种"心理原型"反映了我们的经验偏好。类似地，扩散模型在训练后也应该隐含地编码了每个概念的"典型表示"——但如何可视化它？

现有方法的局限：

**像素空间平均**：需要空间对齐，对齐后细节仍会被平均掉，产生模糊且不真实的结果。对抽象概念（如"自由"）根本无法空间对齐

**语义空间平均**：在 AE/VAE/GAN 中可以编码→平均→解码，但扩散模型**没有显式的语义瓶颈层**。语义信息分散在不同时间步中，且没有直接的解码器

**数据集蒸馏方法**（D4M, MGD3）：面向下游任务优化，产生的原型不自然或不一致

**选择性方法**：从集合中选代表样本，但样本总数有限，无法超越已有生成

核心挑战：扩散模型的语义信息**分布在整个去噪轨迹**中——从早期的粗粒度布局到晚期的细粒度细节，无法在单一层面进行简单平均。

## 方法详解

### 整体框架

DMA 将"平均"重新定义为"轨迹对齐"：同时优化多个噪声潜变量，使它们的去噪轨迹逐步收敛到共享的"从粗到细"语义共识。核心利用 U-Net 瓶颈层（h-space）作为语义表征，该层已被证明具有线性可解释性。

### 关键设计

1. **渐进式轨迹对齐**：在去噪过程中逐步统一语义 → 先对齐全局结构再对齐局部细节 → 符合扩散模型本身的粗到细生成逻辑

    - 初始化 K=1000 个噪声潜变量 $\{\mathbf{z}_k^{(0)}\}_{k=1}^K$
    - 在每个时间步 t：
        - 计算所有潜变量的平均 h-space 激活：$\bar{A}^{(t)} = \frac{1}{K}\sum_{k} H(\mathbf{z}_k^{(t)})$
        - 对每个潜变量优化：$\min_{\mathbf{z}_i^{(t)}} \|H(\mathbf{z}_i^{(t)}) - \bar{A}^{(t)}\|_2^2$（300次 Adam 迭代）
        - 优化后执行 DDIM 采样到下一步
    - 到达 cutoff $t_{stop}=10$ 后，使用标准 DDIM 完成剩余去噪
    - 最终任意潜变量解码即得原型图像
    - **设计动机**：利用 h-space 的语义线性性，在扩散过程的自然层级上逐步建立共识，先达成全局一致再细化局部

2. **多模态概念的模式发现**：处理"狗"有多种品种的情况 → 在 CLIP 空间聚类后分别平均 → 每个聚类学习轻量条件适配

    - 先生成样本并提取 CLIP 嵌入，用 GMM 聚类
    - 对每个聚类，学习 Textual Inversion 嵌入或 LoRA 来将扩散模型的条件引导到对应的语义子区域
    - **设计动机**：CLIP 空间比 h-space 更稳定，适合做模式分离。但 CLIP 与 h-space 存在语义不一致，因此需要 TI/LoRA 作为跨空间的桥梁

3. **Grounded 聚类**：支持用户定义的分割维度

    - 使用 BLIP-VQA 获取属性聚焦的嵌入（如按肤色分"医生"、按品种分"狗"）
    - 在 PCA 降维后用 GMM 聚类

### 损失函数 / 训练策略

- 核心优化目标极其简洁：$\mathcal{L} = \|H(\mathbf{z}_k^{(t)}) - \bar{A}_t\|_2^2$
- 使用 Adam 优化器，学习率 $2 \times 10^{-2}$，每个时间步每个潜变量优化 300 次迭代
- CFG scale = 7.0，DDIM 采样 20 步
- LoRA 方案：rank-1 LoRA，2000步训练，学习率 $10^{-4}$，CFG 降至 3.0
- Textual Inversion：3000步训练，学习率 $10^{-2}$
- 完整优化约需 10 小时（RTX 4080）

## 实验关键数据

### 主实验

12 个概念（动物、人物、物体、抽象），每组 1000 个样本，10 组重复：

| 方法 | Consistency (CLIP) ↓ | Consistency (DreamSim) ↓ | Representativeness (CLIP) ↓ | ImageReward ↑ |
|------|---------------------|-------------------------|---------------------------|--------------|
| GANgealing | 0 | 0 | 0.386 | -0.684 |
| Avg VAE | 0 | 0 | 0.473 | -2.262 |
| D4M | 0.168 | 0.274 | 0.197 | 0.823 |
| MGD3 | 0.180 | 0.319 | 0.195 | 0.755 |
| **DMA (Ours)** | **0.031** | **0.032** | **0.179** | **1.002** |

- GANgealing 和 Avg VAE 一致性为 0（设计如此），但结果模糊不真实
- DMA 在一致性、代表性和图像质量上全面最优

### 消融实验

| 配置 | 说明 |
|------|------|
| 不同 cutoff $t_{stop}$ | 越大对齐越充分但计算越贵，10步足够 |
| LoRA vs Textual Inversion | LoRA 更好地保留颜色和形状，TI 容量有限 |
| 不同 SD 变体 | 各变体产生风格/偏好不同的原型，验证了方法的泛化性 |
| DiT 架构 | 使用 final transformer block 替代 h-space，同样有效 |

### 关键发现

- **一致性极高**：DMA 从不同随机种子出发收敛到几乎相同的原型，一致性比 D4M/MGD3 好 5-10 倍
- **可揭示模型偏见**：如"soldier"在 SD1.5/Realistic Vision 中总是男性，PixelArt 更中性，Animerge 生成女性卡通
- **抽象概念可行**：对"freedom"一致生成自由女神像，对"Italy"一致生成威尼斯运河，而基线方法无法处理
- **模式发现有效**：对"crane"可分离出"鸟"和"建筑吊车"两种语义模式

## 亮点与洞察

1. **全新研究问题**：首次提出从扩散模型中提取"心理平均"的概念——不是生成多样样本，而是找到概念的"最典型"表示
2. **理念精妙**：将"平均"转化为"轨迹对齐"，完美契合扩散模型从粗到细的生成范式
3. **模型探针工具**：DMA 可作为分析扩散模型内部概念表示和偏见的新工具，具有广泛的分析应用价值
4. **方法通用性**：从 SD1.5 到 DiT 架构均可工作，说明语义层级结构是扩散模型的共性而非特定架构的产物
5. **模式发现 + 条件适配的组合**应对多义词概念，是一个优雅的解决方案

## 局限与展望

1. **计算成本高**：1000个潜变量 × 20个时间步 × 300次优化 = 约10小时（RTX 4080），实际应用受限
2. **依赖 h-space 的存在**：对非 U-Net 架构（如 DiT）需要手动寻找类似的语义层，缺乏自动化方法
3. **CFG 和样本数影响结果**：高变化概念需更多样本，高 CFG 提高一致性但减少多样性
4. **聚类依赖外部编码器**：继承了 CLIP/BLIP 的偏见
5. **评估主观性**：代表性指标依赖于特定嵌入空间的选择

## 相关工作与启发

- **Kwon et al.**：发现 U-Net 瓶颈层（h-space）具有线性语义性质，是本文的关键基础
- **GANgealing**：在 GAN 中通过空间对齐做像素平均，但依赖预训练 GAN 且无法处理抽象概念
- **D4M / MGD3**：利用扩散模型做数据集蒸馏的最新工作，但面向分类任务而非概念总结
- **Textual Inversion / LoRA**：作为轻量级模型适配方法，在这里被创新性地用于跨空间语义对齐
- 启示：扩散模型的时间步维度蕴含了从抽象到具体的语义层级，这种结构可以被更多任务利用

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 全新问题定义，"心理平均"概念极具启发性
- 实验充分度: ⭐⭐⭐⭐ — 多概念、多变体、多架构实验全面，但计算成本未深入分析
- 写作质量: ⭐⭐⭐⭐⭐ — 故事线流畅，从认知科学类比引入，可视化丰富
- 价值: ⭐⭐⭐⭐ — 作为分析工具和概念可视化方法很有意义，但实际应用场景有限

<!-- RELATED:START -->

## 相关论文

- [GenIR: Generative Visual Feedback for Mental Image Retrieval](../../NeurIPS2025/image_generation/genir_generative_visual_feedback_for_mental_image_retrieval.md)
- [PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](pixelrush_ultra-fast_training-free_high-resolution_image_generation_via_one-step.md)
- [Reviving ConvNeXt for Efficient Convolutional Diffusion Models](reviving_convnext_for_efficient_convolutional_diffusion_models.md)
- [Exploring Conditions for Diffusion Models in Robotic Control](exploring_conditions_for_diffusion_models_in_robotic_control.md)
- [Diffusion Probe: Generated Image Result Prediction Using CNN Probes](diffusion_probe_generated_image_result_prediction_using_cnn_probes.md)

<!-- RELATED:END -->
