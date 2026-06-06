---
title: >-
  [论文解读] Efficient Input-Level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation
description: >-
  [ICCV 2025][图像生成][后门防御] NaviT2I 发现了文生图扩散模型中后门触发器导致的"早期步骤激活变化"（Early-step Activation Variation）现象，基于此提出了一种仅需分析第一步扩散迭代的高效输入级后门防御框架，在 8 种主流攻击上平均 AUROC 达 96.3%…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "后门防御"
  - "文生图模型"
  - "神经元激活变化"
  - "输入级检测"
  - "扩散模型"
---

# Efficient Input-Level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation

**会议**: ICCV 2025  
**arXiv**: [2503.06453](https://arxiv.org/abs/2503.06453)  
**代码**: [GitHub](https://github.com/zhaisf/NaviT2I)  
**领域**: AI 安全/文生图后门防御  
**关键词**: 后门防御, 文生图模型, 神经元激活变化, 输入级检测, Stable Diffusion

## 一句话总结
NaviT2I 发现了文生图扩散模型中后门触发器导致的"早期步骤激活变化"（Early-step Activation Variation）现象，基于此提出了一种仅需分析第一步扩散迭代的高效输入级后门防御框架，在 8 种主流攻击上平均 AUROC 达 96.3%，耗时仅为已有方法的 3.8%~16.7%。

## 研究背景与动机

文生图（T2I）扩散模型（如 Stable Diffusion）的开源生态带来了后门威胁：攻击者可在模型中植入触发器，部署时通过特定文本触发恶意行为。

**现有防御的困境**：

**触发器主导假设失效**：已有输入级防御方法（T2IShield、UFID）假设触发器主导模型输出（Trigger Dominance），即修改良性特征不影响模型预测。然而在 T2I 中，后门目标多样——可能只修改图像局部（BadT2I）、替换物体（EvilEdit）或改变风格（RickBKD），此时修改良性 token 也会改变生成结果，该假设不成立。

**计算开销巨大**：传统方法需要多次完整图像生成来分析输出差异，在 T2I 场景下产生 2~5 倍的额外计算开销。

核心矛盾：需要一种不依赖"触发器主导"假设、且不需完整生成过程的通用高效防御方法。

切入角度：**从模型内部神经元激活状态入手，而非分析生成结果**。关键发现是触发器 token 在扩散生成的前几步就引起了异常大的神经元激活变化，且这种差异在早期最显著。

## 方法详解

### 整体框架

NaviT2I 的流程：
1. 对输入 prompt 中的每个非停用词 token，生成 mask 版本
2. 仅执行一步扩散迭代，计算 mask 前后的逐层神经元激活变化
3. 通过异常值检测识别触发器 token，判定输入是否恶意

### 关键设计

1. **Early-step Activation Variation 现象发现**:

    - 功能：揭示后门触发器对扩散模型神经元激活的影响在生成早期最显著
    - 核心思路：实验发现 mask 触发器 token 后，模型的 Neuron Coverage 变化显著大于 mask 正常 token。理论分析证明，当 $t$ 较大时（即迭代前期），不同条件 $c, c'$ 下的预测差异上界为：
    $\|\epsilon(\mathbf{x}_t, t, c) - \epsilon(\mathbf{x}_t, t, c')\|_2 \leq O\left(\frac{1}{\alpha}\exp\left(-\frac{1}{2\sigma_t^2}\right)\right)$
      当 $\sigma_t$ 较大时（前期步骤），差异可以很大；当 $\sigma_t$ 较小时（后期步骤），差异以指数速率衰减
    - 设计动机：仅需第一步扩散就能捕获触发器信号，大幅提升效率

2. **逐层激活变化度量**:

    - 功能：精确量化每个 token 对模型各层激活的影响
    - 核心思路：
        - 线性层：$\delta^{(\ell)}(c, c') = \frac{1}{N_\ell d_\ell}\|\mathbf{A}^{(\ell)}(c) - \mathbf{A}^{(\ell)}(c')\|_1$
        - 卷积层：先在空间维度平均池化，再计算 1-范数差异
        - 总激活变化：$\delta_\theta(c, c') = \sum_{\ell \in \mathcal{L}_{set}} \delta^{(\ell)}(c, c')$
    - 设计动机：逐层计算比粗粒度的 Neuron Coverage 更精确，能捕捉不同层对触发器的差异响应

3. **异常值检测与评分函数**:

    - 功能：将每个 token 的激活变化归一化为评分，检测异常值
    - 核心思路：对每个 token $k$ 计算归一化特征值：
    $V_k = \frac{\delta_\theta(c, c_k)}{\mathcal{D}(c, c_k)}$
      其中 $\mathcal{D}(c, c_k) = \|\mathcal{T}(c) - \mathcal{T}(c_k)\|_2$ 是文本嵌入距离（归一化语义差异的影响）
     
      评分函数：$\mathcal{S}(c) = \frac{\max(\mathbf{V})}{\text{mean}(\mathbf{V}')}$，其中 $\mathbf{V}'$ 去除了前 25% 的值
    - 设计动机：除以文本嵌入距离使得不同语义重要性的 token 可以公平比较；max/mean 比值有效检测异常

### 损失函数 / 训练策略

NaviT2I 是**无需训练**的检测方法。阈值通过对少量干净样本做高斯拟合自动设定：
$$\mathcal{D}(c) = \mathbb{1}[\mathcal{S}(c) > \mu_\text{clean} + m \cdot \sigma_\text{clean}]$$
其中 $m = 1.2$ 为默认的平衡系数。

## 实验关键数据

### 主实验

| 攻击方法 | NaviT2I AUROC | T2IShield_CDA | UFID | 提升 |
|--------|------|------|----------|------|
| RickBKD_TPA | **99.9** | 94.1 | 72.9 | +5.8 |
| RickBKD_TAA | **99.8** | 80.2 | 69.1 | +19.6 |
| BadT2I_Tok | **97.0** | 62.1 | 47.6 | +34.9 |
| BadT2I_Sent | **89.7** | 70.7 | 62.4 | +19.0 |
| VillanBKD_one | **98.9** | 92.6 | 95.7 | +3.2 |
| VillanBKD_mul | **99.9** | 98.0 | 99.9 | +0.0 |
| PersonalBKD | **99.8** | 68.5 | 64.0 | +31.3 |
| EvilEdit | **85.5** | 57.8 | 42.7 | +27.7 |
| **平均** | **96.3** | 78.0 | 69.3 | **+18.3** |

### 消融实验（效率）

| 方法 | 扩散迭代数/样本 | 耗时（秒/样本） | 说明 |
|------|---------|------|------|
| T2IShield_FTT | 50 | 7.445 | 需完整生成过程 |
| T2IShield_CDA | 50 | 7.467 | 同上 + 协方差分析 |
| UFID | 200 | 33.041 | 需 4 张完整图像 |
| **NaviT2I** | **≈7** | **1.242** | 仅需第一步×(K+1)次 |

### 关键发现

- NaviT2I 在**所有 8 种攻击**上均达到 85%+ AUROC，而 baseline 方法在 BadT2I、EvilEdit 等不满足"触发器主导"假设的攻击上退化至接近随机猜测（~50%）
- 仅需正常生成过程 **7%~14%** 的时间即可完成检测，适合实时部署
- 对多 token 触发器（VillanBKD_mul）和句子触发器（BadT2I_Sent）同样有效
- 风格触发器（Style Trigger）由于其 ASR 仅 28.5%、FAR 高达 16.3%，本身就不构成有效攻击

## 亮点与洞察

1. **Early-step Activation Variation 是关键贡献**：从理论和实验两方面说明触发器影响集中在生成早期，这个发现有很强的可推广性
2. **归一化设计精巧**：除以文本嵌入距离消除了不同 token 语义重要性差异的干扰
3. **实用性极强**：无需训练、无需额外数据、仅一步扩散、即插即用
4. **对 DiT 架构也有效**：不限于 UNet，展示了方法的架构通用性

## 局限与展望

- 对 EvilEdit（85.5% AUROC）的检测效果相对较弱，因为该攻击更加隐蔽
- 假设触发器是当前 T2I 攻击中的常见形式（文本 token），纯视觉触发器未讨论
- 阈值设定依赖于少量干净样本的高斯拟合，分布偏移可能影响效果
- 最坏情况下（77 个非停用词的长 prompt）计算开销接近一次完整生成

## 相关工作与启发

- 将软件测试领域的 Neuron Coverage 概念巧妙引入 T2I 安全领域
- Early-step 的观察可能对扩散模型可解释性研究有启发
- 激活变化度量方法可以推广到其他条件生成模型（如视频生成）的安全检测

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Early-step Activation Variation 现象的发现和理论分析是原创且深刻的
- 实验充分度: ⭐⭐⭐⭐⭐ 8 种攻击、多数据集、效率分析、自适应攻击分析、消融实验齐全
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机清晰，方法讲解配合图示非常直观
- 价值: ⭐⭐⭐⭐⭐ 直接解决了 T2I 部署中的安全痛点，方法简洁高效，具有很高实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Disrupting Model Merging: A Parameter-Level Defense Without Sacrificing Accuracy](disrupting_model_merging_a_parameter-level_defense_without_sacrificing_accuracy.md)
- [\[ICCV 2025\] Your Text Encoder Can Be An Object-Level Watermarking Controller](your_text_encoder_can_be_an_object-level_watermarking_controller.md)
- [\[ICCV 2025\] Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation](dense2moe_restructuring_diffusion_transformer_to_moe_for_efficient_text-to-image.md)
- [\[ICCV 2025\] PolarAnything: Diffusion-based Polarimetric Image Synthesis](polaranything_diffusion-based_polarimetric_image_synthesis.md)
- [\[ICCV 2025\] Towards Robust Defense against Customization via Protective Perturbation Resistant to Diffusion-based Purification](towards_robust_defense_against_customization_via_protective_perturbation_resista.md)

</div>

<!-- RELATED:END -->
