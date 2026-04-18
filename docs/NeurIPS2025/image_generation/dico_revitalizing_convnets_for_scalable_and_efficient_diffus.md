---
title: >-
  [论文解读] DiCo: Revitalizing ConvNets for Scalable and Efficient Diffusion Modeling
description: >-
  [NeurIPS 2025 (Spotlight)][图像生成][ConvNet] 重新发掘卷积网络在扩散模型中的潜力——发现预训练DiT的全局自注意力主要捕获局部模式（冗余），提出用标准ConvNet模块+紧凑通道注意力构建纯卷积扩散模型DiCo，在ImageNet-256上以2.05 FID超越DiT-XL/2且速度快2.7倍。
tags:
  - NeurIPS 2025 (Spotlight)
  - 图像生成
  - ConvNet
  - 扩散模型
  - 卷积替代注意力
  - 通道冗余
  - 注意力机制
  - 高效生成
---

# DiCo: Revitalizing ConvNets for Scalable and Efficient Diffusion Modeling

**会议**: NeurIPS 2025 (Spotlight)  
**arXiv**: [2505.11196](https://arxiv.org/abs/2505.11196)  
**代码**: [https://github.com/shallowdream204/DiCo](https://github.com/shallowdream204/DiCo)  
**领域**: 图像生成  
**关键词**: ConvNet, 扩散模型, 通道注意力, 高效生成, U-shaped架构

## 一句话总结
发现预训练DiT的全局self-attention在生成任务中主要捕获局部模式存在大量冗余，提出用标准卷积模块+紧凑通道注意力（CCA）构建纯卷积扩散模型DiCo，在ImageNet-256上FID达2.05超越DiT-XL/2且推理速度快2.7倍，512分辨率下更快3.1倍。

## 研究背景与动机

**扩散模型的架构演进**：扩散模型经历了从U-Net（ADM、Stable Diffusion）到全Transformer架构（DiT）的过渡。DiT凭借Transformer的强大建模能力在ImageNet生成上取得了出色表现，并成为Stable Diffusion 3、FLUX、Sora等主流模型的backbone。然而，self-attention的二次计算复杂度在高分辨率图像生成时带来了巨大的计算瓶颈——在512×512分辨率下尤为严重。

**替代架构的探索及其不足**：为缓解这一问题，近年涌现了一系列基于线性复杂度架构的尝试，如基于Mamba的DiS/DiM和基于门控线性注意力的DiG。然而这些模型的因果设计与视觉生成的双向性存在天然冲突，且即便使用高度优化的CUDA实现，其实际运行速度优势在高分辨率场景下仍然有限。

**关键反直觉发现**：作者对预训练好的DiT-XL/2、PixArt-α和FLUX三个代表性DiT模型的注意力图进行了系统分析，发现了一个令人惊讶的现象——在几乎所有层中，self-attention的权重高度集中在锚点token的邻近空间位置上，远距离token的注意力权重极低。这与视觉识别任务中attention捕获全局依赖的常规认知截然不同，表明在生成任务中，全局注意力计算存在大量冗余，本质上只需要局部空间建模就足够了。

**简单替换的失败与根因分析**：受上述发现启发，自然的想法是用卷积（天然擅长局部模式捕获）替代self-attention。但直接替换导致生成质量下降。通过channel activation score分析，作者发现根本原因在于ConvNet的**通道冗余**——与Transformer相比，ConvNet中大量通道的激活值停留在极低水平，特征多样性严重不足。self-attention作为动态的、内容依赖的操作，天然具有更强的表示能力来促进通道多样化，而卷积的静态权重则缺乏这种能力。DiCo的设计正是要在保持卷积高效性的同时，通过轻量级的通道注意力机制弥补这一表示能力差距。

## 方法详解

### 整体框架
DiCo采用三阶段U-shaped架构，由堆叠的DiCo Block组成。输入图像经VAE编码器获得空间表征$z$（256×256图像对应32×32×4的latent），首先通过3×3卷积映射到$D$通道的初始特征图$z_0$。条件信息（时间步$t$和类别标签$y$）分别通过MLP和Embedding层处理。在各stage内，编码器和解码器之间通过skip connection传递中间特征（拼接后用1×1卷积降维）。跨stage的多尺度处理使用pixel-unshuffle下采样和pixel-shuffle上采样。最终输出特征$z_L$经归一化后通过3×3卷积头预测噪声和协方差。整个模型不包含任何self-attention或cross-attention操作，完全由标准卷积模块构成。

### 关键设计

1. **Conv Module（卷积模块）**:

    - 功能：替代DiT中的self-attention模块，实现高效的空间和通道特征提取
    - 核心思路：采用1×1 pointwise卷积聚合逐像素的跨通道信息，然后用3×3 depthwise卷积捕获通道内的空间上下文，再接GELU非线性激活。整个过程可表示为$Y = W_{p_2} \text{CCA}(\text{GELU}(W_d W_{p_1} X))$，其中$W_{p_1}$和$W_{p_2}$是pointwise卷积，$W_d$是depthwise卷积。与现代识别ConvNet使用大尺寸kernel（如31×31）不同，DiCo仅依赖1×1和3×3的标准小卷积核，设计极其简洁
    - 设计动机：从注意力局部性分析中得知生成任务的有效感受野很小，3×3 depthwise卷积已足以捕获关键的局部空间模式，同时保持了卓越的硬件友好性和推理效率

2. **Compact Channel Attention（紧凑通道注意力, CCA）**:

    - 功能：动态激活更多信息丰富的通道，解决ConvNet的通道冗余问题，提升特征多样性
    - 核心思路：CCA首先对空间维度做全局平均池化（GAP）将特征压缩为通道描述符，然后经1×1卷积和Sigmoid激活生成通道级注意力权重，最后与输入逐通道相乘：$\text{CCA}(X) = X \odot \text{Sigmoid}(W_p \text{GAP}(X))$。这是一种轻量级的全局通道建模方式，仅引入极低的计算开销
    - 设计动机：通道激活score分析显示，直接用Conv替换attention后大量通道处于"死亡"状态（激活值接近零），导致有效特征通道数远少于Transformer。CCA通过数据自适应的通道加权机制，迫使网络激活更多不同的通道，从而恢复了Transformer级别的特征多样性。实验验证加入CCA后通道冗余显著降低

3. **U-shaped多尺度架构设计**:

    - 功能：构建三阶段的层次化编码器-解码器结构，利用多尺度特征表示提升去噪能力
    - 核心思路：不同于DiT的各向同性（isotropic）架构，DiCo采用U-shaped设计，各stage间通过pixel-shuffle/unshuffle实现分辨率变化。编码器和解码器之间通过skip connection传递特征，拼接后用1×1卷积融合。模型提供S/B/L/XL/H五个变体，参数量分别对齐DiT对应规模，但GFLOPs仅为DiT的70.1%-74.6%
    - 设计动机：多尺度特征在图像去噪中具有关键作用——低分辨率特征捕获全局结构，高分辨率特征保留细节纹理。作者通过消融实验系统比较了isotropic、isotropic+skip connection和U-shaped三种架构，U-shaped在所有规模上均表现最优

### 损失函数 / 训练策略
遵循DiT的标准扩散训练流程。噪声预测器$\epsilon_\theta$使用简化损失$\mathcal{L}_{simple}(\theta) = \|\epsilon_\theta(x_t) - \epsilon_t\|_2^2$训练，协方差$\Sigma_\theta$使用完整的变分下界损失$\mathcal{L}$优化。采用classifier-free guidance（CFG）增强采样质量。训练配置包括：学习率$1 \times 10^{-4}$、batch size 256、无weight decay、EMA decay 0.9999。对于最大的DiCo-H（1B参数），学习率提高至$2 \times 10^{-4}$、batch size扩大至1024以加速训练。

## 实验关键数据

### 主实验
| 模型 | 类型 | GFLOPs | 吞吐(img/s) | FID↓ | IS↑ |
|------|------|--------|-------------|------|-----|
| DiT-XL/2 (w/ CFG) | Attn | 118.66 | 76.90 | 2.27 | 278.24 |
| DiG-XL/2 (w/ CFG) | Conv+Attn | 89.40 | 71.74 | 2.07 | 278.95 |
| **DiCo-XL (w/ CFG)** | **Conv** | **87.30** | **208.47** | **2.05** | **282.17** |
| DiCo-H (w/ CFG) | Conv | 194.15 | 117.57 | **1.90** | **284.31** |

**ImageNet 512×512结果**：

| 模型 | GFLOPs | 吞吐(img/s) | FID↓ | IS↑ |
|------|--------|-------------|------|-----|
| DiT-XL/2 (w/ CFG) | 524.70 | 18.58 | 3.04 | 240.82 |
| DiS-H/2 | - | 8.59 | 2.88 | 272.33 |
| **DiCo-XL (w/ CFG)** | **349.78** | **57.45** | **2.53** | **275.74** |

### 消融实验
| 配置 | FID↓ | 说明 |
|------|------|------|
| Isotropic架构 | 58.23 | DiT风格平坦架构 |
| Isotropic + Skip | 54.10 | 加长跳跃连接 |
| **U-shaped架构** | **49.97** | 三阶段层次化，最优 |
| Conv替换attention（无CCA） | 62.06 | 大量通道"死亡" |
| **Conv + CCA** | **49.97** | 通道冗余显著降低 |

**各规模400K步无CFG对比**：

| 规模 | DiT FID | DiCo FID | FID改进 | 速度提升 |
|------|---------|----------|---------|---------|
| S | 68.40 | 49.97 | -18.43 | 1.37× |
| B | 43.47 | 27.20 | -16.27 | 2.17× |
| L | 23.33 | 13.66 | -9.67 | 2.51× |
| XL | 19.47 | 11.67 | -7.80 | 2.71× |

### 关键发现
- DiCo在所有模型规模上均优于DiT，且规模越大优势越明显
- 分辨率越高加速比越大：256上2.7× → 512上3.1×，源于卷积$O(n)$ vs attention $O(n^2)$
- DiCo-XL比基于Mamba的DiS-H/2快6.7×，比DiM-H快7.8×
- 1B参数的DiCo-H进一步将FID推至1.90，证明架构可扩展性
- MS-COCO T2I实验中，动态DWC替代cross-attention后仍保持竞争力

## 亮点与洞察
- **ConvNet在Transformer时代的复兴**：在所有人都认为Transformer是扩散模型最优架构的背景下，用严谨的注意力局部性分析为ConvNet的回归提供了坚实的理论依据，NeurIPS Spotlight当选实至名归
- **通道冗余这一关键洞察**：精准定位了Conv替代Attention性能下降的根本原因——不是感受野不够大，而是通道多样性不足。CCA模块以极低代价（仅一个GAP+1×1 Conv+Sigmoid）解决了此问题，设计的简洁性令人赞叹
- **分辨率友好的效率优势**：卷积$O(n)$复杂度使得加速比随分辨率增长而扩大，对高分辨率T2I应用极具价值

## 局限与展望
- 卷积的固有局部性在需要全局空间关系建模的场景（如复杂物体间的空间布局一致性）可能受限
- 仅在class-conditional ImageNet和小规模MS-COCO T2I验证，缺乏大规模T2I数据集上的训练实验
- 与MM-DiT范式（如FLUX/SD3中的多模态DiT）的兼容性尚未探索
- 动态DWC文本注入方案将CLIP 77 tokens填充到81后reshape为9×9 kernel，固定reshape可能限制文本条件灵活性

## 相关工作与启发
- **vs DiT**: DiT用Transformer做扩散backbone，DiCo用纯ConvNet——同等质量下快2.7×（256）到3.1×（512），GFLOPs降低26.4%-33.3%
- **vs DiG**: DiG使用门控线性注意力仍依赖全局token mixing；DiCo彻底放弃全局交互只用局部卷积，反而取得更好的FID和更高的推理速度
- **vs ConvNeXt**: ConvNeXt在识别任务中证明现代ConvNet可与ViT竞争，DiCo将此理念拓展到生成任务

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 在Transformer主导的时代重新证明ConvNet的价值，通道冗余洞察精准且解法优雅
- 实验充分度: ⭐⭐⭐⭐ ImageNet-256/512全面对比，多规模消融充分，但缺乏大规模T2I实验
- 写作质量: ⭐⭐⭐⭐ 从发现问题→定位原因→解决的logic chain清晰流畅
- 价值: ⭐⭐⭐⭐⭐ Spotlight当选，为高效扩散模型架构开辟了ConvNet新路径
---
title: >-
  [论文解读] DiCo: Revitalizing ConvNets for Scalable and Efficient Diffusion Modeling
description: >-
  [NeurIPS 2025 (Spotlight)][图像生成][ConvNet] 重新发掘卷积网络在扩散模型中的潜力——发现预训练DiT的全局自注意力主要捕获局部模式（冗余），提出用标准ConvNet模块+紧凑通道注意力构建纯卷积扩散模型DiCo，在ImageNet-256上以2.05 FID超越DiT-XL/2且速度快2.7倍。
tags:
  - NeurIPS 2025 (Spotlight)
  - 图像生成
  - ConvNet
  - 扩散模型
  - 卷积替代注意力
  - 通道冗余
  - 注意力机制
  - 高效生成
---

# DiCo: Revitalizing ConvNets for Scalable and Efficient Diffusion Modeling

**会议**: NeurIPS 2025 (Spotlight)  
**arXiv**: [2505.11196](https://arxiv.org/abs/2505.11196)  
**代码**: [https://github.com/shallowdream204/DiCo](https://github.com/shallowdream204/DiCo)  
**领域**: 图像生成 / 扩散模型 / 高效架构  
**关键词**: ConvNet, 扩散模型, 卷积替代注意力, 通道冗余, compact channel attention, 高效生成

## 一句话总结
重新发掘卷积网络在扩散模型中的潜力——发现预训练DiT的全局自注意力主要捕获局部模式（冗余），提出用标准ConvNet模块+紧凑通道注意力构建纯卷积扩散模型DiCo，在ImageNet-256上以2.05 FID超越DiT-XL/2且速度快2.7倍。

## 背景与动机
Diffusion Transformer (DiT)展现了强大的生成能力但计算开销大。一个反直觉的发现是：分析预训练DiT的注意力模式，发现全局self-attention在大多数层实际上主要捕获的是局部pattern而非全局依赖——这意味着注意力机制在此场景中有很大冗余，可以用更高效的局部操作（如卷积）替代。但简单替换会导致性能下降，需要找到原因并解决。

## 核心问题
能否用纯卷积网络替代DiT中的Transformer实现高效且高质量的扩散建模？为什么简单替换会掉点？如何修复？

## 方法详解

### 整体框架
DiCo是一族完全由标准ConvNet模块构建的扩散模型，不使用任何self-attention操作。每个DiCo block用深度可分离卷积替代self-attention，用紧凑通道注意力解决卷积网络的通道冗余问题。

### 关键设计
1. **DiT注意力冗余的发现**：通过分析预训练DiT各层的注意力图，发现大部分层的注意力集中在局部邻域——全局self-attention的有效感受野实际上很小（类似卷积）。这为用卷积替代提供了理论基础。

2. **通道冗余问题的定位**：简单将attention替换为convolution导致性能下降的根本原因是ConvNet的通道冗余高于Transformer——ConvNet倾向于让很多通道学到相似的特征（通道间相关性高），而Transformer的attention天然促进通道多样性。

3. **紧凑通道注意力（Compact Channel Attention）**：为解决通道冗余，引入一个轻量级的通道注意力模块，动态调整每个通道的激活权重，促进更多样化的通道特征学习。这使得ConvNet能学到与Transformer同样丰富的特征表示。

### 损失函数 / 训练策略
标准扩散训练loss（epsilon/velocity prediction），与DiT相同的训练配置。

## 实验关键数据
| 模型 | 分辨率 | FID↓ | 速度提升 |
|------|--------|------|----------|
| DiT-XL/2 | 256 | 2.27 | 1x |
| **DiCo-XL** | **256** | **2.05** | **2.7x** |
| DiT-XL/2 | 512 | - | 1x |
| **DiCo-XL** | **512** | **2.53** | **3.1x** |

- ImageNet-256上FID 2.05，超越DiT-XL/2 (2.27)
- 256分辨率2.7倍加速，512分辨率3.1倍加速
- 分辨率越高加速比越大（卷积复杂度O(n)对比attention的O(n²)）
- 在MS-COCO上也展示了T2I生成潜力

### 消融实验要点
- 简单Conv替代attention：FID下降
- +紧凑通道注意力：FID恢复并超越DiT
- 通道冗余分析：ConvNet的通道相关性显著高于Transformer
- 通道注意力有效降低通道冗余，提升特征多样性

## 亮点
- **NeurIPS Spotlight**：挑战了"Transformer是扩散模型最优架构"的共识
- **ConvNet的文艺复兴**：在Transformer主导的时代证明了卷积在扩散模型中的独特优势——更高效且可以更好
- **通道冗余是关键洞察**：精准定位了Conv替代Attention掉点的原因，并用轻量通道注意力解决
- **高分辨率优势更大**：511分辨率3.1x加速——对高分辨率T2I应用特别有价值
- **代码开源**，方便社区复现和扩展

## 局限与展望
- 卷积的局部性可能在需要全局建模的场景（如复杂空间关系、长距离一致性）受限
- 仅在class-conditional ImageNet和MS-COCO上验证，大规模T2I（如Laion数据）未测试
- 与MM-DiT（如FLUX/SD3的cross-modal attention）的兼容性未探索

## 与相关工作的对比
- **vs. DiT**：DiT用Transformer做扩散backbone；DiCo用ConvNet——同等质量下快2.7x
- **vs. SANA**：SANA用Linear Attention替代标准Attention加速；DiCo完全去掉Attention用Conv——从不同角度追求效率
- **vs. Dense2MoE**：Dense2MoE通过稀疏化减少每步计算；DiCo通过Conv替代Attention减少每步计算——正交策略可叠加

## 启发与关联
- DiCo + SANA-Sprint（步骤蒸馏）= 每步更快×步数更少 = 可能实现超实时T2I
- 通道冗余分析方法可以迁移到视频扩散模型——视频DiT的冗余可能更严重
- "ConvNet在扩散模型中的复活"可能引发新一轮架构创新

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 在Transformer主导的时代重新证明ConvNet的价值，通道冗余洞察精准
- 实验充分度: ⭐⭐⭐⭐ ImageNet-256/512+MS-COCO，效率和质量全面对比
- 写作质量: ⭐⭐⭐⭐ 从发现问题（注意力冗余）→定位原因（通道冗余）→解决（通道注意力）逻辑清晰
- 价值: ⭐⭐⭐⭐⭐ Spotlight当选，为高效扩散模型架构开辟了ConvNet新路径
