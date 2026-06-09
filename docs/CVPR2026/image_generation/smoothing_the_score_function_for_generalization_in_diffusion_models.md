---
title: >-
  [论文解读] Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework
description: >-
  [CVPR 2026][图像生成][扩散模型] 本文从理论上证明扩散模型的记忆化问题源于经验得分函数中 softmax 权重的尖锐性（单个训练样本主导），并提出 Noise Unconditioning 和 Temperature Smoothing 两种平滑方法来缓解记忆化、增强泛化，同时保持生成质量。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "扩散模型"
  - "记忆化"
  - "泛化"
  - "得分函数平滑"
  - "温度缩放"
---

# Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework

**会议**: CVPR 2026  
**arXiv**: [2601.19285](https://arxiv.org/abs/2601.19285)  
**代码**: [GitHub](https://github.com/xinyu-zhou/score-smoothing)  
**领域**: 图像生成  
**关键词**: 扩散模型, 记忆化, 泛化, 得分函数平滑, 温度缩放

## 一句话总结
本文从理论上证明扩散模型的记忆化问题源于经验得分函数中 softmax 权重的尖锐性（单个训练样本主导），并提出 Noise Unconditioning 和 Temperature Smoothing 两种平滑方法来缓解记忆化、增强泛化，同时保持生成质量。

## 研究背景与动机

1. **领域现状**：扩散模型已成为图像生成的主流框架，通过逐步添加噪声再学习反转过程实现高质量生成。其核心机制是通过神经网络逼近不同噪声水平下的得分函数。
2. **现有痛点**：大量研究发现扩散模型会"记忆化"训练数据——生成的样本可能与训练样本完全相同。理论分析表明，如果神经网络完美学习了经验得分函数，采样将退化为复制训练数据，无法产生新样本。
3. **核心矛盾**：理论预测的完美记忆化与实践中观察到的泛化能力之间存在根本性矛盾。虽然已知神经网络的有限容量和正则化起了作用，但"为什么神经网络能部分解决记忆化"这一核心问题缺乏系统性的理论解释。
4. **本文目标**：(1) 建立理论框架解释记忆化的根本原因；(2) 解释神经网络为何能实现泛化；(3) 基于理论提出进一步增强泛化的方法。
5. **切入角度**：将经验得分函数分解为高斯分量得分的加权和，发现权重本质上是 softmax 函数。在高维空间中，几何上每个训练样本对应一个"壳层"，采样点在低噪声时只落入单个壳层内，导致单点主导。
6. **核心 idea**：神经网络通过隐式平滑得分函数权重实现泛化，使采样受到局部流形（多个近邻样本）而非单个点的影响；可以通过显式平滑方法进一步增强这一效果。

## 方法详解

### 整体框架
这篇论文要回答一个看似矛盾的问题：理论上如果神经网络完美学到了经验得分函数，扩散采样就会退化成"复制训练样本"，可现实里模型却能生成新图像——泛化到底是从哪冒出来的？作者的切入点是把经验得分函数拆开看：它本质上是各训练样本对应的高斯分量得分的加权和，而权重是一个 softmax。于是"记忆化 vs 泛化"被翻译成"softmax 权重有多尖锐"的问题——权重越尖（单个样本独占），采样越坍缩到训练点；权重越平（多个近邻共同发声），采样越受局部流形约束、越能泛化。沿着这条线，全文先用高维几何说清记忆化为什么会发生，再给出两种把权重显式抹平的办法：Noise Unconditioning 在训练侧去掉噪声条件化，Temperature Smoothing 在采样侧引入温度参数，两者按噪声水平自适应切换。

### 关键设计

**1. 高维壳层几何分析：把"权重为何尖锐"翻译成"薄壳层是否重叠"**

要解释记忆化，先得看清权重在高维空间里长什么样。在 $d$ 维空间中，以训练样本 $\mu_j$ 为中心的高斯分布，其概率质量并不堆在中心，而是集中在半径约 $\sigma_i\sqrt{d}$ 的一层薄壳上。经验得分函数对各样本的权重 $w_{ij}(x) = \text{Softmax}(f(x, \mu_j, \sigma_i))$ 因此呈现两条尖锐性质：一是 $\sigma$-主导，对固定的中心和采样位置，存在一个最优噪声水平 $\sigma_j^*$，它对应的权重会把别的噪声水平远远甩开（比值可达 $e^6 \approx 403$）；二是 $\mu$-主导，最近训练样本相对其余样本的权重，随距离差异呈指数级拉大。两条性质叠加的后果是：高噪声时各样本的壳层还彼此重叠、权重相对均摊，但随着采样进入低噪声阶段，壳层不断收缩直到不再相交，某个训练样本就独占了得分函数，采样轨迹被它牢牢拽过去——这正是记忆化的几何根源。这套分析不是为了缓解而缓解，而是给后面两种平滑方法提供了明确的下手点：要泛化，就得阻止低噪声阶段的单点主导。

**2. Noise Unconditioning（噪声去条件化）：让每个训练样本都以自己的最优壳层发声**

第一条干预针对训练侧。标准扩散的得分函数 $s_\theta(x, \sigma_i)$ 把噪声水平当作输入条件，问题在于采样点未必落在大多数训练样本的最优壳层上，那些"壳层没对上"的样本权重就被压住、贡献被抹掉。去条件化的做法是直接拿掉这个条件，让网络只学 $s_\theta(x)$。这等价于把原本按噪声水平切开的一系列分布，合并成一个 $M \times N$ 项的高斯混合 $p_{\text{MN}}(x)$，并对它做得分匹配——于是采样可以重新解释为对 $\log p_{\text{MN}}(x)$ 的梯度上升。训练点仍是这个目标的最优解，但因为现在每个样本都能以它自己的最优壳层参与进来，单点主导被推迟，坍缩时间随之延后，泛化窗口被拉长。实现上几乎是"减法"：损失 $\mathcal{L}_u$ 与标准扩散完全相同，唯一区别是不再把噪声喂给网络；采样时因为没有了噪声水平这个输入，预设步长不再可靠，需换成自适应步长 $\alpha \sigma_{n*}^2$。

**3. Temperature Smoothing（温度平滑）：用 softmax 温度直接旋钮控制权重的尖锐度**

第二条干预针对采样侧，且专门管低噪声这段最危险的区间。既然尖锐性来自 softmax，那就给它加一个温度：$w_j^*(x;T) = \frac{\exp(f/T_j^*)}{\sum_l \exp(f/T_l^*)}$。作者设一个阈值 $\sigma_{\text{collapse}}$，只在 $\sigma_i \leq \sigma_{\text{collapse}}$（即壳层快要分离、坍缩在即）时启用温度缩放，并令 $T_i = \sigma_{\text{collapse}}/\sigma_i$——噪声越小、温度越高、抹得越平。升高温度会压低主导比 $a$、缩小扩展因子 $\gamma_{ex}$，把原本独占的权重重新分给近邻。为省开销，得分函数只用 top-K 最近邻样本近似；且 KNN 必须在特征空间而非像素空间里做，因为特征空间的局部曲率更小、近邻更贴合真实流形，这一点在实验里被像素空间 FID 崩到 50.81 的对比直接验证。这样温度就成了一个可调旋钮，在泛化和生成质量之间给出连续的权衡。

### 损失函数 / 训练策略
两种方法按噪声水平自适应拼接：$\sigma_i > \sigma_{\text{collapse}}$ 时（壳层尚重叠）走 Unconditioning 损失 $\mathcal{L}_u$，$\sigma_i \leq \sigma_{\text{collapse}}$ 时（壳层将分离）切到 Temperature 损失 $\mathcal{L}_T$。整体沿用 VE-SDE 框架，Unconditioning 分支仅去掉时间嵌入层，网络其余结构不变。

## 实验关键数据

### 主实验

| 数据集 | 方法 | FID(G,Train) | FID(G,Test) | 说明 |
|--------|------|-------------|------------|------|
| CIFAR-10 | Conditioning (baseline) | 6.49 | 6.56 | 标准 VE-SDE |
| CIFAR-10 | Unconditioning | 7.33 | 7.34 | FID 微升但泛化增强 |
| CIFAR-10 | Temp T=7/σ, K=100 (feat) | 7.96 | 7.98 | 像素空间 KNN 会崩溃(50.81) |
| CelebA | Conditioning | 7.25 | 7.81 | 标准 VE-SDE |
| CelebA | Unconditioning | 7.07 | 7.34 | FID 反而下降 |
| CelebA | Temp T=10/σ, K=100 (feat) | 8.40 | 8.19 | 特征 KNN 显著优于像素 KNN |

### 消融实验

| 配置 | 扩展因子 γ_ex | 说明 |
|------|-------------|------|
| 经验得分函数 (Conditioning) | ~10³ (低噪声时) | 极度尖锐，记忆化 |
| NN 学习的得分函数 | ~1-2 | 隐式平滑 |
| Unconditioning (经验) | 中等 | 比 Conditioning 平滑 |
| Temperature T=10 | ~1 | 显式平滑效果好 |
| Temperature T=100 | <1 | 接近非扩展 |
| Temperature T=1000 | ≈1 | 极度平滑 |

### 关键发现
- **特征空间 KNN 一致优于像素空间 KNN**：在 CIFAR-10 上 T=7/σ, K=100 时，像素空间 FID 崩溃到 50.81 而特征空间仅 7.96，验证了"局部流形曲率小有助于平滑"的理论
- **Unconditioning 在 CelebA 上反而改善 FID**（7.07 vs 7.25），说明平滑不仅不损害质量，可能还有正面效果
- **ODE 采样器在 Unconditioning 下会失败**，因为预设噪声水平与实际不匹配导致步长爆炸，SDE 采样器的随机项提供自校正机制保持稳定

## 亮点与洞察
- **壳层几何直觉**极其优美：将高维高斯混合的抽象数学分析转化为"薄壳层是否重叠"的几何图景，使记忆化的本质一目了然。这种几何化思维可迁移到其他涉及高斯混合的问题中。
- **统一分布视角**是最大的"啊哈"时刻：Noise Unconditioning 将扩散模型的逐步去噪重新解释为对固定目标函数的梯度上升，这不仅解释了泛化，还开启了用投影梯度法施加约束的可能性（如物理定律约束的视频生成）。
- **温度平滑作为即插即用方法**几乎不增加额外成本，可直接应用于现有扩散模型框架中，具有很强的工程实用性。

## 局限与展望
- 实验主要基于 VE-SDE 框架，未验证在更主流的 VP-SDE 和 Flow Matching 框架上的效果
- Temperature Smoothing 需要 KNN 查询，对大规模数据集有额外开销
- 温度参数和 $\sigma_{\text{collapse}}$ 的选择需要调参，缺少自动化策略
- 未扩展到潜在扩散模型（Latent Diffusion），作者提到这是重要的未来方向
- 未来可探索将此框架与 Consistency Model、Rectified Flow 等新范式结合

## 相关工作与启发
- **vs Carlini et al. (2023) 的记忆化检测工作**: 它们从攻击角度证明记忆化的存在，本文从理论角度解释记忆化的根本原因并提出缓解方法，二者互补
- **vs Bonnaire et al. (2025) 的隐式正则化分析**: 该工作研究训练动态中的隐式正则化如何防止记忆化，本文聚焦于得分函数的结构分析，提供了更直接的几何解释和显式干预方法

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从得分函数权重的 softmax 结构出发建立完整理论框架，视角独特且优美
- 实验充分度: ⭐⭐⭐⭐ 理论验证充分，但实验规模有限（主要在小数据集上）
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨，几何直觉清晰，论述逻辑流畅
- 价值: ⭐⭐⭐⭐ 为扩散模型泛化提供了深刻的理论洞察，实用方法简单有效

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework](taming_score-based_denoisers_in_admm_a_convergent_plug-and-play_framework.md)
- [\[ICML 2025\] Towards a Mechanistic Explanation of Diffusion Model Generalization](../../ICML2025/image_generation/towards_a_mechanistic_explanation_of_diffusion_model_generalization.md)
- [\[CVPR 2026\] Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems](fractals_made_practical_denoising_diffusion_as_partitioned_iterated_function_sys.md)
- [\[CVPR 2026\] Reviving ConvNeXt for Efficient Convolutional Diffusion Models](reviving_convnext_for_efficient_convolutional_diffusion_models.md)
- [\[CVPR 2026\] Bias at the End of the Score: Demographic Biases in Reward Models for T2I](bias_reward_models_t2i.md)

</div>

<!-- RELATED:END -->
