---
title: >-
  [论文解读] Transferable Black-Box One-Shot Forging of Watermarks via Image Preference Models
description: >-
  [NeurIPS 2025][图像生成][水印伪造] 本文提出一种基于图像偏好模型的黑盒水印伪造方法，仅需单张水印图像即可通过反向传播从中提取水印并粘贴到任意新图像上，在不访问水印算法的条件下有效伪造多种后处理水印方案。
tags:
  - NeurIPS 2025
  - 图像生成
  - 水印伪造
  - 偏好模型
  - 单样本攻击
  - 黑盒攻击
  - 后处理水印
---

# Transferable Black-Box One-Shot Forging of Watermarks via Image Preference Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.20468](https://arxiv.org/abs/2510.20468)  
**代码**: https://github.com/facebookresearch/videoseal/tree/main/wmforger  
**领域**: AI安全 / 图像水印攻击  
**关键词**: 水印伪造, 偏好模型, 单样本攻击, 黑盒攻击, 后处理水印

## 一句话总结

本文提出一种基于图像偏好模型的黑盒水印伪造方法，仅需单张水印图像即可通过反向传播从中提取水印并粘贴到任意新图像上，在不访问水印算法的条件下有效伪造多种后处理水印方案。

## 研究背景与动机

数字水印是确保内容真实性和溯源的关键技术，尤其在生成式 AI 时代，后处理水印（post-hoc watermarking）因其模块化和易部署性被广泛采用（如 Google DeepMind 的 SynthID、Meta 的 Video Seal 等）。EU AI Act 和美国 AI 行政令也明确要求 AI 生成内容标注水印。

然而，现有水印安全研究主要聚焦于**水印去除**（removal），对**水印伪造**（forging）——即攻击者窃取合法水印并加到恶意内容上——关注甚少。水印伪造的威胁在于：它可以让虚假内容看起来像是合法来源生成的，也可以用大量假阳性淹没检测系统。

已有的伪造方法面临严重的实际限制：
- Wang et al. (2021) 需要成对的原始/水印图像
- Warfare (Li et al., 2023) 和 Dong et al. (2025) 需要数千张含相同隐藏消息的水印图像
- 这些条件在实际黑盒场景中几乎无法满足

本文的切入点是：**能否在只有一张水印图像、不知道水印算法的条件下完成伪造？**

## 方法详解

### 整体框架

两步流程：（1）训练一个图像偏好模型 $R$，学会区分"有伪影"和"干净"的图像；（2）用偏好模型作为代理损失，通过梯度上升从水印图像中提取水印，再粘贴到新图像上。

### 关键设计

1. **偏好模型训练**：

    - 架构：ConvNeXt V2-Tiny，输入 RGB 图像，输出标量分数 $R(\mathbf{x}) \in \mathbb{R}$
    - 损失函数：Bradley-Terry 排名损失 $-\mathbb{E}[\log \sigma(R(\mathbf{x}^+) - R(\mathbf{x}^-))]$，其中 $\mathbf{x}^+$ 为原始图像（偏好），$\mathbf{x}^-$ 为加入合成伪影的图像（不偏好）
    - **核心创新——合成伪影训练数据**：完全不使用任何真实水印！在傅里叶空间中随机生成三种类型的伪影：（a）波形伪影——非零振幅集中在若干随机极坐标点；（b）噪声——振幅按高斯衰减随机采样；（c）线形伪影——振幅在随机水平/垂直线上非零。伪影缩放到 $[-0.05, 0.05]$ 范围内，随机选择 RGB 或灰度，50% 概率乘以 JND 图
    - **对抗训练**：每隔一个 batch 用梯度扰动生成的对抗样本 $\tilde{\mathbf{x}}^- = \mathbf{x}^- + \epsilon \cdot \nabla R(\mathbf{x}^-)$ 替换负样本，确保模型产生语义可解释的梯度。没有对抗训练时，反向传播会产生棋盘格伪影

2. **水印提取与伪造**：

    - 给定水印图像 $\mathbf{x}_w$，通过最大化偏好分数来估计水印：$\hat{w} = \arg\max_\delta R(\mathbf{x}_w - \delta)$
    - 使用 SGD 优化器，固定学习率 0.05，50-500 步梯度上升
    - 提取的水印 $\hat{w} = \mathbf{x}_w - \hat{\mathbf{x}}$ 可直接加到任意新图像：$\mathbf{y}_{\hat{w}} = \mathbf{y} + \hat{w}$
    - 对于不同分辨率的图像，先将水印图像缩放到较小分辨率提取水印，再将水印上采样到目标分辨率

3. **水印去除**：同样的方法，只需计算 $\hat{\mathbf{x}} = \mathbf{x}_w - \text{resize}(\hat{w}')$ 即可得到去水印图像

### 损失函数 / 训练策略

- 排名损失（Equation 2），不使用二分类交叉熵或 hinge 损失（消融实验证明排名损失最优）
- 训练数据：SA-1b 数据集，图像缩放到 768×768 后随机裁剪 256×256
- 从头训练 120k 步，8 GPU，batch size 16/GPU，AdamW 优化器，学习率 $1 \times 10^{-5}$
- 训练时间约 60 小时（V100 GPU）

## 实验关键数据

### 主实验（水印伪造）

| 方法 | CIN Bit acc.↑ | MBRS Bit acc.↑ | TrustMark Bit acc.↑ | Video Seal Bit acc.↑ | PSNR↑ |
|------|-------------|---------------|-------------------|-------------------|-------|
| Gray image blending* | 1.00 | 0.80 | 0.54 | 0.83 | 52.9 |
| Warfare (n=1000) | 0.93 | 0.50 | 0.53 | 0.74 | 39.6 |
| DiffPure (FLUX) | 1.00 | 0.83 | 0.59 | 0.75 | 26.6 |
| Image averaging (n=100) | 1.00 | 0.91 | 0.61 | 0.59 | 26.2 |
| **Ours (n=1)** | **1.00** | **0.83** | **0.61** | **0.83** | **31.3** |

*Gray image blending 需要访问水印 API，不可用于实际攻击

### 消融实验

| 配置 | CIN | MBRS | TrustMark | Video Seal | PSNR |
|------|-----|------|-----------|-----------|------|
| 二分类交叉熵损失 | 0.60 | 0.53 | 0.52 | 0.47 | 39.9 |
| Hinge 损失 | 0.62 | 0.55 | 0.52 | 0.47 | 44.1 |
| 无对抗扰动 | 0.97 | 0.65 | 0.52 | 0.49 | 34.7 |
| 用真实水印训练 | 1.00 | 0.67 | 0.58 | 0.77 | 36.9 |
| **完整方法** | **1.00** | **0.83** | **0.61** | **0.83** | **31.3** |

### 关键发现

- **仅需 1 张水印图像** 即可达到甚至超过需要 100-1000 张图像的方法的性能
- 对 Video Seal 等**内容感知**水印方法，Image averaging 完全失效（0.59），而本文方法仍能达到 0.83
- 对 CIN 和 MBRS 等内容无关水印，简单的 Image averaging 就很有效，因为这些方法的水印模式基本固定
- TrustMark 最难伪造，因为其编码器和解码器都高度依赖输入图像内容
- 排名损失远优于分类损失——因为正负样本极为相似，不存在全局分类边界
- **惊人发现**：用程序化合成伪影训练的模型比用真实水印训练的效果更好（line 4 vs. line 5），原因是真实水印多样性不足导致过拟合
- 对抗训练是获得可解释梯度的关键，缺了它梯度方向完全无效

## 亮点与洞察

- **极低资源威胁模型**：黑盒 + 单样本的设定是目前最现实的攻击场景
- **偏好模型的创造性应用**：借鉴 LLM 的 RLHF 思路，用排名损失训练图像质量判别器作为水印检测的代理
- **合成伪影替代实际水印**：完全不需要任何水印模型参与训练，而且效果更好——这一反直觉的发现深具启发性
- **对抗训练确保梯度可解释性**：与对抗鲁棒性文献的联系（Santurkar et al., 2019）巧妙且深刻
- 方法同时支持水印去除和伪造，并且在去除任务上也极具竞争力

## 局限与展望

- 仅针对后处理水印方法，无法伪造语义水印（如 Tree-Ring、RingID 等通过改变生成内容结构实现的水印）
- 优化步数较多时可能模糊图像中的高频纹理区域（如水面、树木）
- 单张水印图像提取的水印质量受限于该特定图像的特征
- 防御建议：确保解码器真正具备内容感知能力（如训练解码器拒绝来自不同源图像的水印）

## 相关工作与启发

- Warfare 和 Image averaging 代表了需要大量水印图像的传统伪造方法
- DiffPure 和 CtrlRegen 是基于扩散模型的去除方法，但会引入幻觉细节
- UnMarker 在频域层面操作去除水印
- 本文为水印安全研究敲响警钟：即使是内容感知的后处理水印也可能被低资源攻击破解，需要重新思考解码器的设计

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （偏好模型+合成伪影的组合非常原创）
- 实验充分度: ⭐⭐⭐⭐⭐ （4种水印方法、多种对比方法、充分消融）
- 写作质量: ⭐⭐⭐⭐⭐ （结构清晰，威胁模型定义明确）
- 价值: ⭐⭐⭐⭐⭐ （揭示了当前水印方案的根本性安全缺陷）

<!-- RELATED:START -->

## 相关论文

- [WMCopier: Forging Invisible Image Watermarks on Arbitrary Images](wmcopier_forging_invisible_image_watermarks_on_arbitrary_images.md)
- [PPO-MI: Efficient Black-Box Model Inversion via Proximal Policy Optimization](../../ICML2025/image_generation/ppo-mi_efficient_black-box_model_inversion_via_proximal_policy_optimization.md)
- [Amortized Sampling with Transferable Normalizing Flows](amortized_sampling_with_transferable_normalizing_flows.md)
- [BlackMirror: Black-Box Backdoor Detection for Text-to-Image Models via Instruction-Response Deviation](../../CVPR2026/image_generation/blackmirror_black-box_backdoor_detection_for_text-to-image_models_via_instructio.md)
- [Where's the Liability in the Generative Era? Recovery-Based Black-Box Detection of AI-Generated Content](../../CVPR2025/image_generation/wheres_the_liability_in_the_generative_era_recovery-based_black-box_detection_of.md)

<!-- RELATED:END -->
