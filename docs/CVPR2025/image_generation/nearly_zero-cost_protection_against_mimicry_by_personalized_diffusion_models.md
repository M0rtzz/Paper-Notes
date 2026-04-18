---
title: >-
  [论文解读] Nearly Zero-Cost Protection Against Mimicry by Personalized Diffusion Models
description: >-
  [CVPR 2025][图像生成][图像保护] 本文提出FastProtect，首个关注延迟的图像保护框架，通过预训练Mixture-of-Perturbations (MoP)替代传统逐图迭代优化，配合Multi-Layer Protection Loss增强训练效果、Adaptive Targeted Protection和Adaptive Protection Strength优化推理，实现了比现有最快方法PhotoGuard快175×（A100 GPU上0.04秒 vs 7秒处理512²图像）的实时保护，同时保持相当的保护效力和更优的不可见性。
tags:
  - CVPR 2025
  - 图像生成
  - 图像保护
  - 对抗扰动
  - 扩散模型
  - 个性化防御
  - 实时保护
  - 混合扰动
---

# Nearly Zero-Cost Protection Against Mimicry by Personalized Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2412.11423](https://arxiv.org/abs/2412.11423)  
**代码**: https://webtoon.github.io/impasto  
**领域**: 图像生成/图像保护  
**关键词**: 图像保护, 对抗扰动, 扩散模型, 个性化防御, 实时保护, 混合扰动

## 一句话总结

本文提出FastProtect，首个关注延迟的图像保护框架，通过预训练Mixture-of-Perturbations (MoP)替代传统逐图迭代优化，配合Multi-Layer Protection Loss增强训练效果、Adaptive Targeted Protection和Adaptive Protection Strength优化推理，实现了比现有最快方法PhotoGuard快175×（A100 GPU上0.04秒 vs 7秒处理512²图像）的实时保护，同时保持相当的保护效力和更优的不可见性。

## 研究背景与动机

**领域现状**：扩散模型的个性化技术（如DreamBooth、LoRA、Textual Inversion）已使恶意用户能够用少量参考图像模仿他人的艺术风格或生成深度伪造。现有保护方法（AdvDM、PhotoGuard、Mist、Glaze、Anti-DB、Impasto）通过在图像上添加对抗扰动使个性化微调失效，但全部依赖推理时迭代优化（PGD），保护一张512²图像需要7-225秒（A100 GPU）。

**现有痛点**：(1) **延迟**是最大障碍——CPU上处理一张图要5-120分钟，普通用户根本无法使用；(2) 扰动不可见性与保护效力的trade-off仍不理想，尤其在纹理简单的卡通/插画上痕迹明显；(3) 随图像分辨率增大（如2048²），延迟呈指数增长，而现代艺术作品多为高分辨率。这三点共同阻碍了保护技术的实际普及。

**核心矛盾**：已有Universal Adversarial Perturbation (UAP)技术可通过预训练消除推理时优化，但UAP的单一扰动是image-agnostic的，直接用于图像保护任务会严重降低保护效力（FID从227.6降到207.6）。如何在保持UAP的速度优势的同时恢复image-specific优化的保护效力？

**本文目标** 设计一个满足三重需求——有效保护、不可见性、实时延迟——的图像保护框架，使普通用户也能在低算力设备上保护自己的图像。

**切入角度**：(1) 训练多个扰动（而非一个）并根据输入图像的VAE latent code自适应选择；(2) 在VAE多层特征空间计算保护loss增强训练效果；(3) 推理时根据输入纹理复杂度自适应选择目标图像和扰动强度。

**核心 idea**：用Mixture-of-Perturbations (MoP)预训练替代逐图PGD迭代，通过基于VAE latent的聚类分配实现半image-specific保护，将推理成本从秒级降至毫秒级。

## 方法详解

### 整体框架

FastProtect分训练和推理两阶段。**训练阶段**：先用K-means++对训练图像的VAE latent聚类建立K组分配函数$\mathcal{A}$；然后用Multi-Layer Protection Loss训练一个全局扰动$\delta_g$和K个组扰动$\Delta = \{\delta_1,...,\delta_K\}$。对3种不同模式重复度的目标图像分别训练3套MoP。**推理阶段**：输入图像经VAE encoder得latent code，通过熵距离选择最匹配的目标图像和对应MoP，从K组中选择扰动，用LPIPS距离图自适应调节区域保护强度，最终输出保护图像。

### 关键设计

1. **Mixture-of-Perturbations (MoP)**：
    - 功能：在保持零推理优化成本的同时提供半image-specific的保护能力
    - 核心思路：预训练K=4个扰动$\Delta=\{\delta_1,...,\delta_4\}$加一个全局扰动$\delta_g$。对每张输入图像，先用VAE encoder提取latent $\mathbf{z}$，通过预训练的K-means++分配函数$\mathcal{A}$选择对应组的扰动$\Delta_k$。保护图像$\hat{\mathbf{x}} = \mathbf{x} + \delta_g + \Delta_k$，其中$k = \mathcal{A}(\mathcal{E}(\mathbf{x}))$。两个扰动分别约束在$(\eta/2)$-ball内
    - 设计动机：UAP的单一扰动无法覆盖图像的多样性（纹理简单vs复杂、自然照片vs卡通），MoP通过聚类让相似图像共享扰动——既增大总容量又保留了部分图像适应性

2. **Multi-Layer Protection (MLP) Loss**：
    - 功能：在预训练时增强扰动的保护效力，不增加推理成本
    - 核心思路：传统texture loss仅在VAE latent $\mathbf{z}$上计算$\|\mathbf{z} - \mathbf{z}_y\|_2^2$。MLP Loss额外利用VAE encoder的中间层特征$\mathcal{F} = \{\mathbf{f}^1,...,\mathbf{f}^L\}$：$\mathcal{L}_T = -\|\mathbf{z} - \mathbf{z}_y\|_2^2 - \frac{\lambda}{L}\sum_{l=1}^L \|\mathcal{F}^l - \mathcal{F}_y^l\|_2^2$
    - 设计动机：仅z-space loss不总能充分将$\mathbf{z}$推向$\mathbf{z}_y$。多层特征空间的辅助loss从多个抽象层次施加约束，增强扰动在后续fine-tuning时的破坏力

3. **Adaptive Targeted Protection + Adaptive Protection Strength**：
    - 功能：推理时自适应选择最佳目标图像和每个区域的扰动强度
    - 核心思路：准备低/中/高纹理重复度的目标图像各一张（对应3套MoP）。推理时用latent entropy距离$t = \arg\min_{i\in\{l,m,h\}} \|\mathcal{H}(\mathbf{z}) - \mathcal{H}(\mathbf{z}_y^i)\|_1$选择最匹配的目标。然后用LPIPS生成感知距离图$\mathbf{M} = \text{LPIPS}(\mathbf{x}, \hat{\mathbf{x}})$，按$\hat{\mathbf{x}} = \mathbf{x} + \mathcal{S}(1-\mathbf{M}) \cdot (\delta_g^t + \Delta_k^t)$调节——纹理复杂区域加强扰动（人眼不敏感），平坦区域减弱
    - 设计动机：简单纹理图像需要低重复度目标而复杂纹理需要高重复度（Fig.3的实验发现）。LPIPS距离图比Impasto使用的传统JND maps更准确且更快

### 损失函数

- 训练：MLP-enhanced texture loss $\mathcal{L}_T = -\|\mathbf{z} - \mathbf{z}_y\|_2^2 - \frac{\lambda}{L}\sum_{l=1}^L \|\mathcal{F}^l - \mathcal{F}_y^l\|_2^2$
- Adam优化器更新扰动参数，初始分辨率512×512，不同分辨率推理时双线性插值

## 实验关键数据

### 主实验表

**512²图像保护对比（LoRA个性化攻击）**：

| 方法 | CPU延迟 | GPU延迟 | Object DISTS↓/FID↑ | Face DISTS↓/FID↑ | Cartoon DISTS↓/FID↑ |
|------|---------|---------|---------------------|-------------------|---------------------|
| PhotoGuard | 370s | 7s | 0.203/223.0 | 0.189/308.7 | 0.209/219.1 |
| Mist | 1440s | 40s | 0.185/217.2 | 0.154/307.5 | 0.223/223.7 |
| Anti-DB | 7278s | 225s | 0.239/214.4 | 0.162/301.4 | 0.294/225.4 |
| **FastProtect** | **2.9s** | **0.04s** | **0.155/223.0** | **0.149/308.9** | **0.186/220.3** |

FastProtect比第二快的PhotoGuard在GPU上快175×，同时在大多数域的不可见性（DISTS）更优。

### 消融实验表

**MoP各组件贡献（Object域FID↑）**：

| 配置 | FID↑ |
|------|------|
| PhotoGuard (iterative PGD) | 227.6 |
| UAP (baseline) | 207.6 |
| MoP (w/o assignment $\mathcal{A}$) | 214.5 |
| MoP (with $\mathcal{A}$) | 225.9 |
| + MLP Loss | 234.6 |
| + Adaptive Target | **238.8** |

MoP+分配函数将UAP的207.6提升到225.9（接近PGD的227.6），MLP Loss和Adaptive Target进一步提升至238.8超越PGD。

### 关键发现

- FastProtect在2048²图像上仍保持实时性（GPU ~0.04s），而其他方法延迟呈指数增长——对高分辨率艺术作品的保护至关重要
- MoP的分配函数$\mathcal{A}$是核心：有分配（225.9）vs无分配的平均（214.5），后者甚至劣于有分配的单一UAP
- 目标图像的纹理重复度与输入图像存在匹配关系（Fig.3）——简单纹理配低重复度目标更有效
- 在黑盒场景中（SD-v2.1、SD-XL、Textual Inversion、DreamStyler），FastProtect的保护效力与PhotoGuard相当或更优
- 推理VRAM仅需1.7GB vs其他方法>8GB——在消费级GPU甚至CPU上可用

## 亮点与洞察

1. **首次聚焦延迟问题**：之前所有保护方法都忽略了"用户等不起"这个最实际的障碍，本文将推理时间从分钟级降到毫秒级是真正的paradigm shift
2. MoP设计巧妙地在image-agnostic (UAP) 和image-specific (PGD) 之间找到了**最佳平衡点**——用VAE latent聚类做粗粒度分配，代价极低但效果接近逐图优化
3. **MLP Loss的"免费午餐"**效应：多层特征约束只在训练时增加成本，推理时完全免费，是一种高效的知识注入方式
4. Adaptive Protection Strength用LPIPS感知图调节区域扰动强度的设计**高度对齐人类视觉感知**

## 局限性

- K=4的扰动数量是手工设定的，更多扰动是否能继续提升有待研究
- 预训练扰动的分辨率固定为512²，其他分辨率通过双线性插值可能引入伪影
- 仅在Stable Diffusion v1.5上训练和主要评估，对新一代模型（SD3、FLUX）的迁移性未验证
- 对JPEG压缩等反制措施的鲁棒性在某些域（如cartoon）上仍需改进（Table 4）
- MoP的聚类分配是一次性预计算的，无法适应分布外的新图像类型

## 相关工作与启发

- **Impasto** [Ahn et al.]: 提出感知导向的不可见保护，但依赖传统JND maps且仍需迭代优化，本文用LPIPS替代更快更准
- **Mist** [Liang & Wu]: 统一texture和semantic loss的保护方法，本文的MLP Loss继承了texture loss并扩展到多层
- **UAP** [Moosavi-Dezfooli et al.]: 通用对抗扰动的概念，本文将其从分类攻击迁移并改进为MoP用于图像保护
- **PhotoGuard** [Salman et al.]: 之前最快的保护方法（GPU 7s），FastProtect比它快175×
- **启发**：预训练扰动+自适应推理的范式可能适用于更广泛的对抗攻防场景——任何需要per-instance优化的任务都可能受益于类似的"离线预训练+在线选择"策略

## 评分

⭐⭐⭐⭐ — 首次将图像保护的延迟降至实时水平，解决了该领域最关键的实用性障碍。MoP设计简单有效，消融实验清晰地验证了每个组件的贡献。来自NAVER WEBTOON AI的产业背景使研究动机很接地气。美中不足是对新一代扩散模型的泛化性验证不足。
