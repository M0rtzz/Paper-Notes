---
title: >-
  [论文解读] Attention to Neural Plagiarism: Diffusion Models Can Plagiarize Your Copyrighted Images!
description: >-
  [ICCV 2025][目标检测][神经抄袭] 揭示"神经抄袭"威胁——扩散模型可轻松复制受版权保护的图像（包括受水印保护的图像），提出基于"锚点与垫片"的通用攻击框架，通过在交叉注意力机制中搜索扰动实现从粗到细的语义修改，绕过从可见商标到隐形水印的各类版权保护。
tags:
  - ICCV 2025
  - 目标检测
  - 神经抄袭
  - 扩散模型
  - 版权保护
  - 水印去除
  - 注意力扰动
---

# Attention to Neural Plagiarism: Diffusion Models Can Plagiarize Your Copyrighted Images!

**会议**: ICCV 2025  
**arXiv**: [2603.00150](https://arxiv.org/abs/2603.00150)  
**代码**: [https://github.com/zzzucf/Neural-Plagiarism](https://github.com/zzzucf/Neural-Plagiarism)  
**领域**: AI安全 / 版权保护  
**关键词**: 神经抄袭, 扩散模型, 版权保护, 水印去除, 注意力扰动

## 一句话总结
揭示"神经抄袭"威胁——扩散模型可轻松复制受版权保护的图像（包括受水印保护的图像），提出基于"锚点与垫片"的通用攻击框架，通过在交叉注意力机制中搜索扰动实现从粗到细的语义修改，绕过从可见商标到隐形水印的各类版权保护。

## 研究背景与动机
扩散模型的生成能力日益强大，引发了数据版权侵犯的严重担忧。这种被称为"神经抄袭"的威胁有两种形式：

**伪造攻击（Forgery Attack）**：生成视觉相似的副本同时移除水印，使版权验证失败$\mathcal{V}(x^*) \neq w$

**模糊攻击（Ambiguity Attack）**：替换为新水印制造所有权争议$\mathcal{V}(x^*) = w^*$

现有防护措施包括法律框架（GDPR、版权法）和技术手段（可见/隐形水印）。但现有的水印去除方法（如Regen、Rinse）仅通过简单的加噪-去噪过程去除水印，效果有限且往往引入明显噪声。

直接优化目标函数$\min_{x_T^*} d_{visual}(x_0^w, x_0^*) - \gamma d_{latent}(x_T, x_T^*)$面临三个挑战：

**显存爆炸**：每个时间步需计算梯度$\frac{\partial x_{t-1}^*}{\partial x_t^*}$，10步需>100GB显存

**过度平滑**：跳步估计梯度导致图像模糊

**噪声输出**：高扰动产生噪声而非有意义的语义变化

核心idea：用"锚点"（反转隐变量序列）保持轨迹，用"垫片"（学习的扰动）在特定时间步微调语义，类似安装门时用垫片调整间距——在保持门框对齐的同时调整局部空间。

## 方法详解

### 整体框架
攻击流水线分为两阶段：
1. **锚点获取**：将版权图像通过VAE编码为$\hat{x}_0$，用DPM solver反转生成过程得到锚点序列$\{\hat{x}_1, ..., \hat{x}_T\}$
2. **垫片搜索**：从选定时间步K开始反向采样，在特定时间步$\mathcal{T}_{select}$上优化垫片$\delta_t$（注入到交叉注意力的文本嵌入中），使生成图像视觉相似但隐变量偏离锚点

### 关键设计

1. **锚点与垫片优化（Anchors and Shims）**:

    - 功能：解耦隐变量的长链依赖，逐时间步独立优化扰动
    - 核心思路：保存完整的反转隐变量轨迹作为锚点$\{\hat{x}_t\}$，定义垫片$\delta_t$为搜索变量：$x_t^* = \Delta_{\delta_t \in \mathcal{S}}(x_t, \delta_t)$
    - 范数约束：$\mathcal{L}_{norm}(t) = \max(0, \hat{\varepsilon}_t - \|\delta_t\|)$，确保垫片足够大以偏离锚点（从而破坏水印）
    - 设计动机：通过保存锚点解耦时间步链，每步仅需一步反向传播的显存，解决了>100GB的显存问题

2. **基于注意力的语义搜索（Semantic Search via Attention Perturbation）**:

    - 功能：在语义空间中进行可控的修改搜索
    - 核心思路：不直接扰动隐变量，而是扰动交叉注意力中的文本嵌入——在空字符串嵌入$\mathbf{e}_\emptyset$上注入垫片$\delta_t$
    - 语义保持损失：$\mathcal{L}_{semantic}(t) = -\frac{\mathbf{e}_\emptyset \cdot (\mathbf{e}_\emptyset + \delta_t)}{\|\mathbf{e}_\emptyset\| \|\mathbf{e}_\emptyset + \delta_t\|}$（最大化余弦相似度）
    - 对齐损失：$\mathcal{L}_{align}(t) = d(x_{t-1}, \hat{x}_{t-1})$，保证扰动后输出接近锚点
    - 关键洞察：不同时间步的垫片控制不同粒度的语义变化——大时间步改变大语义（如颜色、形状），小时间步改变细节纹理
    - 设计动机：交叉注意力是扩散模型中语义控制的核心机制，通过扰动$K$和$V$的输入实现精确语义操控

3. **迭代搜索过程**:

    - 功能：在选定时间步上联合优化垫片
    - 联合目标：$\min_{\delta_t} \mathcal{L}_{norm}(t) + \gamma_1 \mathcal{L}_{semantic}(t) + \gamma_2 \mathcal{L}_{align}(t)$
    - 超参数：$\gamma_1 = 10^5, \gamma_2 = 0.1, \hat{\varepsilon}_t = 10$
    - 使用Adam优化器（学习率0.01），权重衰减$10^{-3}$，梯度裁剪（最大范数1.0）
    - DPM solver，50个时间步
    - 两种启动模式：
        - **晚期启动 + 噪声初始化**（$K=140$，垫片在100和60步）：小扰动，适合隐形水印去除
        - **早期启动 + 反转初始化**（$K=1000$，垫片在600和200步）：大扰动，大幅语义修改

### 攻击算法（Algorithm 1）
```
输入：版权图像 x^w
1. VAE编码 → x̂_0
2. DPM Solver反转 → 锚点序列 {x̂_1,...,x̂_T}
3. 初始化 x*_K = x̂_K 或 noisy latent
4. For t = K to 1:
   If t ∈ T_select:  # 需要扰动的时间步
     初始化 δ_t = 0
     While 未收敛:
       x*_{t-1} = x*_t - ζ_t·ε_θ(x*_t, t, e_∅ + δ_t)
       δ_t -= η·∇L(t, δ_t, x*_{t-1}, x̂_{t-1})
   Else:  # 正常去噪
     x*_{t-1} = x*_t - ζ_t·ε_θ(x*_t, t, e_∅)
5. 输出 x* = D(x*_0)
```

## 实验关键数据

### 主实验（隐形水印去除，MS-COCO）

| 方法 | DwtDctSvd BA↓ | DwtDctSvd ACC↓ | RivaGAN BA↓ | RivaGAN ACC↓ | PSNR↑ | SSIM↑ |
|------|--------------|----------------|-------------|-------------|-------|-------|
| Regen | 0.64 | 0.15 | 0.60 | 0.05 | 26.21 | 0.75 |
| Rinse | 0.54 | 0.04 | 0.52 | 0.01 | 23.68 | 0.68 |
| **Ours(晚期)** | **0.52** | **0.01** | 0.56 | 0.02 | 25.27 | 0.73 |
| **Ours(早期)** | **0.52** | **0.03** | **0.52** | **0.00** | 21.16 | 0.70 |

| 方法 | Stable Signature T@1%F↓ | Tree-Ring T@1%F↓ |
|------|------------------------|------------------|
| Regen | 0.00 | **1.00** |
| **Ours** | **0.00** | 0.98 |

### 消融实验（语义修改程度 vs 时间步选择）

| 启动模式 | 垫片时间步 | 效果 | 适用场景 |
|---------|-----------|------|---------|
| 晚期(K=140) + 噪声 | 100, 60 | 小语义变化，保持图像质量 | 隐形水印去除 |
| 早期(K=1000) + 反转 | 600, 200 | 大语义变化（改变发型、服装等） | 可见版权绕过 |
| - | 大时间步单步 | 改变颜色、形状等全局语义 | IP字符修改 |
| - | 小时间步单步 | 改变纹理、细节等局部语义 | 精细伪造 |

### 关键发现
- Bit准确率降至接近50%（随机水平），说明水印被有效移除
- 与Regen/Rinse相比，本方法在保持图像质量的同时水印去除更彻底
- 对Stable Signature有效（T@1%F降至0），但**无法去除Tree-Ring水印**——因为Tree-Ring在傅里叶空间注入局部模式，隐变量的全局偏移不影响局部傅里叶特征
- 模糊攻击成功：可在已攻击图像上嵌入新水印，两个水印同时可检测（Tree-Ring T@1%F双方均为1.00/0.99）
- 生成100张Elon Musk变体，GPT仲裁无法确定性地识别任何一张为本人
- 可见版权攻击：成功修改Disney Elsa的标志性发型和服装、Monet画作风格、Van Gogh签名

## 亮点与洞察
- "锚点与垫片"的比喻直观且方法设计优雅——既解决了显存问题，又提供了多粒度的语义控制
- 注意力扰动是一种低成本且精确的语义操控方式，仅修改文本嵌入而非整个网络
- 纯梯度搜索，不需要额外训练或微调，适用于任何预训练扩散模型
- 清晰揭示了现有版权保护的脆弱性，为防御研究提供了重要的攻击基线
- 时间步选择提供了coarse-to-fine的语义控制能力

## 局限与展望
- 无法去除Tree-Ring水印（傅里叶域局部模式对隐变量全局偏移免疫）
- 早期启动模式下FID较高，语义变化过大时图像质量下降
- 未结合ControlNet、额外图像编码器、负面提示词等增强手段（论文明确指出留作future work）
- 从防御角度看，研究如何使水印抵抗这种基于注意力的扰动搜索是急需的方向
- 仅在Stable Diffusion上测试，对其他扩散架构（如DiT）的效果未知

## 相关工作与启发
- **vs Regen**: Regen简单加噪再去噪，水印去除不彻底且引入明显噪声；本方法通过优化垫片精确控制语义修改
- **vs Rinse**: Rinse多次迭代Regen过程，改善了水印去除但图像质量严重下降（FID 82-87）
- **vs Tree-Ring水印**: Tree-Ring在傅里叶空间嵌入水印的策略被证明对隐变量扰动具有天然鲁棒性，值得防御方法借鉴
- **vs Stable Signature**: 虽然是sota水印方法，但因水印在VAE decoder中实现，隐变量的任何偏移都足以绕过

## 评分
- 新颖性: ⭐⭐⭐⭐ "锚点与垫片"框架设计精巧，注意力扰动搜索视角新颖
- 实验充分度: ⭐⭐⭐⭐ 覆盖多种水印方法、可见/隐形版权、伪造/模糊攻击、真实版权图像
- 写作质量: ⭐⭐⭐⭐ 问题形式化清晰，攻击类型定义严谨，但部分符号和术语偏多
- 价值: ⭐⭐⭐⭐⭐ 揭示了紧迫的AI安全威胁，对版权保护社区具有警示意义和防御研究的基准价值

<!-- RELATED:START -->

## 相关论文

- [DiffDoctor: Diagnosing Image Diffusion Models Before Treating](diffdoctor_diagnosing_image_diffusion_models_before_treating.md)
- [Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention](../../CVPR2025/object_detection/efficient_event-based_object_detection_a_hybrid_neural_network_with_spatial_and_.md)
- [Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion](diffusion_curriculum_synthetic-to-real_data_curriculum_via_image-guided_diffusio.md)
- [Generalized Diffusion Detector: Mining Robust Features from Diffusion Models for Domain-Generalized Detection](../../CVPR2025/object_detection/generalized_diffusion_detector_mining_robust_features_from_diffusion_models_for_.md)
- [Can OOD Object Detectors Learn from Foundation Models?](../../ECCV2024/object_detection/can_ood_object_detectors_learn_from_foundation_models.md)

<!-- RELATED:END -->
