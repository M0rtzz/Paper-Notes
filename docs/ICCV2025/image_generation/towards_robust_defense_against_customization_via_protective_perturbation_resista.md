---
title: >-
  [论文解读] Towards Robust Defense against Customization via Protective Perturbation Resistant to Diffusion-based Purification
description: >-
  [ICCV 2025][图像生成][Protective Perturbation] 提出 AntiPure，一种针对扩散模型净化（purification）过程的对抗扰动方法，通过 Patch-wise Frequency Guidance 和 Erroneous Timestep Guidance 两种引…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "Protective Perturbation"
  - "Anti-Purification"
  - "DreamBooth"
  - "扩散模型"
  - "adversarial attack"
---

# Towards Robust Defense against Customization via Protective Perturbation Resistant to Diffusion-based Purification

**会议**: ICCV 2025  
**arXiv**: [2509.13922](https://arxiv.org/abs/2509.13922)  
**代码**: 无（未提及）  
**领域**: 图像生成 / 对抗防御 / 扩散模型安全  
**关键词**: Protective Perturbation, Anti-Purification, DreamBooth, 频域引导, 时间步引导

## 一句话总结
提出AntiPure——一种抗净化（anti-purification）保护性扰动方法，通过Patch-wise频域引导和错误时间步引导两种机制，使保护性扰动在被扩散净化后仍能有效干扰定制化生成（DreamBooth/LoRA），实现最小感知差异和最大输出失真的双重目标。

## 研究背景与动机

扩散模型（如Stable Diffusion）的定制化生成能力（DreamBooth、LoRA）带来了深度伪造和版权侵权的安全风险。保护性扰动（Protective Perturbation）方法通过向图像注入不可感知的对抗噪声来阻碍定制化fine-tuning，是目前的主要防御手段。

然而，**扩散净化**（Diffusion-based Purification，如DiffPure、GrIDPure）可以在定制化之前移除保护性扰动，使图像重新暴露于恶意伪造风险中。现有保护性扰动方法（AdvDM、Mist、Anti-DB、SimAC）几乎不考虑"净化-定制化"（Purification-Customization, P-C）工作流，导致在净化后效果大幅下降。

本文首次形式化了抗净化任务，分析了抗净化比抗定制化更困难的三个核心原因：

**缺乏脆弱组件**：DDPM中只有UNet，不像LDM有脆弱的VAE编码器可被攻击

**无训练冻结参数**：净化过程不涉及fine-tuning，对抗样本无法影响模型先验

**固定高时间步去噪**：净化本质是固定高时间步的生成过程，低频结构难以被修改

## 方法详解

### 整体框架
AntiPure不试图在净化后保留反定制化扰动，而是**直接攻击净化模型本身**，利用净化的弱点来产生扰动后的失真。攻击目标为最大化净化输出与原始图像的差异：

$$\delta^{adv}_{max} = \arg\max_{\|\delta\|_\infty \leq \eta} \|\text{Pure}(x_0 + \delta) - (x_0 + \delta)\|_\infty$$

总攻击损失结合三个组件：

$$\mathcal{L}_{pgd}(x_0; \delta^{adv}) = \mathbb{E}_{\epsilon, t}\left(\mathcal{L}_{ddpm} + \lambda_1 e^{\bar{\alpha}_t - 1}\mathcal{L}_{fre} + \lambda_2 e^{\mathcal{L}_{err-t}}\right)$$

其中 $t \sim \mathcal{U}(1, t^p)$ 限制在净化步数范围内，$\lambda_1 = \lambda_2 = 0.5$。

### 关键设计一：Patch-wise Frequency Guidance (PFG)
冻结网络参数中的clean先验使净化模型能恢复高质量图像，但对**高频分量**的控制力较弱。PFG利用这一弱点：

1. 将对抗样本扩散到时间步 $t$，用UNet预测去噪图像 $\hat{x}_0$
2. 将 $\hat{x}_0$ 分割为 $P$ 个 $s \times s$ 的patch，对每个patch做DCT变换
3. 提取每个patch频谱图中右下角1/4（高频区域），求平均后经sigmoid归一化

$$\mathcal{L}_{fre}(x_0; \delta^{adv}) = \sigma\left(\mathbb{E}_P \frac{4}{s^2} \sum_{m,n=s/2}^{s-1} \text{PatchDCT}(\hat{x}_0, s)_{m,n}\right)$$

该损失增强净化模型去噪预测中的高频分量，间接强化对抗扰动的高频元素，形成均匀网格图案。由于攻击目标是高频信息，局部结构改变极小，确保对人眼的感知一致性。

权重系数 $e^{\bar{\alpha}_t - 1}$ 使 $\mathcal{L}_{fre}$ 的影响随时间步 $t$ 减小而增大，在低时间步更积极地调制高频。

### 关键设计二：Erroneous Timestep Guidance (ETG)
净化本质上是高时间步去噪已固定的生成过程，但可以鼓励噪声预测器在不同时间步的输出尽可能接近——即注入使UNet无法区分"当前应做什么"的对抗输入：

$$\mathcal{L}_{err-t}(x_0; \delta^{adv}) = -\|\epsilon_\theta(x_t, t_{err}) - \epsilon_\theta(x_t, t)\|_2^2$$

选择一个错误时间步 $t_{err}$ 作为UNet输入，最小化正确与错误时间步预测的差异。由于MSE值很小，用指数函数 $e^{\mathcal{L}_{err-t}}$ 使优化更激进。

### 损失函数
总损失用PGD（Projected Gradient Descent）最大化：
- $\mathcal{L}_{ddpm}$：标准DDPM训练损失（最大化以破坏去噪）
- $\mathcal{L}_{fre}$：Patch-wise频域引导（调制高频分量）
- $\mathcal{L}_{err-t}$：错误时间步引导（混淆时间步感知）

## 实验

### 实验设置
- **数据集**：CelebA-HQ和VGGFace2各50个ID、每ID 12张512×512图像
- **基线方法**：AdvDM、Mist、Anti-DB、SimAC
- **净化设置**：GrIDPure，2轮×20迭代，$t^p=10$
- **定制化方法**：DreamBooth、LoRA
- **评估指标**：FID↑、ISM↓、FDFR、BRISQUE↑、LPIPS↓

### 主实验结果（DreamBooth，P-C工作流）

| 数据集 | 方法 | FID↑ | ISM↓ | BRISQUE↑ |
|--------|------|------|------|----------|
| CelebA-HQ | AdvDM | 77.51 | 0.6561 | 31.33 |
| CelebA-HQ | Mist | 70.23 | 0.6688 | 37.00 |
| CelebA-HQ | Anti-DB | 78.84 | 0.6422 | 31.76 |
| CelebA-HQ | SimAC | 67.37 | 0.6734 | 33.73 |
| CelebA-HQ | **AntiPure** | **81.15** | **0.6112** | **43.60** |
| VGGFace2 | AdvDM | 83.90 | 0.5923 | 37.42 |
| VGGFace2 | Anti-DB | 90.29 | 0.5938 | 38.35 |
| VGGFace2 | **AntiPure** | **90.77** | **0.5475** | **46.01** |

AntiPure在两个数据集上所有指标均最优。净化后其他方法效果显著下降，AntiPure则能更好地保持保护效果。

### LoRA定制化结果

| 数据集 | 方法 | FID↑ | ISM↓ | BRISQUE↑ |
|--------|------|------|------|----------|
| VGGFace2 | Anti-DB | 117.89 | 0.5723 | 58.56 |
| VGGFace2 | **AntiPure** | **127.67** | **0.5428** | **69.97** |

在LoRA设置下AntiPure同样优于所有基线，特别是ISM指标（面部特征相似度）差距明显。

### 消融实验：不同净化迭代次数

| 扰动方法 | 净化迭代 | FID↑ | ISM↓ | BRISQUE↑ |
|---------|---------|------|------|----------|
| Anti-DB | 10 | 124.62 | 0.6020 | 32.74 |
| Anti-DB | 40 | 77.30 | 0.6391 | 30.34 |
| AntiPure | 10 | 54.45 | 0.6362 | 40.27 |
| AntiPure | 40 | 78.21 | 0.5994 | 47.54 |

关键发现：Anti-DB随净化迭代增加效果逐渐减弱，而AntiPure反而**越来越鲁棒**——在30-40次迭代后超越Anti-DB。这验证了直接攻击净化过程本身（而非试图保留反定制化扰动）的策略优越性。

### 感知一致性（LPIPS）

AntiPure在相同 $l_\infty$ 约束下实现了**最小的感知差异**（LPIPS最低），同时提供最好的P-C工作流攻击效果。这归功于PFG有效避免了对低频信息的修改。

## 亮点与洞察

- **首次形式化抗净化任务**：系统分析了为何anti-purification比anti-customization更难（缺乏脆弱组件、冻结参数、固定高时间步），为后续研究奠定理论基础
- **攻击策略转变**：不试图保留反定制化扰动穿过净化，而是直接攻击净化模型本身——思路简洁而有效
- **频域弱点利用**：发现净化模型对高频分量的控制力弱于低频，PFG精准利用了这一结构性弱点
- **反直觉的鲁棒性**：AntiPure随净化迭代增多反而更强，与其他方法的递减趋势形成鲜明对比
- **最小感知-最大失真**：同时实现了对人眼最不可见（LPIPS最低）和对模型最有破坏力（FID/ISM最优）的双重目标

## 局限性

- 无法实现语义级别的结构性失真（如改变人脸身份），只能产生局部纹理级别的退化
- 白盒攻击设定——假设知道净化模型的架构和参数
- 仅在人脸数据集上验证，对艺术风格、物体等其他场景的泛化性未测试
- PGD攻击的计算开销较大（需要多步梯度回传）

## 相关工作

- **保护性扰动**：AdvDM、Mist、Anti-DB、SimAC等通过白盒攻击生成对抗噪声
- **扩散净化**：DiffPure、DensePure、GrIDPure利用扩散模型的去噪能力移除对抗噪声
- **定制化生成**：DreamBooth、LoRA、Textual Inversion等 fine-tuning 方法

## 评分
- 新颖性：⭐⭐⭐⭐ — 首次系统性地解决"净化-定制化"工作流下的保护性扰动问题
- 有效性：⭐⭐⭐⭐ — 在两个数据集、两种定制化方法上全面超越基线
- 实用性：⭐⭐⭐ — 白盒攻击假设和计算开销限制实际部署
- 写作质量：⭐⭐⭐⭐ — 问题形式化清晰，分析逻辑严密
# Towards Robust Defense against Customization via Protective Perturbation Resistant to Diffusion-based Purification

**会议**: ICCV 2025  
**arXiv**: [2509.13922](https://arxiv.org/abs/2509.13922)  
**代码**: 无（未提及）  
**领域**: 图像生成 / 对抗防御 / 扩散模型安全  
**关键词**: Protective Perturbation, Anti-Purification, DreamBooth, Diffusion Purification, adversarial attack

## 一句话总结
提出 AntiPure，一种针对扩散模型净化（purification）过程的对抗扰动方法，通过 Patch-wise Frequency Guidance 和 Erroneous Timestep Guidance 两种引导机制，生成在净化后仍能持续干扰定制化微调的保护性扰动，在"净化-定制化"工作流中全面超越现有防护方法。

## 研究背景与动机

### 问题背景
Stable Diffusion 等扩散模型的定制化微调技术（如 DreamBooth、LoRA）虽然强大，但也带来了深度伪造和版权侵权的严重安全威胁。保护性扰动（Protective Perturbation）通过在图像中注入不可感知的对抗噪声来干扰微调输出，是一种有前景的防御手段。

### 核心挑战
现有保护性扰动（如 AdvDM、Anti-DreamBooth）可以被扩散模型净化（如 DiffPure、GrIDPure）轻松去除。净化过程通过向对抗图像添加噪声再去噪，有效消除了对抗扰动，使保护失效。在实际场景中，恶意用户可以先净化图像再微调，形成"净化-定制化"（P-C）工作流，导致现有防护方法几乎完全失效。

### 关键洞察
作者深入分析了 anti-purification 比 anti-customization 更困难的三个核心原因：
1. **缺乏脆弱组件**：LDM 有易受攻击的 VAE encoder，而 DDPM 净化模型只有鲁棒的 UNet
2. **无需训练的冻结参数**：净化过程不需要微调，对抗样本无法通过数据毒化影响模型先验
3. **固定高时间步去噪**：净化的去噪从高时间步开始，低频结构已被锁定，攻击被限制在高频分量

## 方法详解

### 整体框架
AntiPure 不试图保留 anti-customization 扰动通过净化，而是直接攻击净化模型本身。核心思路是：即使后续定制化正常运行，净化过程引入的失真也会导致学到的概念偏离原始图像。

### 问题形式化
理想的抗净化扰动优化目标为：

$$\delta^{adv} = \arg\max_{\|\delta\|_\infty \leq \eta} \min_{\theta_c} \mathbb{E}_x \mathcal{L}_{ldm}(\text{Pure}(x_0 + \delta); \theta_c)$$

由于直接反向传播计算图过深，作者将其分解为最大化净化前后的差异：

$$\delta^{adv'} = \arg\max_{\|\delta\|_\infty \leq \eta} \|\text{Pure}(x_0 + \delta) - (x_0 + \delta)\|_\infty$$

### 关键设计一：Patch-wise Frequency Guidance (PFG)
冻结参数中的清洁图像先验使净化模型能良好恢复低频结构，但对高频分量控制较弱。PFG 利用这一弱点：

1. 对噪声对抗样本 $x_t$ 使用 UNet 预测去噪图像 $\widehat{x}_0$
2. 将 $\widehat{x}_0$ 分解为 patch，对每个 patch 做 DCT 变换
3. 提取高频分量（DCT 谱图右下角四分之一）并最大化：

$$\mathcal{L}_{fre}(x_0; \delta^{adv}) = \sigma\left(\mathbb{E}_P \frac{4}{s^2} \sum_{m,n=s/2}^{s-1} \text{PatchDCT}(\widehat{x}_0, s)_{m,n}\right)$$

PFG 增强净化模型预测中的高频分量，间接强化对抗扰动的高频元素，形成均匀网格模式。由于攻击目标是高频，局部结构信息变化最小，保证人类感知一致性。

### 关键设计二：Erroneous Timestep Guidance (ETG)
净化过程可视为高时间步去噪被固定的生成过程。ETG 通过注入对抗噪声，使 UNet 难以区分不同时间步的适当行为：

$$\mathcal{L}_{err\text{-}t}(x_0; \delta^{adv}) = -\|\epsilon_\theta(x_t, t_{err}) - \epsilon_\theta(x_t, t)\|_2^2$$

选择错误时间步 $t_{err}$ 作为 UNet 输入获取更高时间步的噪声预测，最小化错误时间步和正确时间步预测间的差异，瓦解模型的时间步感知能力。

### 总体损失函数
将 PFG 和 ETG 与原始 $\mathcal{L}_{ddpm}$ 结合，通过 PGD 梯度上升优化：

$$\mathcal{L}_{pgd}(x_0; \delta^{adv}) = \mathbb{E}_{\epsilon,t}\left(\mathcal{L}_{ddpm} + \lambda_1 e^{\bar{\alpha}_t - 1} \mathcal{L}_{fre} + \lambda_2 e^{\mathcal{L}_{err\text{-}t}}\right)$$

其中 $\lambda_1 = \lambda_2 = 0.5$，攻击时间步 $t \sim \mathcal{U}(1, t^p)$ 被限制在净化步范围内。系数 $e^{\bar{\alpha}_t - 1}$ 使 PFG 随 $t$ 降低时影响增大；指数函数应用于 ETG 实现更积极的优化。

## 实验

### 实验设置
- **数据集**：CelebA-HQ 和 VGGFace2，各 50 个 ID × 12 张 512×512 图像
- **基线方法**：AdvDM、Mist、Anti-DreamBooth、SimAC
- **净化方法**：GrIDPure（2 轮 × 20 迭代，$t^p=10$）
- **评估指标**：FID↑、ISM↓、FDFR、BRISQUE↑（定制化输出质量），LPIPS↓（扰动感知差异）

### 主实验：DreamBooth P-C 工作流

| 数据集 | 方法 | FID↑ | ISM↓ | BRISQUE↑ |
|--------|------|------|------|----------|
| CelebA-HQ | AdvDM | 77.51 | 0.6561 | 31.33 |
| CelebA-HQ | Mist | 70.23 | 0.6688 | 37.00 |
| CelebA-HQ | Anti-DB | 78.84 | 0.6422 | 31.76 |
| CelebA-HQ | SimAC | 67.37 | 0.6734 | 33.73 |
| CelebA-HQ | **AntiPure** | **81.15** | **0.6112** | **43.60** |
| VGGFace2 | AdvDM | 83.90 | 0.5923 | 37.42 |
| VGGFace2 | Anti-DB | 90.29 | 0.5938 | 38.35 |
| VGGFace2 | **AntiPure** | **90.77** | **0.5475** | **46.01** |

AntiPure 在所有指标和两个数据集上均取得最优表现。

### LoRA 微调验证

| 数据集 | 方法 | FID↑ | ISM↓ | BRISQUE↑ |
|--------|------|------|------|----------|
| VGGFace2 | Anti-DB | 117.89 | 0.5723 | 58.56 |
| VGGFace2 | **AntiPure** | **127.67** | **0.5428** | **69.97** |

在 LoRA 微调场景下仍全面领先，ISM 指标差距尤为显著。

### 净化迭代消融

| 方法 | 迭代=10 ISM | 迭代=20 ISM | 迭代=30 ISM | 迭代=40 ISM |
|------|------------|------------|------------|------------|
| Anti-DB | 0.6020 | 0.6352 | 0.6473 | 0.6391 |
| AntiPure | 0.6362 | 0.6271 | 0.6075 | **0.5994** |

Anti-DB 随迭代增加逐渐失效（ISM 升高），而 AntiPure 反而越来越强——这与其直接攻击净化过程本身的设计一致。

### 感知一致性
AntiPure 在相同 $\eta$ 约束下实现了最小的 LPIPS 感知差异，这归功于 PFG 有效避免了对低频信息的修改。

## 亮点与洞察
- **首次形式化 anti-purification 任务**：系统分析了为什么对抗净化比对抗定制化更困难，为后续研究奠定理论基础
- **"以毒攻毒"的巧妙思路**：不试图让扰动"幸存"净化过程，而是让净化过程本身产生失真，间接扰动后续微调
- **频率域与时间步的双重攻击**：PFG 攻击净化模型的高频控制弱点，ETG 瓦解时间步感知，两者协同增效
- **随净化加深反而增效**：独特的"越净化越有效"特性，展示了方法的鲁棒性

## 局限性
- 无法实现语义结构级别的严重失真（如完全扭曲人脸），只能引入可辨识的伪影
- 依赖白盒攻击，需要知道净化模型的具体架构和参数
- 评估主要限于人脸数据集和 DreamBooth/LoRA 两种微调方式
- 使用 JPEG 压缩时性能有所下降（CelebA-HQ 结果明显差于 VGGFace2）

## 相关工作
- **定制化微调**：DreamBooth、LoRA、Textual Inversion、Custom Diffusion、ControlNet
- **保护性扰动**：AdvDM、Mist、Anti-DreamBooth、SimAC、MetaCloak、CAAT
- **扩散净化**：DiffPure、DensePure、GrIDPure

## 评分
- 新颖性：⭐⭐⭐⭐ — 首次系统定义并解决 anti-purification 问题
- 技术深度：⭐⭐⭐⭐ — 对三个核心挑战的分析透彻
- 实验充分度：⭐⭐⭐⭐ — 多数据集、多微调方法、多净化配置
- 实用价值：⭐⭐⭐ — 白盒假设限制了实际部署

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] DCT-Shield: A Robust Frequency Domain Defense against Malicious Image Editing](dct-shield_a_robust_frequency_domain_defense_against_malicious_image_editing.md)
- [\[ICCV 2025\] Efficient Input-Level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation](efficient_input-level_backdoor_defense_on_text-to-image_synthesis_via_neuron_act.md)
- [\[ICCV 2025\] DPoser-X: Diffusion Model as Robust 3D Whole-Body Human Pose Prior](dposer-x_diffusion_model_as_robust_3d_whole-body_human_pose_prior.md)
- [\[AAAI 2026\] VoiceCloak: A Multi-Dimensional Defense Framework against Unauthorized Diffusion-based Voice Cloning](../../AAAI2026/image_generation/voicecloak_a_multi-dimensional_defense_framework_against_unauthorized_diffusion-.md)
- [\[NeurIPS 2025\] Token Perturbation Guidance for Diffusion Models](../../NeurIPS2025/image_generation/token_perturbation_guidance_for_diffusion_models.md)

</div>

<!-- RELATED:END -->
