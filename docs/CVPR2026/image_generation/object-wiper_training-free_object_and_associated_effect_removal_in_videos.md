---
title: >-
  [论文解读] Object-WIPER: Training-Free Object and Associated Effect Removal in Videos
description: >-
  [CVPR 2026][图像生成][视频物体移除] 提出 Object-WIPER，首个无训练的视频物体及其关联效应（阴影、反射、镜像等）移除框架，利用 DiT 中的文本-视觉交叉注意力和视觉自注意力定位关联效应区域，通过前景重初始化和注意力缩放实现干净移除…
tags:
  - "CVPR 2026"
  - "图像生成"
  - "视频物体移除"
  - "关联效应"
  - "训练免费"
  - "注意力机制"
  - "扩散模型"
---

# Object-WIPER: Training-Free Object and Associated Effect Removal in Videos

**会议**: CVPR 2026  
**arXiv**: [2601.06391](https://arxiv.org/abs/2601.06391)  
**代码**: 即将发布  
**领域**: 图像生成 / 视频编辑  
**关键词**: 视频物体移除, 关联效应, 训练免费, 注意力机制, 扩散模型

## 一句话总结
提出 Object-WIPER，首个无训练的视频物体及其关联效应（阴影、反射、镜像等）移除框架，利用 DiT 中的文本-视觉交叉注意力和视觉自注意力定位关联效应区域，通过前景重初始化和注意力缩放实现干净移除，并提出 TokSim 指标和 WIPER-Bench 真实世界基准。

## 研究背景与动机

**领域现状**：视频物体移除是影视制作和隐私保护的关键技术。经典方法（PatchMatch/图割）和学习方法（Propainter）专注填充物体区域，完全忽视关联效应（阴影/反射）。近期扩散方法（VACE/Videopainter）也保留关联效应。

**现有痛点**：(a) 几乎所有现有方法保留阴影/反射导致视觉伪影；(b) ROSE 能处理关联效应但需大量合成数据训练；(c) Omnimatte-Zero 从用户 mask 扩展关联区域但依赖外部点追踪模型（TAP-Net），在快速运动/透明物体下失败，且扩展策略次优。

**核心矛盾**：物体移除不等于区域填充——必须同时移除物体的"视觉痕迹"（阴影、反射、镜像等）才算干净移除。

**本文目标**：无训练地同时移除物体及其所有关联视觉效应。

**切入角度**：利用 MMDiT 中文本-视觉共享嵌入空间直接定位关联效应，不依赖外部模型。

**核心 idea**：交叉注意力定位关联效应种子 → 自注意力精修 → 前景重初始化 + 注意力缩放 → 自适应时序 mask。

## 方法详解

### 整体框架
Object-WIPER 想干净地移除视频里的某个物体——不只是它本身，还包括它投下的阴影、在水面/镜子里的反射这些"视觉痕迹"，而且全程不训练任何参数，只在一个预训练的文生视频 DiT 上做推理期操控。输入是一段 RGB 视频 $\mathcal{I}_k$、框出物体的 mask $\mathbf{M}^{obj}$，以及两段文本提示 $\{P_s, P_T\}$（分别描述物体和它的效应）。

整条流水线分三步走：先在 DiT 的注意力里定位出所有关联效应的位置，得到效应 mask $\mathbf{M}^{AE}$；再把视频反转回噪声、同时把要保留的背景 latent 值原样存下来；最后把前景（物体 + 效应）区域换成纯噪声重新去噪，让模型凭背景上下文把这块"补"成干净的样子。难点在于第一步的定位（关联效应没有现成 mask）和第三步的重生成（要既抹干净又不破坏背景），下面几个设计正是围绕这两点展开。

### 关键设计

**1. 关联效应定位：不靠外部模型，直接从 DiT 注意力里挖出阴影和反射**

阴影、反射这些效应没有用户标注，最朴素的做法（Omnimatte-Zero）是从物体 mask 往外扩，但弱激活的边缘区域会被漏掉，还得额外挂一个点追踪模型。Object-WIPER 改用 MMDiT 里文本与视觉共享嵌入空间的天然语义关联，两步定位。第一步从文本到图像的交叉注意力里，取与物体/效应文本 token 高相关的视觉 token，按头平均后做 Otsu 阈值，得到一张提议 mask $m^{PRO}$：

$$\bar{\mathbf{A}}^{\tilde{T}\to I} = \text{Mean}\Big(\text{Softmax}\big(\tfrac{\mathbf{Q}_{\tilde{T}}\cdot\mathbf{K}_I^\top}{\sqrt{d}}\big)\Big)$$

交叉注意力给出的语义定位虽对，但往往内部有孔洞、不完整。于是第二步用视觉自注意力 $\mathbf{A}^{I\to I}$ 补全：算每个视觉 token 对 $m^{PRO}$ 区域的响应比，阈值化后得到最终的关联效应 mask $\mathbf{M}^{AE}$。背后的直觉很简单——同属一个物体（含它的影子、倒影）的 token 之间自注意力必然偏高，自注意力因此能把交叉注意力漏掉的孔洞和弱边缘填回来。相比 Omnimatte-Zero 依赖 TAP-Net 点追踪、在快速运动或透明物体上失灵，这套纯内在注意力的方案更鲁棒，也不引入任何外部依赖。

**2. 时步自适应 Masking：让 mask 跟着噪声扩散一起"长大"**

把视频反转到噪声分布的过程中，自注意力会让物体的"存在感"不断向四周扩散，一张固定 mask 到了高噪声步根本盖不住物体真正影响到的范围。这里在反转每一步重新算物体响应分数

$$RS_p(j) = \frac{\sum_{y\in\mathbf{M}^{obj}(j)}A_{p,y}^{I\to I}}{\sum_{x\in\mathcal{I}(j)}A_{p,x}^{I\to I}}$$

即第 $p$ 个 token 的注意力里有多大比例落在物体区域上，再阈值化得到随时步动态膨胀的自适应 mask $\hat{M}_t^{obj}$。这样物体影响扩散到哪，mask 就跟到哪，避免后续重初始化时漏掉已经"渗"出去的物体信息——在高速行驶的车这类快速运动场景里尤其关键。

**3. 注意力缩放：反转时切断"污染"，去噪时引入背景语义**

要让前景区域被干净替换，得控制前景与背景之间的信息流向，而且反转和去噪两个阶段的诉求恰好相反。反转阶段，缩小背景对前景的注意力，让背景 latent 少受前景"污染"、尽量保留纯背景信息：

$$\tilde{\mathbf{A}}^{bg\to obj} = \text{Softmax}\big(\tfrac{\mathbf{Q}_I^{bg}\cdot(c\mathbf{K}_I^{obj})^\top}{\sqrt{d}}\big),\quad c<1$$

去噪阶段则反过来，放大前景对背景的注意力，让已被重置为噪声的前景主动去背景里"取景"、按周围语义把空洞补合理：

$$\tilde{\mathbf{A}}^{obj\to bg} = \text{Softmax}\big(\tfrac{\mathbf{Q}_I^{obj}\cdot(b\mathbf{K}_I^{bg})^\top}{\sqrt{d}}\big),\quad b>1$$

一个缩一个放，正好对应"先把背景护干净、再让前景照着背景重画"的两阶段逻辑。

**4. 前景重初始化：把残留先验彻底清零，从噪声重画**

只缩放注意力还不够——反转得到的前景 latent 里仍然带着物体和效应的结构先验，去噪时容易"复活"出原物体。重初始化干脆把前景（物体 mask 与效应 mask 的并集）那块直接换成高斯噪声，背景值原样保留：

$$\tilde{\mathbf{Z}}_1 = \mathbf{Z}_1\odot\big(1-\mathbf{M}^{obj}\cup\mathbf{M}^{AE}\big) + \varepsilon\odot\big(\mathbf{M}^{obj}\cup\mathbf{M}^{AE}\big)$$

抹掉所有残留先验后，这块区域就只能靠背景上下文重新生成，从根上断了物体复现的可能。消融里去掉它 TokSim 掉得最多（降 2.44），说明它是整套方案里最吃重的一环。

**5. TokSim 指标：一个能真正区分"移没移干净"的评测分数**

现有指标（如 BG-PSNR）有个根本缺陷：完全不动物体、只做 VAE 重建反而能拿高分，根本测不出移除质量。TokSim 把三件事揉进一个分数——

$$\text{TokSim} = 100\cdot\frac{1}{F}\sum_z\sum_i \lambda_z^k\cdot(1-\eta_z^k)\cdot\tau_z^k$$

其中 $\lambda$ 奖励时序一致、$\eta$ 惩罚物体残留、$\tau$ 奖励前景与背景的融合度。三项相乘意味着任何一环（残留没清、时序闪烁、补得突兀）拉胯都会拖垮总分，因此它能把"假装移除"的方法暴露出来。
> ⚠️ TokSim 各项符号（$\lambda/\eta/\tau$ 的精确定义、$z/i/k$ 的求和范围）以原文为准。

### 一个完整示例：移除一辆行驶中的汽车及其阴影
以一段地面有清晰投影的行车视频为例。用户只给出框住车身的 $\mathbf{M}^{obj}$ 和文本"car / shadow"。第一步定位：交叉注意力先在车身和地面投影上各点亮一片，但投影边缘有孔洞；自注意力按响应比把孔洞填实，合出完整的 $\mathbf{M}^{AE}$，把车下那条拖影也圈了进去。第二步反转回噪声，由于车速快，固定 mask 盖不住车尾"渗"出去的影响，时步自适应 Masking 让 $\hat{M}_t^{obj}$ 随噪声步逐步膨胀、始终罩住车的影响范围，同时注意力缩放（$c<1$）把路面背景的 latent 护干净存下来。第三步重初始化：车身 + 投影区域全部置为高斯噪声，路面、远处建筑的 latent 原样保留；去噪时注意力放大（$b>1$）让这块噪声区照着周围沥青路面重画，最终输出一段没有车、也没有那条拖影的干净视频。

### 损失函数 / 训练策略
完全无训练，直接复用预训练的文生视频 DiT；推理期只做注意力操控（缩放）和背景 latent 的值复制，不更新任何参数、也不需要任何合成数据。

## 实验关键数据

### 主实验

| 方法 | 训练 | DAVIS TokSim↑ | WIPER TokSim↑ | DAVIS BG-PSNR↑ | DAVIS Text-align↑ |
|------|------|--------------|---------------|----------------|-------------------|
| Propainter | ✓ | 28.24 | 20.99 | 34.01 | 26.18 |
| ROSE | ✓ | 29.36 | 30.02 | 26.97 | 26.13 |
| VACE | ✓ | 15.86 | 11.53 | 24.48 | 24.01 |
| Gen-Prop | ✓ | 30.52 | - | 24.27 | 25.89 |
| KV-Edit-Video | ✗ | 28.68 | 23.26 | 25.78 | 25.21 |
| Attentive-Eraser | ✗ | 30.82 | 25.28 | 28.07 | 26.31 |
| **Object-WIPER** | **✗** | **32.80** | **33.09** | 23.02 | **26.63** |

### 消融实验

| 配置 | TokSim↑ | BG-PSNR↑ | Text-align↑ |
|------|---------|----------|-------------|
| Full Object-WIPER | 32.80 | 23.02 | 26.63 |
| w/o 注意力缩放 | 32.97 | 21.92 | 26.42 |
| w/o 自适应 mask | 32.10 | 22.73 | 26.44 |
| w/o 重初始化 | 30.36 | 23.47 | 25.92 |
| w/o $\mathbf{M}^{AE}$ | 32.18 | 23.10 | 26.17 |

### 关键发现
- Object-WIPER 无训练即在 TokSim 上赶超所有训练方法（包括专门训练关联效应的 ROSE）
- TokSim 比 BG-PSNR 区分力强得多：VAE 重建（不移除物体）BG-PSNR 34.05 但 TokSim 仅 0.32
- 重初始化是最关键组件（去掉后 TokSim 降 2.44）
- 关联效应 mask $\mathbf{M}^{AE}$ 对 WIPER-Bench 至关重要——只有加上才能移除阴影/反射
- 自适应 mask 在快速运动场景（如高速行驶的车）中必不可少

## 亮点与洞察
- **MMDiT 内在注意力做关联效应定位**：完全不依赖外部模型，利用文本-视觉共享空间的语义关联精准定位。这个技巧可迁移到任何 MMDiT-based 编辑任务
- **TokSim 指标设计**精巧：同时度量移除完整性、时序一致性和背景融合，暴露了现有指标的根本缺陷
- **WIPER-Bench**是首个包含镜像、透明物体、多关联效应等真实场景的物体移除基准

## 局限与展望
- BG-PSNR 不如训练方法（因为背景也被扩散模型重新生成）
- 依赖文本描述物体和效应类型，自动化程度有限
- 视频分辨率受预训练模型限制
- 仅处理动态物体，静态物体移除未讨论

## 相关工作与启发
- **vs Omnimatte-Zero**: 不依赖 TAP-Net 点追踪，定位策略更完善
- **vs ROSE/Gen-Prop**: 训练型方法需大量合成数据，Object-WIPER 零成本
- **vs KV-Edit**: KV-Edit 针对图像设计，简单扩展到视频效果差

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在 DiT 内解决关联效应定位和移除，TokSim 指标重要
- 实验充分度: ⭐⭐⭐⭐ 两数据集+新基准+新指标+消融完整
- 写作质量: ⭐⭐⭐⭐ 问题定义和方法层层递进
- 价值: ⭐⭐⭐⭐⭐ WIPER-Bench + TokSim 对社区有持久价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] EffectErase: Joint Video Object Removal and Insertion for High-Quality Effect Erasing](effecterase_joint_video_object_removal_and_insertion_for_high-quality_effect_era.md)
- [\[CVPR 2026\] Precise Object and Effect Removal with Adaptive Target-Aware Attention](precise_object_and_effect_removal_with_adaptive_target-aware_attention.md)
- [\[ICML 2026\] AdaEraser: Training-Free Object Removal via Adaptive Attention Suppression](../../ICML2026/image_generation/adaeraser_training-free_object_removal_via_adaptive_attention_suppression.md)
- [\[CVPR 2026\] SketchDeco: Training-Free Latent Composition for Precise Sketch Colourisation](sketchdeco_training-free_latent_composition_for_precise_sketch_colourisation.md)
- [\[CVPR 2026\] ViHOI: Human-Object Interaction Synthesis with Visual Priors](vihoi_human-object_interaction_synthesis_with_visual_priors.md)

</div>

<!-- RELATED:END -->
