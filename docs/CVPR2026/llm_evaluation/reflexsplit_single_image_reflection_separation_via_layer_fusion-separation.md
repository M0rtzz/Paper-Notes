---
title: >-
  [论文解读] ReflexSplit: Single Image Reflection Separation via Layer Fusion-Separation
description: >-
  [CVPR 2026][图像复原][反射分离] 提出ReflexSplit双流框架，通过跨尺度门控融合(CrGF)自适应聚合多源多尺度特征、层级融合-分离块(LFSB)借助差分注意力实现透射-反射层显式解纠缠、课程训练渐进增强分离强度，在合成和真实世界基准上达到SOTA。
tags:
  - CVPR 2026
  - 图像复原
  - 反射分离
  - 双流架构
  - 差分注意力
  - 跨尺度特征融合
  - 课程学习
---

# ReflexSplit: Single Image Reflection Separation via Layer Fusion-Separation

**会议**: CVPR 2026  
**arXiv**: [2601.17468](https://arxiv.org/abs/2601.17468)  
**代码**: https://github.com/wuw2135/ReflexSplit (有)  
**领域**: 图像复原  
**关键词**: 反射分离, 双流架构, 差分注意力, 跨尺度融合, 课程学习

## 一句话总结
针对单图反射分离中的透射-反射混淆问题（尤其是在深层解码器中），提出ReflexSplit双流框架，通过跨尺度门控融合(CrGF)稳定多尺度特征流、层级融合-分离块(LFSB)的差分双维注意力实现跨流减法解纠缠、课程训练渐进增强分离强度，在合成和真实世界数据集上达到SOTA性能。

## 研究背景与动机

**领域现状**：单图反射分离（SIRS）将混合图像分解为透射层和反射层。早期线性叠加模型 $\mathbf{I}=\mathbf{T}+\mathbf{R}$ 过于简单，后续引入非线性残差公式 $\mathbf{I}=\mathbf{T}+\mathbf{R}+\Phi(\mathbf{T},\mathbf{R})$ 来建模复杂的层间交互。近年来YTMT、DSRNet、DSIT等方法通过激活函数交换、通道分割、双流注意力等增强层间交互。

**现有痛点**：(1) 在强反射或模糊场景下，网络经常错误地混淆透射层和反射层（transmission-reflection confusion），例如将水池中的强光源反射或墙上的月亮画误判为另一层；(2) 网络深度增加时信息丢失，导致层内和层间特征变得不可分离，在深层解码器中尤为严重；(3) 现有方法缺乏有效的跨尺度特征协调机制——DSIT有梯度不稳定问题，RDNet用可逆编码器但缺乏显式尺度平衡。

**核心矛盾**：隐式融合机制和不充分的多尺度协调，使得特征在解码层中逐渐纠缠，深层更加严重。

**本文目标** (1) 跨尺度特征的自适应聚合和梯度稳定性；(2) 透射-反射层特征的显式解纠缠；(3) 差分分离强度的渐进训练策略。

**切入角度**：受Differential Transformer启发，将注意力取消机制从单流噪声抑制扩展到双流层分离——通过跨流减法 $\mathbf{A}^t - \lambda_\ell \mathbf{A}^r$ 实现显式的层特定解纠缠。

**核心 idea**：通过"先融合共享结构、再差分分离层特定特征"的交替范式，配合跨尺度自适应门控和课程训练，防止透射-反射混淆贯穿整个网络层次。

## 方法详解

### 整体框架
ReflexSplit是一个双流编码器-解码器架构。输入混合图像后：(1) 双分支特征提取器分别用Swin Transformer（GFEB）提取全局语义先验和基于MuGI的CNN（LFEB）提取局部纹理细节；(2) 在解码器的每一层，CrGF自适应聚合多源多尺度特征；(3) LFSB在融合（提取共享退化模式）和差分分离（层特定解纠缠）之间交替；(4) 课程训练渐进增强差分分离强度。输出透射层 $\mathbf{T}$、反射层 $\mathbf{R}$ 和非线性残差 $\mathbf{RR}$。

### 关键设计

1. **跨尺度门控融合 (CrGF)**:

    - 功能：在解码器Level 4/3/2自适应聚合语义先验、纹理细节和解码器上下文
    - 核心思路：扩展MuGI的门控原理到跨尺度聚合。先将三种来源特征相加得到原始特征 $\mathbf{F}_\ell^{\text{raw}} = \mathbf{F}_{\ell+1} + \mathbf{P}_\ell + \mathbf{E}_\ell$，然后通过双向门控路径生成主路 $\mathbf{F}_\ell^{\text{main}} = \mathcal{G}_1(\mathbf{F}_\ell^{\text{raw}}) \odot \mathcal{G}_2(\mathbf{F}_{\ell+1})$ 和辅助路 $\mathbf{F}_\ell^{\text{aux}}$，最后通过softmax加权融合。双向路径让当前层特征和上下文特征互相门控
    - 设计动机：MuGI只在单一尺度做双流交互，缺乏跨尺度协调。RobustSIRR直接concatenation没有自适应门控，RDNet的固定可逆路径也缺乏灵活性。CrGF的自适应双向门控能根据上下文动态选择和重组多尺度特征，防止渐进退化

2. **层级融合-分离块 (LFSB)**:

    - 功能：在融合（共享结构提取）和分离（层特定解纠缠）之间交替操作
    - 核心思路：分三步——(a) **早期融合**：通过双向跨流投影 $\mathbf{F}_\ell^{t'} = \mathbf{W}^t[\mathbf{F}_\ell^t \| \mathbf{F}_\ell^r]$ 对齐语义空间；(b) **差分双维注意力**：沿batch维拼接做自注意力（SA）建模空间相关性，沿序列维拼接做交叉注意力（CA）捕获层间依赖，然后执行差分运算 $\mathbf{A}_{\text{diff}}^t = (\mathbf{A}_{\text{SA}}^t + \mathbf{A}_{\text{CA}}^t) - \sigma(\lambda_\ell)(\mathbf{A}_{\text{SA}}^r + \mathbf{A}_{\text{CA}}^r)$；(c) **后期聚合**：通过FFN+残差连接整合分离后的特征
    - 设计动机：DSIT直接聚合SA和CA输出没有分离约束，导致渐进混淆。本文的跨流减法 $-\sigma(\lambda_\ell)\mathbf{A}^r$ 确保透射特定和反射特定特征在整个解码阶段保持可区分，从根本上解决深层混淆问题

3. **课程训练策略**:

    - 功能：渐进增强差分分离强度，避免早期训练不稳定
    - 核心思路：两个互补机制——(a) **深度依赖初始化**：$\lambda_\ell^{\text{init}} = 0.8 - 0.6 e^{-0.3\ell}$，深层更强分离（$\lambda \to 0.8$），浅层弱分离（$\lambda \to 0.2$）保留细节；(b) **逐epoch预热**：前30个epoch将全局缩放因子从0.1线性增到1.0，最终有效系数为 $\lambda_\ell(e) = \lambda_\ell^{\text{init}} \cdot \lambda_{\text{diff}}(e)$
    - 设计动机：差分分离的 $\lambda$ 需要精心控制——早期过强会不稳定因为特征尚未结构化，过弱则无法执行有效分离。课程学习让网络先学习整体重构再聚焦层特定分离

### 损失函数 / 训练策略
总损失为多项加权和：$\mathcal{L}_{\text{total}} = \lambda_{\text{rec}}\mathcal{L}_{\text{rec}} + \lambda_{\text{refl}}\mathcal{L}_{\text{refl}} + \lambda_{\text{vgg}}\mathcal{L}_{\text{vgg}} + \lambda_{\text{color}}\mathcal{L}_{\text{color}} + \lambda_{\text{exclu}}\mathcal{L}_{\text{exclu}} + \lambda_{\text{recons}}\mathcal{L}_{\text{recons}}$，包括Charbonnier损失、L1反射损失、VGG感知损失、颜色一致性损失、排他性损失和重构一致性损失。

## 实验关键数据

### 主实验

| 数据集 | 指标 | ReflexSplit | 之前SOTA(RDNet) | 对比 |
|--------|------|------|----------|------|
| Real20 | PSNR/SSIM | 25.22/0.846 | 25.17/0.841 | +0.05/+0.005 |
| SIR2 Objects | PSNR/SSIM | 27.08/0.929 | 27.11/0.925 | -0.03/+0.004 |
| SIR2 Postcard | PSNR/SSIM | 25.38/0.927 | 25.04/0.910 | +0.34/+0.017 |
| SIR2 Wild | PSNR/SSIM | 27.30/0.933 | 27.86/0.931 | -0.56/+0.002 |
| Nature | PSNR/SSIM | 27.03/0.854 | 26.75/0.846 | +0.28/+0.008 |
| Average (540) | PSNR/SSIM | 26.40/0.898 | 26.38/0.890 | +0.02/+0.008 |

注：RDNet参数量266.4M，ReflexSplit仅174M，参数效率更高。在Postcard子集上优势最明显(+0.34dB PSNR, +0.017 SSIM)。

### 消融实验
| 配置 | PSNR↑ | SSIM↑ | 说明 |
|------|---------|------|------|
| Baseline(DSIT) | 24.71 | 0.831 | 基线 |
| + CrGF | - | - | 替换原有特征聚合，稳定梯度流 |
| + LFSB(w/o diff) | - | - | 仅融合无差分 |
| + LFSB(w/ diff) | - | - | 加差分注意力 |
| + Curriculum | 25.22 | 0.846 | 完整模型 |

### 关键发现
- **CrGF的跨尺度自适应门控**解决了DSIT的梯度不稳定问题，稳定了整个解码器中的特征流
- **差分注意力可视化** 清晰展示了跨流减法如何抑制层间干扰——减法后注意力分布更均衡，层特定注意力模式更清晰
- **在Postcard子集上优势最大**，该子集包含更多非线性混合的挑战场景
- 虽然参数量(174M)少于RDNet(266.4M)，但在平均指标上达到或超越，说明框架设计更高效
- **课程训练策略是不可或缺的**——直接用强差分系数训练会导致收敛不稳定

## 亮点与洞察
- **将Differential Transformer的思想巧妙迁移到双流分离场景**：原始DiffTransformer是在同一注意力头内做减法抑制噪声，本文扩展为跨流减法 $\mathbf{A}^t - \lambda\mathbf{A}^r$ 抑制层间干扰，语义上更自然——透射的注意力减去反射的注意力就是纯透射特定的信号
- **融合-分离交替范式**是解决双流任务的通用范式——先融合提取共享结构信息，再分离得到任务特定表示。可迁移到其他需要"分解"的任务如去雾（雾层+清晰层）
- **课程训练的深度依赖初始化**设计很精巧——深层需要更强分离（因为信息损失更严重），浅层保持弱分离保留细节

## 局限与展望
- 在Wild子集上低于RDNet 0.56dB PSNR，说明在完全非受控的真实场景中仍有改进空间
- 174M参数量仍然较大，轻量化设计（如DExNet的9.6M）在移动端部署更有优势
- 当前的差分系数 $\lambda_\ell$ 是可学习标量，如果改为空间自适应（逐像素）可能能更好处理局部强反射区域
- 未在视频反射分离场景中验证，时序一致性是一个潜在挑战

## 相关工作与启发
- **vs DSIT**: DSIT在深层解码器中出现渐进混淆（论文Fig.2清晰展示），ReflexSplit通过差分注意力在所有深度维持清晰层区分
- **vs RDNet**: RDNet用可逆编码器实现无损梯度流但缺乏显式尺度平衡且参数量大(266.4M)；ReflexSplit用CrGF实现更灵活的跨尺度协调，参数更少
- **vs DSRNet**: DSRNet的MuGI只在单尺度做交互缺乏注意力分离约束；LFSB的融合-分离范式是更系统性的解决方案

## 评分
- 新颖性: ⭐⭐⭐⭐ 差分注意力到双流分离的迁移有创新性，融合-分离范式设计系统
- 实验充分度: ⭐⭐⭐⭐ 涵盖多个合成和真实数据集，有可视化分析，但消融细节可以更详细
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表说服力强，动机论证充分
- 价值: ⭐⭐⭐⭐ 对反射分离领域有推进，融合-分离范式有一定通用性
---
title: >-
  [论文解读] ReflexSplit: Single Image Reflection Separation via Layer Fusion-Separation
description: >-
  [CVPR 2026][图像恢复][反射分离] 提出ReflexSplit双流框架，通过跨尺度门控融合(CrGF)、层融合-分离模块(LFSB)中的差分双维度注意力和课程训练策略，解决单图反射分离中的透射-反射混淆问题，在合成和真实世界基准上达到SOTA性能。
tags:
  - CVPR 2026
  - 图像恢复
  - 反射分离
  - 差分注意力
  - 双流架构
  - 课程学习
  - 跨尺度融合
---

# ReflexSplit: Single Image Reflection Separation via Layer Fusion-Separation

**会议**: CVPR 2026  
**arXiv**: [2601.17468](https://arxiv.org/abs/2601.17468)  
**代码**: https://github.com/wuw2135/ReflexSplit (有)  
**领域**: 图像恢复  
**关键词**: 单图反射分离, 差分注意力, 跨尺度融合, 课程学习, 双流架构

## 一句话总结
ReflexSplit提出一种显式层融合-分离框架，通过跨尺度门控融合(CrGF)自适应聚合多尺度特征，层融合-分离模块(LFSB)中的差分双维度注意力 $\mathbf{A}^t - \lambda_\ell \mathbf{A}^r$ 进行跨流干扰抑制，配合深度依赖初始化+epoch-wise warmup的课程训练，在合成和真实世界反射分离基准上取得SOTA。

## 研究背景与动机

1. **领域现状**：单图反射分离（SIRS）需要将混合图像 $\mathbf{I}$ 分解为透射层 $\mathbf{T}$ 和反射层 $\mathbf{R}$。近年方法从简单线性叠加模型 $\mathbf{I}=\mathbf{T}+\mathbf{R}$ 发展到非线性残差模型 $\mathbf{I}=\mathbf{T}+\mathbf{R}+\Phi(\mathbf{T},\mathbf{R})$，通过YTMT、DSRNet、DSIT等方法增强层间交互。

2. **现有痛点**：当遇到强反射（如水面强光反射）或语义模糊场景（如墙上的月亮画被误识为反射）时，网络会错误地混淆透射和反射层（"透射-反射混淆"）。随着网络深度增加，特征信息损失导致层内和层间特征不可分，这在深层decoder中尤为严重。

3. **核心矛盾**：现有方法在两个维度上存在不足：(a) 层级特征聚合不充分导致梯度不稳定——DSIT缺乏梯度稳定性，RDNet缺少显式尺度协调，MuGI只在单尺度操作；(b) 隐式融合机制导致渐进式层混淆——DSIT直接聚合双维度注意力输出而无分离约束。

4. **本文目标** (a) 如何在多尺度上自适应聚合来自不同来源（语义先验、纹理细节、decoder上下文）的特征？(b) 如何在融合共享结构信息的同时强制执行层特异性的分离？(c) 如何在训练早期避免过强的分离约束导致不稳定？

5. **切入角度**：将反射分离显式建模为"融合-分离"的交替过程——先融合获得共享结构信息，再用差分注意力进行层特异性的分离。将Differential Transformer的注意力消除思想从单流噪声抑制扩展到双流层分离。

6. **核心 idea**：通过在双流架构中交替执行融合（共享结构提取）和差分注意力分离（跨流减法 $\mathbf{A}^t - \lambda_\ell \mathbf{A}^r$），结合课程训练渐进增强分离强度，实现鲁棒的透射-反射分离。

## 方法详解

### 整体框架

ReflexSplit采用双流编码器-解码器架构。编码端包含双分支特征提取：预训练Swin Transformer作为全局特征提取模块（GFEB）提取语义先验 $\{\mathbf{P}_\ell\}$，MuGI-based CNN作为局部特征提取模块（LFEB）捕捉纹理细节 $\{\mathbf{E}_\ell\}$。解码端通过CrGF自适应聚合多尺度特征，LFSB在每个解码层交替执行融合和差分分离。输出透射层 $\mathbf{T}$、反射层 $\mathbf{R}$ 和残差 $\mathbf{RR}$（捕捉非线性交互）。

### 关键设计

1. **跨尺度门控融合（CrGF）**:

    - 功能：在decoder各层级自适应聚合语义先验、纹理细节和decoder上下文，稳定梯度流并防止特征退化
    - 核心思路：在decoder Level {4,3,2}，将原始特征 $\mathbf{F}_\ell^{\text{raw}} = \mathbf{F}_{\ell+1} + \mathbf{P}_\ell + \mathbf{E}_\ell$（decoder上下文+语义+纹理）与decoder上下文通过双向门控路径融合：$\mathbf{F}_\ell^{\text{main}} = \mathcal{G}_1(\mathbf{F}_\ell^{\text{raw}}) \odot \mathcal{G}_2(\mathbf{F}_{\ell+1})$ 和 $\mathbf{F}_\ell^{\text{aux}} = \mathcal{G}_1(\mathbf{F}_{\ell+1}) \odot \mathcal{G}_2(\mathbf{F}_\ell^{\text{raw}})$，其中 $\mathcal{G}$ 通过通道分割选择互补通道。最终用softmax加权融合
    - 设计动机：MuGI只在单尺度上操作，RobustSIRR直接拼接无自适应门控，RDNet固定可逆路径缺乏显式尺度协调——CrGF通过双向自适应门控解决了跨尺度、跨来源特征协调的问题

2. **层融合-分离模块（LFSB）**:

    - 功能：在每个decoder阶段交替执行融合（共享结构提取）和分离（层特异性解纠缠），防止渐进式透射-反射混淆
    - 核心思路：分三步执行——
      (a) **早期融合**：通过双向跨流投影对齐语义空间 $\mathbf{F}^{t'}_\ell = \mathbf{W}^t[\mathbf{F}^t_\ell \| \mathbf{F}^r_\ell]$，使每个流获得互补信息；
      (b) **差分双维度注意力**：沿batch维度拼接做自注意力（SA，空间精炼），沿序列维度拼接做交叉注意力（CA，层间依赖）。关键是引入差分算子 $\mathbf{A}^t_{\text{diff}} = (\mathbf{A}^t_{\text{SA}} + \mathbf{A}^t_{\text{CA}}) - \sigma(\lambda_\ell)(\mathbf{A}^r_{\text{SA}} + \mathbf{A}^r_{\text{CA}})$，通过跨流减法抑制层间干扰；
      (c) **后融合**：FFN + 残差连接整合分离后的特征
    - 设计动机：不同于Differential Transformer在同一注意力头内减法消除噪声，LFSB将减法扩展到跨流——用反射流的注意力模式抑制透射流中的反射残留，反之亦然。DSIT直接聚合SA和CA输出而无分离约束，导致渐进混淆

3. **课程训练策略**:

    - 功能：渐进增强差分分离强度，让网络先学习整体重建再聚焦层特异性分离
    - 核心思路：通过两个互补机制控制 $\lambda_\ell$：
      (a) **深度依赖初始化** $\lambda_\ell^{\text{init}} = 0.8 - 0.6 e^{-0.3\ell}$：深层接收更强的分离权重（$\lambda \to 0.8$），浅层保持弱权重（$\lambda \to 0.2$）以保留细粒度细节；
      (b) **Epoch-wise warmup** $\lambda_{\text{diff}}(e)$：前30个epoch内从0.1线性增加到1.0。最终有效系数 $\lambda_\ell(e) = \lambda_\ell^{\text{init}} \cdot \lambda_{\text{diff}}(e)$
    - 设计动机：早期训练时特征结构不完善，过强的差分分离会导致训练不稳定；过弱又无法有效分离。课程训练在空间（深度依赖）和时间（epoch-wise）两个维度上自适应控制

### 损失函数 / 训练策略

总损失函数包含6项：Charbonnier重建损失 $\mathcal{L}_{\text{rec}}$（透射层）、$\ell_1$反射损失 $\mathcal{L}_{\text{refl}}$、VGG感知损失 $\mathcal{L}_{\text{vgg}}$（layers {2,7,12,21,30}）、颜色一致性损失 $\mathcal{L}_{\text{color}}$、排他性损失 $\mathcal{L}_{\text{exclu}}$和重建约束损失 $\mathcal{L}_{\text{recons}}$。

## 实验关键数据

### 主实验

| 数据集 | PSNR↑ / SSIM↑ | ReflexSplit | 之前SOTA (RDNet) | 比较 |
|--------|------|------|----------|------|
| Real20 | PSNR / SSIM | 25.22 / 0.846 | 25.17 / 0.841 | +0.05 / +0.005 |
| Objects | PSNR / SSIM | 27.08 / 0.929 | 27.11 / 0.925 | -0.03 / +0.004 |
| Postcard | PSNR / SSIM | 25.38 / 0.927 | 25.04 / 0.910 | +0.34 / +0.017 |
| Wild | PSNR / SSIM | 27.30 / 0.933 | 27.86 / 0.931 | -0.56 / +0.002 |
| Nature | PSNR / SSIM | 27.03 / 0.854 | 26.75 / 0.846 | +0.28 / +0.008 |
| 平均 (540张) | PSNR / SSIM | 26.40 / 0.898 | 26.38 / 0.890 | +0.02 / +0.008 |

备注：ReflexSplit参数量174M vs RDNet 266.4M，参数效率更高。

### 消融实验

从论文中LFSB差分注意力可视化和层级特征分离对比可得出以下要点：

| 配置 | 关键效果 | 说明 |
|------|---------|------|
| DSIT (baseline) | 深层出现透射-反射混淆 | 无分离约束，渐进退化 |
| + CrGF | 稳定梯度流 | 自适应跨尺度聚合 |
| + LFSB (w/o diff) | 融合但未分离 | 共享结构但layer混淆未解 |
| + LFSB (w/ diff) | 有效分离各层特征 | 差分算子抑制跨流干扰 |
| + 课程训练 | 训练稳定性提升 | 渐进增强分离强度 |

### 关键发现
- Postcard子集上提升最显著（+0.34 PSNR / +0.017 SSIM），因为该子集反射较强且存在明显的非线性混合
- 差分注意力可视化（Figure 6）清晰展示了跨流减法如何抑制重叠attention模式，将模糊的混合attention转化为层特异性的均衡分布
- 相比RDNet（266.4M参数，两阶段训练），ReflexSplit用更少参数（174M）和更简洁的训练流程达到了可比甚至更好的性能

## 亮点与洞察
- **从Differential Transformer到双流分离的迁移**：原版Diff Transformer在同一head内做减法消除噪声，本文将其扩展到跨流——用另一个流的attention来"减掉"当前流中的层间干扰。这种跨模态/跨流减法的思想可广泛迁移到任何需要分离纠缠信号的多流架构
- **课程训练的空间-时间协同设计**：深度依赖初始化+epoch-wise warmup形成了一个2D的分离强度控制面，使网络在不同阶段和不同深度都有最优的融合-分离平衡，这种细粒度的训练强度控制策略可推广到其他multi-scale分解任务

## 局限与展望
- 在某些子集上（Objects, Wild）PSNR略低于RDNet，说明对某些场景类型的适应性还不够强
- 依赖预训练Swin Transformer提取全局语义，对训练数据域外的泛化能力有待验证
- 差分系数 $\lambda_\ell$ 的初始化公式是手动设计的，可能对不同数据分布不够通用
- 论文未提供计算效率（FLOPs、推理延迟）的详细对比，174M参数相比DSIT（136M）更大但比RDNet（266M）小

## 相关工作与启发
- **vs DSIT**: DSIT用双维度注意力但直接聚合输出无分离约束，导致深层渐进混淆。ReflexSplit用差分算子显式解纠缠
- **vs RDNet**: RDNet用可逆编码器实现无损梯度流但参数量大（266M）且需两阶段训练。ReflexSplit用CrGF实现自适应跨尺度协调，参数更少
- **vs DSRNet**: DSRNet引入MuGI做层间交互但仅在单尺度操作，CrGF将其门控思想扩展到跨尺度聚合

## 评分
- 新颖性: ⭐⭐⭐⭐ 差分注意力在双流分离中的应用有创意，但整体框架是增量式改进
- 实验充分度: ⭐⭐⭐⭐ 多个数据集评估+可视化分析，但缺少详细消融数字
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述详细，图表丰富
- 价值: ⭐⭐⭐ 在反射分离这一较小子领域内有价值，但对更广泛的视觉社区影响有限
