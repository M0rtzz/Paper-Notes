---
title: >-
  [论文解读] Empowering Semantic-Sensitive Underwater Image Enhancement with VLM
description: >-
  [CVPR 2026][图像分割][underwater image enhancement] 提出 VLM 驱动的语义敏感学习策略，通过 VLM 生成目标物体描述、BLIP 构建空间语义引导图、双重引导机制（cross-attention + 语义对齐损失）注入 UIE decoder，使增强结果在感知质量和检测/分割下游任务上同时提升。
tags:
  - CVPR 2026
  - 图像分割
  - underwater image enhancement
  - VLM
  - semantic guidance
  - 注意力机制
  - downstream tasks
---

# Empowering Semantic-Sensitive Underwater Image Enhancement with VLM

**会议**: CVPR 2026  
**arXiv**: [2603.12773](https://arxiv.org/abs/2603.12773)  
**代码**: 无  
**领域**: 水下图像增强 / 语义引导  
**关键词**: underwater image enhancement, VLM, semantic guidance, cross-attention, downstream tasks

## 一句话总结

提出一种 VLM 驱动的语义敏感学习策略，通过 LLaVA 生成物体描述、BLIP 构建空间语义引导图、cross-attention 与语义对齐损失双重引导 UIE decoder 重建，使增强图像在感知质量和下游检测/分割任务上同时获得显著提升。

## 研究背景与动机

**领域现状**：水下图像增强（UIE）已有大量基于深度学习的方法，在 PSNR/SSIM 等感知质量指标上取得了显著进展。然而，学术界逐渐发现一个令人困扰的"增强悖论"——视觉质量更好的增强结果未必能帮助下游的目标检测或语义分割任务，有时甚至会导致性能下降。

**现有痛点**：现有 UIE 方法本质上是"语义盲"的，对图像所有区域施加全局均匀增强，无法区分语义焦点（如海洋生物、人工物体）与非焦点（如背景水体）。这种"一刀切"的增强策略会引入不可察觉的伪影或分布偏移，破坏下游模型所依赖的语义线索。早期语义引导方法依赖高质量的像素级标注来训练分割模型，但这类标注在水下场景中极为稀缺；而基于全局文本提示（如"a clear underwater photo"）的 VLM 引导方法虽然不需要像素标注，但仍是风格层面的全局引导，无法实现细粒度的内容感知。

**核心矛盾**：UIE 追求全局视觉质量与下游任务需要保护关键物体语义特征之间存在根本性矛盾。

**本文目标** 如何让 UIE 具备内容感知能力，在恢复视觉质量的同时保护甚至增强关键物体的语义特征，使增强结果同时服务于人眼观感和机器认知。

**切入角度**：借助 VLM 的开放世界理解能力自动识别图像中的关键物体，生成空间化的语义引导图，再通过双重引导机制（结构引导 + 显式监督）将语义先验注入 UIE 网络的 decoder。

**核心 idea**：用 VLM 告诉增强网络"图中有什么"、用 BLIP 告诉它"在哪里"、用 cross-attention + alignment loss 告诉它"重点增强哪里"。

## 方法详解

### 整体框架

整体流程分三个阶段：输入退化水下图像 $I_d$ → (1) 用 LLaVA 生成关键物体的文本描述 $T$ → (2) 用 BLIP 的视觉-文本对齐计算空间语义引导图 $M_{sem}$ → (3) 将 $M_{sem}$ 通过 cross-attention 注入和语义对齐损失双重机制注入任意 encoder-decoder 结构的 UIE 网络的 decoder 阶段，输出语义敏感的增强图像 $I_e$。该策略设计为可插拔模块，已在 PUIE、SMDR、UIR、PFormer、FDCE 五个 baseline 上验证通用性。

### 关键设计

1. **语义引导图生成（Semantic Guidance Map）**：

    - 功能：为每张退化图像生成一张单通道空间语义引导图 $M_{sem} \in \mathbb{R}^{H \times W}$，精确量化每个空间位置与关键物体的语义相关度
    - 核心思路：首先用 LLaVA 从退化图像自动生成关键物体的文本描述 $T$，然后用 BLIP 的视觉编码器 $\Phi_v$ 提取 patch 特征 $F_v = \{f_v^1, \ldots, f_v^N\}$，文本编码器 $\Phi_t$ 提取全局文本特征 $f_t$，计算每个 patch 与文本的余弦相似度 $s_i = \hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}$，再经语义锐化函数 $\Psi_{sharp}(s_i; \gamma, \delta) = (\max(0, \mathcal{N}(s_i) - \delta))^\gamma$ 进行阈值过滤和非线性放大，最终上采样至原图分辨率
    - 设计动机：对比了 ViT class attention（粗糙弥散）、CLIP（有背景噪声）和 BLIP（干净、边界清晰）三种方案，BLIP 的融合对齐策略生成的引导图质量最优，能精确高亮文本描述的物体且背景噪声极低

2. **Cross-Attention 注入机制**：

    - 功能：在 decoder 各阶段通过 cross-attention 将语义引导图注入重建过程，使网络优先从语义"高亮"区域提取编码器特征
    - 核心思路：在 decoder 第 $l$ 阶段，decoder 特征 $d_l$ 作为 Query，encoder 的 skip-connection 特征 $e_l$ 经下采样后的引导图 $\tilde{M}^{(l)}$ 加权后线性投影为 Key 和 Value，计算 $d_l' = \text{softmax}(Q_l K_l^\top / \sqrt{d_k}) V_l$
    - 设计动机：对比了 Encoder only、All stages、Decoder only 三种注入位置，Decoder only 效果最优——decoder 阶段直接影响图像重建过程，在此注入语义引导能最高效地分配重建资源；编码器阶段注入反而可能干扰特征提取

3. **显式语义对齐损失 $\mathcal{L}_{align}$**：

    - 功能：为 cross-attention 的隐式引导提供显式、可量化的监督信号，确保 decoder 中间特征的空间分布与语义引导图对齐
    - 核心思路：对 decoder 第 $l$ 阶段特征图 $\mathbf{F}^{(l)}$ 施加双项约束——背景抑制项 $\|\mathbf{F}^{(l)} \odot (1 - \tilde{M}^{(l)})\|_F^2$ 惩罚非关键区域的过强激活，前景增强项 $-\eta \langle \mathbf{F}^{(l)}, \tilde{M}^{(l)} \rangle$ 奖励关键物体区域的强响应
    - 设计动机：cross-attention 提供的是结构性引导，其效果是隐式的；alignment loss 从损失函数层面直接约束特征分布，二者协同工作效果最优

### 损失函数 / 训练策略

总损失为重建损失与语义对齐损失的加权和：$\mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda_{align} \sum_{l \in L} \mathcal{L}_{align}^{(l)}$。其中重建损失 $\mathcal{L}_{recon} = \|I_e - I_{gt}\|_1 + \lambda_{percep} \sum_j \|\phi_j(I_e) - \phi_j(I_{gt})\|_1$ 包含 L1 像素损失和基于 VGG-19 的感知损失。$\lambda_{align}$ 设为 0.1。在 UIEB 训练集（790 对配对图像）上训练，策略作为可插拔模块分别应用于五个 baseline。

## 实验关键数据

### 主实验

**UIE 感知质量（UIEB 测试集）**：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| PUIE | 21.05 | 0.869 | 0.117 |
| PUIE-SS | **23.20**(+2.15) | **0.884**(+0.015) | **0.092**(-0.025) |
| SMDR | 22.44 | 0.899 | 0.106 |
| SMDR-SS | **23.28**(+0.84) | **0.909**(+0.010) | **0.099**(-0.007) |
| UIR | 22.89 | 0.885 | 0.124 |
| UIR-SS | **24.62**(+1.73) | **0.901**(+0.016) | **0.113**(-0.011) |
| PFormer | 23.53 | 0.877 | 0.113 |
| PFormer-SS | **24.97**(+1.44) | **0.933**(+0.056) | **0.087**(-0.026) |
| FDCE | 23.66 | 0.909 | 0.111 |
| FDCE-SS | **24.63**(+0.97) | **0.927**(+0.018) | **0.093**(-0.018) |

**下游任务性能（检测 mAP / 分割 mIoU）**：

| 方法 | mAP↑ | mIoU↑ |
|------|------|-------|
| 原图（无增强） | 95.43 | 68.10 |
| PUIE → PUIE-SS | 95.40 → **96.28**(+0.88) | 66.20 → **70.80**(+4.60) |
| SMDR → SMDR-SS | 95.76 → **96.98**(+1.22) | 68.18 → **73.51**(+5.33) |
| UIR → UIR-SS | 94.37 → **95.31**(+0.94) | 68.52 → **70.45**(+1.93) |
| PFormer → PFormer-SS | 95.50 → **96.87**(+1.37) | 69.34 → **74.75**(+5.41) |
| FDCE → FDCE-SS | 95.72 → **97.01**(+1.29) | 69.78 → **72.36**(+2.58) |

### 消融实验

**引导图生成模型对比**：

| 模型 | 引导图质量 | 特点 |
|------|-----------|------|
| ViT class attention | 差 | 粗糙弥散，无法精确定位物体 |
| CLIP | 中等 | 注意力较锐利但背景有噪声激活 |
| BLIP | 最优 | 干净、边界清晰、无背景噪声 |

**语义引导注入位置对比**：

| 注入位置 | 效果 |
|---------|------|
| Encoder only | 最差，干扰特征提取 |
| All stages | 中等 |
| Decoder only | 最优，直接影响重建过程 |

**双重引导机制消融**：cross-attention 和 alignment loss 单独使用均有提升，二者协同使用效果最优，验证了结构引导与显式监督的互补性。

### 关键发现

- 所有 5 个 baseline 加上 -SS 策略后 PSNR/SSIM 均获提升，PUIE-SS 的 PSNR 提升最大（+2.15 dB）
- 分割 mIoU 提升最为显著：PFormer-SS +5.41，SMDR-SS +5.33，验证了语义引导对像素级分类的巨大帮助
- 部分 baseline（如 PUIE、UIR）增强后下游性能反而低于原图（"增强悖论"），但 -SS 版本一致超过原图
- 在无参考数据集 U45 和 Challenge60 上 UIQM/UCIQE 指标也呈正面趋势，说明语义引导未导致过拟合

## 亮点与洞察

- **VLM 作为零标注语义先验源**：通过 LLaVA + BLIP 的组合巧妙绕过了水下场景像素级标注稀缺的痛点，实现了无需额外标注即可获得空间化语义引导的能力。这种策略具有很强的通用性，可推广到任何缺少密集标注的图像增强场景。

- **"增强悖论"的直接对抗**：实验中清楚展示了部分 baseline 增强后下游性能反而下降的现象，而 -SS 策略在所有 baseline 上一致性地解决了这一问题，说明语义感知增强是连接低层视觉改善与高层机器认知的关键桥梁。

- **可插拔设计的工程价值**：该策略不修改 UIE 网络本身的架构，仅在 decoder 端注入引导信号和损失项，因此可直接应用于任意 encoder-decoder 结构的 UIE 模型，五个差异很大的 baseline 上的一致提升证明了方法的通用性。

## 局限与展望

- VLM（LLaVA）和对齐模型（BLIP）均为预训练冻结模型，推理时需要额外的前向传播开销，在资源受限的水下设备上部署可能受限
- 语义引导图的质量完全依赖 VLM 对退化图像的理解能力，当退化极为严重时（如极低能见度），VLM 可能无法正确识别物体
- 当前仅在 encoder-decoder 结构的 UIE 网络上验证，对于 GAN-based 或 diffusion-based 的增强方法是否同样有效尚未探讨

## 相关工作与启发

- **vs 传统语义引导（Liao et al., Yan et al.）**：传统方法依赖像素级标注训练分割模型来获取语义先验，在水下场景标注稀缺时易引入错误先验；本文用 VLM 实现零标注的语义先验生成
- **vs CLIP 全局文本引导（Liu et al.）**：CLIP-based 方法使用全局风格提示引导增强，本质仍是全局策略；本文生成内容特异的物体描述并映射回空间，实现细粒度区域引导
- **与下游感知联合训练的对比**：联合训练方法将增强网络和下游任务网络端到端优化，但每个下游任务需要定制模型；本文策略与下游任务解耦，一次增强即可服务多个下游任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 VLM 的开放世界理解能力引入 UIE 并设计了完整的空间化语义引导管线，视角新颖
- 实验充分度: ⭐⭐⭐⭐ 5 个 baseline、3 个 UIE 数据集、2 个下游任务、完整消融，实验非常充分
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，方法描述严谨，图表丰富
- 价值: ⭐⭐⭐⭐ "增强悖论"问题具有实际意义，可插拔设计实用性强，对水下视觉和其他退化场景有启发
---
title: >-
  [论文解读] Empowering Semantic-Sensitive Underwater Image Enhancement with VLM
description: >-
  [CVPR 2026][图像分割][underwater image enhancement] 提出 VLM 驱动的语义敏感学习策略，通过 VLM 生成目标物体描述、BLIP 构建空间语义引导图、双重引导机制（cross-attention + 语义对齐损失）注入 UIE decoder，使增强结果在感知质量和检测/分割下游任务上同时提升。
tags:
  - CVPR 2026
  - 图像分割
  - underwater image enhancement
  - VLM
  - semantic guidance
  - 注意力机制
  - downstream tasks
---

# Empowering Semantic-Sensitive Underwater Image Enhancement with VLM

**会议**: CVPR 2026  
**arXiv**: [2603.12773](https://arxiv.org/abs/2603.12773)  
**代码**: 待确认  
**领域**: 水下图像增强 / 语义引导 / VLM 应用  
**关键词**: underwater image enhancement, VLM, semantic guidance, cross-attention, downstream tasks  

## 一句话总结
提出 VLM 驱动的语义敏感学习策略，通过 VLM 生成目标物体描述、BLIP 构建空间语义引导图、双重引导机制（cross-attention + 语义对齐损失）注入 UIE decoder，使增强结果在感知质量和检测/分割下游任务上同时提升。

## 背景与动机
水下图像增强（UIE）已有大量深度学习方法，但存在"增强悖论"：增强后的图像视觉质量好但下游检测/分割性能反而下降。原因在于现有方法是"语义盲"的——全局均匀增强所有区域，无法区分语义焦点（海洋生物、人工物体）和背景（水体），导致分布偏移破坏下游模型所依赖的语义线索。早期语义引导方法依赖高质量像素级标注（在水下场景极为稀缺），全局文本提示（如"a clear underwater photo"）虽利用了 VLM 但仍是一刀切策略。

## 核心问题
如何让水下图像增强具备内容感知能力，在恢复视觉质量的同时保护/增强关键物体的语义特征，使下游机器视觉任务受益？

## 方法详解

### 整体框架
三阶段策略：(1) 用 VLM（LLaVA）从退化图像生成关键物体的文本描述 → (2) 用 BLIP 的视觉-文本对齐构建空间语义引导图 M_sem → (3) 通过双重引导机制将 M_sem 注入 UIE 网络的 decoder。

### 关键设计

1. **语义引导图生成**：

    - 用 LLaVA 自动生成退化图像中关键物体的文本描述 T
    - 用 BLIP 的视觉编码器 Φ_v 提取 patch 特征 F_v = {f_v1,...,f_vN}，文本编码器 Φ_t 提取全局文本特征 f_t
    - 计算每个 patch 与文本的余弦相似度 s_i = v̂_i^T · t̂
    - 语义锐化函数 Ψ_sharp：先 min-max 归一化，减去阈值 δ 过滤低相关噪声，再取 γ 次幂（γ>1）非线性放大差异
    - 上采样至原图分辨率得到单通道引导图 M_sem
    - 对比了 ViT class attention、CLIP、BLIP 三种方案，BLIP 效果最优（干净、边界清晰、无背景噪声）

2. **Cross-Attention 注入机制**：

    - 在 decoder 各阶段 l，decoder 特征 d_l 作为 Query
    - encoder skip-connection 特征 e_l 经 M_sem 加权后生成 Key 和 Value
    - M_sem 下采样至对应分辨率 M̃(l)，e_l 乘以 M̃(l) 后投影
    - d_l' = softmax(Q_l · K_l^T / √d_k) · V_l
    - 使 decoder 优先从语义"高亮"区域提取编码器特征

3. **显式语义对齐损失 L_align**：

    - 对 decoder 第 l 阶段特征图 F(l) 施加双项约束：
    - 背景抑制项：‖F(l) ⊙ (1 - M̃(l))‖²_F → 惩罚非关键区域的过强激活
    - 前景增强项：-η⟨F(l), M̃(l)⟩ → 奖励关键物体区域的强响应
    - η 是平衡超参

### 损失函数 / 训练策略
- 总损失：L_total = L_recon + λ_align · Σ_l L_align(l)
- L_recon = L1(I_e, I_gt) + λ_percep · Σ_j ‖φ_j(I_e) - φ_j(I_gt)‖₁（VGG-19 感知损失）
- λ_align = 0.1
- 在 UIEB 训练集（790 对图像）上训练
- 策略设计为可插拔模块，已在 PUIE、SMDR、UIR、PFormer、FDCE 五个 baseline 上验证

## 实验关键数据

**UIE 感知质量（UIEB 测试集）**：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| PFormer | 23.53 | 0.877 | 0.113 |
| PFormer-SS | **24.97**(+1.44) | **0.933**(+0.056) | **0.087**(-0.026) |
| UIR | 22.89 | 0.885 | 0.124 |
| UIR-SS | **24.62**(+1.73) | **0.901**(+0.016) | **0.113**(-0.011) |

**下游任务（检测 mAP / 分割 mIoU）**：

| 方法 | mAP↑ | mIoU↑ |
|------|------|-------|
| 原图（无增强） | 95.43 | 68.10 |
| PFormer | 95.50 | 69.34 |
| PFormer-SS | **96.87**(+1.37) | **74.75**(+5.41) |
| SMDR | 95.76 | 68.18 |
| SMDR-SS | **96.98**(+1.22) | **73.51**(+5.33) |

- 所有 5 个 baseline 加上 -SS 后 PSNR/SSIM 均提升
- 分割 mIoU 提升最显著，PFormer-SS 达到 +5.41，SMDR-SS +5.33
- 某些 baseline 增强后下游性能反而低于原图，但 -SS 版本一致超过原图

### 消融实验要点
- 引导图模型对比：BLIP > CLIP > ViT（BLIP 无背景噪声、边界清晰）
- 注入位置对比：Decoder only > All stages > Encoder only（decoder 阶段直接影响重建过程）
- 消融验证了 cross-attention 和 alignment loss 二者协同最优

## 亮点
- 精准识别了"增强悖论"问题：全局增强破坏语义线索导致下游性能下降
- VLM→文本→BLIP→空间引导图的管线巧妙地避免了对水下标注数据的依赖
- 可插拔设计使策略适用于任意 encoder-decoder UIE 架构
- 双重引导（结构性 cross-attention + 显式 alignment loss）比单一机制更有效
- 同时评估感知质量和下游任务，实验协议更务实

## 局限与展望
- VLM (LLaVA) 和 BLIP 的推理开销较大，影响实时性
- 语义引导图的质量依赖于 VLM 对退化图像的理解能力，严重退化场景下可能失效
- 仅在 UIEB 上训练，水下场景多样性有限
- 锐化函数中 δ 和 γ 的选择可能需要针对不同场景调整
- 未评估对更多下游任务（如重识别、跟踪）的影响

## 与相关工作的对比
- vs 传统 UIE (PUIE/SMDR 等)：后者语义盲，本文赋予语义感知能力
- vs 语义分割引导方法 (Liao/Yan)：后者需要高质量像素级标注，本文用 VLM 零标注生成语义先验
- vs CLIP 风格引导 (Liu et al.)：CLIP 提供全局文本引导（"清晰的水下照片"），本文构建空间化的目标级语义图
- vs VINE/Watermark 方向的 VLM 应用：不同任务但都展示 VLM 语义能力在低层视觉中的价值

## 启发与关联
- VLM→文本描述→空间引导图的管线可推广到其他退化场景（雾天、低光照）
- 双重引导机制（architectural guidance + loss supervision）的组合思路有通用性
- 下游任务感知的增强是图像恢复领域的重要趋势

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将 VLM 空间语义引导引入水下增强，管线设计新颖
- 实验充分度: ⭐⭐⭐⭐ 5 个 baseline、3 个评估数据集、检测+分割下游评估、消融完整
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，方法逻辑连贯，图表直观
- 价值: ⭐⭐⭐⭐ 可插拔策略实用性强，对水下视觉和下游感知应用有实际意义

<!-- RELATED:START -->

## 相关论文

- [EReCu: Pseudo-label Evolution Fusion and Refinement with Multi-Cue Learning for Unsupervised Camouflage Detection](erecu_pseudolabel_evolution_unsupervised_camouflage.md)
- [Reasoning with Pixel-level Precision: QVLM Architecture and SQuID Dataset for Quantitative Geospatial Analytics](reasoning_with_pixel-level_precision_qvlm_architecture_and_squid_dataset_for_qua.md)
- [Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation](spatio-semantic_expert_routing_architecture_with_mixture-of-experts_for_referrin.md)
- [CTFS: Collaborative Teacher Framework for Forward-Looking Sonar Image Semantic Segmentation with Extremely Limited Labels](ctfs_collaborative_teacher_framework_for_forward-looking_sonar_image_semantic_se.md)
- [Towards Context-Aware Image Anonymization with Multi-Agent Reasoning](towards_context-aware_image_anonymization_with_multi-agent_reasoning.md)

<!-- RELATED:END -->
