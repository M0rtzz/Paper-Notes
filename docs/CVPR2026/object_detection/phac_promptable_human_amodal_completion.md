---
title: >-
  [论文解读] PHAC: Promptable Human Amodal Completion
description: >-
  [CVPR 2026][目标检测][人体非模态补全] 提出可提示人体非模态补全（PHAC）新任务，通过基于点的用户提示（姿态/边界框）配合 ControlNet 注入条件信号，并设计基于修复的精炼模块保留可见区域外观，实现高质量、可控的遮挡人体图像补全。
tags:
  - CVPR 2026
  - 目标检测
  - 人体非模态补全
  - 扩散模型
  - ControlNet
  - 姿态引导生成
  - 图像修复
---

# PHAC: Promptable Human Amodal Completion

**会议**: CVPR 2026  
**arXiv**: [2603.14741](https://arxiv.org/abs/2603.14741)  
**代码**: 无  
**领域**: 目标检测  
**关键词**: 人体非模态补全, 扩散模型, ControlNet, 姿态引导生成, 图像修复

## 一句话总结

提出可提示人体非模态补全（PHAC）新任务，通过基于点的用户提示（姿态/边界框）配合 ControlNet 注入条件信号，并设计基于修复的精炼模块保留可见区域外观，实现高质量、可控的遮挡人体图像补全。

## 研究背景与动机

**人体非模态补全（HAC）的局限**：现有 HAC 方法只能从遮挡图像中幻想不可见区域，无法接受用户指定的约束（如目标姿态或空间范围），用户需反复采样才能获得满意结果。

**姿态引导人像合成（PGPIS）的不足**：PGPIS 允许姿态条件输入，但难以保持特定实例的可见外观，倾向于生成训练分布偏差的内容（如 DeepFashion 数据集的服装特征）。

**可见外观退化问题**：潜空间去噪和 VAE 重建会丢失精细可见细节，已有的解码器微调方案会引入模糊和边界伪影，UV 坐标方案在坐标噪声时丢失细节。

**缺乏多类型提示支持**：现有方法不支持多种类型的用户提示（姿态+边界框），无法灵活权衡性能与用户交互成本。

**零样本泛化能力弱**：PGPIS 基线（如 PIDM、MCLD）在训练集外的真实图像上常出现严重外观幻觉，将老年男性变为白人女性等极端失败情况。

**边界伪影**：直接将可见区域和生成区域拼接会在掩码边界处引入明显伪影，缺乏平滑过渡机制。

## 方法详解

### 整体框架

PHAC 框架分为两个阶段：（A）粗略图像生成 和（B+C）精炼阶段。

- **输入**：遮挡人体图像 $I_{ic}$ + 用户提示 $P$（支持 5 种类型：姿态 $p_{po}$、兴趣区域 bbox $p_{ib}$、全区域 bbox $p_{eb}$、姿态+兴趣 bbox $p_{poib}$、姿态+全 bbox $p_{poeb}$）
- **粗略生成**：将用户提示转为提示图像 $I_p$，通过专用 ControlNet $\Phi_{CN}$ 编码为条件信号 $c_{pr}$，注入去噪 U-Net $\epsilon_{cig}$，从随机噪声去噪 $T$ 步生成粗略补全 $I_{cc}$
- **精炼阶段**：轻量 U-Net $\mathcal{U}_{iv}$ 预测不可见区域掩码 $M_{iv}$ 并膨胀，构建基线合成图后仅对掩码区域注入低幅噪声，精炼网络 $\Phi_{RF}$ 执行少量去噪步得到最终输出 $I_{rc}$

### 关键设计

1. **多类型点提示**：用户只需提供少量点——姿态提示为额外关节点坐标，bbox 提示为两个角点。提示被渲染为图像送入对应的 ControlNet 模块。
2. **仅微调交叉注意力块**：为保持预训练扩散模型的生成先验，仅微调去噪 U-Net 的交叉注意力块，其余参数冻结，兼顾提示对齐与生成质量。
3. **SAM 预测可见掩码**：不依赖真实可见区域掩码，而用 SAM 自动从输入图像预测 $M_v$，提升实用性。
4. **不可见掩码预测与膨胀**：轻量 U-Net 以 $I_{ic}$、$I_{cc}$、$M_v$ 为输入预测不可见掩码 $M_{iv}$，并通过膨胀操作扩大掩码范围以避免边界像素遗漏。
5. **基于修复的精炼**：不是重新生成掩码区域的 RGB，而是对粗略补全注入少量噪声（$s=0.5$），仅执行约 40% 的标准去噪步数（20步），保持可见区域不变并消除边界伪影。
6. **即插即用精炼模块**：精炼网络可直接应用于其他扩散模型的输出，作为通用后处理组件提升性能。

### 损失函数与训练策略

- **粗略生成损失**：标准扩散去噪目标 $\mathcal{L} = \mathbb{E}[\|\epsilon - \epsilon_{cig}(z_t, t, c_{te}, c_{pr})\|_2^2]$
- **掩码预测损失**：BCE + Dice 加权组合 $\mathcal{L} = \mathcal{L}_{BCE} + 0.5 \cdot \mathcal{L}_{Dice}$
- **精炼网络**：直接使用预训练 SDXL Inpainting 模型，无需额外训练
- **训练设置**：DM 学习率 $5 \times 10^{-6}$，ControlNet 学习率 $5 \times 10^{-5}$，batch=14，训练 1750 epoch（4×A6000，约 16 小时）；掩码 U-Net 训练 40 epoch（1×RTX 3090，约 30 分钟）
- **随机推理**：DM 为每个训练图像生成 $N=16$ 个粗略输出，掩码 U-Net 训练时随机采样其一进行监督

## 实验关键数据

### 主实验

**数据集**：OccThuman2.0（合成，5260 张）+ AHP（真实，56 张）

| 方法 | 提示类型 | LPIPS*↓ | SSIM↑ | KID*↓ | PSNR↑ | Joint Err.↓ |
|------|----------|---------|-------|-------|-------|-------------|
| PIDM | 2D pose | 126.33 | 0.797 | 56.91 | 16.80 | 113.72 |
| MCLD | UV map | 115.90 | 0.833 | 41.11 | 18.37 | 53.38 |
| pix2gestalt | - | 90.11 | 0.911 | 16.51 | 22.63 | 36.65 |
| SDHDO | 2D pose | 81.39 | 0.924 | 16.41 | 23.80 | 43.49 |
| **Ours** | **2D pose** | **49.47** | **0.948** | **6.12** | **25.86** | **23.33** |

*OccThuman2.0 数据集结果，值 ×10³*

| 方法 | LPIPS*↓ | SSIM↑ | KID*↓ | PSNR↑ | Joint Err.↓ |
|------|---------|-------|-------|-------|-------------|
| SDHDO | 64.19 | 0.956 | 6.05 | 24.45 | 9.24 |
| **Ours** | **38.77** | **0.970** | **1.25** | **26.93** | **6.37** |

*AHP 真实数据集结果*

### 消融实验

**不同提示类型对比（OccThuman2.0）**：

| 提示类型 | LPIPS*↓ | SSIM↑ | PSNR↑ | Joint Err.↓ |
|----------|---------|-------|-------|-------------|
| 姿态 $p_{po}$ | 49.47 | 0.948 | 25.86 | 23.33 |
| 兴趣 bbox $p_{ib}$ | 51.83 | 0.942 | 24.99 | 24.01 |
| 全区域 bbox $p_{eb}$ | 52.28 | 0.941 | 25.07 | 28.23 |
| 姿态+兴趣 bbox | 49.35 | 0.947 | 25.69 | 22.15 |
| 姿态+全 bbox | 49.42 | 0.946 | 25.49 | 21.96 |

**噪声强度消融（$s$ 参数）**：$s=0.5$ 在 OccThuman2.0 和 AHP 上均为最优，过小（0.1）去噪不充分，过大（0.9）破坏原有外观。

**精炼模块即插即用效果**：应用于 MCLD 后 MSE 降低约 60%、KID 降低约 71%；应用于 pix2gestalt 后 LPIPS 降低 37%。

### 关键发现

1. 姿态提示提供最有效的单提示引导，bbox 提示主要约束空间范围但留下姿态歧义
2. 组合提示（姿态+bbox）在保持感知质量的同时一致性降低关节误差
3. 精炼模块即使在本方法精炼前已超越所有基线，精炼后进一步提升全部指标
4. 兴趣区域 bbox 的每点增益最高（$\Delta$JE pp=2.86），用户交互效率最优

## 亮点与洞察

- **任务定义新颖**：首次提出可提示人体非模态补全任务，在 HAC 和 PGPIS 之间建立了自然连接
- **实用的提示设计**：基于点的提示极度轻量（几个点即可），避免了 3D 信息或密集掩码等难以获取的输入
- **精炼模块通用性**：基于修复的精炼是通用即插即用组件，对其他方法也有显著提升
- **仅微调交叉注意力**：以极小参数量变动获得强提示对齐，保留预训练先验
- **训练效率高**：主模型 4 卡 16 小时，掩码网络单卡 30 分钟，总训练成本可控

## 局限性

- 仅在合成数据（OccThuman2.0，526 个 3D 人体）上训练，真实场景多样性受限
- AHP 测试集仅 56 张图像，真实场景评估规模偏小
- 精炼网络依赖预训练 SDXL Inpainting，单张精炼约 4 秒，实时性不足
- 姿态提示需用户手动标注缺失关节，遮挡严重时点数增加交互负担
- 缺少与最新 DiT 架构或视频扩散模型的比较
- 未探索文本提示或其他高层语义条件的融合

## 相关工作

- **HAC 方法**：pix2gestalt（通用非模态补全+扩散先验）、SDHDO（人体专用+2D 姿态先验），但均不支持用户提示且可见外观退化
- **PGPIS 方法**：PIDM（2D pose map 条件扩散）、MCLD（UV map 条件），但训练数据偏差导致外观幻觉严重
- **ControlNet**：本文多类型提示注入的基础架构，每种提示类型训练专用 ControlNet
- **SAM**：用于自动预测可见区域掩码，替代依赖真值掩码的传统方案

## 评分

- 新颖性: ⭐⭐⭐⭐ — 任务定义新颖，提示机制设计巧妙
- 实验充分度: ⭐⭐⭐⭐ — 多基线比较+多维消融（提示类型/噪声强度/即插即用），但真实数据规模偏小
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式完整，图表丰富
- 价值: ⭐⭐⭐⭐ — 精炼模块的通用性和提示设计的实用性有较好应用前景

<!-- RELATED:START -->

## 相关论文

- [StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars](streamavatar_streaming_diffusion_models_for_real-time_interactive_human_avatars.md)
- [FusionAgent: A Multimodal Agent with Dynamic Model Selection for Human Recognition](fusionagent_a_multimodal_agent_with_dynamic_model_selection_for_human_recognitio.md)
- [Show, Don't Tell: Detecting Novel Objects by Watching Human Videos](show_dont_tell_detecting_novel_objects_by_watching.md)
- [ESCAPE: Equivariant Shape Completion via Anchor Point Encoding](../../CVPR2025/object_detection/escape_equivariant_shape_completion_via_anchor_point_encoding.md)
- [Mining Instance-Centric Vision-Language Contexts for Human-Object Interaction Detection](mining_instance-centric_vision-language_contexts_for_human-object_interaction_de.md)

<!-- RELATED:END -->
