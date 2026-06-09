---
title: >-
  [论文解读] MagicMirror: ID-Preserved Video Generation in Video Diffusion Transformers
description: >-
  [ICCV 2025][视频生成][身份保持视频生成] MagicMirror 是首个基于 Video Diffusion Transformer（CogVideoX）实现零样本身份保持视频生成的框架，通过双分支面部特征提取、条件自适应归一化（CAN）和图像预训练+视频微调两阶段策略…
tags:
  - "ICCV 2025"
  - "视频生成"
  - "身份保持视频生成"
  - "Transformer"
  - "面部特征提取"
  - "自适应归一化"
  - "两阶段训练"
---

# MagicMirror: ID-Preserved Video Generation in Video Diffusion Transformers

**会议**: ICCV 2025  
**arXiv**: [2501.03931](https://arxiv.org/abs/2501.03931)  
**代码**: [https://github.com/dvlab-research/MagicMirror/](https://github.com/dvlab-research/MagicMirror/)  
**领域**: 扩散模型  
**关键词**: 身份保持视频生成、扩散Transformer、面部特征提取、自适应归一化、两阶段训练

## 一句话总结

MagicMirror 是首个基于 Video Diffusion Transformer（CogVideoX）实现零样本身份保持视频生成的框架，通过双分支面部特征提取、条件自适应归一化（CAN）和图像预训练+视频微调两阶段策略，在保持人脸身份一致性的同时生成高质量动态视频。

## 研究背景与动机

**领域现状**：扩散模型在文本到图像生成领域已取得巨大成功，身份保持（ID-preserving）图像生成方法如 PhotoMaker、InstantID、PuLID 等实现了在不微调的情况下保持特定人物身份。然而在视频生成领域，这一能力仍然欠缺。

**现有痛点**：现有 ID 保持视频生成方法存在两类局限：(1) MagicMe 等基于微调的方法需要对每个身份进行单独优化，效率低且泛化性差；(2) ID-Animator 等基于 inflated UNet 的方法受限于基础模型能力，生成的视频动态范围有限，面部表情几乎是静态的"复制粘贴"，无法产生自然的面部动作。另一类两阶段方法先用图像个性化生成参考图再做 I2V，在长序列中身份稳定性差。

**核心矛盾**：先进的视频生成模型（如 CogVideoX）基于 full-attention DiT 架构，与传统的 cross-attention 条件注入方式不兼容。DiT 使用层级分布调制（layer-wise modulation）而非独立的 cross-attention，这使得将身份条件整合到 DiT 中变得复杂。此外，高质量的身份-视频配对训练数据极度匮乏。

**本文目标**：在 Video DiT 架构上实现无需针对特定人物微调的 ID 保持视频生成，同时保持动态自然的面部运动。

**切入角度**：利用 CogVideoX 已有的层级调制机制，设计轻量级的身份条件适配器；通过合成数据 + 渐进式训练解决数据稀缺问题。

**核心 idea**：设计条件自适应归一化（CAN）模块来预测身份相关的分布偏移，结合双分支面部特征提取（高层身份 + 结构细节），实现身份信息在 DiT 中的高效融合。

## 方法详解

### 整体框架

MagicMirror 基于 CogVideoX-5B 构建。输入是一张或多张人脸参考图和文本 prompt。左侧的双分支特征提取器分别提取身份嵌入和面部结构嵌入。这些嵌入通过跨模态适配器（包含 CAN 和解耦交叉注意力）注入到 DiT 的交替层中。训练分两个阶段：先在图像数据上预训练身份保持能力，再在视频数据上微调时序一致性。

### 关键设计

1. **双分支面部特征提取（Decoupled Facial Feature Extraction）**:

    - 功能：同时捕获高层身份语义和面部结构细节信息
    - 核心思路：从预训练的 CLIP ViT 提取稠密特征图 $\mathbf{f}$。ID 分支使用 ArcFace 提取高层身份特征 $\mathbf{q}_{id}$，通过 Q-Former 架构对 $\mathbf{f}$ 做交叉注意力得到 $x_{id}$，再通过融合 MLP 映射到文本嵌入空间，在身份相关的 token 位置替换文本嵌入。Face 分支使用可学习的 32-token query $\mathbf{q}_{face}$，通过另一个 Q-Former 从 $\mathbf{f}$ 提取面部结构特征 $x_{face}$，用于后续的 full-attention 和交叉注意力
    - 设计动机：单一的身份嵌入不足以同时保持发型、脸型等结构信息和身份特征。解耦设计让 ID 特征通过文本通道引导语义，面部结构特征通过注意力通道提供细节参考

2. **条件自适应归一化（Conditioned Adaptive Normalization, CAN）**:

    - 功能：将身份条件信息高效注入到 DiT 的分布调制中
    - 核心思路：CogVideoX 已有针对 text 和 video 两个模态的层级调制模块 $\varphi_{txt}, \varphi_{vid}$，分别预测各自的 scale/shift/gate 参数。MagicMirror 新增面部模态的调制模块 $\varphi_{face}$ 来处理面部特征。关键创新是 CAN 模块 $\varphi_{cond}$，它以时间嵌入 $\mathbf{t}$、层索引 $l$、视频调制因子 $\mu_{vid}^1$ 和 ID 嵌入 $x_{id}$ 为条件，预测对 text 和 video 模态的分布偏移 $\hat{m}_{vid}, \hat{m}_{txt}$。最终调制因子通过残差相加获得：$m_{vid} = \hat{m}_{vid} + \varphi_{vid}(\mathbf{t}, l)$
    - 设计动机：直接添加 cross-attention 在 full-attention DiT 中效果有限，因为 DiT 的条件控制主要通过分布调制实现。CAN 让身份信息直接影响 text/video 特征的分布，加速收敛且提升身份保真度。实验表明没有 CAN 时模型连最基本的发型特征都无法学习

3. **两阶段渐进训练策略**:

    - 功能：解决 ID-视频配对数据稀缺的问题
    - 核心思路：第一阶段在多样化的图像数据上预训练（LAION-Face 50K + SFHQ + FFHQ 合成配对），学习鲁棒的身份保持能力，训练 30K 步，batch size 64。第二阶段在高质量视频数据上微调（Pexels + Mixkit + 少量自采集视频），增强时序一致性，训练 5K 步，batch size 8。合成数据通过 PhotoMakerV2 生成同一身份的不同姿态图像，用 ArcFace 相似度 >0.65 过滤
    - 设计动机：直接用视频训练数据量不足且身份多样性有限。图像预训练先建立强的身份嵌入能力，视频微调再将这种能力迁移到时序域。仅用图像训练会导致视频推理时出现色偏，两阶段策略解决了调制因子在不同训练阶段的不一致问题

### 损失函数 / 训练策略

损失函数包含去噪损失和身份感知损失：$\mathcal{L} = \mathcal{L}_{noise} + \lambda (1 - \cos(q_{face}, D(x_0)))$，其中 $D(\cdot)$ 是解码去噪后的 latent。50% 的训练样本仅在面部区域计算去噪损失。

## 实验关键数据

### 主实验

与 I2V 和 ID 保持方法的定量对比（使用 VBench 和自定义指标）：

| 方法 | 面部相似度↑ | 动态度↑ | Prompt一致性↑ | IS↑ | 面部运动(FM_ref)↑ | 用户偏好↑ |
|------|-----------|---------|-------------|------|-----------------|---------|
| DynamiCrafter | 0.455 | 0.168 | 8.20 | 0.896 | 0.237 | 5.87 |
| CogVideoX-I2V | 0.660 | 0.213 | 9.85 | 0.901 | 0.413 | 6.22 |
| ID-Animator | 0.140 | 0.211 | 7.57 | 0.923 | 0.652 | 5.63 |
| **MagicMirror** | **0.705** | **0.240** | **10.59** | 0.911 | **0.704** | **6.97** |

### 消融实验

| 配置 | 效果说明 |
|------|---------|
| w/o Face 分支 | 缺少结构细节引导，身份保真度显著下降 |
| w/o CAN | 收敛困难，预训练阶段连发型都学不到 |
| 仅图像训练 | 视频推理出现色偏伪影 |
| 仅视频训练 | 身份保持能力弱 |
| 完整两阶段 | 最优，ID 高保真 + 动态面部运动 |

计算开销对比：

| 模型 | GPU 显存 | 参数量 | 推理时间(49帧 480P) |
|------|---------|--------|-------------------|
| CogVideoX-5B | 24.9 GiB | 10.5B | 204s |
| MagicMirror | 28.6 GiB | 12.8B | 209s |

### 关键发现

- **CAN 对收敛至关重要**：没有 CAN 时模型在图像预训练阶段就无法拟合基本的外观特征，加入 CAN 后收敛速度和质量都大幅提升
- **面部运动指标优势明显**：MagicMirror 在 FM_ref（相对面部运动）上达到 0.704，远超 CogVideoX-I2V 的 0.413，说明生成的视频真正有动态面部表情
- **计算开销极小**：仅增加 2.3B 参数（大部分集中在特征提取器，只需单次前向传播）和 5s 推理时间
- **用户研究全面领先**：在运动动态、文本对齐、视频质量和身份一致性四个维度均获得最高评分

## 亮点与洞察

- **CAN 的设计哲学**：不是简单叠加一个 cross-attention 分支，而是利用 DiT 已有的分布调制机制，通过预测身份条件下的分布偏移来注入身份信息。这种"顺应架构"的适配方式比"强行嫁接"更优雅高效，对其他 DiT 适配任务有启发
- **合成数据管线**：利用 PhotoMakerV2 生成同一身份的多样化参考图像来构造训练对，搭配严格的 ArcFace 相似度过滤，是一套实用的数据生产策略
- **Average Similarity 指标**：提出用与参考图像集的平均相似度而非单张相似度来评估 ID 保持，避免了"复制粘贴"行为获得虚高分数

## 局限与展望

- 不支持多人身份定制视频生成，目前仅处理单人场景
- 主要关注面部身份特征，服装、配饰等细粒度属性的保持能力有限
- 基于 CogVideoX-5B，受限于基础模型的生成质量上限和视频时长
- 存在深度伪造（deepfake）风险，需关注社会影响和肖像权保护

## 相关工作与启发

- **vs ID-Animator**: 使用 inflated UNet + face adapter，生成的视频面部几乎静态。MagicMirror 基于更先进的 DiT 架构，面部运动量大 3 倍以上
- **vs MagicMe**: 需要对每个身份进行单独微调，MagicMirror 零样本即可工作，效率远高
- **vs IP-Adapter / PhotoMaker (图像版)**: 这些方法在图像域已成熟，MagicMirror 将其思路首次成功扩展到 Video DiT 域，关键突破在于 CAN 的条件注入方式

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个基于 Video DiT 的零样本 ID 保持视频生成，CAN 设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 定量指标+用户研究+消融+分布可视化+计算开销分析，非常全面
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，Appendix 提供了丰富的补充细节
- 价值: ⭐⭐⭐⭐⭐ 个性化视频生成的里程碑工作，方法实用性强，代码已开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer](decouple_and_track_benchmarking_and_improving_video_diffusion_transformers_for_m.md)
- [\[CVPR 2025\] DiTFlow: Video Motion Transfer with Diffusion Transformers](../../CVPR2025/video_generation/video_motion_transfer_with_diffusion_transformers.md)
- [\[ICML 2025\] RIFLEx: A Free Lunch for Length Extrapolation in Video Diffusion Transformers](../../ICML2025/video_generation/riflex_a_free_lunch_for_length_extrapolation_in_video_diffusion_transformers.md)
- [\[ICML 2025\] AsymRnR: Video Diffusion Transformers Acceleration with Asymmetric Reduction and Restoration](../../ICML2025/video_generation/asymrnr_video_diffusion_transformers_acceleration_with_asymmetric_reduction_and_.md)
- [\[CVPR 2025\] Towards Precise Scaling Laws for Video Diffusion Transformers](../../CVPR2025/video_generation/towards_precise_scaling_laws_for_video_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
