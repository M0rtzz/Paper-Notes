---
title: >-
  [论文解读] RobuSTereo: Robust Zero-Shot Stereo Matching under Adverse Weather
description: >-
  [ICCV 2025][3D视觉][立体匹配] 提出 RobuSTereo 框架，通过基于扩散模型的立体数据生成管线和结合去噪 ViT 与 VGG19 的鲁棒特征编码器，大幅提升立体匹配模型在雨、雾、雪等恶劣天气下的零样本泛化能力。 立体匹配是计算机视觉中的基础任务，通过估计左右图像之间的视差来获取深度信息…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "立体匹配"
  - "恶劣天气"
  - "零样本泛化"
  - "扩散模型数据生成"
  - "鲁棒特征编码器"
  - "深度估计"
---

# RobuSTereo: Robust Zero-Shot Stereo Matching under Adverse Weather

**会议**: ICCV 2025  
**arXiv**: [2507.01653](https://arxiv.org/abs/2507.01653)  
**代码**: 待确认  
**领域**: 3D视觉  
**关键词**: 立体匹配, 恶劣天气, 零样本泛化, 扩散模型数据生成, 鲁棒特征编码器, 深度估计

## 一句话总结

提出 RobuSTereo 框架，通过基于扩散模型的立体数据生成管线和结合去噪 ViT 与 VGG19 的鲁棒特征编码器，大幅提升立体匹配模型在雨、雾、雪等恶劣天气下的零样本泛化能力。

## 研究背景与动机

立体匹配是计算机视觉中的基础任务，通过估计左右图像之间的视差来获取深度信息，广泛应用于自动驾驶、机器人和增强现实等领域。现有方法（IGEV、StereoBase、StereoAnything 等）在正常条件下表现良好，但在雨天、雾天、雪天等恶劣天气下性能严重退化。

两大核心挑战：

**训练数据稀缺**：现有立体数据集以正常天气为主。传统方法用图形渲染（如 vKITTI）模拟天气，但无法捕捉复杂光效（如湿路面的镜面反射），与真实场景存在域差距。实际采集恶劣天气数据则受限于 LiDAR 等传感器在恶劣条件下精度差且采集成本高。

**特征提取困难**：预训练于正常条件的编码器（如 MobileNetV2、ImageNet 预训练模型）面对低可见度、高噪声的退化图像时，提取的特征不稳定、含大量噪声，直接影响后续匹配精度。

这两个问题在零样本场景下尤其突出——模型未经恶劣天气微调就直接部署，性能损失更大。论文的核心动机就是同时从数据和模型两个维度解决恶劣天气立体匹配的鲁棒性问题。

## 方法详解

### 整体框架

RobuSTereo 包含三个核心组件：

1. **Prompts Generation**：利用 LLM（GPT-4）和深度估计网络生成天气描述提示词和深度图
2. **Data Generation**：基于 ControlNet + Stable Diffusion 的扩散数据生成管线，含一致性增强模块
3. **Robust Stereo Matching Model**：集成鲁棒特征编码器的立体匹配网络

### 关键设计一：基于扩散模型的立体数据生成

**目标**：将正常天气域的立体数据 $(I_R, I_L, D)_{norm}$ 转换为恶劣天气域 $(I'_R, I'_L, D)_{adv}$，同时保持视差标注 $D$ 不变。

**流程**：

- **提示词生成**：将源图像输入 LLM，自动生成对应天气条件的文本提示（如 *"Rainy, dark clouds, wet pavement, raindrops, reflections, and misty air"*）
- **深度条件控制**：使用 DepthAnythingV2 预测深度图 $D_{pred}$，作为 Depth2Image ControlNet 的输入条件，确保生成图像与原始视差 GT 一致
- **图像生成**：ControlNet 提供条件特征 $c$，引导 Stable Diffusion 1.5 生成目标天气风格的图像。使用 DDIM 调度器，50 步采样
- **数据来源**：使用 KITTI 和 vKITTI 作为源数据，生成的合成数据集命名为 RST-Dataset

理论上可无限量生成覆盖雨、雾、雪等多种天气的训练数据，且保留准确的视差标注。

### 关键设计二：一致性增强模块 (Coherence-Enhanced Consistency Module)

扩散模型生成内容多样性强，但左右图像可能出现内容不一致，导致无法用于立体网络训练。论文提出 Disparity Fusion Method (DFM)：

- 将左右图像的特征 patch 分为 src 和 dst 集合
- 基于视差相似度和图像相似度计算 patch 对的匹配关系，找到 top-n 相似 patch 对
- 执行 patch 融合 $\mathcal{M}$ 操作，通过自注意力机制 $Attn(\cdot)$ 精炼一致特征
- 最后执行 un-fusion $\mathcal{U}$ 恢复为独立的左右图像

该模块直接嵌入扩散生成过程中，确保生成的立体图像对在几何和内容上保持对齐。

### 关键设计三：鲁棒特征编码器

替换传统的单一 CNN 编码器，设计双分支结构：

- **VGG19 分支**：提取多尺度金字塔特征 $f_c^{(i)}$（$i \in \{4, 8, 16\}$ 倍下采样），捕获细粒度局部结构
- **DVT（Denoising Vision Transformer）分支**：生成 $1/32$ 分辨率的高维鲁棒特征 $f_c^{(32)}$，能在特征级别降噪，捕获语义和上下文信息
- **视差精炼网络**：沿用 StereoBase 的迭代精炼策略

DVT 的核心优势：在特征层面执行去噪，解决恶劣天气下图像质量差导致特征不稳定的问题。

### 损失函数

训练使用标准的 L1 损失，基于生成的合成数据和真实视差标注进行监督。

## 实验关键数据

### 主实验：DrivingStereo 数据集对比 (Table 1)

| 方法 | 数据集 | Rainy EPE↓ | Rainy D1↓ | Foggy EPE↓ | Foggy D1↓ | Overall EPE↓ | Overall D1↓ |
|------|--------|-----------|----------|-----------|----------|-------------|------------|
| StereoAnything | MIX | 1.144 | 5.395 | 1.134 | 4.821 | 1.042 | 3.865 |
| MonSter | MIX | 1.153 | 5.335 | 1.152 | 5.275 | 1.081 | 4.325 |
| LightStereo | MIX | 1.105 | 4.846 | 1.155 | 4.927 | 1.088 | 4.107 |
| StereoBase | SceneFlow | 1.695 | 8.610 | 1.224 | 5.980 | 1.302 | 5.974 |
| **RobuSTereo** | **RST** | **0.973** | **1.939** | **0.853** | **1.610** | **0.836** | **1.598** |

**RobuSTereo 的 Overall D1 仅 1.598%，比第二名 StereoAnything（3.865%）降低了 58.6%。**

### SeeingThroughFog 数据集 (Table 2)

| 方法 | Snow EPE↓ | Rain EPE↓ | Dense Fog EPE↓ | Overall EPE↓ | Overall D1↓ |
|------|----------|----------|---------------|-------------|------------|
| StereoAnything | 4.265 | 3.440 | 6.055 | 4.204 | 26.526 |
| LightStereo | 3.894 | 3.034 | 5.904 | 3.853 | 28.431 |
| **RobuSTereo** | **3.409** | **2.577** | **5.317** | **3.359** | **20.881** |

在极端天气数据集上同样取得最优。

### 数据集有效性验证 (Table 3)

仅使用 RST-Dataset（不加模型增强）训练 StereoBase：

| 训练数据 | Overall EPE↓ | Overall D1↓ |
|---------|-------------|------------|
| SceneFlow | 1.302 | 5.974 |
| KITTI | 0.927 | 2.279 |
| vKITTI | 1.159 | 5.157 |
| RST-Dataset | **0.875** | **2.050** |

**RST-Dataset 超越所有现有数据集**，证明数据生成管线本身的价值。

### 消融实验 (Table 4)

| 组件 | 设置 | Overall EPE↓ | Overall D1↓ |
|------|------|-------------|------------|
| 一致性模块 | Off | 1.308 | 5.997 |
| 一致性模块 | **On** | **0.875** | **2.050** |
| 源数据 | vKITTI | 1.039 | 4.689 |
| 源数据 | **KITTI** | **0.875** | **2.050** |
| 编码器 | MobileNetV2 | 0.875 | 2.050 |
| 编码器 | DINOv2 | 0.852 | 1.793 |
| 编码器 | **Robust Encoder** | **0.836** | **1.598** |

**关键发现**：
- 一致性模块贡献最大：关闭后 D1 从 2.050 飙升到 5.997（+192%）
- 真实数据源（KITTI）优于合成数据源（vKITTI），推测 vKITTI 纹理过于简单导致域差距
- 鲁棒编码器进一步将 D1 从 2.050 压低到 1.598

## 亮点与洞察

1. **数据生成思路巧妙**：利用扩散模型做风格迁移而非从头渲染，保留了原始视差标注，解决了恶劣天气数据"标注难"的根本问题。理论上可无限生成数据，且质量超越传统 CG 方法
2. **一致性模块是关键**：DFM 借鉴视频编辑中的 token merging 思想，通过视差相似度驱动的 patch 融合保证左右图像一致性，是数据生成管线能用于立体匹配训练的核心
3. **鲁棒编码器的特征去噪**：DVT 在特征空间执行去噪，而非在图像空间，这是一个值得注意的设计选择——直接处理退化特征而非退化图像
4. **数据本身就足够强**：即使不用鲁棒编码器，仅用 RST-Dataset 训练现有模型就能超越 SOTA，说明数据质量是恶劣天气立体匹配的核心瓶颈
5. **实际应用价值高**：湿路面镜面反射是自动驾驶的真实痛点，论文的点云可视化（Figure 6）展示了其他方法在湿路面出现严重伪影而本文方法有效避免

## 局限性

1. **依赖 Stable Diffusion 1.5**：使用较旧的 SD 版本，生成质量和多样性受限；升级到更新的基础模型（如 SDXL、SD3）可能进一步提升效果
2. **极端天气泛化未充分验证**：主要在 DrivingStereo 和 SeeingThroughFog 上测试，缺少对沙尘暴、冰雹、强逆光等更极端条件的评估
3. **推理效率未讨论**：DVT + VGG19 双分支编码器的计算开销相比单编码器会增加，但论文未给出推理速度对比
4. **数据生成效率较低**：50 步 DDIM 采样 + ControlNet + LLM 提示词生成的完整管线，生成速度可能较慢
5. **对深度估计质量的依赖**：ControlNet 以 DepthAnythingV2 预测的深度为条件，深度估计误差会传播到生成图像中
6. **仅验证了有限的基础模型**：消融中只测试了 PSMNet、IGEV、StereoBase，未验证在更多最新架构上的通用性

## 相关工作与启发

- **数据生成范式**：与直接采集或 CG 渲染不同，本文的"扩散模型风格迁移 + 保留标注"范式为其他需要特殊条件数据的任务（如夜间检测、水下感知）提供了通用思路
- **一致性约束**：DFM 模块借鉴了视频编辑中的 VidToMe（CVPR 2024）思想，将多帧一致性方法迁移到立体一致性，是跨任务方法迁移的好案例
- **特征去噪 vs 图像去噪**：DVT 的特征级去噪策略启发了一个方向——在退化条件下，可能直接在特征空间处理比先恢复图像再提取特征更高效
- **与 MonSter（CVPR 2025）对比**：MonSter 结合单目深度和立体匹配，但未专门针对恶劣天气优化，本文方法在天气鲁棒性上明显更强

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 数据生成+鲁棒编码的双管齐下策略有新意，一致性模块设计巧妙
- **技术深度**: ⭐⭐⭐⭐ — 扩散生成管线与立体匹配网络的结合较完整，消融分析充分
- **实验说服力**: ⭐⭐⭐⭐⭐ — 在两个恶劣天气基准上全面 SOTA，性能提升幅度非常大（D1 降低 58%+）
- **实用价值**: ⭐⭐⭐⭐ — 直接面向自动驾驶恶劣天气场景，数据生成管线可复用
- **总评**: ⭐⭐⭐⭐ — 问题定义清晰、方法设计合理、实验有力的扎实工作

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] ZeroStereo: Zero-shot Stereo Matching from Single Images](zerostereo_zero-shot_stereo_matching_from_single_images.md)
- [\[ICCV 2025\] Learning Robust Stereo Matching in the Wild with Selective Mixture-of-Experts](learning_robust_stereo_matching_in_the_wild_with_selective_mixture-of-experts.md)
- [\[CVPR 2026\] Lite Any Stereo: Efficient Zero-Shot Stereo Matching](../../CVPR2026/3d_vision/lite_any_stereo_efficient_zero-shot_stereo_matching.md)
- [\[CVPR 2025\] FoundationStereo: Zero-Shot Stereo Matching](../../CVPR2025/3d_vision/foundationstereo_zero-shot_stereo_matching.md)
- [\[CVPR 2025\] MVSAnywhere: Zero-Shot Multi-View Stereo](../../CVPR2025/3d_vision/mvsanywhere_zero-shot_multi-view_stereo.md)

</div>

<!-- RELATED:END -->
