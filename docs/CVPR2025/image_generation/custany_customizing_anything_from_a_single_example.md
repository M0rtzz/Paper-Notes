---
title: >-
  [论文解读] CustAny: Customizing Anything from A Single Example
description: >-
  [CVPR 2025][图像生成][零样本定制] 本文构建了首个大规模通用物体定制数据集MC-IDC（315K图像、10K+类别），并提出CustAny框架，通过多模型ID提取、全局-局部双层ID注入和ID感知解耦模块，实现从单张参考图像对任意物体的零样本定制生成。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "零样本定制"
  - "身份保持"
  - "扩散模型"
  - "通用物体"
  - "双层注入"
---

# CustAny: Customizing Anything from A Single Example

**会议**: CVPR 2025  
**arXiv**: [2406.11643](https://arxiv.org/abs/2406.11643)  
**代码**: [https://github.com/LingjieKong-fdu/CustAny](https://github.com/LingjieKong-fdu/CustAny)  
**领域**: 图像生成 / 定制化生成  
**关键词**: 零样本定制, 身份保持, 扩散模型, 通用物体, 双层注入

## 一句话总结
本文构建了首个大规模通用物体定制数据集MC-IDC（315K图像、10K+类别），并提出CustAny框架，通过多模型ID提取、全局-局部双层ID注入和ID感知解耦模块，实现从单张参考图像对任意物体的零样本定制生成。

## 研究背景与动机

**领域现状**：扩散模型驱动的图像定制生成分为两类——物体专用方法（如DreamBooth，需多张参考图+fine-tune）和物体无关方法（如PhotoMaker、InstantID，零样本但限于人脸等特定领域）。

**现有痛点**：零样本通用物体定制面临两大障碍：(1) 缺乏大规模通用物体ID一致性数据集用于预训练；(2) 人脸等特定领域的方法（如用CLIP提取ID）无法泛化到一般物体，因为单一视觉编码器难以同时捕获细节和颜色信息。

**核心矛盾**：通用性与保真度的矛盾——通用方法需要处理玩具、动物、衣服等差异巨大的物体类别，但保持每个物体的独特ID特征是极难的。同时ID信息常与非ID元素（姿态、方向）纠缠，影响文本可编辑性。

**本文目标**：实现"任意物体+单张参考图+文本描述→保持ID的多样化生成"。

**切入角度**：通过构建专门的多类别数据集解决数据瓶颈，通过多模型互补ID提取解决特征不全问题，通过ID解耦训练解决ID/非ID纠缠问题。

**核心 idea**：结合DINOv2（擅长细节但颜色不敏感）和MAE（通过重建保留颜色信息）进行互补ID提取，双层注入将语义和细粒度ID分别嵌入扩散模型，训练时加入解耦分支迫使模型区分ID与非ID信息。

## 方法详解

### 整体框架
给定参考图像+分割mask+文本提示，CustAny通过三步生成：(1) 通用ID提取器从DINOv2和MAE获取class token和patch token；(2) 全局注入将class token融入文本embedding的类别词位置，局部注入在UNet上采样块添加交叉注意力层嵌入patch token；(3) ID感知解耦模块在训练时将ID特征和非ID特征分离。推理时50步去噪，CFG scale=7。

### 关键设计

1. **多模型互补ID提取**:

    - 功能：从参考图像提取全面的ID特征表示
    - 核心思路：DINOv2因对比学习擅长捕捉物体细节结构，但其ColorJitter数据增强导致颜色不敏感；MAE基于重建训练天然保留颜色和结构信息。因此将参考图（经mask遮挡背景后）分别送入两个编码器，各提取class token和patch token，再通过MLP对齐维度
    - 设计动机：实验证明仅用CLIP/DINO/MAE任一种都有明显ID保真度缺陷，互补组合在FID、CLIP-i、DINO-i三个指标上全面最优

2. **全局-局部双层ID注入**:

    - 功能：将ID信息最大化嵌入扩散模型且保留文本编辑能力
    - 核心思路：全局注入将DINOv2和MAE的class token与文本中类别词embedding拼接后经MLP融合，替换回文本embedding中的类别词位置，通过标准cross-attention与UNet交互。局部注入将patch token融合后作为额外cross-attention层的key/value，在UNet每个上采样块注入细粒度空间信息
    - 设计动机：全局注入提供语义级ID引导（"这是什么物体"），局部注入提供像素级细节（"物体长什么样"），二者缺一不可——仅全局时FID 49.78/DINO-i 60.67，仅局部时DINO-i 62.89，双层组合达到47.50/65.12

3. **ID感知解耦模块**:

    - 功能：训练时分离ID和非ID信息，提升生成多样性
    - 核心思路：加入"解耦分支"，用CLIP提取目标图像embedding后通过可学习mask $m_{id}$ 遮掉ID信息得到 $f_{msk}$（仅含非ID信息如姿态、方向）。解耦分支用 $f_{msk} \oplus f_{fuse}^C$ 去噪，正常分支仅用 $f_{fuse}^C$ 去噪，两条路径都预测目标图。加对比损失 $\mathcal{L}_{contrast} = Sim(f_{fuse}^C, f_{msk})$ 确保ID和非ID特征正交
    - 设计动机：如果不解耦，参考图中"站着的人"的姿态信息会泄入ID表示，导致无论文本写什么姿势都生成站姿

### 损失函数 / 训练策略
总损失 $\mathcal{L} = 2.0 \cdot \mathcal{L}_{normal} + 1.0 \cdot \mathcal{L}_{decouple} + 0.5 \cdot \mathcal{L}_{contrast}$，其中前两个是标准去噪MSE损失，最后一个是余弦相似度对比损失。训练使用SD1.5，lr=1e-5，batch=32，32块V100训练6个epoch约30小时。

## 实验关键数据

### 主实验

| 领域 | 方法 | FID↓ | CLIP-i↑ | DINO-i↑ | FaceSim↑ |
|---|---|---|---|---|---|
| 通用物体 | IP-Adapter | 70.32 | 77.18 | 44.94 | - |
| 通用物体 | **CustAny** | **47.09** | **82.16** | **65.13** | - |
| 人物定制 | PhotoMaker | 106.35 | 71.80 | 44.62 | 64.10 |
| 人物定制 | InstantID | 113.18 | 75.87 | 49.26 | 63.26 |
| 人物定制 | **CustAny** | **86.40** | **79.60** | **57.44** | **78.54** |
| 虚拟试穿 | MagicClothing | 126.09 | 76.53 | 29.10 | - |
| 虚拟试穿 | **CustAny** | **50.65** | **83.82** | **66.24** | - |

### 消融实验

| 配置 | FID↓ | CLIP-i↑ | DINO-i↑ |
|---|---|---|---|
| 仅CLIP提取 | 49.11 | 79.58 | 59.45 |
| 仅DINO提取 | 48.89 | 80.82 | 63.71 |
| 仅MAE提取 | 49.00 | 79.09 | 59.79 |
| DINO+MAE（本文） | **47.50** | **81.86** | **65.12** |
| 仅全局注入 | 49.78 | 78.76 | 60.67 |
| 仅局部注入 | 48.66 | 81.24 | 62.89 |
| 双层注入 | **47.50** | **81.86** | **65.12** |

### 关键发现
- CustAny在通用物体定制上大幅超越IP-Adapter（DINO-i提升20.19%）
- 即使在人脸定制等专用任务上，CustAny也超越专用方法（FaceSim 78.54 vs InstantID 63.26）
- DINOv2颜色不敏感的发现很有趣——ColorJitter增强导致的"副作用"通过MAE互补完美解决
- ID解耦模块对文本编辑多样性有明显提升，但对ID保真度指标影响很小

## 亮点与洞察
- **MC-IDC数据集**：首个大规模通用物体定制数据集（315K样本/10K+类别），构建pipeline（数据收集→实例分割→图像对生成→文本标注）可复用。数据集的贡献可能比方法本身更有持久价值
- **多编码器互补的洞察**：DINOv2缺颜色是因为ColorJitter训练策略，MAE保留颜色是因为重建目标——这种从训练目标反推特征特性的分析思路很精准
- **通用打败专用**：在人脸和虚拟试穿等专用任务上也超过专用方法，证明通用数据集+通用架构的路线可行

## 局限与展望
- 基于SD1.5，生成质量受限于backbone能力，迁移到SDXL/SD3.0可能进一步提升
- MC-IDC数据集依赖现有分割和追踪模型，数据质量受限于这些工具的准确率
- 推理时50步去噪较慢，未探索加速推理方案
- 可探索将该框架扩展到视频定制生成

## 相关工作与启发
- **vs IP-Adapter**: 使用CLIP作为唯一编码器且缺乏ID解耦，CustAny在所有指标上大幅领先
- **vs PhotoMaker/InstantID**: 专注人脸领域，CustAny用通用方案反而超越它们
- DINO+MAE互补提取的策略可迁移到其他需要全面视觉特征表示的任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 数据集构建和多编码器互补的insight有原创性
- 实验充分度: ⭐⭐⭐⭐ 三个领域全面对比+消融覆盖每个模块
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示丰富
- 价值: ⭐⭐⭐⭐ 数据集和通用定制框架对社区有较大贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ScribbleLight: Single Image Indoor Relighting with Scribbles](scribblelight_single_image_indoor_relighting_with_scribbles.md)
- [\[CVPR 2025\] ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models](ice_intrinsic_concept_extraction_from_a_single_image_via_diffusion_models.md)
- [\[CVPR 2025\] Traversing Distortion-Perception Tradeoff Using a Single Score-Based Generative Model](traversing_distortion-perception_tradeoff_using_a_single_score-based_generative_.md)
- [\[CVPR 2025\] DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models](difflocks_generating_3d_hair_from_a_single_image_using_diffusion_models.md)
- [\[CVPR 2025\] DeClotH: Decomposable 3D Cloth and Human Body Reconstruction from a Single Image](decloth_decomposable_3d_cloth_and_human_body_reconstruction_from_a_single_image.md)

</div>

<!-- RELATED:END -->
