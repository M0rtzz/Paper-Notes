---
title: >-
  [论文解读] CObL: Toward Zero-Shot Ordinal Layering without User Prompting
description: >-
  [ICCV 2025][自监督学习][object layers] 本文提出 CObL，一种基于多个冻结 Stable Diffusion UNet 并行生成的架构，能在无需用户提示、不知物体数量的前提下，从单张图像推断出遮挡排序的物体层叠表示（每层一个 amodal 完整物体）…
tags:
  - "ICCV 2025"
  - "自监督学习"
  - "object layers"
  - "amodal completion"
  - "扩散模型"
  - "zero-shot generalization"
  - "occlusion ordering"
  - "scene decomposition"
---

# CObL: Toward Zero-Shot Ordinal Layering without User Prompting

**会议**: ICCV 2025  
**arXiv**: [2508.08498](https://arxiv.org/abs/2508.08498)  
**代码**: [项目页面](https://vision.seas.harvard.edu/cobl/)  
**领域**: 自监督学习 / 场景分解 / 扩散模型 / 感知组织  
**关键词**: object layers, amodal completion, diffusion model, zero-shot generalization, occlusion ordering, scene decomposition

## 一句话总结

本文提出 CObL，一种基于多个冻结 Stable Diffusion UNet 并行生成的架构，能在无需用户提示、不知物体数量的前提下，从单张图像推断出遮挡排序的物体层叠表示（每层一个 amodal 完整物体），并且仅用数千张合成桌面场景就能零样本泛化到真实世界照片。

## 研究背景与动机

**领域现状**：感知组织（Perceptual Organization）是计算机视觉的基础能力——人类能自然地将图像分解为多个物体、判断遮挡关系、甚至推断被遮挡部分的形状和颜色（amodal completion）。然而让机器做到这些仍然很难，特别是在 zero-shot 场景下。

**现有痛点**：
   - **Amodal completion 方法**（如 pix2gestalt）：每次只能处理一个物体，且需要人工提供掩码指定待完成的物体
   - **Inpainting 方法**（如 LaMa、SDXL Inpainting）：同样需要人工提供遮挡掩码
   - **Object-centric representation learning**（如 MONet、IODINE）：可以无监督分解场景，但局限于训练过的封闭世界（如 CLEVR），不能泛化到新物体
   - **缺少合适的训练数据**：现有数据集要么只提供 amodal 边界（无外观），要么只有遮挡排序（无 amodal completion），要么场景不够真实

**核心矛盾**：要同时实现——(1) 无用户提示、(2) 自动处理所有物体、(3) 推断遮挡顺序和 amodal 完成、(4) 零样本泛化到新物体——这四个目标在之前的方法中从未被同时满足。

**本文目标**
   - 设计一个模型能从单张图片自动推断完整的物体层叠表示
   - 不需要知道物体数量、不需要任何人工提示
   - 在仅用合成数据训练的情况下，能零样本泛化到真实世界场景

**切入角度**：利用 Stable Diffusion 作为自然图像的强先验。单个 SD 实例可能在基本的感知组织任务上也会挣扎（例如识别被遮挡断开的物体部分），但多个 SD 实例通过交叉注意力协作可以更好地解决这一问题。

**核心 idea**：用 N 个冻结的 Stable Diffusion UNet 并行去噪生成 N 个物体层，通过可学习的跨层注意力同步各层生成，并用推理时的合成约束（compositional guidance）保证各层复合回原图。

## 方法详解

### 整体框架

CObL 由三个核心部分组成：
1. **合成数据生成流水线**：通过 3D 建模 + ControlNet 文到图生成，创建合成桌面场景及对应的物体层标注
2. **模型架构**：N 个冻结 SD UNet 副本，通过跨层注意力连接，用条件适配器注入输入图像信息
3. **引导采样**：推理时用合成损失（compositional loss）和先验得分匹配（PSM loss）引导扩散去噪过程

输入是一张多物体图像，输出是 N 个按遮挡顺序排列的 RGBA 物体层（每层包含一个 amodally-completed 的物体），复合这些层即可重建原图。

### 关键设计

1. **合成数据生成流水线**:

    - 功能：生成训练用的合成桌面场景图像及对应的物体层 ground truth
    - 核心思路："部分渲染 + 文到图生成"的两步法。第一步在 Blender 中用 3D 素材摆放场景，只渲染深度图、掩码和阴影图（不渲染纹理）；第二步用 ControlNet-depth 将深度图+文字提示转化为带自然纹理的图像，再用掩码合成
    - 设计动机：相比完全虚拟渲染（如 CLEVR），这种方法生成的图像落在 SD 的自然图像分布内，有效弥合 sim-to-real gap。同一几何可以生成多种纹理，提供高效的数据增强。总共只用 600 个 3D 素材生成 2250 个训练场景

2. **并行多层扩散架构**:

    - 功能：同时生成所有物体层，而非逐个处理
    - 核心思路：N 个冻结 SD 2.1 UNet 副本（仅训练跨层注意力权重 $\phi$ 和条件适配器 $\psi$）。跨层注意力让各 UNet 使用可学习的 lateral attention 相互通信（灵感来自视频生成中的帧间注意力）。条件适配器 $c_\psi(I)$ 使用冻结 MiDaS 生成伪深度图 + 图像特征注入
    - 训练目标：标准扩散去噪损失 $\mathcal{L}(\phi,\psi) = \mathbb{E}_{t,\epsilon,(I,z_0)} \|\epsilon - \epsilon_\phi(z_0, t, c_\psi(I))\|^2$

3. **推理时引导采样**:

    - 功能：用额外约束引导去噪过程，确保输出层的物理一致性
    - 核心思路：两个互补的引导损失
        - **合成损失（Compositional Loss）** $\mathcal{L}_c$：将生成的层按遮挡顺序用 $\alpha$ 合成，要求复合结果与输入图像一致 $\|I - \bar{x}^N\|^2$。掩码通过冻结的前景分割模型自动估计
        - **先验得分匹配损失（PSM Loss）** $\mathcal{L}_{psm}$：鼓励每个层的 latent 保持在原始 SD 的自然分布内 $\|\hat{\epsilon}_t - \epsilon_\phi(\hat{z}_{0;t}, t, c_\psi(I))\|^2$，防止生成不自然的结果
    - 额外的非微分操作：周期性的层排列优化、空层清理和排序

### 损失函数 / 训练策略

- 训练：标准扩散去噪损失，10% dropout 条件嵌入（用于 classifier-free guidance）
- 推理：DDIM 30 步，guidance 参数 $w = 10^4$，$\lambda = 10^{-7}$
- 在单块 H100 GPU 上训练约 1 天，batch size = 2

## 实验关键数据

### 主实验

在自建的 TABLETOP 真实世界数据集（100 张桌面照片，2-6 个物体）上，与需要额外人工掩码输入的对比方法比较：

| 模型 | LPIPS ↓ (Best/Avg) | CLIP ↑ (Best/Avg) | 需要额外掩码? |
|------|---------|---------|------|
| LaMa (Inpainting-GAN) | .113 / — | .914 / — | ✓ (oracle) |
| SDXL Inpainting | .373 / .384 | .760 / .758 | ✓ (oracle) |
| pix2gestalt (Completion) | .128 / .153 | .889 / .872 | ✓ (oracle) |
| **CObL (Ours)** | **.094 / .122** | **.935 / .914** | ✗ |

CObL 在**不使用任何额外掩码**的情况下，全面超越了使用 oracle 掩码的对比方法。

可见分割质量（TABLETOP 子集）：
- CObL ARI = **83.5%**
- Mask2Former（未微调）ARI = 66.3%

### 消融实验

| 配置 | LPIPS ↓ | 说明 |
|------|---------|------|
| Full CObL | .094 | 完整模型 |
| w/o depth cues (MiDaS) | +4% | 移除深度条件，性能略降 |
| w/o frozen SD prior (UNet unfrozen) | +9% | 解冻 UNet 导致过拟合，泛化能力严重下降 |
| w/o guidance (PSM + compositional) | +8% | 无引导时层无法正确复合回原图 |

### 关键发现

- **冻结 SD 是关键**：解冻 UNet 虽然在合成数据上可能更好拟合，但在真实世界中泛化失败。冻结的自然图像先验是零样本泛化的核心
- **合成引导是质量保障**：推理时的 compositional guidance 让 CObL 更好地保留可见区域的细节（如茶壶的颜色花纹、书本上的文字），而对比方法倾向于"幻觉"
- **物体数量>4 时性能明显下降**：这是当前架构的主要限制，更多物体增加了层分离的难度
- **深度先验有帮助但非必需**：SD 本身已包含很强的深度先验

## 亮点与洞察

- **无提示多物体分解**：首次实现了无需用户提供任何掩码/提示就能同时完成所有物体的 amodal completion 和遮挡排序。这在交互式应用中极具价值——用户不需要逐个标记物体。
- **Sim-to-Real 的巧妙桥接**：数据流水线的设计非常聪明——用 3D 模型提供精确几何，用 ControlNet 生成自然纹理，只渲染不烘焙纹理。同一几何多种纹理即为数据增强。
- **多 SD 实例协作**：灵感来自视频生成中的帧间注意力，但物体层之间的关系比视频帧之间弱得多且更异质。用 CFG + PSM 两种互补引导来约束协作。
- **推理时组合约束**：compositional loss 直接约束"层合成后=原图"，这种物理一致性约束是方法可靠性的重要来源。

## 局限与展望

- **仅限桌面场景**：当前训练和评估都限于桌面俯拍视角，未验证更复杂的自然场景（如室内、户外多层遮挡场景）
- **物体数量限制**：N=7 个 SD UNet 副本的显存开销大，且性能在 >4 个物体时下降。需要存储 7 个 UNet 的梯度用于 guidance
- **推理速度慢**：30 步 DDIM + guidance 计算，比不使用引导的方法慢很多
- **继承 SD 偏见**：模型继承了 Stable Diffusion 的生成偏差，某些物体类型可能系统性地完成得不好
- **可改进思路**：可以探索用轻量级扩散模型替代 SD 降低计算成本；可以尝试迭代式的层数自适应策略而非固定 N

## 相关工作与启发

- **vs pix2gestalt**：pix2gestalt 用 SD 做 amodal completion 但需要人工提供待完成物体的可见掩码，且一次只处理一个物体。CObL 无需任何提示且并行处理所有物体。
- **vs Slot Attention / SLATE 等 object-centric 方法**：这些方法在 CLEVR 等封闭世界上有效，但输出是 slot 而非有排序和清晰边界的物体层，且不做 amodal completion，更本质的局限是不能泛化到训练分布外的物体。
- **vs 视频扩散模型的帧间注意力**：CObL 借鉴了这一架构，但视频帧间有强时间相关性，而物体层之间关系更加异质和松散。
- 这套方法对于 AR/VR 中的场景编辑、机器人抓取规划（需要理解物体堆叠关系）等下游任务有直接的应用潜力。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 问题定义新颖，多 SD 实例协作+推理时引导的思路巧妙
- 实验充分度: ⭐⭐⭐⭐ 建了新数据集、多角度评估、消融完整，但场景局限于桌面
- 写作质量: ⭐⭐⭐⭐ 论文组织清晰，图示精美直观
- 价值: ⭐⭐⭐⭐ 开辟了零样本感知组织的新方向，但场景泛化性有待验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection](../../ICML2026/self_supervised/from_zero_to_hero_advancing_zero-shot_foundation_models_for_tabular_outlier_dete.md)
- [\[ICML 2025\] Alpha-SQL: Zero-Shot Text-to-SQL using Monte Carlo Tree Search](../../ICML2025/self_supervised/alpha-sql_zero-shot_text-to-sql_using_monte_carlo_tree_search.md)
- [\[ICML 2026\] InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation](../../ICML2026/self_supervised/infoatlas_a_foundation_model_for_zero-shot_statistical_dependence_estimate.md)
- [\[CVPR 2025\] Transformers without Normalization](../../CVPR2025/self_supervised/transformers_without_normalization.md)
- [\[CVPR 2025\] Few-Shot Implicit Function Generation via Equivariance](../../CVPR2025/self_supervised/few-shot_implicit_function_generation_via_equivariance.md)

</div>

<!-- RELATED:END -->
