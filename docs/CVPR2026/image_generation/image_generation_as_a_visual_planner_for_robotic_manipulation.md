---
title: >-
  [论文解读] Image Generation as a Visual Planner for Robotic Manipulation
description: >-
  [CVPR 2026][图像生成][视觉规划] 将预训练图像生成模型（DiT）通过 LoRA 微调适配为机器人操作的视觉规划器，以 3×3 网格图像形式生成时序连贯的操作序列，支持文本条件和轨迹条件两种控制模式。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "视觉规划"
  - "机器人操作"
  - "扩散模型"
  - "网格图像生成"
  - "LoRA"
---

# Image Generation as a Visual Planner for Robotic Manipulation

**会议**: CVPR 2026  
**arXiv**: [2512.00532](https://arxiv.org/abs/2512.00532)  
**代码**: [GitHub](https://github.com/pangye202264690373/Image-Generation-as-a-Visual-Planner-for-Robotic-Manipulation)  
**领域**: 图像生成 / 机器人操作  
**关键词**: 视觉规划, 机器人操作, 扩散模型, 网格图像生成, LoRA

## 一句话总结

将预训练图像生成模型（DiT）通过 LoRA 微调适配为机器人操作的视觉规划器，以 3×3 网格图像形式生成时序连贯的操作序列，支持文本条件和轨迹条件两种控制模式。

## 研究背景与动机

生成真实的机器人操作视频是实现感知-规划-动作统一的关键步骤。现有视频扩散模型需要大规模领域特定数据集，泛化能力有限且计算成本高。同时，大规模图像生成模型（如 FLUX.1-dev）在语言-图像对上训练后展现出**强组合生成能力**——能在单张网格布局中排列多个语义一致的子图像，隐式地呈现出类似短视频的时序过渡。

本文的核心假设是：**预训练图像生成器已经编码了可迁移的时序先验**，通过轻量级 LoRA 微调即可成为机器人操作的视觉规划器，无需设计专门的视频架构。

## 方法详解

### 整体框架

这篇论文的核心赌注是：预训练图像生成模型其实已经偷偷学会了时序先验，不用专门设计视频架构，轻量微调一下就能当机器人操作的视觉规划器。具体做法是把一段操作视频拆成 9 帧，按蛇形排进 $3 \times 3$ 网格拼成一张图，于是「生成视频」就变成了「生成一张网格图」。训练时只保留左上角第一帧当条件、其余遮黑，让基于 FLUX.1-dev 的 DiT 学着补全整张 9 帧网格，全程用 LoRA 做参数高效微调。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["操作视频<br/>抽 9 帧"] --> B["蛇形网格排列<br/>时序相邻帧在 3×3 网格里物理相邻"]
    B --> C{"条件模式"}
    C -->|语义控制| D["双模态条件控制·文本<br/>指令经 CLIP+T5 编码注入"]
    C -->|空间控制| E["双模态条件控制·轨迹<br/>首帧叠加 2D 末端轨迹"]
    D --> F["LoRA 参数高效适配<br/>DiT(FLUX.1-dev) 单次补全整张网格"]
    E --> F
    F --> G["输出 9 帧网格<br/>解析回操作序列"]
```

### 关键设计

**1. 蛇形网格排列：让时序相邻帧在空间上也相邻**

把视频塞进网格的难点是时序关系——如果帧的空间布局和时间顺序对不上，Transformer 的局部注意力就抓不住连续动作。蛇形排列正是为此：9 帧按 $1\to2\to3$、$6\leftarrow5\leftarrow4$、$7\to8\to9$ 的回环顺序铺进网格，

$$\mathbf{D} = \begin{bmatrix} D^{\text{img}_1} & D^{\text{img}_2} & D^{\text{img}_3} \\ D^{\text{img}_6} & D^{\text{img}_5} & D^{\text{img}_4} \\ D^{\text{img}_7} & D^{\text{img}_8} & D^{\text{img}_9} \end{bmatrix}$$

这样任意两个时序相邻帧在网格里都物理相邻，局部注意力顺手就能建模短程时序依赖，不需要任何显式的时序模块就能保证帧间一致。

**2. 双模态条件控制：文本管语义、轨迹管空间**

同一个规划器给了两种控制方式，对应两类需求。文本条件生成接收语言指令（如 "pick up the red cup"）加首帧，把文本经 CLIP + T5 编码成 $c_{\text{text}} = \{e_{\text{clip}}, E_{\text{t5}}\}$ 再 cross-attention 注入 DiT，擅长的是从高层语义推出合理动作序列。轨迹条件生成则在首帧上把末端执行器的 2D 轨迹渲染出来（红→蓝表示时间推进），用叠加后的图 $\tilde{\mathbf{D}}^{\tau}$ 替换首帧当条件，让模型沿着给定路径走，擅长的是空间上的精确控制。两种模式刚好互补：语义推理找文本，几何精度找轨迹。

**3. LoRA 参数高效适配：用低秩增量完成跨域迁移**

要把通用图像生成器搬到机器人视频领域，全量微调既贵又容易过拟合。这里只对 DiT 里 self-attention 的 query/value 投影和前馈层挂 LoRA（低秩 $r \ll d$），训练参数从 $O(d^2)$ 砍到 $O(rd)$，既低成本完成迁移又不增加推理延迟。消融里这一点也是命门——去掉 LoRA 整个模型直接崩。

### 损失函数 / 训练策略

使用潜空间 MSE 损失 $\mathcal{L}_{\text{lat}} = \|\mathcal{E}(\mathbf{D}_{gt}) - \mathcal{E}(\hat{\mathbf{D}})\|_2^2$，其中 $\mathcal{E}$ 为 VAE 编码器。模型做**单次网格生成**（非自回归逐帧），一次性预测完整 9 帧网格，靠图像生成模型的组合先验完成隐式时序推理。

## 实验关键数据

### 主实验

| 数据集 | 方法 | FVD↓ | SSIM↑ | MSE↓ | Success↑ |
|--------|------|------|-------|------|----------|
| JacoPlay | Text | 490.7 | 0.797 | 0.00695 | **80.6%** |
| JacoPlay | Traj | 503.37 | **0.802** | **0.00680** | 74.0% |
| BridgeV2 | Text | **644.2** | **0.733** | **0.0135** | **73.2%** |
| BridgeV2 | Traj | 693.2 | 0.726 | 0.0152 | 70.9% |
| RT-1 | Text | 698.0 | 0.727 | 0.0118 | 72.4% |
| RT-1 | Traj | **688.1** | **0.731** | **0.0117** | **81.7%** |

### 消融实验（BridgeV2）

| 配置 | FVD↓ | SSIM↑ | Success↑ | 说明 |
|------|------|-------|----------|------|
| Full (Traj) | 644.2 | 0.733 | 73.2% | 完整模型 |
| Full (Text) | 693.2 | 0.726 | 70.9% | 完整模型 |
| w/o LoRA | 4377.1 | 0.064 | 0% | 冻结骨干完全失败 |
| w/o Prompt Template | 843.4 | 0.754 | 2.5% | 语义引导严重退化 |
| w/o Trajectory Overlay | 720.0 | 0.749 | 3.9% | 空间控制丧失 |

### 关键发现

- LoRA 是关键组件：不使用 LoRA 时 FVD 从 644 飙升至 4377，成功率降为 0%
- 文本模板对语义理解至关重要：移除后成功率从 73.2% 降至 2.5%
- 文本条件在 JacoPlay/BridgeV2 上更优（语义跟随），轨迹条件在 RT-1 上更优（空间跟随）
- 两种条件模式互补：文本擅长语义推理，轨迹擅长几何精度

## 亮点与洞察

1. **新颖视角**：首次系统验证预训练图像生成模型可作为机器人视觉规划器——仅通过 LoRA 微调即可将图像生成器转化为视频合成器
2. **极端简洁的时序建模**：完全不使用时序模块，仅通过网格布局+局部注意力实现帧间一致性
3. **成本效益**：无需大规模视频数据集或专用视频架构，利用预训练图像生成器的组合先验，LoRA 微调即可

## 局限与展望

1. 网格拼接块之间偶有色调/纹理不一致，拼接边界可能出现轻微错位
2. 9 帧的序列长度较短，难以覆盖长时间操作任务
3. 成功率基于视觉判断，未与实际机器人执行闭环验证
4. 仅在 3 个数据集上测试，未验证跨域泛化能力（如从 JacoPlay 到 BridgeV2）
5. 轨迹条件需预先提供 2D 轨迹，限制了自主规划的实用性

## 相关工作与启发

- **RIGVid**：用 AI 生成的任务视频估计 6-DoF 轨迹在真实机器人上执行
- **Gen2Act**：生成人类执行视频并条件化策略以泛化到新场景
- **ControlNet**：空间条件分支注入到冻结的文本-图像模型中，启发了本文的轨迹条件设计
- **启发**：图像生成模型的组合先验可能适用于更多规划任务——如导航路径规划、装配序列规划等

## 评分

- 新颖性: ⭐⭐⭐⭐ 将图像生成器重新定位为视觉规划器的idea新颖且具启发性
- 实验充分度: ⭐⭐⭐ 3个数据集+消融完整，但缺乏与视频生成baseline的直接对比和真机验证
- 写作质量: ⭐⭐⭐ 结构清晰但数学描述略冗余，部分内容重复
- 价值: ⭐⭐⭐ 提供了一个有趣的研究方向，但实用价值受限于缺少闭环执行验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation](../../ICCV2025/image_generation/a0_affordance_aware_hierarchical_model_robotic_manipulation.md)
- [\[AAAI 2026\] MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation](../../AAAI2026/image_generation/mp1_meanflow_tames_policy_learning_in_1-step_for_robotic_manipulation.md)
- [\[ICLR 2026\] Self-Improving Loops for Visual Robotic Planning](../../ICLR2026/image_generation/self-improving_loops_for_visual_robotic_planning.md)
- [\[CVPR 2026\] Exploring Conditions for Diffusion Models in Robotic Control](exploring_conditions_for_diffusion_models_in_robotic_control.md)
- [\[ICCV 2025\] EC-Flow: Enabling Versatile Robotic Manipulation from Action-Unlabeled Videos via Equivariant Flow Matching](../../ICCV2025/image_generation/ec-flow_enabling_versatile_robotic_manipulation_from_action-unlabeled_videos_via.md)

</div>

<!-- RELATED:END -->
