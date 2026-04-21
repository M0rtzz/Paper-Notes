---
title: >-
  [论文解读] RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation
description: >-
  [ICCV 2025][3D视觉][机器人操作] 提出多模态机器人操作大模型 RoboTron-Mani 和综合数据集 RoboData，通过 3D 感知增强（UVFormer + 占据监督）与模态隔离掩码（MIM）实现多数据集联合训练，首次作为通才策略在多个数据集上同时超越专家模型。
tags:
  - ICCV 2025
  - 3D视觉
  - 机器人操作
  - 多模态大模型
  - 3D感知
  - 跨embodiment泛化
  - 数据对齐
---

# RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation

**会议**: ICCV 2025  
**arXiv**: [2412.07215](https://arxiv.org/abs/2412.07215)  
**代码**: [GitHub](https://github.com/EmbodiedAI-RoboTron/RoboTron-Mani)  
**领域**: 3D视觉  
**关键词**: 机器人操作, 多模态大模型, 3D感知, 跨具身泛化, 数据对齐

## 一句话总结

提出多模态机器人操作模型 RoboTron-Mani 和综合数据集 RoboData，通过相机参数与占用监督增强3D感知、Modality-Isolation-Mask 实现灵活多模态融合，首次作为通才策略在多个数据集上同时超越专家模型。

## 研究背景与动机

近年来，大模型在 NLP 和 CV 领域取得了巨大成功，研究者开始将其引入具身智能（EAI）。然而，当前方法面临两个核心挑战：

**2D到3D的鸿沟**：现有多模态模型主要聚焦2D图像理解和生成，但机器人需要与物理3D空间交互。直接将2D多模态模型应用于机器人操作并非最优解，缺乏对3D空间的深度理解严重限制了操作精度。

**数据集构建成本高昂**：收集机器人操作数据极其耗时（如RT-1收集13万episodes用了17个月）。虽然 Open X-Embodiment 等项目整合了多数据集，但缺少多视角图像、相机参数和深度图等关键3D信息，且不同数据集间缺乏空间对齐，导致直接融合训练反而降低性能（RT-1-X 表现不如 RT-1）。

因此，本文从**模型架构**和**数据标准化**两方面同时入手，解决异构数据有效利用的核心问题。

## 方法详解

### 整体框架

RoboTron-Mani 基于 OpenFlamingo 架构，由四个核心组件构成：Vision Encoder → 3D Perception Adapter → Feature Fusion Decoder → Multimodal Decoders。模型接收文本指令 $T$、多视角多帧图像 $I \in \mathbb{R}^{H \times N \times H \times W \times 3}$ 和相机参数 $Cam$ 作为输入，输出动作 $O_A$、图像 $O_I$ 和占用图 $O_O$：

$$(O_A, [O_I, O_O]) = \text{RoboTron-Mani}(T, I, Cam)$$

### 关键设计

1. **3D Perception Adapter (UVFormer)**：该模块是实现3D环境感知的核心。它接收多视角图像特征 $X^h = \{F_I^{h,n}\}_n^N$、相机参数 $Cam^h$ 和可学习的统一视角查询 $Q = \{Pos, Emb\}$ 作为输入，通过交叉注意力生成统一视角表示 $U_I^h$：

$$U_I^h = \text{UVFormer}(Q, X^h, Cam^h)$$

其中 $Pos \in \mathbb{R}^{L \times B \times 3P}$ 定义操作空间内3D网格的位置，$Emb \in \mathbb{R}^{L \times B \times C}$ 是可学习特征。这种设计的精妙之处在于：通过相机参数将多视角2D特征投影到统一的3D空间中，使模型获得真正的空间感知能力。即使相机参数不同，3D场景表示保持一致，这为跨数据集训练奠定了基础。

2. **Modality-Isolation-Mask (MIM)**：为了灵活引入多模态监督，作者设计了即插即用的 MIM 机制。具体地，先构建包含文本和多模态读出token的序列 $T'$，其中包括静态图像token $T_{simg}$、腕部图像token $T_{gimg}$、占用token $T_{occ}$ 和动作token $T_{act}$。MIM 通过 KQ mask 控制不同模态间的注意力交互：暗色方块允许注意力连接，白色方块禁止注意力传播。这使得训练时可以使用辅助模态监督提升性能，而推理时可以省略不必要的模态输出，极大增强了模型灵活性。

3. **Multimodal Decoders**：包含三种解码器：

    - **Image Decoder**：2层注意力解码层，输出图像patches并按坐标拼装为完整图像（静态图或腕部图）
    - **Occupancy Decoder**：先生成特征 $U_{occ}^h$，再经reshape、上采样和3D卷积重建完整3D占用 $O_o^h = \{o_{pos}^h, o_{rgb}^h\}$
    - **Action Decoder**：使用MLP或DiT输出delta 6D位姿 $a_{pose}^h$ 和1-DoF夹持器动作 $a_g^h$

### 损失函数 / 训练策略

综合损失函数设计如下：

$$l = l_a + \lambda_{\text{image}}(l_{simg} + l_{gimg}) + \lambda_{\text{occ}} l_o$$

- **动作损失** $l_a$：位姿MSE + 夹持器BCE，$l_a = \sum_h (\text{MSE}(a_{pose}^h, \hat{a}_{pose}^h) + \lambda_g \text{BCE}(a_g^h, \hat{a}_g^h))$
- **图像损失**：预测图像与下一帧的L2距离
- **占用损失** $l_o$：空间位置MSE + RGB颜色MSE

**RoboData 数据对齐**：整合了9个仿真数据集和RT-1真实数据，花费数百人天完成以下对齐：
- **3D空间对齐**：统一坐标系（X右、Y前、Z上），工作空间限制在[-0.5,-0.5,0]到[0.5,0.5,1]
- **动作表示对齐**：全部使用Composite Rotation Matrix Method (CRMM)重新生成动作
- **缺失数据补全**：重建原始仿真并重新渲染以获取缺失的相机内外参

训练使用32块80G A100 GPU，40亿参数bf16精度，210万样本训练10个epoch，约50小时。

## 实验关键数据

### 主实验

| 数据集 | 指标(SR) | RoboTron-Mani | 之前SOTA | 提升 |
|--------|---------|---------------|---------|------|
| LIBERO | 成功率 | **91.7%** | QueST 89.8% | +1.9% |
| RoboCasa | 成功率 | **47.4%** | GR00T-N1 40.9% | +6.5% |
| CALVIN (Avg Len) | 序列长度 | **3.51** | MDT 93.7% (单任务) | 1.7→3.5 |
| Meta-World | 成功率 | **80.1%** | PRISE 80.4% | 可比 |
| RT-1 | 成功率 | **60.0%** | RT-2-X(55B) 60.7% | 可比 |

关键发现：RoboTron-Mani 是**唯一**在多个数据集上同时评估的通才策略，且表现接近甚至超越各数据集的专家模型。相比同为通才策略的 RoboFlamingo*，LIBERO 提升19.6%，Meta-World 提升14.8%。

### 消融实验

| 配置 | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Avg Len |
|------|--------|--------|--------|--------|--------|---------|
| Baseline | 81.0% | 48.1% | 25.7% | 14.5% | 8.6% | 1.77 |
| +FFA | 85.0% | 63.3% | 42.0% | 28.7% | 18.8% | 2.37 |
| +FFA+Image | 88.5% | 74.7% | 60.7% | 49.1% | 39.6% | 3.13 |
| +UVFormer | 94.2% | 74.7% | 55.1% | 38.3% | 25.8% | 2.88 |
| +FFA+Image+UVFormer | 94.1% | 78.9% | 63.7% | 48.0% | 36.4% | 3.21 |
| All (MLP head) | 94.7% | 80.3% | 65.1% | 51.4% | 39.0% | 3.31 |
| All (DiT head) | **96.9%** | **83.0%** | **68.1%** | **56.5%** | **46.8%** | **3.51** |

### 关键发现

- UVFormer（3D感知）对首任务提升最显著（81%→94.2%），说明3D空间理解对操作至关重要
- 图像辅助输出虽质量不理想，但仍显著提升动作性能，说明多模态监督的价值
- 数据对齐至关重要：未对齐的 RoboTron-Mani- 在 LIBERO 上仅64.2%，对齐后达90.7%
- DiT动作头比MLP动作头带来额外0.2的平均序列长度提升

## 亮点与洞察

- **首个跨数据集通才策略超越专家模型**：打破了之前"通才不如专家"的固有认知
- **3D感知是关键**：通过UVFormer统一多视角特征到3D空间，实现了相机参数不变的操作表示
- **MIM设计精巧**：训练时利用多模态监督提升性能，推理时可灵活裁剪，兼顾效果与效率
- **数据标准化贡献大**：RoboData的空间对齐工作耗费数百人天，但为跨平台联合训练提供了坚实基础

## 局限与展望

- 辅助模态（图像/占用）的生成质量仍有较大提升空间
- 训练成本较高（32×A100，50小时），实际部署需要模型压缩
- RT-1数据集缺少深度和腕部相机数据，限制了完整3D感知的发挥
- 仅在仿真中完成初步验证，真实场景的泛化效果有待进一步验证

## 相关工作与启发

- 与 Open X-Embodiment 相比，RoboData 补充了关键3D信息并实现空间对齐
- MIM 思路可推广到其他多模态学习场景
- 数据对齐的成功证明：异构数据有效融合的关键不在于数据量，而在于数据质量和一致性

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个跨数据集通才策略超越专家，MIM设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 5个数据集全面评估，消融实验充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但数据集部分略显冗长
- 价值: ⭐⭐⭐⭐⭐ 为具身智能领域的数据标准化和跨平台联合训练开辟了新方向
# RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation

**会议**: ICCV 2025  
**arXiv**: [2412.07215](https://arxiv.org/abs/2412.07215)  
**代码**: [GitHub](https://github.com/EmbodiedAI-RoboTron/RoboTron-Mani)  
**领域**: 3D视觉  
**关键词**: 机器人操作, 多模态大模型, 3D感知, 跨embodiment泛化, 数据对齐

## 一句话总结

提出多模态机器人操作大模型 RoboTron-Mani 和综合数据集 RoboData，通过 3D 感知增强（UVFormer + 占据监督）与模态隔离掩码（MIM）实现多数据集联合训练，首次作为通才策略在多个数据集上同时超越专家模型。

## 研究背景与动机

当前将大模型应用于机器人操作领域面临两大核心挑战：

1. **2D 到 3D 的鸿沟**：现有多模态大模型（如 LLaVA、Flamingo）主要聚焦于 2D 图像理解，但机器人需要与物理 3D 空间交互。直接将 2D 多模态模型用于具身智能并非最优解——机器人需要理解空间深度、遮挡关系和物体的 3D 几何信息才能精确操作。

2. **数据收集成本高昂**：RT-1 收集约 13 万个 episode 就花了 17 个月。而现有跨平台数据集（如 Open X-Embodiment）虽然整合了多个数据集，但缺少关键的 3D 信息（多视角图像、相机参数、深度图），且不同数据集的坐标系和动作空间不统一——直接融合反而导致性能下降（如 RT-1-X 弱于 RT-1）。

这两个问题相互关联：要让通才模型在异构数据上有效学习，既需要统一的 3D 输入表示（消除不同相机参数带来的 2D 特征差异），也需要对齐的输出空间（统一不同机器人的动作表示）。

## 方法详解

### 整体框架

RoboTron-Mani 基于 OpenFlamingo 架构，接收多视角图像 $I$、文本指令 $T$ 和相机参数 $Cam$ 作为输入，输出动作 $O_A$，以及可选的图像 $O_I$ 和占据图 $O_O$。整体由四个核心组件串联：Vision Encoder → 3D Perception Adapter → Feature Fusion Decoder → Multimodal Decoders。

### 关键设计

1. **3D Perception Adapter (UVFormer)**：解决多视角特征统一和 3D 空间感知问题。利用 UVFormer 将 $H$ 个时间步、$N$ 个视角的图像特征 $X^h$ 和对应的相机参数 $Cam^h$ 转换为统一视图表示：

$$U_I^h = \text{UVFormer}(Q, X^h, Cam^h)$$

其中 $Q = \{Pos, Emb\}$ 是可学习的查询，$Pos \in \mathbb{R}^{L \times B \times 3P}$ 定义了机器人操作空间内 3D 网格的位置。这一设计的关键优势在于：无论相机参数如何变化，同一 3D 场景的统一视图表示 $U_I^h$ 保持一致，从而实现输入空间对齐。

2. **模态隔离掩码 (Modality-Isolation-Mask, MIM)**：在 Feature Fusion Decoder 的自注意力层中引入 KQ 掩码，控制不同模态 token（文本、图像、动作、占据）之间的注意力连接。深色方块表示允许注意力交互，白色方块禁止。MIM 的核心价值是实现灵活的模态融合——训练时可使用辅助模态监督（图像重建、占据预测），推理时可按需省略不必要的模态输出，显著提升了模态组合的灵活性。

3. **多模态解码器**：设计三种不同的解码器以适配不同模态输出：
    - **图像解码器**：2 层注意力解码器，输出图像 patch 后拼装为完整图像（静态视图或手腕视图）
    - **占据解码器**：先生成特征 $U_{occ}^h$，再 reshape + 上采样 + 3D 卷积重建完整 3D 占据 $O_o^h = \{o_{pos}^h, o_{rgb}^h\}$
    - **动作解码器**：MLP 或 DiT 块输出 delta 6D 位姿 $a_{pose}^h$ 和 1-DoF 夹爪动作 $a_g^h$

4. **RoboData 数据对齐**：整合 CALVIN、Meta-World、LIBERO、RT-1 等 10 个数据集，进行三维度对齐：
    - **3D 空间对齐**：统一世界坐标系为 X→右、Y→前、Z→上，工作空间限制在 $[-0.5, -0.5, 0]$ 到 $[0.5, 0.5, 1]$
    - **动作表示对齐**：统一使用复合旋转矩阵方法 (CRMM) 重新生成动作
    - **缺失数据补全**：重建原始仿真环境，补充缺失的相机内外参

### 损失函数 / 训练策略

综合损失函数：

$$l = l_a + \lambda_{\text{image}}(l_{simg} + l_{gimg}) + \lambda_{\text{occ}} l_o$$

- **动作损失** $l_a$：位姿用 MSE，夹爪用 BCE
- **图像损失** $l_{simg}, l_{gimg}$：预测下一帧与真值的 L2 损失
- **占据损失** $l_o$：位置 MSE + RGB MSE（$\lambda_{rgb}$ 调节权重）

训练细节：4B 参数（bf16），32×A100 训练约 50 小时，2.1M 样本，10 epochs。

## 实验关键数据

### 主实验

| 数据集 | 指标 | RoboTron-Mani | 之前 SOTA | 提升 |
|--------|------|---------------|-----------|------|
| LIBERO | 成功率 | **91.7%** | QueST 89.8% | +1.9% |
| RoboCasa | 成功率 | **47.4%** | GR00T-N1 40.9% | +6.5% |
| CALVIN | Avg Len | **3.51** | MDT 93.7%(Task1) | 竞争力 |
| Meta-World | 成功率 | **80.1%** | PRISE 80.4% | 持平 |
| RT-1 | 成功率 | **60.0%** | RT-2-X(55B) 60.7% | 持平(参数量远小) |

注：RoboTron-Mani 是唯一在所有 5 个数据集上同时评估的通才策略，其余均为针对单一数据集优化的专家模型。

### 消融实验

| 配置 | Task1 | Task2 | Task3 | Task4 | Task5 | Avg Len | 说明 |
|------|-------|-------|-------|-------|-------|---------|------|
| Baseline | 81.0% | 48.1% | 25.7% | 14.5% | 8.6% | 1.77 | 仅最后帧输出动作 |
| +FFA | 85.0% | 63.3% | 42.0% | 28.7% | 18.8% | 2.37 | 逐帧动作输出 |
| +FFA+Image | 88.5% | 74.7% | 60.7% | 49.1% | 39.6% | 3.13 | 加图像预测 |
| +FFA+UVFormer | 94.2% | 74.7% | 55.1% | 38.3% | 25.8% | 2.88 | 3D感知 |
| +All(MLP) | 94.7% | 80.3% | 65.1% | 51.4% | 39.0% | 3.31 | 完整框架 |
| +All(DiT) | **96.9%** | **83.0%** | **68.1%** | **56.5%** | **46.8%** | **3.51** | DiT动作头 |

### 关键发现

- UVFormer 对首个任务提升显著（81% → 94.2%），说明 3D 感知对任务启动至关重要
- 即使生成的图像和占据质量不理想，辅助模态监督仍能显著提升动作性能
- 数据对齐是跨数据集训练的关键：未对齐时 LIBERO 仅 64.2%，对齐后 90.7%
- DiT 动作头相比 MLP 在长序列任务上优势明显（Avg Len: 3.31 → 3.51）

## 亮点与洞察

- **首次实现通才策略全面超越专家模型**：在 5 个异构数据集上联合训练和评估，打破了"通才不如专家"的常规认知
- **3D 感知是跨 embodiment 泛化的关键**：同一 3D 场景在不同相机参数下的 2D 特征不同，但 UVFormer 的 3D 特征保持一致
- **MIM 设计巧妙**：允许训练时用辅助模态监督增强学习，推理时灵活裁剪，是一种免费的性能提升手段
- **数据工程的深度投入**：花费数百人天对齐数据，补充缺失模态，这种工程投入被充分验证有价值

## 局限与展望

- 当前仅在仿真环境中验证数据对齐方案，真实世界的异构数据对齐更加困难
- 4B 参数模型的训练成本仍然较高（32×A100，50小时）
- 辅助模态生成质量较低，若能提升生成质量可能带来更大性能增益
- 未探索实时在线学习和自适应能力

## 相关工作与启发

- 与 Open X-Embodiment 的对比说明，简单数据融合不如精细的空间对齐 + 架构设计
- OpenFlamingo 的交叉注意力机制自然适配多帧/视频输入，优于 LLaVA 的自回归机制
- RoboData 的对齐方法论可推广到更多机器人数据集的统一

## 评分

- **新颖性**: ⭐⭐⭐⭐ 3D 感知 + 模态隔离掩码的设计新颖，数据对齐方案系统性强
- **实验充分度**: ⭐⭐⭐⭐⭐ 5 个数据集全面评估，消融实验详尽
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，公式规范，但部分章节较冗长
- **价值**: ⭐⭐⭐⭐⭐ 为跨 embodiment 机器人学习提供了完整的数据 + 模型解决方案

<!-- RELATED:START -->

## 相关论文

- [Matrix3D: Large Photogrammetry Model All-in-One](../../CVPR2025/3d_vision/matrix3d_large_photogrammetry_model_all-in-one.md)
- [RoboPearls: Editable Video Simulation for Robot Manipulation](robopearls_editable_video_simulation_for_robot_manipulation.md)
- [RayZer: A Self-supervised Large View Synthesis Model](rayzer_a_self-supervised_large_view_synthesis_model.md)
- [One Diffusion to Generate Them All](../../CVPR2025/3d_vision/one_diffusion_to_generate_them_all.md)
- [Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh](../../CVPR2025/3d_vision/mani-gs_gaussian_splatting_manipulation_with_triangular_mesh.md)

<!-- RELATED:END -->
