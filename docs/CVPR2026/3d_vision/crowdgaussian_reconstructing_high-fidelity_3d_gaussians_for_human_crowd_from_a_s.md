---
title: >-
  [论文解读] CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image
description: >-
  [CVPR 2026][3D视觉][人体重建] CrowdGaussian 提出了从单张图像重建多人 3D 高斯泼溅表示的统一框架，通过自监督适配的大型遮挡人体重建模型（LORM）恢复被遮挡区域的完整几何，再通过自校准学习（SCL）训练的单步扩散精炼器（CrowdRefiner）提升纹理细节质量。
tags:
  - CVPR 2026
  - 3D视觉
  - 人体重建
  - 3D高斯泼溅
  - 遮挡恢复
  - 扩散模型精炼
  - 人群场景
---

# CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image

**会议**: CVPR 2026  
**arXiv**: [2603.17779](https://arxiv.org/abs/2603.17779)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 人体重建, 3D高斯泼溅, 遮挡恢复, 扩散模型精炼, 人群场景

## 一句话总结

CrowdGaussian 提出了从单张图像重建多人 3D 高斯泼溅表示的统一框架，通过自监督适配的大型遮挡人体重建模型（LORM）恢复被遮挡区域的完整几何，再通过自校准学习（SCL）训练的单步扩散精炼器（CrowdRefiner）提升纹理细节质量。

## 研究背景与动机

**领域现状**：单图 3D 人体重建近年取得显著进展，大型重建模型（LRM）利用 Transformer 和大规模数据集实现了从单图快速前馈重建。但现有方法几乎都只处理清晰、近距离的单人图像。

**现有痛点**：
   - **严重遮挡**：人群场景中人与人、人与物之间频繁遮挡，导致身体部位不完整。现有方法直接处理这类输入会产生透明空洞和不完整几何。
   - **低分辨率**：人群中每个个体的裁剪图分辨率很低，导致外观模糊、缺乏高频细节。
   - **多人场景的效率需求**：需要同时重建大量人体，逐个处理效率太低。

**核心矛盾**：现有的大型人体重建模型虽然有强大的 2D-to-3D 生成先验，但缺乏遮挡感知训练。直接送入遮挡输入，Transformer 无法整合不完整的视觉特征，导致输出破碎。而用有限的 3D 监督做微调往往放大单目歧义的几何偏差，反而损害预训练先验。

**本文目标**：(a) 如何从被严重遮挡的裁剪图恢复完整 3D 人体？(b) 如何从低分辨率输入恢复高频纹理细节？(c) 如何高效地同时处理多人场景？

**切入角度**：不做 3D 标注监督微调，而是用"自监督蒸馏"——让冻结的教师模型在完整图像上生成伪 GT，学生模型学习从遮挡输入恢复完整几何。

**核心idea**：两阶段框架——Stage 1 用自监督适配的 LORM 生成粗糙但完整的多人 3DGS，Stage 2 用 SCL 训练的 CrowdRefiner 精炼渲染结果并回蒸到 3DGS 中。

## 方法详解

### 整体框架

输入为包含 $N$ 个人的单张图像。Stage 1：用多人 HMR 估计每人的 SMPL-X 参数和3D位置 → SAM 分割每个人 → LORM 从遮挡裁剪图恢复完整 3DGS → 组装初始粗糙多人 3DGS 场景。Stage 2：渲染粗糙场景 → CrowdRefiner 精炼 → 精炼结果作为伪 GT 通过可微渲染蒸馏回 3DGS。

### 关键设计

1. **LORM (Large Occluded Human Reconstruction Model)**

    - 功能：从被遮挡的单人裁剪图恢复完整的 3D 高斯表示，包括几何和纹理
    - 核心思路：基于预训练的大型人体重建模型（LHM-500M），该模型使用 Sapiens 编码器（MAE 架构）+ 多模态身体-头部 Transformer (MBHT) + 高斯解码器。为了适配遮挡输入，**冻结** Sapiens 编码器和高斯解码器，仅在 MBHT Transformer 中注入**可训练的 LoRA 模块**。
    - **自监督适配框架**：
        - 教师流：冻结的预训练模型处理完整图像 $I_{\text{full}}$，生成完整 3D 高斯 $\mathcal{G}_{\text{full}}$，从 $V$ 个新视角渲染得到干净伪 GT $R_{\text{clean}}^{(v)}$
        - 学生流：对 $I_{\text{full}}$ 施加随机遮挡 mask（Bézier 曲线 + 关键点椭圆）得到 $I_{\text{occ}}$，LORM 预测 3DGS 渲染得到粗糙视图 $R_{\text{coarse}}^{(v)}$
        - 自蒸馏损失：$\mathcal{L}_{\text{self-distill}} = \sum_v (\lambda_{\text{rgb}} \|R_{\text{clean}}^{(v)} - R_{\text{coarse}}^{(v)}\|_2 + \lambda_{\text{ssim}} (1 - \text{SSIM}(R_{\text{clean}}^{(v)}, R_{\text{coarse}}^{(v)})))$
    - 设计动机：避免使用外部 3D 标注（会放大单目歧义），利用预训练模型本身作为教师，仅通过 2D 一致性来教会学生处理遮挡。LoRA 仅微调 Transformer 的注意力权重，保护了预训练的视觉特征提取和高斯生成能力。

2. **CrowdRefiner (单步扩散精炼器)**

    - 功能：将粗糙的多人 3DGS 渲染结果精炼为高保真图像，作为伪 GT 用于 3DGS 优化
    - 核心思路：基于 SD-Turbo 的单步扩散模型。输入为粗糙 RGB 渲染 $R_{\text{coarse}}$ 和对应的 SMPL 法线图 $N$（几何先验）。法线图通过轻量级 PoseNet 编码，RGB 通过冻结的 VAE 编码器编码，两路特征注入 UNet 引导生成。VAE 解码器使用 LoRA 适配微调。
    - 设计动机：直接用 LORM 输出虽然几何完整但纹理过度平滑（受限于重建模型分辨率）。扩散模型可以利用 2D 生成先验增补高频细节。选择单步推理（而非迭代采样）确保效率。

3. **自校准学习 (Self-Calibrated Learning, SCL)**

    - 功能：训练 CrowdRefiner 时防止过度精炼导致的面部扭曲和伪影
    - 核心思路：在训练中随机混合两类样本对——(a) 标准退化对 $(R_{\text{coarse}}, R_{\text{gt}})$：从粗糙恢复到高质量；(b) 身份保持对 $(R_{\text{gt}}, R_{\text{gt}})$：输入和目标都是 GT，教模型"不要改动已经好的区域"。
    - 设计动机：如果只用退化对训练，模型会过于激进地"增强"所有区域，导致原本重建良好的区域（如面部）产生扭曲。混合身份保持样本让模型学会**自适应地**判断哪些区域需要增强、哪些需要保留。

4. **多人场景的聚类精炼策略**

    - 功能：避免逐人精炼的高计算成本
    - 核心思路：使用 DBSCAN 根据根位置将个体聚类为空间连贯的组。对每个组同时渲染和精炼，而非逐人处理。精炼结果通过 L1 + SSIM 损失蒸馏回 3DGS。
    - 设计动机：人群场景中人数众多，逐个处理效率太低。聚类策略保证空间邻近的人被一起处理，同时保持全局场景一致性。

### 损失函数 / 训练策略

- **LORM 自蒸馏损失**: $\mathcal{L}_{\text{self-distill}} = \sum_v (\lambda_{\text{rgb}} \| \cdot \|_2 + \lambda_{\text{ssim}} (1 - \text{SSIM}))$，从 24 个固定视角渲染
- **CrowdRefiner 训练损失**: $\mathcal{L}_{\text{diff}} = \lambda_{L2}\mathcal{L}_{\text{L2}} + \lambda_{\text{lpips}}\mathcal{L}_{\text{LPIPS}} + \lambda_{\text{ssim}}\mathcal{L}_{\text{SSIM}} + \lambda_{\text{gram}}\mathcal{L}_{\text{Gram}}$
- **3DGS 优化损失**: $\mathcal{L}_{\text{optim}} = \|R_{\text{refined}} - R_{\text{coarse}}\|_1 + \lambda_{\text{ssim}}(1 - \text{SSIM})$
- LORM 训练数据：HuGe100K 中的 1002 张正面图像
- CrowdRefiner 训练数据：THuman2.1 的 114 个合成多人场景（91 训练/23 测试），每场景 126 个视角

## 实验关键数据

### 主实验

遮挡人体重建的定量比较（THuman2.1，随机遮挡 mask）：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| IDOL | 18.063 | 0.919 | 0.994 |
| LHM | 18.171 | 0.918 | 1.012 |
| LORM (Ours) | 18.566 | 0.923 | 0.956 |
| LORM + CrowdRefiner | **18.619** | **0.931** | **0.914** |

不同遮挡率下的鲁棒性（THuman2.1）：

| 方法 | 遮挡率 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|--------|---------|
| IDOL | 20% | 18.196 | 0.921 | 0.978 |
| IDOL | 60% | 16.667 | 0.909 | 1.063 |
| LHM | 20% | 17.945 | 0.919 | 1.006 |
| LHM | 60% | 17.551 | 0.915 | 1.037 |
| **LORM** | **20%** | **18.428** | **0.923** | **0.947** |
| **LORM** | **60%** | **18.116** | **0.919** | **0.972** |

### 消融实验

CrowdRefiner 的 SCL 策略和几何条件输入消融：

| SCL | Normal Map | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-----|-----------|--------|--------|---------|
| ✗ | ✗ | 20.013 | 0.888 | 0.141 |
| ✗ | ✓ | 20.130 | 0.892 | 0.138 |
| ✓ | ✗ | 20.382 | 0.896 | 0.129 |
| ✓ | ✓ | **20.790** | **0.901** | **0.122** |

### 关键发现

- **LORM 在大遮挡率下退化极小**：遮挡从 20% 增加到 60%，LORM 的 PSNR 仅下降 0.31（18.43→18.12），而 IDOL 下降 1.53（18.20→16.67），LHM 下降 0.39。LORM 的自监督适配有效地注入了遮挡处理能力。
- **SCL 是防止过度精炼的关键**：没有 SCL 时 PSNR 下降 0.77（20.79→20.01），定性上出现面部扭曲。SCL 中身份保持样本教会模型不要过度修改。
- **法线图条件可提升几何一致性**：增加 SMPL 法线图输入后 LPIPS 从 0.129 降至 0.122，为精炼器提供了明确的几何约束。
- **Mesh-based 方法在遮挡下全面失败**：PSHuman 和 SyncHuman 无法恢复被遮挡部分的几何，而基于 3DGS 的 IDOL 和 LHM 虽能输出某些结果但有透明伪影和扭曲纹理。

## 亮点与洞察

- **自监督适配策略绝妙**：利用预训练模型本身作为教师，通过合成遮挡+自蒸馏来学习遮挡恢复，不需要任何外部 3D 标注。这个范式可以迁移到任何需要让预训练模型适应新退化类型的场景——不改变模型的生成先验，只教它处理新的输入分布。
- **SCL 策略的直觉优雅**：在训练数据中混入"输入=输出"的身份保持样本，本质上是告诉模型"如果输入已经足够好，就不要改动"。这个简单的 trick 有效解决了生成式精炼中的过度修改问题。
- **从单人模型到多人场景的完整路径**：LORM + CrowdRefiner + DBSCAN 聚类构成了一个完整的、可扩展的多人 3D 重建方案，展示了如何基于单人模型构建多人系统。

## 局限与展望

- 依赖 off-the-shelf 的姿态估计和分割（Multi-HMR、SAM），初始化严重错误会传播到最终结果，尤其是手部重建挑战很大
- 在极低分辨率下精炼可能产生与真实不一致的幻觉细节（如特定 logo）
- 训练数据仅使用 THuman2.1 的 114 个合成场景，多样性有限
- 需要 SMPL-X 参数估计，对非标准体型或极端服装可能不适用
- 聚类策略使用 DBSCAN，在人群密度极高时可能将太多人聚为一组导致精炼分辨率不足

## 相关工作与启发

- **vs LHM**: 本文直接基于 LHM-500M 做适配。LHM 在遮挡输入下产生透明伪影，LORM 通过 LoRA + 自蒸馏解决了这个问题，且仅用 1002 张图像就足够。
- **vs CHROME**: CHROME 使用多视角扩散生成无遮挡图像，但合成视图间的不一致导致纹理损坏（尤其面部）。LORM 直接在 3DGS 空间做恢复，避免了多视图不一致的问题。
- **vs DIFIX/GSFix3D**: 通用场景 3DGS 精炼方法。CrowdRefiner 专注于人体场景，通过 SMPL 法线条件和 SCL 策略更好地保持身份和面部细节。

## 评分

- 新颖性: ⭐⭐⭐⭐ 自监督适配和 SCL 策略有创新性，但整体框架是模块化组合
- 实验充分度: ⭐⭐⭐⭐ 定量+定性覆盖充分，遮挡率梯度实验有说服力，但缺少更大规模的真实世界基准
- 写作质量: ⭐⭐⭐⭐ 结构清晰，pipeline 图表直观
- 价值: ⭐⭐⭐⭐ 填补了多人 3D 重建的空白，对 VR/远程会议等应用有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Human Interaction-Aware 3D Reconstruction from a Single Image](human_interaction-aware_3d_reconstruction_from_a_single_image.md)
- [\[CVPR 2026\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_singleforward_gaussian_splatting_for_hi.md)
- [\[CVPR 2026\] HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars](hypergaussians_high-dimensional_gaussian_splatting_for_high-fidelity_animatable_.md)
- [\[CVPR 2026\] Catalyst4D: High-Fidelity 3D-to-4D Scene Editing via Dynamic Propagation](catalyst4d_highfidelity_3dto4d_scene_editing_via_d.md)
- [\[CVPR 2026\] TopoMesh: High-Fidelity Mesh Autoencoding via Topological Unification](topomesh_high-fidelity_mesh_autoencoding_via_topological_unification.md)

</div>

<!-- RELATED:END -->
