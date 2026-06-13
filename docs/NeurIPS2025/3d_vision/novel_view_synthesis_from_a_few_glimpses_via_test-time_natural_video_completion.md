---
title: >-
  [论文解读] Novel View Synthesis from A Few Glimpses via Test-Time Natural Video Completion
description: >-
  [NeurIPS 2025][3D视觉][新视角合成] 将稀疏输入新视角合成重新定义为测试时自然视频补全问题，利用预训练视频扩散模型的先验生成中间伪视图，并通过不确定性感知机制与 3D 高斯泼溅（3D-GS）迭代优化，在极稀疏输入下实现高保真场景重建。 稀疏输入的新视角合成（Sparse-input Novel View S…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "新视角合成"
  - "视频扩散模型"
  - "3D 高斯泼溅"
  - "稀疏输入"
  - "测试时推理"
---

# Novel View Synthesis from A Few Glimpses via Test-Time Natural Video Completion

**会议**: NeurIPS 2025  
**arXiv**: [2511.17932](https://arxiv.org/abs/2511.17932)  
**作者**: Yan Xu, Yixing Wang, Stella X. Yu
**代码**: 无  
**领域**: 3D 视觉 / 新视角合成  
**关键词**: 新视角合成, 视频扩散模型, 3D 高斯泼溅, 稀疏输入, 测试时推理

## 一句话总结

将稀疏输入新视角合成重新定义为测试时自然视频补全问题，利用预训练视频扩散模型的先验生成中间伪视图，并通过不确定性感知机制与 3D 高斯泼溅（3D-GS）迭代优化，在极稀疏输入下实现高保真场景重建。

## 研究背景与动机

稀疏输入的新视角合成（Sparse-input Novel View Synthesis）是 3D 视觉的核心挑战：仅给定少量输入视角（如 3-5 张图像），需要生成任意新视角的渲染结果。

**现有方法的局限**：
- 基于 NeRF/3D-GS 的方法在输入极度稀疏时严重退化，因为欠约束区域缺乏监督
- 基于扩散模型的方法（如 Zero-1-to-3）通常需要场景特定的微调或仅支持单物体
- 基于几何的方法（如 MVS）在宽基线下容易失败

**本文的洞察**：将"稀疏视角的空间插值"重新想象为"相机在场景中滑行时生成的自然视频的补全"。这样就可以直接利用预训练视频扩散模型对自然视频运动的强大先验知识。

## 方法详解

### 整体框架

本文提出一个 **零样本、生成引导** 的框架，包含三个核心组件的迭代循环：

1. **视频补全模块**：利用预训练视频扩散模型，在给定的稀疏关键帧之间生成中间视图
2. **不确定性感知筛选**：通过多次采样估计生成视图的不确定性，过滤低质量的伪视图
3. **3D-GS 重建模块**：利用筛选后的伪视图与原始输入共同训练 3D 高斯泼溅

### 关键设计

**测试时视频补全**：
- 将稀疏输入图像视为视频关键帧，沿预设的相机轨迹排列
- 使用视频扩散模型（如 Stable Video Diffusion）对关键帧之间的"缺失帧"进行生成
- 通过条件采样确保生成帧与已知关键帧的一致性

**不确定性感知机制**：
- 对同一位置进行 $K$ 次独立采样，得到 $K$ 个候选视图
- 计算像素级方差作为不确定性指标：$U(x) = \text{Var}_{k=1}^K [I_k(x)]$
- 高不确定性区域在 3D-GS 训练时降低权重，避免错误伪视图误导重建

**迭代反馈循环**：
- 第一轮：仅用稀疏输入训练初始 3D-GS
- 后续轮次：用当前 3D-GS 渲染中间视角 → 与视频补全结果融合 → 更新 3D-GS
- 3D 几何约束与 2D 生成先验相互促进，逐步提升质量

### 损失函数 / 训练策略

3D-GS 的训练损失为加权重建损失：

$$\mathcal{L} = \sum_{i \in \text{real}} \mathcal{L}_1(I_i, \hat{I}_i) + \lambda_{\text{SSIM}} \mathcal{L}_{\text{SSIM}}(I_i, \hat{I}_i) + \sum_{j \in \text{pseudo}} w_j \cdot \mathcal{L}_1(I_j, \hat{I}_j)$$

其中伪视图的权重 $w_j$ 与其不确定性成反比：$w_j = \exp(-\beta \cdot U_j)$

## 实验关键数据

### 主实验

**LLFF 数据集（3 输入视角）：**

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| 3D-GS | 15.82 | 0.412 | 0.498 |
| DNGaussian | 18.95 | 0.571 | 0.342 |
| FSGS | 19.34 | 0.589 | 0.328 |
| ReconFusion | 20.12 | 0.623 | 0.285 |
| **本文方法** | **21.87** | **0.672** | **0.241** |

**DTU 数据集（3 输入视角）：**

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| 3D-GS | 12.45 | 0.521 | 0.412 |
| SparseNeRF | 16.82 | 0.645 | 0.335 |
| FSGS | 17.91 | 0.672 | 0.298 |
| **本文方法** | **19.56** | **0.718** | **0.252** |

### 消融实验

| 组件 | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| 完整模型 | 21.87 | 0.672 | 0.241 |
| 去掉不确定性感知 | 20.43 | 0.638 | 0.278 |
| 去掉迭代反馈 | 20.89 | 0.651 | 0.262 |
| 仅用单次补全 | 19.75 | 0.612 | 0.301 |
| 随机伪视图权重 | 20.15 | 0.625 | 0.289 |

### 关键发现

1. **不确定性感知是关键**：移除后 PSNR 下降 1.44dB，说明伪视图质量参差不齐，需要可靠的筛选机制
2. **迭代反馈逐轮提升**：典型情况下 3 轮迭代即可收敛，继续迭代收益递减
3. **在更大场景（MipNeRF-360、DL3DV）上同样有效**：相比基线提升 2-3dB
4. **零样本能力**：无需对任何目标场景进行训练或微调

## 亮点与洞察

- **将 NVS 问题巧妙转化为视频补全**：利用视频扩散模型的场景运动先验，绕过了传统几何方法在极稀疏下的困境
- **不确定性感知的伪标签策略**：可推广到其他需要利用生成模型输出作为监督的任务
- **零样本泛化**：不依赖数据集特定的训练，在多个 benchmark 上一致优于专用方法
- **3D 与 2D 的闭环协同**：迭代反馈让重建和生成相互改进，是一种优雅的系统设计

## 局限与展望

- 推理速度较慢：视频扩散模型的多次采样带来显著计算开销
- 对相机轨迹的预设有依赖，不同的轨迹选择可能影响结果
- 视频扩散模型对大尺度场景变化（如室外远景）的先验可能不够strong
- 伪视图的几何一致性仍有待加强，可考虑引入多视图一致性约束

## 相关工作与启发

- **视频扩散模型**：Stable Video Diffusion、Sora 等
- **稀疏视角 3D 重建**：DNGaussian、FSGS、SparseNeRF、ReconFusion
- **生成引导的 3D**：DreamFusion、Score Jacobian Chaining

本文展示了大规模视频生成模型如何赋能 3D 视觉任务，为 foundation model 在 3D 领域的落地提供了新范式。

## 评分

| 维度 | 分数 (1-10) |
|------|------------|
| 创新性 | 8 |
| 理论深度 | 5 |
| 实验充分性 | 8 |
| 写作质量 | 8 |
| 实用价值 | 7 |
| 总体推荐 | 7.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Reconstruct, Inpaint, Test-Time Finetune: Dynamic Novel-View Synthesis from Monocular Videos](reconstruct_inpaint_test-time_finetune_dynamic_novel-view_synthesis_from_monocul.md)
- [\[NeurIPS 2025\] PointMAC: Meta-Learned Adaptation for Robust Test-Time Point Cloud Completion](pointmac_meta-learned_adaptation_for_robust_test-time_point_cloud_completion.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](../../ICCV2025/3d_vision/self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)
- [\[NeurIPS 2025\] NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods](nerfbaselines_consistent_and_reproducible_evaluation_of_novel_view_synthesis_met.md)
- [\[NeurIPS 2025\] HyRF: Hybrid Radiance Fields for Memory-efficient and High-quality Novel View Synthesis](hyrf_hybrid_radiance_fields_for_memory-efficient_and_high-quality_novel_view_syn.md)

</div>

<!-- RELATED:END -->
