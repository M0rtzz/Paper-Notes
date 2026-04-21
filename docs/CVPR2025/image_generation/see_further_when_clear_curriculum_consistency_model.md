---
title: >-
  [论文解读] See Further When Clear: Curriculum Consistency Model
description: >-
  [CVPR 2025][图像生成][一致性模型] 本文提出 Curriculum Consistency Model (CCM)，发现一致性蒸馏中不同时间步的学习难度（知识差异）高度不均衡，通过基于 PSNR 的 KDC 指标动态调整教师模型的迭代步数以保持课程难度一致，在 CIFAR-10 单步 FID 达到 1.64，并成功扩展到 SDXL 和 SD3。
tags:
  - CVPR 2025
  - 图像生成
  - 一致性模型
  - 课程学习
  - 扩散蒸馏
  - 少步生成
  - Flow Matching
---

# See Further When Clear: Curriculum Consistency Model

**会议**: CVPR 2025  
**arXiv**: [2412.06295](https://arxiv.org/abs/2412.06295)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 一致性模型, 课程学习, 扩散蒸馏, 少步生成, Flow Matching

## 一句话总结

本文提出 Curriculum Consistency Model (CCM)，发现一致性蒸馏中不同时间步的学习难度（知识差异）高度不均衡，通过基于 PSNR 的 KDC 指标动态调整教师模型的迭代步数以保持课程难度一致，在 CIFAR-10 单步 FID 达到 1.64，并成功扩展到 SDXL 和 SD3。

## 研究背景与动机

- **少步生成的需求**：扩散模型和 Flow Matching 的采样效率不理想，一致性模型（CM）通过强制 ODE 轨迹上的自一致性大幅减少采样步数，但训练效率仍有提升空间。
- **知识差异不均衡问题**：在一致性蒸馏中，学生模型在不同时间步面对的学生-教师输出差异（知识差异）高度不均——在大噪声（$t \to 0$）时差异大，小噪声（$t \to 1$）时差异小。这导致模型在低噪声区学习不充分。
- **现有方法的局限**：iCT 和 ECM 通过逐渐减小蒸馏步长解决累积误差，但更小的步长反而加剧了知识差异减小的问题，使学习更低效。
- **"看得清时应看得远"**：当输入噪声小（$t$ 大）时，模型的感知更清晰但学习信号更弱。此时应让教师模型"看得更远"——通过多步迭代到更远的时间步，增加知识差异。
- **统一框架**：方法同时适用于扩散模型（如 Stable Diffusion）和 Flow Matching 模型（如 SD3），填补了 Flow Matching 一致性蒸馏研究的空白。

## 方法详解

### 整体框架

CCM 在标准一致性蒸馏的基础上引入三个关键组件：
1. **KDC 指标**：基于 PSNR 衡量当前课程的知识差异
2. **动态目标调整**：根据 KDC 自适应确定教师模型的目标时间步 $u$
3. **多步迭代生成**：教师模型通过多次小步迭代到达 $u$，保证预测质量

### 关键设计

**1. Knowledge Discrepancy of the Curriculum (KDC)**
- **功能**：以稳定、可比较的方式量化每次蒸馏迭代中学生与教师之间的学习难度
- **核心思路**：$\text{KDC}_t^u = 100 - \text{PSNR}(\boldsymbol{x}_{\text{est}}, \boldsymbol{x}_{\text{target}})$，其中 $\boldsymbol{x}_{\text{est}} = f_\theta(\boldsymbol{x}_t, t, 1)$ 为学生输出，$\boldsymbol{x}_{\text{target}} = f_{\theta^-}(\text{Solver}(\boldsymbol{x}_t, t, u; \phi), u, 1)$ 为教师输出。在 CIFAR-10、ImageNet、CC3M 不同数据集和模型上实验表明 KDC 曲线趋势一致
- **设计动机**：PSNR 直接衡量图像差异且尺度稳定，减去 100 使得大 KDC 对应大差异。实验验证 KDC 在 DM 和 FM 模型间具有跨模型一致性

**2. 动态课程目标调整**
- **功能**：保证所有时间步上的学习难度大致相等，避免"课程太简单"或"课程太难"
- **核心思路**：设定 KDC 阈值 $T_{\text{KDC}}$。在每个训练步骤中，教师模型从时间步 $t$ 开始，每次向前迭代一个小步 $s$，直到估计的 KDC 超过 $T_{\text{KDC}}$。最终的 $u$ 即为满足目标难度的时间步
- **设计动机**：在早期训练阶段，模型较弱，大时间步（$t$ 大）处 KDC 自然较大无需调整；随训练进行，模型变强，需要更大的蒸馏步长 $l = u - t$ 来维持足够的知识差异

**3. 多步迭代教师生成**
- **功能**：在蒸馏步长变大时保证教师预测的准确性
- **核心思路**：当 $u$ 远大于 $t$ 时，直接一步从 $t$ 到 $u$ 的 ODE 求解不够准确。因此教师模型从 $t$ 开始，每次迭代一小步 $s$ 直到达到 $u$，确保 $\boldsymbol{x}_u = \text{Solver}(\boldsymbol{x}_t, t, u; \phi)$ 的质量
- **设计动机**：大蒸馏步长的好处是减少累积误差和提升全局一致性，但前提是教师的轨迹预测必须准确

### 损失函数

与标准一致性蒸馏相同，但使用动态调整后的目标：
$$\mathcal{L}_{\text{CCM}} = \mathbb{E}_{t \sim \mathcal{U}(0,1)}\left[\lambda(\sigma_t) \cdot d\left(f_\theta(\boldsymbol{x}_t, t, 1), f_{\theta^-}(\boldsymbol{x}_u^{\text{KDC}}, u, 1)\right)\right]$$

其中 $u$ 由 KDC 动态确定，$d(\cdot,\cdot)$ 为距离度量。

## 实验关键数据

### 主实验：单步采样 FID

| 方法 | CIFAR-10 FID ↓ | ImageNet 64×64 FID ↓ |
|------|:---:|:---:|
| CM (Song et al.) | 2.93 | 6.20 |
| iCT (Song & Dhariwal) | 2.83 | 3.25 |
| ECM | 1.68 | 2.58 |
| **CCM (Ours)** | **1.64** | **2.18** |

### 大规模 T2I 模型扩展

| 基础模型 | 方法 | FID ↓ | CLIP Score ↑ |
|------|------|:---:|:---:|
| SDXL (DM) | LCM | 较高 | 较低 |
| SDXL (DM) | **CCM** | **更低** | **更高** |
| SD3 (FM) | LCM-adapt | 较高 | 较低 |
| SD3 (FM) | **CCM** | **更低** | **更高** |

### 关键发现

- KDC 从 $t=0$（约 60）到 $t=1$（约 35）逐渐下降，证实了知识差异不均衡的存在
- 减小蒸馏步长 $l$ 会进一步降低 KDC，使 iCT/ECM 的问题更严重
- CCM 的自适应 $l$ 随训练进行逐渐增大（与 iCT 相反），在后期使用更大步长
- 在大规模 T2I 模型上，CCM 显著改善图文对齐和语义结构质量
- CCM 是首个系统研究 Flow Matching (SD3) 一致性蒸馏的工作

## 亮点与洞察

1. **"看得清时应看得远"**：精辟的直觉——在噪声低的区域模型能力强但学习信号弱，应扩大蒸馏范围以保持学习效率
2. **KDC 指标的通用性**：基于 PSNR 的知识差异度量在不同模型（DM/FM）、数据集和分辨率上表现一致
3. **与 iCT 的对比洞察**：iCT 逐步减小步长以减少误差，CCM 逐步增大步长以维持难度——两种互补的策略
4. **DM + FM 统一框架**：首次将一致性蒸馏同时应用于扩散模型和 Flow Matching 模型

## 局限与展望

- 多步迭代教师生成增加了训练时间成本
- KDC 阈值 $T_{\text{KDC}}$ 需要手动设定，不同场景可能需要不同值
- 当前实验主要在单步采样上验证，多步采样（2-4步）的效果有待更全面研究
- 未来可结合自适应步长调度与更先进的 ODE solver 进一步提升质量

## 相关工作与启发

- **Consistency Models (CM)**：开创性地通过自一致性实现少步生成
- **iCT/ECM**：通过减小蒸馏步长解决累积误差，但加剧了训练不均衡
- **LCM**：将一致性蒸馏拓展到潜在空间和文本条件图像合成
- **PCM 和 SCott**：分段轨迹和噪声控制的改进方案
- 启发：训练效率的问题往往不在于模型架构本身，而在于训练信号的质量和均衡性——课程学习思想在加速学习中有广泛适用性

## 评分

⭐⭐⭐⭐ — 问题分析深入，"知识差异不均衡"的发现有洞察力且有实验支撑。KDC 指标简洁有效，CCM 在 CIFAR-10/ImageNet 上达到 SOTA 单步 FID。成功扩展到 SDXL/SD3 展示了方法的通用性。主要代价是多步教师迭代的训练开销。

<!-- RELATED:START -->

## 相关论文

- [Riemannian Consistency Model](../../NeurIPS2025/image_generation/riemannian_consistency_model.md)
- [PCM: Picard Consistency Model for Fast Parallel Sampling of Diffusion Models](pcm_picard_consistency_model_for_fast_parallel_sampling_of_diffusion_models.md)
- [Learning to See in the Extremely Dark](../../ICCV2025/image_generation/learning_to_see_in_the_extremely_dark.md)
- [DDIS: When Model Knowledge Meets Diffusion Model](../../ICML2025/image_generation/when_model_knowledge_meets_diffusion_model_diffusion-assisted_data-free_image_synthesis.md)
- [Zero-Shot Image Restoration Using Few-Step Guidance of Consistency Models (and Beyond)](zero-shot_image_restoration_using_few-step_guidance_of_consistency_models_and_be.md)

<!-- RELATED:END -->
