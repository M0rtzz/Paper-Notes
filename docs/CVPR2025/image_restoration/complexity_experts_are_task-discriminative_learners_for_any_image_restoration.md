---
title: >-
  [论文解读] Complexity Experts are Task-Discriminative Learners for Any Image Restoration
description: >-
  [CVPR 2025][图像恢复][全合一图像修复] 提出 MoCE-IR，用具有不同计算复杂度和感受野大小的"复杂度专家"替代传统均匀 MoE 的统一架构，配合偏向低复杂度的弹簧式路由机制，意外地实现了任务判别性分配——不同退化类型自动路由到适当复杂度的专家，可在推理时跳过无关专家。 领域现状：全合一图像修复（all-in…
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "全合一图像修复"
  - "混合专家"
  - "复杂度感知路由"
  - "任务判别学习"
  - "稀疏计算"
---

# Complexity Experts are Task-Discriminative Learners for Any Image Restoration

**会议**: CVPR 2025  
**arXiv**: [2411.18466](https://arxiv.org/abs/2411.18466)  
**代码**: [https://eduardzamfir.github.io/MoCE-IR/](https://eduardzamfir.github.io/MoCE-IR/)  
**领域**: 图像修复 / 模型压缩  
**关键词**: 全合一图像修复, 混合专家, 复杂度感知路由, 任务判别学习, 稀疏计算

## 一句话总结
提出 MoCE-IR，用具有不同计算复杂度和感受野大小的"复杂度专家"替代传统均匀 MoE 的统一架构，配合偏向低复杂度的弹簧式路由机制，意外地实现了任务判别性分配——不同退化类型自动路由到适当复杂度的专家，可在推理时跳过无关专家。

## 研究背景与动机

**领域现状**：全合一图像修复（all-in-one restoration）用单一模型处理去噪、去雾、去雨等多种退化。密集模型（如 PromptIR）的很多参数在特定任务时处于闲置状态。MoE 架构是自然的扩展，但现有 MoE 方法（AnyIR、InstructIR 等）存在路由不一致问题——某些专家意外地跨任务泛化，其他专家在自己负责的任务上反而表现差。

**现有痛点**：(1) 传统 MoE 的均匀架构：所有专家有相同的计算复杂度和感受野，但不同退化需要不同处理——运动模糊需要局部空间感知，去雾需要全局上下文理解。(2) 路由困难：退化复杂度事先未知，基于语言或退化先验的路由导致优化不平衡。

**核心矛盾**：需要让不同专家专注不同任务以实现推理时跳过无关专家（计算高效），但如何自动确定哪个专家适合哪个任务？

**本文目标** 设计一种 MoE 架构，使专家自动与不同复杂度的退化任务对齐，支持推理时跳过不必要专家。

**切入角度**：不同退化有不同的固有复杂度——给专家设计不同的计算量和感受野，然后用偏向简单专家的路由。令人惊讶的是，这种简单偏好就能让任务自动分配到合适复杂度的专家。

**核心 idea**：设计计算复杂度递增、感受野递增的专家块，用偏向低复杂度的弹簧力路由机制，实现退化任务到专家的自动最优匹配。

## 方法详解

### 整体框架
U 形非对称编码解码器架构。编码器用标准 Transformer 块，解码器块中集成 MoCE 层。每个 MoCE 层包含 n 个复杂度递增的专家 $\mathbf{E}$ 和一个共享专家 $\mathbf{S}$。高频 Sobel 特征向量引导门控函数。最终通过全局残差连接精修高频细节。

### 关键设计

1. **复杂度专家设计（Complexity Experts）**

    - 功能：提供从轻量到重量的一系列处理能力，匹配不同退化的需求
    - 核心思路：第 i 个专家的通道维度为 $r_i = C/2^i$（越高越窄），窗口大小为 $w_i$（越高越大）。这样轻量专家处理局部简单退化，重量专家处理需要全局上下文的复杂退化。每个专家用 FFT 加速的窗口自注意力。专家输出与共享专家（通道维度转置注意力）做元素乘调制，最终通过交叉注意力合并。共享专家捕获跨退化共享特征
    - 设计动机：均匀专家无法反映任务的固有复杂度差异，导致路由混乱。异构设计让路由有了明确的"目的地"

2. **复杂度感知路由（Complexity-aware Routing）**

    - 功能：自动将输入退化图像分配到具有适当复杂度的专家
    - 核心思路：图像级 top-1 路由（非 token 级），softmax 后选择最高分的专家。关键创新是复杂度偏置 $\mathbf{b} = [p_1/p_{max}, p_2/p_{max}, ..., p_n/p_{max}]$，乘以重要性分数，使低复杂度专家获得更高的有效权重。类比弹簧系统：参数量对应位移，归一化因子对应弹簧常数，共同决定偏置力大小。辅助损失 $\mathcal{L}_{aux}$ 用复杂度加权的重要性和负载均衡的变异系数
    - 设计动机：没有偏置时路由随机化，导致模型容量利用不充分。偏向简单专家的直觉是：如果简单专家能处理，就不需要动用复杂专家，只有复杂退化才会"克服弹簧力"被分配到大专家

3. **推理时专家跳过**

    - 功能：基于训练后的路由统计，推理时跳过与当前退化无关的专家，节省计算
    - 核心思路：由于路由的任务判别性，每种退化类型在训练后几乎总是被分配到同一个专家。推理时只需激活该任务对应的专家，跳过其他专家
    - 设计动机：这是 MoCE 的核心实用价值——密集模型总是激活所有参数，MoCE 实现了真正的条件计算

### 损失函数 / 训练策略
$\mathcal{L} = \mathcal{L}_{pixel} + \lambda \mathcal{L}_{aux}$，像素损失 + 复杂度感知路由辅助损失。基于 Restormer 架构。

## 实验关键数据

### 主实验（三退化：去雾+去雨+去噪）

| 方法 | 参数 | 去雾PSNR↑ | 去雨PSNR↑ | 去噪σ25↑ | 平均↑ |
|------|------|---------|---------|---------|------|
| AirNet | 9M | 27.94 | 34.90 | 31.26 | 31.20 |
| PromptIR | 36M | 30.58 | 36.37 | 31.31 | 32.06 |
| Art-PromptIR | 33M | 30.83 | 37.94 | 31.42 | 32.49 |
| **MoCE-IR-S** | **11M** | **30.94** | **38.22** | **31.42** | **32.57** |

11M 参数轻量版即超越 33-36M 参数的 SOTA。

### 消融实验

| 配置 | 平均PSNR | 说明 |
|------|---------|------|
| 均匀专家 + 标准路由 | 32.19 | 传统 MoE |
| 复杂度专家 + 标准路由 | 32.38 | 仅异构设计 |
| **复杂度专家 + 复杂度路由** | **32.57** | 完整方法 |

### 关键发现
- 偏向低复杂度的路由令人惊讶地实现了任务判别性分配——去噪被分配到最简单专家，去雾被分配到最复杂专家
- 11M 轻量版超越 36M PromptIR，证明高效参数利用比盲目增加参数更有效
- 推理时跳过专家可进一步降低 FLOPs 而几乎不掉性能
- 共享专家的通道注意力与复杂度专家的空间注意力互补是性能关键

## 亮点与洞察
- **"偏好简单"就能实现任务判别**是最核心的发现——简单退化自然留在简单专家，复杂退化"克服弹簧力"被推向大专家。这种涌现的任务分配行为非常优雅
- **弹簧力类比**直观且有理论基础——复杂度偏置与物理系统的恢复力完全对应
- **异构专家设计**为 MoE 在低级视觉任务中的应用开辟了新方向

## 局限与展望
- 图像级路由（非 token 级），无法处理单张图像中混合多种退化的情况
- 专家数量和复杂度梯度需要手动设计
- 目前仅在三退化基准上验证，更多退化类型（如超分、去模糊）的扩展未充分测试

## 相关工作与启发
- **vs PromptIR**: PromptIR 用可调提示编码退化特定信息，36M 参数。MoCE-IR 11M 参数即超越，因为条件计算更高效
- **vs AnyIR**: AnyIR 用 MoE 但均匀专家+退化先验路由，路由不一致。本文异构专家+复杂度偏置解决了路由问题
- **vs InstructIR**: 依赖 LLM 文本提示，资源需求大。本文完全视觉驱动，轻量化

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 复杂度专家+弹簧力路由的组合高度原创，涌现的任务判别性令人惊喜
- 实验充分度: ⭐⭐⭐⭐ 多尺度模型+消融+效率分析，但退化类型可更多
- 写作质量: ⭐⭐⭐⭐⭐ 动机→观察→方法→结果的逻辑链完美，弹簧力类比直观
- 价值: ⭐⭐⭐⭐⭐ 对全合一修复和高效 MoE 设计都有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Devil is in the Uniformity: Exploring Diverse Learners within Transformer for Image Restoration](../../ICCV2025/image_restoration/devil_is_in_the_uniformity_exploring_diverse_learners_within_transformer_for_ima.md)
- [\[ICCV 2025\] Exploiting Diffusion Prior for Task-driven Image Restoration](../../ICCV2025/image_restoration/exploiting_diffusion_prior_for_task-driven_image_restoration.md)
- [\[ECCV 2024\] MoE-DiffIR: Task-customized Diffusion Priors for Universal Compressed Image Restoration](../../ECCV2024/image_restoration/moe-diffir_task-customized_diffusion_priors_for_universal_compressed_image_resto.md)
- [\[CVPR 2025\] DarkIR: Robust Low-Light Image Restoration](darkir_robust_low-light_image_restoration.md)
- [\[CVPR 2025\] DPIR: Dual Prompting Image Restoration with Diffusion Transformers](dpir_dual_prompting_restoration_dit.md)

</div>

<!-- RELATED:END -->
