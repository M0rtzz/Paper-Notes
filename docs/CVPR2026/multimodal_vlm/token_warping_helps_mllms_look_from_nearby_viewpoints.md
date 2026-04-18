---
title: >-
  [论文解读] Token Warping Helps MLLMs Look from Nearby Viewpoints
description: >-
  [CVPR 2026][多模态][视角变换] 提出对 MLLM 的 ViT image token 做空间 warping（而非传统的像素级 warping）来模拟视角变换，发现 backward token warping 在保持语义一致性同时对深度估计噪声鲁棒，在自建的 ViewBench 上大幅超越像素级 warping、专用空间推理 MLLM 和生成式 warping 方法。
tags:
  - CVPR 2026
  - 多模态
  - 视角变换
  - token warping
  - 空间推理
  - 心理意象
  - MLLM
---

# Token Warping Helps MLLMs Look from Nearby Viewpoints

**会议**: CVPR 2026  
**arXiv**: [2604.02870](https://arxiv.org/abs/2604.02870)  
**代码**: https://token-warping-mllm.github.io/ (项目页)  
**领域**: 多模态VLM  
**关键词**: 视角变换, token warping, 空间推理, 心理意象, MLLM

## 一句话总结
提出对 MLLM 的 ViT image token 做空间 warping（而非传统的像素级 warping）来模拟视角变换，发现 backward token warping 在保持语义一致性同时对深度估计噪声鲁棒，在自建的 ViewBench 上大幅超越像素级 warping、专用空间推理 MLLM 和生成式 warping 方法。

## 研究背景与动机

**领域现状**：多模态大语言模型在视觉推理上表现出色，但面对视角变化时相当脆弱。即使深度估计已接近完美，将预测深度整合到 MLLM 中也无法带来真正的 3D 理解。专门为空间推理微调的 MLLM（如 SpatialReasoner）在视角变换任务上改善有限。

**现有痛点**：传统做法是用像素级 warping 将源图像变换到目标视角，但像素级操作对深度图中的微小误差极度敏感——即使小的深度不准确，warping 后也会出现明显的几何扭曲和语义退化（如书本变形、物体模糊）。生成式新视角合成方法（如 GenWarp）虽能合成完整图像，但可能幻觉出不存在的物体或丢失已有物体。

**核心矛盾**：视角变换需要对场景进行某种内部表征变换，但变换的粒度选择存在根本性矛盾——物体级表征太粗、丢失空间细节；像素级表征太细、对噪声过于敏感。需要一个中间粒度的表征。

**本文目标** (1) 找到一种对深度误差鲁棒的视角变换表征方式；(2) 探索最佳的 warping 策略（前向/后向、最近/自适应）；(3) 构建评估 MLLM 视角推理能力的标准基准。

**切入角度**：受认知科学中"心理意象"理论启发——Shepard、Minsky、Pylyshyn、Hinton 等人提出心理图像依赖于"部件级结构描述"而非整体表征。ViT 中的 image token 恰好处于像素和物体之间的中间粒度，天然是"部件级"表征单元。

**核心 idea**：将视角变换操作从像素级提升到 token 级，利用 image token 作为视角变换的鲁棒语义单元，实现 MLLM 的近视角推理。

## 方法详解

### 整体框架
输入是一张源视角图像、其深度图、源和目标相机位姿；目标是让 MLLM 能回答"从目标视角看，场景是什么样的"。方法核心是在 MLLM 的 ViT 层面对 image token 做几何 warping，而非在像素级操作。整个流程无需额外训练，仅在推理时增加少量 warping 计算。

### 关键设计

1. **Token 对位置扰动的鲁棒性验证**:

    - 功能：证明 image token 是进行视角变换的合适表征粒度
    - 核心思路：设计"获取位置噪声敏感性测试"——对每个 token 的网格中心坐标施加高斯位移扰动（从 0 到 20 像素），然后用这些扰动后的位置获取 patch 输入 ViT。实验发现 Qwen2.5-VL 在 CV-Bench-2D 上的准确率几乎不变，即使扰动量接近 patch 大小。相比之下，同样的扰动施加在像素级表征上会导致性能明显下降。
    - 设计动机：为后续 token warping 提供理论支撑——既然 token 对位置信息不敏感，那么 warping 时由深度误差引入的位置偏移就不会严重影响 MLLM 理解。

2. **Backward Token Warping（核心方法）**:

    - 功能：将源视角的 token 重新排列到目标视角的规则网格上
    - 核心思路：从目标视角出发定义密集规则网格，对每个目标网格点经由反向投影函数 $f_{T \to S}$ 映射回源图像平面，找到源图像中对应的 patch/token。具体实现是从源图像深度图构建轻量 3D 代理网格（proxy mesh），通过 ray casting 从目标视角的每个网格位置向源图像投射，确定对应的源坐标。与前向 warping 相比，后向 warping 保证目标视角的 token 是密集、规则排列的——这对于在规则网格上训练的 MLLM 至关重要。
    - 设计动机：前向 warping（将源 token 投射到目标平面）会产生稀疏、不规则的 token 分布，大量空洞使 MLLM 收到分布外输入而性能暴跌。后向 warping 从目标视角的规则网格出发，天然保证密集和规则性。

3. **Nearest vs Adaptive Fetching**:

    - 功能：决定如何从源图像获取与目标网格点对应的 token
    - 核心思路：Nearest fetching 直接选择源图像中与映射坐标欧氏距离最近的已有 token；Adaptive fetching 则以映射坐标为 patch 中心重新裁剪源图像并编码为新 token。实验表明两者性能接近——nearest fetching 简单高效却不输 adaptive，这再次验证了 token 对位置偏移的鲁棒性。
    - 设计动机：Nearest 免去重新编码开销，Adaptive 理论上更精确但计算更贵。两者可比的性能提供了实用性指导。

### 损失函数 / 训练策略
本方法无需训练，纯推理时操作——仅需在 MLLM 推理前对 image token 做一次 warping 变换，计算开销极小。

## 实验关键数据

### 主实验

实验在自建的 ViewBench 上进行，基于 ScanNet 真实室内场景，评估三类任务：Text（文本标记的空间关系）、Shape（几何形状的空间关系）、Object（目标视角物体描述）。

| 方法 | ViewBench-Text (5-15%) | ViewBench-Shape (5-15%) | ViewBench-Object (5-15%) |
|------|----------------------|------------------------|-------------------------|
| SpatialReasoner | 46.73 | 33.72 | - |
| VLM-3R | 63.82 | 49.22 | - |
| GenWarp | 69.35 | 53.10 | 4.32 |
| Pixel Backward | 71.86 | 62.40 | 4.53 |
| Token Backward-Nearest | 74.87 | 67.44 | 4.80 |
| **Token Backward-Adaptive** | **77.89** | **67.44** | **4.97** |
| Oracle (GT Target View) | 100.00 | 100.00 | 6.64 |

### 消融实验
| 配置 | ViewBench-Text (5-15%) | ViewBench-Shape (5-15%) | 说明 |
|------|----------------------|------------------------|------|
| Token Forward | 60.30 | 55.04 | 前向 warping 导致不规则 token |
| Token Backward-Nearest | 74.87 | 67.44 | 后向+最近，性能优异 |
| Token Backward-Adaptive | 77.89 | 67.44 | 后向+自适应，计算更贵但提升有限 |
| Pixel Forward | 70.85 | 56.20 | 像素级前向 |
| Pixel Backward | 71.86 | 62.40 | 像素级后向 |

### 关键发现
- **后向 > 前向**是最关键的设计选择：后向 token warping 在 Text 5-15% 场景比前向提升 14.57%，因为 MLLM 需要密集规则的 token 网格
- **Token 级 > 像素级**：后向 token warping 比后向像素 warping 在 Text 上高 6%，Shape 上高 5%，因为 token 对深度噪声更鲁棒
- Nearest fetching 与 Adaptive fetching 性能接近，说明 token 表征的鲁棒性使得精确对齐并非必要
- 使用预测深度 vs GT 深度差距很小，进一步验证方法对深度误差的鲁棒性
- 所有专用空间推理 MLLM（SpatialReasoner、VLM-3R、ViLaSR）均不如 token warping，说明空间微调不能替代显式视角变换

## 亮点与洞察
- **认知科学与工程设计的巧妙结合**：从心理意象理论中抽取"部件级表征"思想，对应到 ViT patch token，实现了从认知理论到工程方法的优雅映射。这个类比不仅有解释力，还直接指导了方法设计。
- **零训练的推理时增强**：整个方法不需要任何额外训练，仅在推理时对 token 做一次 warping，就能显著提升视角推理能力。这种"免费午餐"式的方法具有极高的实用价值。
- **规则密集 token 网格的重要性**：发现 MLLM 对 token 的空间分布模式非常敏感——稀疏不规则的 token（前向 warping 产生）是严重的分布外输入。这个洞察可迁移到其他需要操控 token 布局的任务。

## 局限与展望
- 仅处理**近视角变换**（两视角有重叠），大角度视角变化时 warping 失效（出现大量遮挡和空洞区域）
- 依赖深度图（GT 或预测），虽然对深度噪声鲁棒但仍需深度输入，限制了应用场景
- ViewBench 基于室内场景（ScanNet），对户外场景、动态场景的泛化性未验证
- 仅在 Qwen2.5-VL 上实验，不同架构的 MLLM 对 token perturbation 的鲁棒性可能不同
- 未探索与空间推理微调方法的组合——token warping + SpatialReasoner 微调是否能进一步提升？

## 相关工作与启发
- **vs SpatialReasoner / VLM-3R**：这些方法通过空间数据微调 MLLM 来获得空间推理能力，但本文发现微调不能替代显式视角变换。Token warping 在不需要额外训练的情况下大幅超越它们。
- **vs GenWarp（生成式 warping）**：生成式方法用扩散模型合成目标视角图像，但会幻觉不存在的物体。Token warping 不生成新像素，仅重排已有 token，避免了幻觉问题。
- **vs 像素 warping**：经典 3D 视觉方法，但对深度噪声敏感。Token warping 利用 ViT patch 的粗粒度天然容忍位置误差。

## 评分
- 新颖性: ⭐⭐⭐⭐ 从认知科学出发的 token warping 思路很有创意，但技术实现相对简单
- 实验充分度: ⭐⭐⭐⭐ ViewBench 设计合理，消融全面，但仅限室内场景和单一 MLLM
- 写作质量: ⭐⭐⭐⭐⭐ 论述清晰，从理论到实验的逻辑链完整，图表直观
- 价值: ⭐⭐⭐⭐ 无训练推理时增强有强实用价值，但应用场景受限于近视角变换
