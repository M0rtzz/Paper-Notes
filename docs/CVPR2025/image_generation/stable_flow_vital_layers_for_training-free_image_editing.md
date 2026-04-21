---
title: >-
  [论文解读] Stable Flow: Vital Layers for Training-Free Image Editing
description: >-
  [CVPR 2025][图像生成][图像编辑] Stable Flow 提出自动检测 DiT（FLUX）中的"关键层"（vital layers）并仅在这些层注入参考图像的注意力特征，实现无需训练的多种图像编辑操作，同时引入 latent nudging 技术改善真实图像的流模型反演质量。
tags:
  - CVPR 2025
  - 图像生成
  - 图像编辑
  - DiT
  - 流匹配
  - 注意力注入
  - 训练无关编辑
---

# Stable Flow: Vital Layers for Training-Free Image Editing

**会议**: CVPR 2025  
**arXiv**: [2411.14430](https://arxiv.org/abs/2411.14430)  
**代码**: [项目主页](https://omriavrahami.com/stable-flow)  
**领域**: 图像生成 / 图像编辑  
**关键词**: 图像编辑, DiT, 流匹配, 注意力注入, 训练无关编辑

## 一句话总结
Stable Flow 提出自动检测 DiT（FLUX）中的"关键层"（vital layers）并仅在这些层注入参考图像的注意力特征，实现无需训练的多种图像编辑操作，同时引入 latent nudging 技术改善真实图像的流模型反演质量。

## 研究背景与动机
- FLUX、SD3 等新一代 T2I 模型采用 DiT 架构 + flow matching，生成质量显著提升
- Flow-based 模型沿直线轨迹采样，导致多样性降低——但本文将此视为**编辑优势**
- UNet 时代的编辑方法（如 P2P、MasaCtrl）利用了 UNet 的粗→细→粗层级结构进行注意力注入
- DiT 架构**没有** UNet 的分层结构，所有层角色不明确，无法直接迁移注入策略
- 标准 inverse Euler ODE 反演在 FLUX 上效果差，导致重建失败
- 需要解决两个核心问题：(1) DiT 中**哪些层**适合注入特征？(2) 如何有效反演真实图像？

## 方法详解

### 整体框架
三步法：(1) **Vital Layers 检测**——逐层移除 DiT 中的每一层，测量对生成图像的感知影响，高影响层为 vital layers；(2) **Vital Layers 注入编辑**——并行生成参考图和编辑图，仅在 vital layers 将参考图的图像嵌入替换到编辑图中；(3) **Latent Nudging**——对真实图像反演时，将初始潜在乘以小系数 $\lambda=1.15$ 以改善重建。

### 关键设计

**设计一：自动 Vital Layers 检测**
- **功能**: 找到 DiT 中对图像形成至关重要的层子集
- **核心思路**: 用 64 个多样文本提示生成参考图像集 $G_{ref}$；对每层 $\ell$ 通过残差连接旁路该层，生成 $G_\ell$；用 DINOv2 计算 $G_{ref}$ 和 $G_\ell$ 的感知相似度；定义 vitality 分数 $vitality(\ell) = 1 - \frac{1}{k} \sum d(M_{full}, M_{-\ell})$；超过阈值 $\tau_{vit}$ 的层定义为 vital layers
- **设计动机**: DiT 没有 UNet 的结构化层级；实验发现 vital layers 分散在 Transformer 各处（非集中在某个区域）；旁路实验量化了每层对图像内容的真实贡献

**设计二：Vital Layers 注意力注入编辑**
- **功能**: 仅替换 vital layers 中参考图的图像嵌入，实现多种编辑（非刚性变形、添加物体、全局修改）
- **核心思路**: 并行生成源图像 $x$（原提示+种子）和编辑图 $\hat{x}$（编辑提示+同种子）；在 vital layers $V$ 中将 $\hat{x}$ 的图像嵌入替换为 $x$ 的图像嵌入；non-vital layers 保持 $\hat{x}$ 自己的嵌入
- **设计动机**: 分析发现 vital layers 中的多模态注意力在保持源内容和响应文本编辑之间取得了良好平衡——不变区域主要关注视觉特征，编辑区域主要关注文本 token（如"avocado"）；non-vital layers 几乎全部关注图像，不利于编辑

**设计三：Latent Nudging 真实图像反演**
- **功能**: 改善 FLUX 模型对真实图像的反演重建质量
- **核心思路**: 在反演前将干净潜在 $z_0$ 乘以 $\lambda=1.15$，轻微偏移出训练分布；然后应用标准 inverse Euler ODE：$z_t = z_{t-1} + (\sigma_t - \sigma_{t+1}) \cdot u_t(z_{t-1})$
- **设计动机**: 标准反演假设 $u(z_t) \approx u(z_{t-1})$，但 FLUX 的直线轨迹使此假设不成立；微小偏移使模型在前向过程中更不容易改变图像内容

### 损失函数
- 训练无关方法，无需损失函数
- 仅在 vital layers 检测阶段使用 DINOv2 感知相似度

## 实验关键数据

### 主实验：定量对比

| 方法 | CLIP_txt↑ | CLIP_img↑ | CLIP_dir↑ |
|------|------|------|------|
| SDEdit | 0.24 | 0.71 | 0.07 |
| P2P+NTI | 0.21 | 0.76 | 0.08 |
| Instruct-P2P | 0.22 | 0.87 | 0.07 |
| MagicBrush | 0.24 | 0.88 | 0.11 |
| MasaCTRL | 0.20 | 0.76 | 0.03 |
| **Stable Flow** | **0.23** | **0.92** | **0.14** |

### 消融实验

| 配置 | CLIP_txt↑ | CLIP_img↑ | CLIP_dir↑ |
|------|------|------|------|
| **Stable Flow** | **0.23** | **0.92** | **0.14** |
| 所有层注入 | 0.17 | 0.98 | 0.00 |
| Non-vital 层注入 | 0.25 | 0.72 | 0.09 |
| 无 latent nudging | 0.22 | 0.62 | 0.05 |

### 用户研究（两选一格式，胜率）

| vs 方法 | 提示符合度 | 图像保持 | 真实感 | 总体 |
|------|------|------|------|------|
| vs SDEdit | 69% | 68% | 64% | 71% |
| vs MasaCTRL | 82% | 80% | 80% | 72% |
| vs MagicBrush | 61% | 67% | 77% | 74% |

### 关键发现
- 所有层注入→CLIP_dir 降为 0（完全复制源图，无编辑效果）
- Non-vital 层注入→图像保持度大降（CLIP_img 0.72），编辑过度
- Vital layers 选出约 20 个层（FLUX 共 57 层），分散于 Transformer 各处
- Latent nudging 将 CLIP_img 从 0.62 提升到 0.92

## 亮点与洞察
- **将 flow 模型的低多样性缺点转化为编辑优势**是非常巧妙的视角
- Vital layers 的自动检测方法通用性强，可应用于其他 DiT 模型分析
- 同一个注入机制可处理非刚性编辑、物体添加、场景修改——高度统一
- Latent nudging 极其简单（乘 1.15）但效果显著

## 局限与展望
- 仅在 FLUX 模型上验证，对 SD3 等其他 DiT 模型的适用性需测试
- Vital layers 集合在所有编辑类型中固定，可能并非对所有编辑最优
- Latent nudging 的系数 $\lambda=1.15$ 为经验值，理论解释不足
- 未来可探索自适应 vital layers 选择、可控编辑强度、以及视频编辑扩展

## 相关工作与启发
- P2P/MasaCTRL：UNet 时代的注意力注入编辑方法
- FLUX/SD3：最新 DiT+flow matching T2I 模型
- SDEdit：通过加噪-去噪实现图像编辑的经典方法
- 启发：理解模型内部**层角色分工**是设计免训练编辑方法的关键前提

## 评分
⭐⭐⭐⭐ — 将 DiT 的低多样性转化为编辑优势的视角新颖；vital layers 检测方法通用且有洞见价值；latent nudging 虽简单但效果显著；全面的定量+用户研究验证令人信服。

<!-- RELATED:START -->

## 相关论文

- [Unveil Inversion and Invariance in Flow Transformer for Versatile Image Editing](unveil_inversion_and_invariance_in_flow_transformer_for_versatile_image_editing.md)
- [Training-Free Constrained Generation with Stable Diffusion Models](../../NeurIPS2025/image_generation/training-free_constrained_generation_with_stable_diffusion_models.md)
- [Decoupling Training-Free Guided Diffusion by ADMM](decoupling_training-free_guided_diffusion_by_admm.md)
- [SplitFlow: Flow Decomposition for Inversion-Free Text-to-Image Editing](../../NeurIPS2025/image_generation/splitflow_flow_decomposition_for_inversion-free_text-to-image_editing.md)
- [FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](../../ICCV2025/image_generation/flowedit_inversion-free_text-based_editing_using_pre-trained_flow_models.md)

<!-- RELATED:END -->
