---
title: >-
  [论文解读] Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception
description: >-
  [CVPR 2026][图像生成][图像美学] 提出 DIAE 框架，通过多模态美学感知（MAP）将模糊的美学指令转化为 HSV/轮廓图视觉信号 + 文本联合引导，并构建"不完美配对"数据集 IIAEData 实现弱监督的图像美学增强。
tags:
  - CVPR 2026
  - 图像生成
  - 图像美学
  - 扩散模型
  - 多模态感知
  - 弱监督
  - ControlNet
---

# Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception

**会议**: CVPR 2026  
**arXiv**: [2603.11556](https://arxiv.org/abs/2603.11556)  
**代码**: 无  
**领域**: 图像美学增强 / 扩散模型  
**关键词**: 图像美学, 扩散模型, 多模态感知, 弱监督, ControlNet

## 一句话总结

提出 DIAE 框架，通过多模态美学感知（MAP）将模糊的美学指令转化为 HSV/轮廓图视觉信号 + 文本联合引导，并构建"不完美配对"数据集 IIAEData 实现弱监督的图像美学增强。

## 研究背景与动机

图像美学增强（IAE）要求模型同时具备美学评估和属性增强能力，而现有扩散模型在此任务上面临两大挑战：

**美学感知困难**：美学是高层人类视觉能力，受文化、经验、情感等不可控因素影响；简单的文本编码器难以理解"饱和度不足"、"三分法构图"等抽象美学描述

**配对数据稀缺**：完全监督需要内容完全一致但美学属性不同的"完美配对"图像，这要求专业摄影师/艺术家逐张编辑，成本极高

作者指出，现有图像编辑方法（InstructPix2Pix、MGIE 等）在美学增强方面效果有限，因为它们缺乏专门的美学感知能力，而基于强化学习的方法（DOODL、DiffusionDPO）虽能引导生成方向但不够精细。

## 方法详解

### 整体框架

基于 Stable Diffusion v1.5 + ControlNet 架构，包含三个核心模块：(1) IIAEData 数据集构建，(2) 多模态美学感知 MAP，(3) 双分支监督框架。

### 关键设计

1. **IIAEData 数据集**：构建"不完美配对"数据集，图像具有相同语义但不同美学质量：

    - 从 AVA、TAD66K、KonIQ、FLICKR 收集图像
    - 高 MOS 值图像作为参考图，低 MOS 值图像作为输入图
    - 使用 LLaVA-13B 生成图像描述，基于语义匹配形成配对
    - UNIAA-LLaVA 生成包含色彩、光照、构图等多维度的标准化美学评估
    - 最终得到 **47.5K 样本**（45K 训练 + 1.5K 测试）

2. **Multimodal Aesthetic Perception (MAP)**：将模糊的美学描述转化为可操作的多模态信号：

    - **色彩属性**：HSV 图作为视觉表征（直接反映饱和度、明度、色调），+ 色彩美学文本描述
    - **结构属性**：HED 轮廓图作为视觉表征（反映焦点、构图、拍摄方式），+ 结构美学文本描述
    - 视觉嵌入通过 CNN 提取：$F^I_{col} = \Phi_i(I_{col}), F^I_{str} = \Phi_i(I_{str})$
    - 文本嵌入通过 CLIP 编码：$F^T_{col} = \Phi_t(T_{col}), F^T_{str} = \Phi_t(T_{str})$
    - 通过 ControlNet 的 adding structure 注入扩散模型

3. **双分支监督框架**：解决"不完美配对"数据的弱监督训练问题：

    - 通过时间步参数 $t_s$（默认900/1000）划分两个训练阶段
    - $t \leq t_s$：语义监督分支，由输入图监督（保持内容一致性）
    - $t > t_s$：美学监督分支，由参考图监督（引入高美学属性）

    $L = L_{ref} + \lambda L_{inp}$

   其中 $L_{ref}$ 在所有时间步由参考图监督，$L_{inp}$ 仅在早期时间步由输入图监督。

### 损失函数 / 训练策略

$$L_{ref} = \|\epsilon_{ref} - \epsilon_\theta(x_{ref}(t), c, x_{ref}, t, cond)\|_2^2$$
$$L_{inp} = \|\epsilon_{inp} - \epsilon_\theta(x_{inp}(t\%t_s), c, x_{inp}, t\%t_s, cond)\|_2^2$$

- 基础模型：Stable Diffusion v1.5
- 优化器：AdamW，学习率 1e-5
- 训练 100K 迭代，4×A800 GPU
- 推理 50 步去噪，256×256 约4秒，512×512 约9秒

## 实验关键数据

### 主实验

| 方法 | LAIONs(256) | LAIONs(512) | MLLMs(256) | MLLMs(512) | CLIP-I(256) | CLIP-I(512) |
|------|------------|------------|-----------|-----------|------------|------------|
| Original | 4.962 | 5.123 | 3.243 | 3.300 | 1.000 | 1.000 |
| ControlNet | 4.979 | 5.522 | 3.271 | 3.415 | 0.628 | 0.617 |
| InstructPix2Pix | 4.991 | 5.396 | 3.264 | 3.325 | 0.764 | 0.690 |
| MGIE | 4.947 | 5.519 | 3.045 | 3.411 | 0.557 | 0.770 |
| DOODL | 5.102 | 5.140 | 3.255 | 3.297 | 0.775 | 0.703 |
| **DIAE** | **5.324** | **6.012** | **3.339** | **3.662** | **0.772** | **0.784** |

512分辨率下，LAIONs 提升 17.4%，MLLMs 提升 11.0%。

### 消融实验

| 配置 | LAIONs | MLLMs | CLIP-I | 说明 |
|------|--------|-------|--------|------|
| DIAE (w/o v) | 5.250 | 3.343 | 0.623 | 去除视觉引导 |
| DIAE (w/o t) | 5.428 | 3.410 | 0.792 | 去除文本评估 |
| **DIAE** | **5.668** | **3.501** | **0.778** | 完整MAP |

### 关键发现

- 双视觉模态（HSV+轮廓图）对内容一致性贡献显著（CLIP-I 0.792 vs 0.623）
- 文本模态对美学提升更关键（LAIONs 5.668 vs 5.428）
- 参数 $t_s$ 越大，生成图越接近输入图（保持更多原始美学属性）
- 对低美学质量图像（MOS<4.0）的提升尤为明显

## 亮点与洞察

- **弱监督方案实用性强**：用"不完美配对"替代不可得的"完美配对"，通过双分支框架优雅地解决内容-美学的解耦问题
- **MAP 设计有针对性**：HSV 对应色彩美学、轮廓图对应结构美学，比直接用 RGB 更符合美学评估的感知维度
- **可与 MLLM 集成**：DIAE 可接入美学理解 MLLM 自动生成 MAP 输入，形成端到端流程

## 局限与展望

- **未处理人像场景**：人像的美学涉及面部特征和体态等复杂因素，作者明确排除了此类数据
- **SD v1.5 底座较旧**：相比 SDXL/SD3 在生成质量上有差距
- **语义匹配质量依赖 LLaVA**：配对数据的质量受限于 caption 的准确性
- $t_s$ 参数需手动调节，不同场景可能需要不同设置

## 相关工作与启发

- 区别于 DOODL/RAHF 使用 MOS 作为分类器引导的方式，DIAE 通过多模态条件直接控制美学属性编辑，更精细
- 双分支监督思想受 StyleDiffusion 的内容-风格解耦启发，但具体化为早/晚期时间步的语义/美学分离
- 启发：可将此框架推广到视频美学增强，或结合用户个性化偏好

## 评分

- 新颖性: ⭐⭐⭐ MAP 模块有创意但整体技术组合较常规（ControlNet+SD v1.5）
- 实验充分度: ⭐⭐⭐ 评估维度齐全但与 SOTA 对比对象略旧，缺少 user study
- 写作质量: ⭐⭐⭐ 结构清晰但部分细节（如 $\lambda$ 选择）交代不足
- 价值: ⭐⭐⭐ 图像美学增强方向有实际应用价值，"不完美配对"数据方案有参考意义

<!-- RELATED:START -->

## 相关论文

- [Enhancing Spatial Understanding in Image Generation via Reward Modeling](enhancing_spatial_understanding_in_image_generation_via_reward_modeling.md)
- [PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](pixelrush_ultra-fast_training-free_high-resolution_image_generation_via_one-step.md)
- [Learnability-Guided Diffusion for Dataset Distillation](learnability-guided_diffusion_for_dataset_distillation.md)
- [Reviving ConvNeXt for Efficient Convolutional Diffusion Models](reviving_convnext_for_efficient_convolutional_diffusion_models.md)
- [MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)

<!-- RELATED:END -->
