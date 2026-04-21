---
title: >-
  [论文解读] BeautyGRPO: Aesthetic Alignment for Face Retouching via Dynamic Path Guidance and Fine-Grained Preference Modeling
description: >-
  [CVPR2026][目标检测][人脸修图] 提出 BeautyGRPO，一个基于强化学习的人脸修图框架，通过构建细粒度偏好数据集 FRPref-10K 训练专用奖励模型，并设计动态路径引导（DPG）机制在随机探索与高保真之间取得平衡，实现与人类美学偏好对齐的自然修图效果。
tags:
  - CVPR2026
  - 目标检测
  - 人脸修图
  - 强化学习
  - 美学对齐
  - 流匹配
  - 偏好建模
  - GRPO
  - 动态路径引导
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# BeautyGRPO: Aesthetic Alignment for Face Retouching via Dynamic Path Guidance and Fine-Grained Preference Modeling

**会议**: CVPR2026  
**arXiv**: [2603.01163](https://arxiv.org/abs/2603.01163)  
**代码**: 待确认（有Project Page）  
**领域**: 图像生成/人脸修复  
**关键词**: 人脸修图, 强化学习, 美学对齐, 流匹配, 偏好建模, GRPO, 动态路径引导

## 一句话总结

提出 BeautyGRPO，一个基于强化学习的人脸修图框架，通过构建细粒度偏好数据集 FRPref-10K 训练专用奖励模型，并设计动态路径引导（DPG）机制在随机探索与高保真之间取得平衡，实现与人类美学偏好对齐的自然修图效果。

## 研究背景与动机

**人脸修图需求旺盛**：社交媒体和拍照设备普及驱动高质量人脸修图的需求持续增长，任务要求去除痘痘、瑕疵等缺陷的同时保留痣、毛孔等身份特征

**监督学习的根本局限**：现有方法（CNN-based、Transformer-based）依赖像素级监督训练，只能模仿标注图像而无法捕获复杂的主观美学偏好，产出结果视觉僵硬、不自然

**监督学习天花板问题**：像素级重建目标导致模型过拟合特定修图风格，无法发现超越训练数据美学质量的解决方案

**通用图像编辑模型的不足**：NanoBanana、SeedDream 等大模型虽能执行修图操作，但常产生身份改变、油腻塑料质感皮肤和伪影等不自然效果

**在线 RL 的保真度冲突**：FlowGRPO 等 T2I-RL 框架通过注入随机噪声（SDE）实现探索，但累积随机漂移严重破坏人脸修图所需的高保真度，引入明显噪声伪影

**奖励模型粒度不足**：现有 T2I 奖励模型（ImageReward、HPSv2）主要关注全局美学或文本-图像一致性，缺乏评估皮肤光滑度、瑕疵去除、纹理保留等细粒度感知维度的能力

## 方法详解

### 整体框架

BeautyGRPO 由三大组件构成：(1) 细粒度偏好数据集 FRPref-10K；(2) 专用多维奖励模型；(3) 动态路径引导（DPG）算法。基于 FluxKontext-LoRA 作为骨干网络，先 LoRA 微调后再进行 RL 训练。

### 关键设计一：FRPref-10K 偏好数据集

- **数据来源**：从 FFHQR 和私有高分辨率集合中采集源肖像，确保多样人口统计和拍摄条件
- **候选生成**：对每张输入图使用 NanoBanana、FluxKontext、RetouchFormer 等多模型生成修图候选，变化随机种子
- **偏好对构造**：通过模型间比较（output vs output）和输出-标签比较构建 10,000 组高分偨率偏好对
- **混合标注流程**：(1) 多 VLM（GPT-4o、Qwen2.5-VL-72B、Gemini 2.5 Pro）从 5 个维度（皮肤光滑、瑕疵去除、纹理质量、清晰度、身份保持）评估并提供结构化推理；(2) 人工审核员在相同标准下审核；(3) 分歧情况由高级专家最终裁决

### 关键设计二：三阶段奖励模型训练

基于 Qwen2.5-VL-7B 初始化，采用 UnifiedReward-Thinking 范式：

- **Stage 1（结构化推理初始化）**：在 2K 子集上 SFT，训练模型在 `<think>` 块中生成五维度结构化推理、`<answer>` 块中给出最终偏好决策
- **Stage 2（自训练+一致性过滤）**：Stage 1 模型对剩余 8K 样本生成多条推理轨迹，按偏好正确性和推理连贯性双标准过滤一致轨迹，再做一轮 SFT
- **Stage 3（GRPO 鲁棒性增强）**：对不一致样本使用 GRPO，探索多样推理路径，由结果奖励（偏好准确率）和过程奖励（DeBERTa-V3 评估推理连贯性）联合监督

### 关键设计三：动态路径引导（DPG）

**问题**：FlowGRPO 将 ODE 转为 SDE（注入噪声 $\sigma_t d\omega$）来实现探索，但累积随机漂移导致轨迹偏离高保真流形。

**DPG 核心思想**：在每个采样时间步动态重规划引导轨迹，将当前状态拉回锚点附近的高保真流形，同时保留受控探索。

- **稳定锚点**：从 FRPref-10K 中选取高偏好样本 $x_0^{\text{anchor}}$，仅在采样过程中引入（非监督目标），约束轨迹在高保真流形附近
- **动态轨迹重规划**：在当前状态 $x_t$ 和锚点之间重规划直线 ODE 轨迹，计算下一时间步的目标状态 $x_{t-\Delta t}^* = (\Delta t/t) x_0^{\text{anchor}} + (1 - \Delta t/t) x_t$
- **引导修正向量**：$z_t^{\text{anchor}} = (x_{t-\Delta t}^* - \mu_t) / \sigma_{\text{step}}$，指示将轨迹拉回锚点路径的方向
- **混合噪声**：$z_t^{\text{mix}} = \lambda(t) z_t^{\text{anchor}} + (1-\lambda(t)) z_t^{\text{std}}$，其中 $\lambda(t) = t / \max(1, T-1)$ 是时间依赖系数——早期时间步使用更强的锚点引导修正结构偏差，后期依赖更多随机噪声鼓励细粒度探索
- **效率优化**：将完整采样轨迹分为 $K=3$ 段，每段随机选一个时间步做 DPG 更新，其余做 ODE 更新

### 损失函数

采用 GRPO 目标函数（clipped surrogate objective），配合归一化优势 $\hat{A}^i$。在 DPG 下，转移概率满足高斯分布 $\mathcal{N}(\mu_{\text{new}}, \sigma_{\text{new}}^2 I)$，其中 $\mu_{\text{new}} = (1-\lambda)\mu_t + \lambda x_{t-\Delta t}^*$，$\sigma_{\text{new}} = (1-\lambda)\sigma_{\text{step}}$，用于计算逐步似然比 $r_t(\theta)$ 进行策略更新。

## 实验

### 实验设置

- **骨干**：FluxKontext + LoRA 微调
- **数据**：FFHQR 测试集 1000 张 + In-the-wild 互联网肖像 1000 张
- **评估指标**：无参考感知/美学指标（NIQE↓、NIMA↑、MUSIQ↑、MANIQA↑、NRQM↑、TOPIQ↑）+ ArcFace 身份保持 + FID
- **硬件**：8×H20 GPU

### 主要结果

| 方法 | 类别 | NIQE↓ | NIMA↑ | MUSIQ↑ | MANIQA↑ | TOPIQ↑ | ArcFace↑ |
|------|------|-------|-------|--------|---------|--------|----------|
| RetouchFormer | 专用模型 | 11.153 | 4.723 | 4.465 | 1.036 | 0.605 | 0.986 |
| NanoBanana | 通用编辑 | 11.301 | 4.919 | 4.681 | 1.009 | 0.621 | 0.889 |
| FluxKontext+LoRA | 基线 | 12.913 | 4.694 | 4.459 | 1.035 | 0.601 | 0.973 |
| FlowGRPO | RL基线 | 15.024 | 4.573 | 4.271 | 0.935 | 0.571 | 0.882 |
| **BeautyGRPO (Ours)** | **RL** | **10.831** | **5.123** | **4.906** | **1.079** | **0.676** | 0.952 |

### 用户研究

| 方法 | VRetouchEr | RetouchFormer | NanoBanana | Kontext LoRA | **Ours** |
|------|-----------|---------------|------------|-------------|---------|
| Win Rate | 6.50% | 8.50% | 9.75% | 12.00% | **63.25%** |

100名参与者对20组测试样本进行偏好投票，BeautyGRPO 以 63.25% 的胜率大幅领先所有基线。

### 消融实验

**奖励模型对比**：替换不同编辑奖励模型，本文奖励模型在 NIMA(5.123) / MUSIQ(4.906) / MANIQA(1.079) / TOPIQ(0.676) 上全面超越 EditReward、EditScore、UnifiedReward-Edit。

**骨干泛化性**：将 BeautyGRPO 应用于 Qwen-Image-Edit，NIMA 从 4.571（原始）/4.824（+LoRA）提升至 5.351（+BeautyGRPO），TOPIQ 从 0.563 提升至 0.664。

**DPG 步数 K 的影响**：K=3 与 K=5 效果接近但采样更快，选 K=3 为默认。

### 关键发现

- FlowGRPO 直接应用于人脸修图会导致严重退化（NIQE 15.024 vs BeautyGRPO 10.831），验证了 DPG 的必要性
- 虽然 BeautyGRPO 的 FID 略高于最佳监督基线（4.054 vs 2.229），但这是因为模型探索了超越训练标签分布的更优美学解
- ArcFace 分数保持高位（0.952/0.944），表明美学增强的同时身份保持稳健

## 亮点

- 首次将基于人类美学偏好的 RL 对齐引入人脸修图任务，突破监督学习的天花板
- DPG 机制设计精巧：锚点引导 + 时间依赖混合系数，在保真与探索间取得优雅平衡
- 三阶段奖励模型训练流程（SFT → 自训练 → GRPO）结合五维度评估体系，构建了面向修图的细粒度反馈信号
- 用户研究 63.25% 胜率远超次优的 12%，定性效果令人信服

## 局限性

- FRPref-10K 数据集虽大但依赖特定 VLM 集合标注，标注偏差可能传播到奖励模型
- 锚点选择策略对结果质量影响未充分分析；锚点与输入图像的匹配方式不明确
- 主要在 FFHQR 和互联网肖像上评估，对极端光照、大角度侧脸、严重遮挡等困难场景的鲁棒性未探讨
- ArcFace 评分略低于部分监督方法（0.952 vs 0.986），身份保持与美学增强的权衡可进一步优化
- 计算成本未充分讨论（8×H20 GPU 训练，推理时 DPG 引入额外开销）

## 相关工作

- **专用修图模型**：ABPN、RestoreFormer(++)、VRetouchEr、RetouchFormer — 依赖监督学习，受限于像素级目标
- **通用编辑模型**：NanoBanana、ICEdit、SeedDream4.0、FluxKontext — 能力强但修图不够精细自然
- **扩散模型 RL 对齐**：DDPO、DPOK（策略梯度）、DiffusionDPO（离线）、FlowGRPO（在线 SDE 探索）— BeautyGRPO 在 FlowGRPO 基础上加入 DPG 解决保真问题
- **奖励建模**：ImageReward、HPSv2、EditScore、EditReward、UnifiedReward — 缺乏修图细粒度维度

## 评分

- 新颖性: ⭐⭐⭐⭐ — DPG 机制和面向修图的偏好对齐是新颖的组合，但核心 GRPO 框架借鉴自 FlowGRPO
- 实验充分度: ⭐⭐⭐⭐ — 定量+定性+用户研究+多消融+跨骨干验证，指标覆盖全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机论证充分，公式推导完整
- 价值: ⭐⭐⭐⭐ — 首个将 RL 偏好对齐系统性引入人脸修图的工作，开辟新方向

<!-- RELATED:START -->

## 相关论文

- [Specificity-aware Reinforcement Learning for Fine-grained Open-world Classification](specificity-aware_reinforcement_learning_for_fine-grained_open-world_classificat.md)
- [FG-CLIP: Fine-Grained Visual and Textual Alignment](../../ICML2025/object_detection/fg-clip_fine-grained_visual_and_textual_alignment.md)
- [Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection](../../ICCV2025/object_detection/dynamicdino_finegrained_mixture_of_experts_tuning_for_realti.md)
- [FALCON: False-Negative Aware Learning of Contrastive Negatives in Vision-Language Alignment](falcon_false-negative_aware_learning_of_contrastive_negatives_in_vision-language.md)
- [CTPD: Cross Tokenizer Preference Distillation](../../AAAI2026/object_detection/ctpd_cross_tokenizer_preference_distillation.md)

<!-- RELATED:END -->
