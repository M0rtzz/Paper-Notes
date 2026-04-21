---
title: >-
  [论文解读] DC-Merge: Improving Model Merging with Directional Consistency
description: >-
  [CVPR 2026 (Main Track)][多模态][model merging] DC-Merge 发现模型合并的关键在于保持合并后多任务向量与原始单任务向量之间**奇异空间方向的一致性**，通过奇异值平滑 + 共享正交子空间投影两步操作，在 Vision 和 Vision-Language 任务上均取得 SOTA 合并效果。
tags:
  - CVPR 2026 (Main Track)
  - 多模态
  - model merging
  - task vector
  - singular value decomposition
  - directional consistency
  - LoRA
---

# DC-Merge: Improving Model Merging with Directional Consistency

**会议**: CVPR 2026 (Main Track)  
**arXiv**: [2603.06242](https://arxiv.org/abs/2603.06242)  
**代码**: [https://github.com/Tobeginwith/DC-Merge](https://github.com/Tobeginwith/DC-Merge)  
**领域**: 模型合并 / 多任务学习  
**关键词**: model merging, task vector, singular value decomposition, directional consistency, LoRA

## 一句话总结
DC-Merge 发现模型合并的关键在于保持合并后多任务向量与原始单任务向量之间**奇异空间方向的一致性**，通过奇异值平滑 + 共享正交子空间投影两步操作，在 Vision 和 Vision-Language 任务上均取得 SOTA 合并效果。

## 研究背景与动机

### 领域现状
模型合并旨在将多个任务适配模型整合为一个统一模型，继承各任务知识。现有方法如 Task Arithmetic、TIES、DARE 等通过对 task vector（微调参数 - 预训练参数）进行加权平均/剪枝来实现合并。

### 现有痛点

**能量分布不均**：Task vector 的 SVD 分解中，少数大奇异值主导了总能量（如前 5% 的奇异值可能占 90%+ 能量），合并时弱但语义重要的分量被忽略

**几何方向不一致**：不同任务的 task vector 在参数空间的几何方向相互冲突，直接合并会扭曲各 task vector 的方向结构

### 核心矛盾
简单的加权平均或裁剪在处理高维参数空间中的方向性信息时过于粗糙——它们无法保证合并结果在奇异空间的方向与各个单独 task vector 保持一致。

### 核心 idea
通过两步操作保持**方向一致性（Directional Consistency）**：先平滑各 task vector 的奇异值以平衡能量分布，再将能量平衡后的 task vector 投影到共享正交子空间以对齐几何方向。

## 方法详解

### 整体框架
输入 $N$ 个 task vector $\{\boldsymbol{\tau}_i\}_{i=1}^N$，每个 $\boldsymbol{\tau}_i = \boldsymbol{\theta}_i - \boldsymbol{\theta}_0$。Pipeline 分三步：
1. **奇异值平滑（Singular Value Smoothing）**：对每个 $\boldsymbol{\tau}_i$ 做 SVD，平滑奇异值分布
2. **共享正交子空间投影（Shared Orthogonal Subspace Projection）**：将平滑后的 task vector 投影到一个共享子空间以对齐方向
3. **子空间聚合与反投影（Aggregation & Back-projection）**：在子空间中聚合后投影回原参数空间

### 关键设计

#### 1. 奇异值平滑（SVD Smoothing）
- **功能**：对每个 task vector $\boldsymbol{\tau}_i$ 进行 SVD 分解 $\boldsymbol{\tau}_i = \mathbf{U}_i \boldsymbol{\Sigma}_i \mathbf{V}_i^\top$，然后对奇异值向量 $\boldsymbol{\sigma}_i$ 应用平滑变换
- **核心思路**：使用幂变换 $\sigma_j \leftarrow \sigma_j^\alpha$（$\alpha < 1$，如 $\alpha = 0.5$），压缩大奇异值、提升小奇异值，使能量分布更均匀
- **设计动机**：防止少数大奇异值在合并时完全主导结果，确保语义上重要但能量较弱的知识分量得到充分表达
- **与之前方法的区别**：Task Arithmetic 直接平均不考虑奇异值分布；TIES 只做向量级裁剪，未触及奇异空间

#### 2. 共享正交子空间投影
- **功能**：寻找一个共享正交基 $\mathbf{Q}$，使得所有能量平滑后的 task vector 被投影到该子空间后，与原始方向的重构误差最小
- **核心思路**：求解优化问题 $\min_{\mathbf{Q}} \sum_{i=1}^N \|\tilde{\boldsymbol{\tau}}_i - \mathbf{Q}\mathbf{Q}^\top \tilde{\boldsymbol{\tau}}_i\|_F^2$，其中 $\tilde{\boldsymbol{\tau}}_i$ 是平滑后的 task vector。通过对拼接矩阵做 SVD 取前 $k$ 个奇异向量得到 $\mathbf{Q}$
- **设计动机**：不同 task vector 的奇异基 $\mathbf{U}_i, \mathbf{V}_i$ 方向各异，直接平均会导致方向扭曲。共享子空间提供统一的坐标系，在保证最小重构误差的同时对齐各 task vector 的几何方向

#### 3. 子空间聚合与反投影
- **功能**：在子空间中对对齐后的 task vector 做加权聚合：$\hat{\boldsymbol{\tau}} = \lambda \sum_{i=1}^N \mathbf{Q}^\top \tilde{\boldsymbol{\tau}}_i$，再反投影回原空间：$\boldsymbol{\theta}_{merge} = \boldsymbol{\theta}_0 + \mathbf{Q}\hat{\boldsymbol{\tau}}$
- **设计动机**：在共享子空间内做聚合，天然保证了合并结果与各 task vector 的方向一致性

### 损失函数 / 训练策略
DC-Merge 是**完全无训练（training-free）**的后处理方法——不需要任何额外数据或微调。超参仅有平滑指数 $\alpha$ 和子空间维度 $k$，均通过小规模验证集选取。

## 实验关键数据

### 主实验：Vision 任务（8 任务合并，ViT-B/32）

| 方法 | 平均准确率 (%) | 对比 Pretrained |
|------|---------------|-----------------|
| Pretrained | 48.3 | — |
| Task Arithmetic | 55.4 | +7.1 |
| TIES | 56.3 | +8.0 |
| DARE | 57.0 | +8.7 |
| Consensus | 57.8 | +9.5 |
| **DC-Merge** | **59.6** | **+11.3** |

### 消融实验

| 配置 | 平均准确率 (%) | 说明 |
|------|---------------|------|
| Full DC-Merge | 59.6 | 完整方法 |
| w/o SVD Smoothing | 57.8 | 去掉平滑后降 1.8% |
| w/o Subspace Projection | 56.5 | 去掉子空间投影后降 3.1% |
| w/o Both (baseline) | 55.4 | 等价于 Task Arithmetic |

### LoRA 合并实验

| 方法 | Vision-Language Avg (%) |
|------|------------------------|
| LoRA Arithmetic | 72.1 |
| DARE-LoRA | 73.5 |
| **DC-Merge-LoRA** | **75.8** |

### 关键发现
- 子空间投影是最关键的模块（贡献 3.1%），SVD 平滑贡献 1.8%，两者互补
- $\alpha \in [0.3, 0.6]$ 范围内效果稳定，对超参不敏感
- DC-Merge 在 LoRA 和 Full Fine-tuning 两种场景下都一致优于 baseline
- 方法在 Vision-Language（如 CLIP 微调）场景的提升更显著

## 亮点与洞察
- **从奇异空间方向一致性角度理解模型合并**——这个视角比"裁剪冲突参数"或"权重平均"更深刻，提供了理论基础
- **SVD 平滑的简洁优雅**——一个幂变换就能有效平衡能量分布，几乎没有计算开销
- **共享子空间投影的思路可推广**——不仅限于 task vector 合并，任何需要"在保持方向的前提下合并多个高维向量"的场景都可以借鉴
- **Training-free**——无需额外数据或计算，纯后处理，即插即用

## 局限与展望
- 子空间维度 $k$ 需要通过验证集选取，对于没有验证数据的场景不够方便
- SVD 分解在极大规模模型（如 70B+ LLM）上的计算开销可能较高
- 仅在 ViT 和 CLIP 系列上验证，未扩展到 decoder-only LLM
- 当任务数量很多（>20）时，共享子空间可能无法同时满足所有 task vector 的方向约束
- 未讨论 task vector 质量差异大（如某些任务微调很差）的鲁棒性

## 相关工作与启发
- **vs Task Arithmetic**：Task Arithmetic 简单平均 task vector，不考虑奇异空间结构。DC-Merge 在此基础上加入方向约束，提升 4+ 个百分点
- **vs TIES/DARE**：TIES/DARE 通过裁剪/随机丢弃处理冲突，属于"减法"策略。DC-Merge 是"变换"策略——不丢弃参数，而是变换到方向一致的空间
- **vs RegMean/Fisher Merging**：这些方法需要额外数据做正则化，DC-Merge 不需要
- **启发**：奇异值平滑的思路可以迁移到 LoRA 的初始化中——如果训练 LoRA 时就限制奇异值分布均匀，可能天然更适合后续合并

## 评分
- 新颖性: ⭐⭐⭐⭐ 从方向一致性角度统一理解模型合并是新颖的，但两个技术模块（SVD平滑、子空间投影）本身不新
- 实验充分度: ⭐⭐⭐⭐ Vision + VL 两大场景，Full FT + LoRA 两种设置，消融齐全
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，从分析到方案到实验逻辑连贯
- 价值: ⭐⭐⭐⭐ 解决了模型合并的核心问题，training-free 属性实用价值高

<!-- RELATED:START -->

## 相关论文

- [RobustMerge: Parameter-Efficient Model Merging for MLLMs with Direction Robustness](../../NeurIPS2025/multimodal_vlm/robustmerge_parameter-efficient_model_merging_for_mllms_with_direction_robustnes.md)
- [Label-Free Cross-Task LoRA Merging with Null-Space Compression](label-free_cross-task_lora_merging_with_null-space_compression.md)
- [Self-Consistency for LLM-Based Motion Trajectory Generation and Verification](self-consistency_for_llm-based_motion_trajectory_generation_and_verification.md)
- [Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging](../../ICML2025/multimodal_vlm/bring_reason_to_vision_understanding_perception_and_reasoning_through_model_merg.md)
- [Transferring Textual Preferences to Vision-Language Understanding through Model Merging](../../ACL2025/multimodal_vlm/transferring_textual_preferences_to_vision-language_understanding_through_model_.md)

<!-- RELATED:END -->
