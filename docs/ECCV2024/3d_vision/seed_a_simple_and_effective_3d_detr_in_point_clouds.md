---
title: >-
  [论文解读] SEED: A Simple and Effective 3D DETR in Point Clouds
description: >-
  [ECCV 2024][3D视觉][3D目标检测] SEED 提出了一种简洁高效的 3D DETR 检测器，通过双重查询选择（DQS）模块以粗到精方式获取高质量查询，结合可变形网格注意力（DGA）模块利用 3D 物体的几何结构信息实现灵活的查询交互，在 Waymo 和 nuScenes 上达到新 SOTA。
tags:
  - ECCV 2024
  - 3D视觉
  - 3D目标检测
  - DETR
  - 点云
  - Query选择
  - 可变形注意力
---

# SEED: A Simple and Effective 3D DETR in Point Clouds

**会议**: ECCV 2024  
**arXiv**: [2407.10749](https://arxiv.org/abs/2407.10749)  
**代码**: https://github.com/happinesslz/SEED (有)  
**领域**: 3D视觉  
**关键词**: 3D目标检测, DETR, 点云, Query选择, 可变形注意力

## 一句话总结
SEED 提出了一种简洁高效的 3D DETR 检测器，通过双重查询选择（DQS）模块以粗到精方式获取高质量查询，结合可变形网格注意力（DGA）模块利用 3D 物体的几何结构信息实现灵活的查询交互，在 Waymo 和 nuScenes 上达到新 SOTA。

## 研究背景与动机
DETR 范式在 2D 检测领域已经成为主流方法，通过将目标检测建模为集合预测问题，优雅地消除了手工锚框和 NMS 后处理。然而，在 3D 点云目标检测中，基于 DETR 的方法尚未展现出与 2D 领域类似的出色性能，仍然落后于最先进的非 DETR 3D 检测器。

**两大核心挑战**：
1. **查询选择困难** — 点云高度稀疏且分布不均匀，如何获取合适的物体查询是难题。现有方法（如 TransFusion、ConQueR）仅通过单步 Top-N 选择，未考虑选出查询用于框定位的质量
2. **查询交互不充分** — 如何利用点云丰富的几何结构信息进行有效的查询交互尚未被充分探索。2D 图像中物体可能占据整张图，需要全局感受野；但 3D 物体通常只占据很小的局部区域，局部注意力即可满足

**核心思路**：设计 "粗到精" 的双重查询选择来确保高质量查询，再用网格化的可变形注意力来充分利用 3D 物体的几何信息进行查询交互。

## 方法详解

### 整体框架
1. 点云输入经典的 voxel-based 3D backbone 提取体素特征，转为 BEV 特征
2. BEV 特征添加位置编码后展平，送入 DQS 模块选择高质量查询
3. 选出的查询送入 6 层 SEED Decoder Layer，进行自注意力（inter-query 交互）+ DGA（query-BEV 交互），最终输出检测结果

### 关键设计
1. **Dual Query Selection (DQS) — 双重查询选择**：
   - **做什么**：以粗到精方式从 BEV 特征中挑选出高质量查询
   - **前景查询选择（粗筛）**：用二分类预测器区分前景/背景，按比例 $r = 0.3$ 从置信度最高的 BEV 特征中保留 $N_c = H \times W \times r$ 个粗查询。目标是保证高召回率
   - **质量查询选择（精筛）**：粗查询经过一个 SEED Decoder Layer 增强后，通过三个 FFN 分支预测分类分数 $S_c$、定位分数 $S_l$（预测 3D IoU）和回归框 $B_c$。质量分数计算为：

     $$S_q^i = \begin{cases} (S_c^i)^{1-\beta} \cdot (S_l^i)^{\beta}, & \text{if } S_c^i > \tau \\ S_c^i, & \text{otherwise} \end{cases}$$

     从中选出 $N_f = 1000$ 个最高质量分数的查询，拼接框信息后通过 MLP 生成几何感知的高质量查询
   - **设计动机**：一步选择无法同时保证召回和质量；两步策略先保证覆盖再保证精度，且引入定位分数可以过滤高置信但定位差的查询

2. **Deformable Grid Attention (DGA) — 可变形网格注意力**：
   - **做什么**：在 SEED Decoder Layer 中替代标准 cross attention，利用 3D 物体几何信息进行有效的查询-BEV 交互
   - **核心思路**：
     - 将估计的 proposal box 均匀划分为 $k \times k$（默认 5×5）个网格点作为参考点
     - 通过查询预测偏移量 $\Delta g$，加到网格点上得到最终采样位置
     - 采样 BEV 特征（双线性插值），乘以预测的注意力权重
   - **DGA 公式**：$\text{DGA}(g, F_{bev}) = \sum_{j=1}^{K} A_j \cdot \phi(F_{bev}(g_j + \Delta g_j))$
   - **设计动机**：结合了 box attention（利用几何信息）和 deformable attention（灵活感受野）两者的优势。纯 box attention 依赖框精度；纯 deformable attention 不利用几何结构；DGA 以网格为基础加偏移，兼得两者优势

3. **Quality-aware Hungarian Matching (QHM)**：
   - **做什么**：改进 DETR 的匈牙利匹配策略
   - **核心思路**：在计算分类代价时，用质量分数 $S_f$（融合了分类和定位分数）替代传统的分类分数，使得匹配更倾向于定位质量高的 proposal
   - **匹配代价**：$\mathcal{C}_{match} = \lambda_{cls} \mathcal{C}_{cls} + \lambda_{reg} \mathcal{C}_{reg} + \lambda_{giou} \mathcal{C}_{giou}$

### 损失函数 / 训练策略
- 最终 loss = DETR head loss + DQS loss
- DQS loss：分类分数用 BCE loss，定位分数用 IoU loss，回归用 Smooth-L1 loss
- 使用 AdamW 优化器，初始学习率 0.001
- WOD 上用 20% 数据训练 24 epochs，100% 数据训练 12 epochs
- 8 块 V100 GPU，batch size 24
- 使用 fade strategy（最后一个 epoch 关闭增强）和 query contrast strategy

## 实验关键数据

### 主实验

**Waymo Open Dataset (val, 100% 数据, 单帧)**：

| 方法 | 类型 | Vehicle APH(L2) | Ped APH(L2) | Cyclist APH(L2) | mAPH(L2) |
|------|------|-----------------|-------------|-----------------|----------|
| ConQueR | DETR | 68.2 | 64.7 | 70.1 | 67.7 |
| FocalFormer3D | DETR | 67.6 | 66.8 | 72.6 | 69.0 |
| DSVT-Voxel | 非DETR | 71.0 | 71.5 | 73.7 | 72.1 |
| **SEED-S** | DETR | 69.7 | 68.1 | 74.5 | **70.8** |
| **SEED-B** | DETR | 71.4 | 70.8 | 76.1 | **72.8** |
| **SEED-L** | DETR | **71.5** | **71.8** | **77.3** | **73.5** |

**Waymo (val, 多帧)**：

| 方法 | 帧数 | mAPH(L2) |
|------|------|----------|
| DSVT-Voxel | 3 | 75.0 |
| SEED-B | 3 | 75.8 |
| SEED-L | 3 | **76.1** |

**nuScenes (val)**：

| 方法 | NDS | mAP |
|------|-----|-----|
| TransFusion-L | 70.1 | 65.1 |
| Uni3DETR | 68.5 | 61.7 |
| **SEED** | **71.2** | **66.6** |

### 消融实验

**各组件贡献（Waymo val, 20% 数据）**：

| 配置 | DQS | DGA | mAPH(L2) | 提升 |
|------|-----|-----|----------|------|
| Baseline | ✗ | ✗ | 64.6 | - |
| +DQS | ✓ | ✗ | 67.4 | +2.8 |
| +DGA | ✗ | ✓ | 66.4 | +1.8 |
| SEED | ✓ | ✓ | **68.2** | **+3.6** |

**查询选择策略对比**：

| 策略 | mAPH(L2) | 说明 |
|------|----------|------|
| Learnable (CMT) | 66.6 | 可学习查询 |
| Heatmap (TransFusion) | 65.0 | 最差，query 来自 BEV 本身 |
| Top-N (ConQueR) | 66.8 | 单步选择 |
| **DQS (Ours)** | **68.2** | 粗到精两步选择 |

**注意力机制对比**：

| 注意力类型 | mAPH(L2) | 说明 |
|-----------|----------|------|
| Global Attention | OOM | GPU 内存不足 |
| Deformable Attn | 67.5 | 灵活但不利用几何信息 |
| Box Attention | 67.5 | 利用几何但不够灵活 |
| **DGA (Ours)** | **68.2** | 兼得灵活性和几何信息 |

### 关键发现
- DQS 贡献最大（+2.8 mAPH），验证了粗到精查询选择的有效性
- DGA 比 deformable attention 和 box attention 分别高 0.7 和 0.7 mAPH/L2
- Heatmap-based 选择最差，因为 query 直接从 BEV 特征取不利于 decoder 堆叠
- QHM 对 Vehicle 的提升大于 Pedestrian 和 Cyclist，因为大型刚性物体的定位分数更易估计
- SEED-S 在速度上也优于现有 DETR 方法（13.5 FPS on RTX 3090）

## 亮点与洞察
1. **粗到精的双重查询选择** — 先保证召回再保证精度的策略设计合理，且引入定位分数评估查询质量是个新颖且有效的做法
2. **网格化可变形注意力** — 在 proposal box 内均匀划分网格再加偏移的设计，优雅地融合了几何先验和感受野灵活性
3. **首个超越非 DETR SOTA 的 3D DETR 方法** — SEED-L 超过 DSVT-Voxel(L2 mAPH 73.5 vs 72.1)，证明 DETR 范式在 3D 检测中的潜力
4. **提供三个版本（S/B/L）** — 方便在速度和精度之间做权衡

## 局限性 / 可改进方向
- 远距离小物体检测能力不足，这类物体在 2D 相机图像中更容易识别 — 未来可融合多模态
- 3D backbone 增强（如 DSVT）与 SEED 正交，结合可能进一步提升
- 定位分数的预测精度本身受限于初始 proposal 质量
- DQS 的两步选择增加了推理延迟，虽然整体速度仍可接受

## 相关工作与启发
- 对 2D DETR 方法（DINO、DN-DETR 等）中查询选择策略的分析很有价值
- DGA 的设计思路可推广到其他需要利用几何先验的注意力场景
- 质量感知匈牙利匹配的思路来自 AFDetv2、PillarNet 等非 DETR 方法中的 IoU 校正技术
- 为 DETR 范式在 3D 检测中的发展提供了一个强 baseline

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐
