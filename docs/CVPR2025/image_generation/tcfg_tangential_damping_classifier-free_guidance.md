---
title: >-
  [论文解读] TCFG: Tangential Damping Classifier-Free Guidance
description: >-
  [CVPR 2025][图像生成][分类器自由引导] 从数据流形几何视角出发，通过 SVD 分解去除无条件 score 中与条件 score 不对齐的切向分量，以极低计算开销改善 CFG 采样质量，在 SD1.5/SDXL/SD3/DiT 上均降低 FID。
tags:
  - CVPR 2025
  - 图像生成
  - 分类器自由引导
  - 流形假设
  - 奇异值分解
  - 扩散模型采样
  - 切向分量
---

# TCFG: Tangential Damping Classifier-Free Guidance

**会议**: CVPR 2025  
**arXiv**: [2503.18137](https://arxiv.org/abs/2503.18137)  
**代码**: 无  
**领域**: 扩散模型  
**关键词**: 分类器自由引导, 流形假设, 奇异值分解, 扩散模型采样, 切向分量

## 一句话总结
从数据流形几何视角出发，通过 SVD 分解去除无条件 score 中与条件 score 不对齐的切向分量，以极低计算开销改善 CFG 采样质量，在 SD1.5/SDXL/SD3/DiT 上均降低 FID。

## 研究背景与动机
1. **领域现状**：Classifier-Free Guidance (CFG) 是当前文本到图像扩散模型中最核心的采样策略，通过组合条件 score 和无条件 score 实现高质量条件生成。
2. **现有痛点**：CFG 的引导公式 $\tilde{s}_\theta = s_\theta(z_t) + \omega(s_\theta(z_t, y) - s_\theta(z_t))$ 中，无条件 score 估计的是所有样本的通用过中间流形，其切向分量可能与条件 score 的流形方向不一致，导致采样轨迹偏离目标流形，产生过曝、异常形状等伪影。
3. **核心矛盾**：无条件 score 需要同时服务所有条件，因此其切向分量是"平均化"的，与特定条件 $y$ 对应的流形切空间存在系统性偏差。
4. **本文目标**：在不改变模型权重、不增加推理成本的前提下，消除 CFG 中无条件 score 的流形不对齐问题。
5. **切入角度**：利用扩散模型 score 函数与数据流形法空间/切空间的理论联系——高奇异值对应法向分量（对齐好），低奇异值对应切向分量（不对齐）。
6. **核心 idea**：对条件和无条件 score 做 SVD，保留最大奇异值对应的法向分量，丢弃其余切向分量。

## 方法详解

### 整体框架
TCFG 是一个即插即用的采样策略，替换标准 CFG 中的 score 组合方式。每个采样步中：获取条件 score $s_\theta(z_t, y)$ 和无条件 score $s_\theta(z_t)$ → 拼成矩阵 $A$ → SVD 分解 → 将无条件 score 投影到最大奇异值对应的方向上（法向分量）→ 用修正后的无条件 score 执行 CFG。

### 关键设计

1. **中间流形假设与 SVD 分析**

    - 功能：为方法提供理论基础，证明 score 函数中存在法向/切向分量的几何结构
    - 核心思路：扩展已有定理（score 在 $t \to 0$ 时趋近数据流形法向量），假设在所有时间步 $t \in (0,1)$ 都存在中间流形 $\mathcal{M}_t$，使得 score 函数是其法空间的元素。通过对 SD1.5 收集 17000 个 score 样本做 SVD，观察到在所有时间步都存在 singular value 的明显 gap，验证了中间流形的存在。进一步发现高奇异值对应的奇异向量在条件/无条件 score 之间余弦相似度高（法向分量对齐），低奇异值对应方向相似度低（切向分量不对齐）。
    - 设计动机：说明低奇异值分量的不对齐是 CFG 产生伪影的根源

2. **切向分量丢弃（Tangential Damping）**

    - 功能：去除无条件 score 中与条件 score 不对齐的切向分量
    - 核心思路：将无条件和条件 score 拼成矩阵 $A = [s_\theta(z_t), s_\theta(z_t, y)]$，对 $A$ 做 SVD 得到奇异值 $\sigma_i$ 和右奇异向量 $v_i$。仅保留最大奇异值对应的方向 $v_1$（法向分量），将无条件 score 投影到该方向：$\hat{s}_\theta(z_t) = s_\theta(z_t) \cdot V^T \cdot [v_1, 0]$。然后执行修正版 CFG：$\hat{s}_\theta(z_t, y) = \hat{s}_\theta(z_t) + \omega(s_\theta(z_t, y) - \hat{s}_\theta(z_t))$。由于只需对一个 $2 \times D$ 矩阵做 SVD，计算开销可忽略不计。
    - 设计动机：法向分量负责"拉向流形"（条件和无条件 score 一致），切向分量负责"沿流形移动"（两者不一致），丢弃切向分量减少不对齐干扰

3. **单样本 SVD 的充分性**

    - 功能：验证方法的实用性——不需要收集大量样本就能工作
    - 核心思路：在玩具实验（two moons 数据集）中对比"使用所有样本做 SVD"和"仅用单个样本对做 SVD"的结果，发现两者几乎一致。这使得方法可以在每个采样步仅利用当前样本的两个 score 做 SVD，不增加额外计算。
    - 设计动机：确保方法不引入批处理依赖或额外计算开销

### 损失函数 / 训练策略
TCFG 不需要训练或微调，它是一个纯推理时的采样策略修改。使用预训练模型的原始权重即可。

## 实验关键数据

### 主实验

| 模型 | 指标 | 原始CFG | +TCFG | 提升 |
|------|------|---------|-------|------|
| SD v1.5 | FID↓ | 13.26 | **13.12** | -0.14 |
| SDXL | FID↓ | 13.36 | **12.65** | -0.71 |
| SD v3 | FID↓ | 16.66 | **13.74** | -2.92 |
| DiT (ImageNet) | FID↓ | 32.67 | **29.5** | -3.17 |
| DiT | sFID↓ | 17.92 | **13.27** | -4.65 |
| DiT | Recall↑ | 0.13 | **0.19** | +46% |

### 消融/兼容实验

| 方法 | FID↓ | 说明 |
|------|------|------|
| SAG | 13.53 | - |
| SAG + TCFG | **11.48** | 与 SAG 叠加效果显著 |
| PAG | 14.45 | - |
| PAG + TCFG | **11.87** | 与 PAG 兼容 |
| CFG++ | 13.97 | - |
| CFG++ + TCFG | **13.44** | 与 CFG++ 兼容 |

### 关键发现
- 模型越强（SD3 > SDXL > SD1.5），TCFG 的 FID 提升越大。推测更强模型的流形结构更清晰，切向分量丢弃效果更好
- SD3 基于 Rectified Flow 而非标准扩散，TCFG 同样适用，说明方法具有框架无关性
- TCFG 显著缓解了过曝偏差问题，这是 CFG 的一个已知顽疾
- CLIP Score 几乎不变，说明去除切向分量不影响文本对齐

## 亮点与洞察
- **极简但有效**：整个方法只需在每步做一次 $2 \times D$ 矩阵的 SVD（$D$ 为 latent 维度），零额外训练，零参数调整，却能持续改善 FID。这种"几何视角+极简操作"的组合非常优雅
- **理论与实践的衔接**：从流形假设 → SVD gap 观察 → 切向分量不对齐 → 丢弃操作，逻辑链完整
- **通用性强**：适用于扩散模型、Rectified Flow、文本引导、类别引导，且可与 SAG/PAG/CFG++ 等叠加使用

## 局限与展望
- 理论仍基于假设（中间流形 $\mathcal{M}_t$ 的存在），严格数学证明尚不完备
- 仅丢弃最小奇异值方向（保留 $v_1$），对更高维流形可能需要保留更多方向
- 缺少对 user study 的主观质量评估
- 未来可探索自适应选择保留多少奇异方向，或根据时间步动态调整

## 相关工作与启发
- **vs SAG/PAG**: SAG 用 self-attention map、PAG 用 identity attention map 来增强 CFG，是从注意力机制角度优化；本文从 score 的流形几何角度优化，两者正交且可叠加
- **vs CFG++**: CFG++ 修改了 CFG 的计算公式实现更好的采样，本文修改的是输入给 CFG 的 score 本身，两者也可叠加
- **vs ICG**: ICG 用随机文本嵌入替代空文本，仍是在条件表示层面操作；本文直接在 score 空间操作

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从流形几何视角理解 CFG 不对齐问题，提出极简解法，思路新颖
- 实验充分度: ⭐⭐⭐⭐ 覆盖多个模型和框架，FID-CLIP 曲线、兼容性实验详尽，但缺少 user study
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，图示直观，从玩具实验到实际模型层层递进
- 价值: ⭐⭐⭐⭐⭐ 即插即用、零成本、广泛兼容，实用价值极高

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Classifier-Free Guidance inside the Attraction Basin May Cause Memorization](classifier-free_guidance_inside_the_attraction_basin_may_cause_memorization.md)
- [\[AAAI 2026\] Studying Classifier(-Free) Guidance From A Classifier-Centric Perspective](../../AAAI2026/image_generation/studying_classifier-free_guidance_from_a_classifier-centric_perspective.md)
- [\[ICCV 2025\] TeEFusion: Blending Text Embeddings to Distill Classifier-Free Guidance](../../ICCV2025/image_generation/teefusion_blending_text_embeddings_to_distill_classifier-free_guidance.md)
- [\[NeurIPS 2025\] Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations](../../NeurIPS2025/image_generation/towards_a_golden_classifier-free_guidance_path_via_foresight_fixed_point_iterati.md)
- [\[AAAI 2026\] DICE: Distilling Classifier-Free Guidance into Text Embeddings](../../AAAI2026/image_generation/dice_distilling_classifier-free_guidance_into_text_embedding.md)

<!-- RELATED:END -->
