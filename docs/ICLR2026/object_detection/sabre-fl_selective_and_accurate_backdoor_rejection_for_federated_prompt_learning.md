---
title: >-
  [论文解读] SABRE-FL: Selective and Accurate Backdoor Rejection for Federated Prompt Learning
description: >-
  [ICLR2026][目标检测][联邦学习] 首次研究联邦 Prompt Learning 场景下的后门攻击威胁，并提出 SABRE-FL——一种基于 embedding 空间异常检测的轻量级服务器端防御方法，无需访问客户端原始数据即可有效过滤中毒 prompt 更新。
tags:
  - ICLR2026
  - 目标检测
  - 联邦学习
  - 提示学习
  - Backdoor Attack
  - CLIP
  - Anomaly Detection
---

# SABRE-FL: Selective and Accurate Backdoor Rejection for Federated Prompt Learning

**会议**: ICLR2026  
**arXiv**: [2506.22506](https://arxiv.org/abs/2506.22506)  
**代码**: 待发布  
**领域**: object_detection  
**关键词**: federated learning, Prompt Learning, Backdoor Attack, CLIP, Anomaly Detection

## 一句话总结
首次研究联邦 Prompt Learning 场景下的后门攻击威胁，并提出 SABRE-FL——一种基于 embedding 空间异常检测的轻量级服务器端防御方法，无需访问客户端原始数据即可有效过滤中毒 prompt 更新。

## 背景与动机
- **联邦 Prompt Learning (FPL)** 是近年兴起的范式：客户端仅优化轻量的 prompt 向量（冻结 CLIP 骨干），再将 prompt 上传至服务器聚合，大幅降低通信和计算开销
- 联邦学习天然面临后门攻击风险——恶意客户端通过注入触发器（trigger）污染本地数据，使全局模型在推理时对含触发器的输入产生定向误分类
- 已有后门攻击研究集中在传统单模态 FL（全参数微调），FPL 场景下攻击面仅限 prompt 向量且图像编码器冻结，攻击可行性和防御策略均属空白
- 本文的动机是双重的：**(1)** 验证 FPL 是否真的脆弱；**(2)** 设计针对性防御

## 核心问题
1. **攻击层面**：在 FPL 中，恶意客户端能否通过可学习的 imperceptible noise trigger 成功植入后门，使全局 prompt learner 在推理时对触发样本误分类，同时不影响干净样本精度？
2. **防御层面**：如何在服务器端检测并过滤中毒 prompt 更新，且不依赖客户端原始数据、标签或下游任务信息？

## 方法详解

### 攻击设计
- **威胁模型**：标准 FL 设置，$N$ 个客户端中 $m/N$（默认 25%）被攻击者控制；恶意客户端可修改本地训练数据、添加可学习触发器并重标记为目标类别（dirty-label attack）
- **触发器优化**：恶意客户端在本地训练时联合优化 prompt 向量和触发器 $t$，使含触发图像 $x^\star = x \oplus t$ 的 CLIP 图像 embedding 偏向目标类别文本 embedding：
$$\cos(f_{\text{img}}(x^\star), f_{\text{text}}(y_{\text{target}})) > \cos(f_{\text{img}}(x^\star), f_{\text{text}}(y)), \quad \forall y \neq y_{\text{target}}$$
- 触发器视觉上不可感知，但在 CLIP embedding 空间中产生一致性偏移

### SABRE-FL 防御框架
- **核心洞察**：后门触发器虽然在像素层面不可见，但在 CLIP embedding 空间中会留下可检测的统计"指纹"——中毒样本的 embedding 与干净样本之间存在一致的分离间距 $\|z - z^\star\|_2 > \epsilon$
- **离线训练检测器**：使用与下游任务无关的辅助数据集 Caltech-101，生成干净/中毒 embedding 对，训练二分类器 $D: \mathbb{R}^d \to \{0, 1\}$
- **在线过滤**：每轮聚合时，服务器对每个客户端 $C_k$ 提交的 embedding 集合 $\{z_j^k\}$ 计算平均检测分数 $S_k = \frac{1}{n_k} \sum_j D(z_j^k)$，采用 rank-based 策略排除分数最高的 $m$ 个客户端
- **隐私保护**：仅需客户端共享 CLIP 编码后的 embedding（冻结编码器产生的压缩向量），不需要原始图像、标签或梯度

### 算法流程
1. 预训练阶段：在辅助数据上构建干净/中毒 embedding 数据集，训练检测器 $D$
2. 每轮 FL：服务器分发全局 prompt → 客户端本地训练 → 客户端回传 prompt 与 embedding → 服务器用 $D$ 计算每个客户端的异常分数 → 过滤 top-$m$ 可疑客户端 → 聚合剩余 prompt

## 实验关键数据

### 攻击效果（无防御 / FedAvg）

| 数据集 | 无攻击 CA | 攻击下 CA | 后门 BA |
|---------|----------|----------|---------|
| Flowers | 80.9 | 77.9 | 41.7 |
| Pets | 94.5 | 94.2 | 16.3 |
| DTD | 65.2 | 65.6 | 34.8 |
| Aircraft | 32.3 | 32.8 | **93.9** |
| Food101 | 90.7 | 90.0 | 20.6 |

攻击在保持干净精度的同时成功注入后门，Aircraft 上 BA 高达 93.9%。

### 防御对比（五个数据集上 BA，越低越好）

| 防御方法 | Flowers | Pets | DTD | Aircraft | Food101 |
|----------|---------|------|-----|----------|---------|
| No Defense | 41.7 | 16.3 | 34.8 | 93.9 | 20.6 |
| Trimmed Mean | 12.3 | 5.6 | 31.0 | 83.1 | 6.4 |
| Median | 10.4 | 5.3 | 28.1 | 79.4 | 5.5 |
| Norm Bounding | 22.0 | 22.5 | 37.5 | 86.2 | 17.2 |
| FLAME | 3.8 | 7.8 | 8.7 | 16.4 | 3.2 |
| **SABRE-FL** | **1.1** | **4.4** | **6.8** | **7.6** | **1.9** |

SABRE-FL 在全部五个数据集上 BA 最低，且干净精度与无防御基线持平甚至更优。

### 消融实验
- **Prompt shot 数量**：随 shot 增加（2→16），无防御时 BA 显著上升（Aircraft 和 Food101 超过 85%）；启用 SABRE-FL 后 BA 始终低于 5%
- **恶意客户端比例**：25% 恶意时 Aircraft BA 达 93.9%；50%+ 时多数数据集 BA 超过 80%；干净精度全程几乎不受影响

## 亮点
- **首次研究 FPL 后门安全**：填补了多模态联邦 prompt 学习安全性的研究空白，既建立了攻击基线也提出了防御
- **防御设计优雅**：利用"攻击的成功信号即是防御的检测信号"这一双面性——trigger 能欺骗分类器恰好说明 embedding 偏移可被检测
- **零数据依赖**：检测器在 OOD 辅助集上离线训练，无需客户端数据/标签/任务信息，部署代价极低
- **跨域泛化强**：Caltech-101 上训练的检测器在 Flowers、DTD、Aircraft、Food101、Pets 五个不同域上均有效

## 局限与展望
- **需要知道恶意客户端上界**：rank-based 过滤假设已知恶意客户端数量上界 $m$，实际部署中此信息往往不可得
- **仅测试了 noise trigger**：攻击类型单一（可学习噪声触发器），未验证对 patch-based trigger、语义 trigger 等其他后门攻击的防御效果
- **客户端需额外上传 embedding**：相比纯 prompt 聚合的 FPL，SABRE-FL 要求客户端额外传输图像 embedding，增加了通信和隐私暴露面
- **数据集规模偏小**：五个 fine-grained 数据集均为小规模，未在 ImageNet 等大规模数据上验证
- **自适应攻击缺失**：未考虑攻击者已知防御机制时的自适应攻击场景

## 与相关工作的对比

| 维度 | BadCLIP (CVPR'24) | A3FL / IBA (传统 FL 后门) | SABRE-FL |
|------|-------------------|--------------------------|----------|
| 场景 | 集中式 prompt learning | 单模态 FL（全参数） | 联邦 prompt learning |
| 攻击面 | 全部训练数据 | 模型参数 + 数据 | 仅 prompt 向量 |
| 防御方式 | 无专门防御 | 鲁棒聚合（Trimmed Mean 等） | embedding 空间异常检测 |
| 数据依赖 | — | 需要验证集 | OOD 辅助集，无需客户端数据 |

## 启发与关联
- 核心思路"在 representation space 而非 pixel/parameter space 检测后门"具有通用性，可推广到其他 foundation model 微调场景（如 LoRA adapter 的联邦学习）
- 冻结编码器 + 可学习 prompt 的架构使得 embedding 偏移成为后门的必要条件，这一结构性约束是设计高效防御的关键
- 联邦 prompt learning 的安全性研究仍处早期，自适应攻击、多目标攻击、clean-label attack 等方向值得继续探索
- 检测器使用 OOD 数据训练即可泛化的特性说明后门 embedding 偏移是一种攻击的结构性副产品，这为其他模态（NLP、audio）的后门防御提供了新思路
- 恶意客户端比例超过 50% 时 BA 接近 100%，凸显了 FPL 中 Sybil attack 的潜在威胁

## 评分
- 新颖性: ⭐⭐⭐⭐ (首次系统研究 FPL 后门攻防，切入点好)
- 实验充分度: ⭐⭐⭐⭐ (五数据集+四基线+消融，但缺大规模验证和自适应攻击)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，理论与实验结合紧密)
- 价值: ⭐⭐⭐⭐ (填补了重要研究空白，防御方法实用)

<!-- RELATED:START -->

## 相关论文

- [Advancing Textual Prompt Learning with Anchored Attributes](../../ICCV2025/object_detection/advancing_textual_prompt_learning_with_anchored_attributes.md)
- [Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack](../../AAAI2026/object_detection/tackling_resource-constrained_and_data-heterogeneity_in_federated_learning_with_.md)
- [Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training](breaking_scale_anchoring_frequency_representation_learning_for_accurate_high-res.md)
- [Towards Effective, Stealthy, and Persistent Backdoor Attacks Targeting Graph Foundation Models](../../AAAI2026/object_detection/towards_effective_stealthy_and_persistent_backdoor_attacks_targeting_graph_found.md)
- [Resource Efficient Sleep Staging via Multi-Level Masking and Prompt Learning](../../AAAI2026/object_detection/resource_efficient_sleep_staging_via_multi-level_masking_and_prompt_learning.md)

<!-- RELATED:END -->
