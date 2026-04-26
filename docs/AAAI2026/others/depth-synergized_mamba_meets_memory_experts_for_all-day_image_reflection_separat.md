---
title: >-
  [论文解读] Depth-Synergized Mamba Meets Memory Experts for All-Day Image Reflection Separation
description: >-
  [AAAI 2026][图像反射分离] 提出 DMDNet，通过深度感知扫描策略（DAScan）引导 Mamba 关注显著结构，结合深度协同状态空间模型（DS-SSM）抑制模糊特征传播，并引入记忆专家补偿模块（MECM）利用跨图像历史知识，实现全天候（白天+夜间）的图像反射分离。
tags:
  - AAAI 2026
  - 图像反射分离
  - Mamba
  - 深度感知
  - 记忆专家
  - 夜间图像
---

# Depth-Synergized Mamba Meets Memory Experts for All-Day Image Reflection Separation

**会议**: AAAI 2026  
**arXiv**: [2601.00322](https://arxiv.org/abs/2601.00322)  
**代码**: [github.com/fashyon/DMDNet](https://github.com/fashyon/DMDNet)  
**领域**: 其他  
**关键词**: 图像反射分离, Mamba, 深度感知, 记忆专家, 夜间图像

## 一句话总结

提出 DMDNet，通过深度感知扫描策略（DAScan）引导 Mamba 关注显著结构，结合深度协同状态空间模型（DS-SSM）抑制模糊特征传播，并引入记忆专家补偿模块（MECM）利用跨图像历史知识，实现全天候（白天+夜间）的图像反射分离。

## 研究背景与动机

图像反射分离旨在将透过玻璃拍摄的混合图像 $\bm{I}$ 分解为透射层 $\bm{T}$（玻璃后的场景）和反射层 $\bm{R}$（玻璃表面的反射内容）。

**现有方法的核心挑战**：

**单图信息有限**：现有方法依赖单张图像的有限信息，当 T 和 R 对比度相似时容易将两层混淆

**夜间场景更加困难**：
   - 白天：充足自然光增强 T 同时抑制 R，两层对比度差异大
   - 夜间：人工光源随机分布，T 因全局光照不足而变暗，R 因玻璃表面的强光产生眩光和散射高光，导致 T 和 R 对比度接近

**额外硬件依赖**：多视角、偏振滤镜、红外相机等方法需要特殊设备

**人工干预需求**：语言提示和手动标注方法耗时费力

**关键insight**：深度估计可以提供无需额外硬件或人工干预的物理线索。对混合图像进行深度估计时，**深度图自然地突出 T 的连贯锐利结构**，同时抑制 R 的模糊透明叠加（如图1所示）。这意味着高近距离（proximity）值倾向于携带显著结构。

**Mamba 的两个局限**：

**结构连续性破坏**：固定的顺序扫描会切割透射场景中的连贯轮廓和纹理

**错误传播**：SSM 中早期扫描区域的状态持续影响后续区域，模糊特征的不确定性会扩散到整个图像

## 方法详解

### 整体框架

DMDNet 由三个分支组成：
- **编码分支**：使用 MuGI 块提取 T 和 R 的多尺度特征
- **深度语义调制分支（DSBranch）**：利用深度语义特征调制编码特征
- **解码分支**：通过 DMBlock（DSMamba + MECM + EFFN）执行 T 和 R 的分离

通道配置为 $C_1,...,C_5 = [48, 96, 192, 384, 768]$。

### 关键设计

1. **深度协同解耦 Mamba（DSMamba）**：

   包含深度感知扫描（DAScan）和深度协同状态空间模型（DS-SSM）两个子模块。

   **DAScan 为 T 和 R 定制了不同的扫描策略**：
   - **DA-RScan（用于 T）**：采用"大面积优先 + 近到远"方案。将近距离图分割为区域扫描图 $\bm{M}_{reg}$，从最大区域到最小区域扫描（大区域=更显著的语义），区域内按近到远顺序扫描。保持同一物体内像素的语义连续性。
   - **DA-GScan（用于 R）**：采用"全局近到远"方案，从全局最近像素到最远像素扫描。匹配 R 的稀疏和不连续分布特征。
   - 最后进行反向 DAScan 以补充结构线索。

   **DS-SSM 调制状态更新的灵敏度**：
   $$\bm{h}_t = \bm{A}\bm{h}_{t-1} + \bm{B}_{aware}\bm{x}_t, \quad \bm{y}_t = \bm{C}_{aware}\bm{h}_t + \bm{D}\bm{x}_t$$
   $$\bm{B}_{aware} = (1-\bm{\gamma}) \cdot \bm{B} + \bm{\gamma} \cdot \bm{B}_{depth}$$
   $$\bm{C}_{aware} = (1-\bm{\gamma}) \cdot \bm{C} + \bm{\gamma} \cdot \bm{C}_{depth}$$

   其中 $\bm{\gamma}$ 是 0-1 之间的权重图，由近距离图导出。在结构显著区域，较大的 γ 增强深度引导矩阵的影响，加速清晰结构的整合；在模糊区域，抑制干预以防止模糊特征传播。

   设计动机：DAScan 确保模型在建模早期就遇到显著结构，DS-SSM 协同地根据结构显著性调节状态演化——两者功能互补。

2. **记忆专家补偿模块（MECM）**：

   利用跨图像的历史知识动态激活最相关的专家提供定向补偿。包含**专家门控**（从 $N_{Exp}$ 个候选中选择 $N_{Exp}^K$ 个最相关专家）和**记忆专家**。

   每个记忆专家包含两个流：
   - **GPStream（全局模式交互流）**：
     - 全局模式调整：输入池化为全局表示 $\bm{I}_G$，与记忆库 $\bm{Mem} \in \mathbb{R}^{M \times C}$ 计算相似度，加权聚合记忆产生全局补偿
     - 记忆演化：每个样本选择最响应的记忆条目，通过加权乘法生成更新向量，以残差方式更新记忆库
   - **SCStream（空间上下文精化流）**：
     - 将记忆库重塑为卷积核与输入卷积生成空间相似度图
     - 对每个空间位置选择 Top-k 最相关记忆项
     - 加权求和：$\bm{F}_{comp}[b,hw,d] = \sum_{k=1}^{K} \bm{W}_A[b,k,hw] \cdot \bm{Mem}_K[b,k,hw,d]$

   设计动机：单图信息有限，通过记忆库积累跨图像的特征模式知识——例如纹理细节和结构轮廓专家用于 T，稀疏高光和模糊鬼影专家用于 R。

3. **NightIRS 数据集**：

   构建了1000张夜间反射图像三元组（I, T, R），使用不同厚度的玻璃和亚克力板引入反射，覆盖多种夜间照明条件（路灯、霓虹、照明建筑、低光自然环境），考虑不同的相机-玻璃距离和视角。还提供高分辨率版本（NightIRS-HR）。

### 损失函数 / 训练策略

- Adam 优化器，初始学习率 $10^{-4}$，分阶段衰减（第30 epoch → $5\times10^{-5}$，第50 epoch → $10^{-5}$）
- 训练60 epochs，batch size=1，裁剪为 352×352 patches
- 训练数据：7643 对 PASCAL VOC + 200 对 Nature + 89 对 Real
- MECM 设置：$N_{Exp}=4$ 个专家，选择 $N_{Exp}^K=2$ 个
- 单张 NVIDIA RTX 4090 GPU

## 实验关键数据

### 主实验

公共数据集（白天场景）上透射层的平均性能：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| BDN (ECCV'18) | 20.55 | 0.800 | 0.202 |
| ERRNet (CVPR'19) | 22.77 | 0.837 | 0.141 |
| DSRNet (ICCV'23) | 24.03 | 0.861 | 0.119 |
| DSIT (NIPS'24) | 26.11 | 0.883 | 0.105 |
| RDNet (CVPR'25) | 26.21 | 0.885 | 0.094 |
| **DMDNet (Ours)** | **26.27** | **0.889** | **0.093** |

NightIRS 数据集性能：

| 方法 | T-PSNR↑ | T-SSIM↑ | T-LPIPS↓ | R-PSNR↑ | 参数(M) | FLOPs(G) |
|------|---------|---------|----------|---------|---------|----------|
| DSIT (NIPS'24) | 24.61 | 0.827 | 0.168 | 27.18 | 131.76 | 74.18 |
| RDNet (CVPR'25) | 25.08 | 0.831 | 0.149 | 27.93 | 266.43 | 66.10 |
| **DMDNet (Ours)** | **25.24** | **0.832** | **0.144** | **28.37** | **87.22** | **39.33** |

### 消融实验

**DSMamba 组件消融（公共数据集透射层）**：

| T扫描 | R扫描 | SSM | SPE | PSNR↑ | SSIM↑ | LPIPS↓ |
|-------|-------|-----|-----|-------|-------|--------|
| DA-RScan | DA-GScan | DS-SSM | ✓ | **26.27** | **0.889** | **0.093** |
| DA-RScan | DA-RScan | DS-SSM | ✓ | 25.99 | 0.886 | 0.098 |
| DA-GScan | DA-GScan | DS-SSM | ✓ | 25.87 | 0.886 | 0.100 |
| DA-RScan | DA-GScan | DS-SSM | ✗ | 25.66 | 0.882 | 0.105 |
| DA-RScan | DA-GScan | Original | ✓ | 25.78 | 0.884 | 0.098 |
| Original | Original | DS-SSM | ✓ | 25.69 | 0.884 | 0.096 |

**Mamba 变体比较（公共数据集）**：

| 方法 | T-PSNR↑ | T-SSIM↑ | R-PSNR↑ | 参数(M) |
|------|---------|---------|---------|---------|
| MambaIR | 25.56 | 0.880 | 22.09 | 103.61 |
| VMambaIR | 25.89 | 0.884 | 22.06 | 83.76 |
| MambaIRv2 | 24.84 | 0.868 | 21.66 | 88.38 |
| **DSMamba (Ours)** | **26.27** | **0.889** | **22.31** | **87.22** |

### 关键发现

1. **T 用 DA-RScan + R 用 DA-GScan 是最优搭配**：分别匹配透射层的连贯结构和反射层的稀疏不连续特征
2. **DS-SSM 显著优于原始 SSM**：PSNR 提升 0.49dB（25.78→26.27）
3. **SPE 空间位置编码贡献明显**：PSNR 提升 0.61dB（25.66→26.27）
4. **DSMamba 全面超越 MambaIR/VMambaIR/MambaIRv2**：无论 T 还是 R 恢复质量
5. **DMDNet 在夜间场景优势更突出**：参数量仅为 RDNet 的 1/3，FLOPs 仅 60%
6. **可视化验证**：$\bm{B}_{depth}$ 和 $\bm{C}_{depth}$ 确实在显著结构区域放大激活，在模糊区域抑制

## 亮点与洞察

- **深度估计作为免费物理线索**：深度估计模型能"看穿"反射遮挡，提取底层结构——这一观察非常精妙，是全文核心 insight
- **为 T 和 R 定制不同扫描策略**：不是一刀切，而是根据两层的不同特性（连贯 vs 稀疏）设计不同的扫描顺序
- **DS-SSM 与 DAScan 的协同设计**：DAScan 确保先看到好的结构，DS-SSM 确保好的结构被放大、坏的被抑制——逻辑链完整
- **记忆机制弥补单图局限**：通过跨图像知识积累，为单图推理提供"经验"补偿
- **NightIRS 数据集填补空白**：首个专门用于夜间反射分离的数据集

## 局限与展望

1. **依赖预训练深度估计模型**：深度估计本身的误差会传播到反射分离
2. **记忆库大小固定**：M 的选择需要平衡存储和性能
3. **计算开销**：虽然比 RDNet 轻量，但相比一些轻量级方法仍有 87M 参数
4. **NightIRS 数据集规模有限**：1000 张三元组可能不足以涵盖所有夜间场景
5. **未考虑视频场景**：利用时序一致性可能进一步提升性能

## 相关工作与启发

DMDNet 巧妙地将深度估计、Mamba 状态空间模型和记忆增强 MoE 架构结合。深度感知扫描策略的思想可推广到其他需要结构感知的序列建模任务。记忆专家机制对其他单图恢复任务（如去雾、去雨）也有启发。全天候的设计思路值得图像增强领域学习。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (深度引导Mamba扫描和DS-SSM的协同设计非常新颖)
- 实验充分度: ⭐⭐⭐⭐⭐ (11个对比方法, 完整消融, 新数据集)
- 写作质量: ⭐⭐⭐⭐ (结构清晰, 公式规范, 动机阐述充分)
- 价值: ⭐⭐⭐⭐ (首次解决夜间反射分离, 方法泛化潜力大)

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Depth-Supervised Fusion Network for Seamless-Free Image Stitching](../../NeurIPS2025/others/depth-supervised_fusion_network_for_seamless-free_image_stitching.md)
- [\[ACL 2025\] Meta-Reflection: A Feedback-Free Reflection Learning Framework](../../ACL2025/others/meta-reflection_a_feedback-free_reflection_learning_framework.md)
- [\[AAAI 2026\] CAE: Hierarchical Semantic Alignment for Image Clustering](hierarchical_semantic_alignment_for_image_clustering.md)
- [\[ACL 2025\] S3 - Semantic Signal Separation](../../ACL2025/others/s3_-_semantic_signal_separation.md)
- [\[ACL 2025\] The Hidden Attention of Mamba Models](../../ACL2025/others/the_hidden_attention_of_mamba_models.md)

<!-- RELATED:END -->
