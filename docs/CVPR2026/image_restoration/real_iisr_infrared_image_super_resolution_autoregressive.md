---
title: >-
  [论文解读] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset
description: >-
  [CVPR 2026][图像恢复][红外图像超分辨率] 提出 Real-IISR 统一自回归框架，通过热-结构引导模块、条件自适应码本和热序一致性损失解决真实红外图像超分辨率的特有挑战，并构建了 FLIR-IISR 数据集（1457 对真实 LR-HR 红外图像）。
tags:
  - CVPR 2026
  - 图像恢复
  - 红外图像超分辨率
  - 视觉自回归
  - 热结构引导
  - 条件自适应码本
  - 热序一致性
---

# Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset

**会议**: CVPR 2026  
**arXiv**: [2603.04745](https://arxiv.org/abs/2603.04745)  
**代码**: [https://github.com/JZD151/Real-IISR](https://github.com/JZD151/Real-IISR)  
**领域**: 图像修复 / 红外超分辨率  
**关键词**: 红外图像超分辨率, 视觉自回归, 热结构引导, 条件自适应码本, 热序一致性

## 一句话总结

提出 Real-IISR 统一自回归框架，通过热-结构引导模块、条件自适应码本和热序一致性损失解决真实红外图像超分辨率的特有挑战，并构建了 FLIR-IISR 数据集（1457 对真实 LR-HR 红外图像）。

## 研究背景与动机

1. **领域现状**：可见光图像超分辨率已有显著进展，但红外成像因更长波长、更弱大气散射导致空间变化模糊、不稳定热边界和温度相关辐射漂移等特有退化。
2. **现有痛点**：现有 IISR 方法在模拟数据集（下采样的 IVIF 数据集）上训练，无法捕获真实红外退化。扩散模型的随机采样和缺乏红外退化先验限制其在 IISR 中的适用性。
3. **核心矛盾**：(1) 缺乏真实红外退化数据集；(2) 缺乏红外感知的退化建模——热辐射强度与结构边缘不对应，且非均匀退化引入量化偏差。
4. **本文目标**：同时解决真实 IISR 数据集和方法两个根本缺口。
5. **切入角度**：利用红外成像的温度-亮度单调性作为物理约束。
6. **核心 idea**：热-结构双引导 + 退化自适应码本 + 热序保持损失。

## 方法详解

### 整体框架

TSG 模块融合热先验进行退化感知编码 → VAR 骨干通过逐尺度预测生成 → CAC 动态调整量化嵌入 → 热序一致性损失保持物理一致性。

### 关键设计

1. **热-结构引导模块（TSG）**:

    - 功能：缓解热辐射与结构边缘的失配
    - 核心思路：从 LR 输入构建热图 $I_{\text{Heat}}$ 和边缘图 $I_{\text{Edge}}$，分别用 DINOv3 编码器提取特征。可学习注意力门 $W = \sigma(L(A) + G(A))$ 自适应平衡两种特征的贡献。融合特征通过交叉注意力引导 LR 特征。
    - 设计动机：汽车发动机作为强热源，其热辐射区域常偏离车辆实际轮廓。直接训练会使模型过拟合热峰值而忽略真实边缘。

2. **条件自适应码本（CAC）**:

    - 功能：动态修正量化偏差以增强纹理真实性
    - 核心思路：每个码嵌入通过低秩扰动动态调制：$Z'(g)[i] = Z[i] + \tanh(\alpha)[(U_i \odot h(g))V^\top]$，其中条件向量 $h(g)$ 来自 TSG 特征。同一离散索引在不同退化条件下可解码为不同嵌入向量。
    - 设计动机：标准 VQ-VAE 量化引入的离散化误差在红外非均匀退化下更严重，静态码本无法适应空间变化的退化模式。

3. **热序一致性损失 $\mathcal{L}_{\text{TOC}}$**:

    - 功能：保持温度-亮度的单调物理关系
    - 核心思路：对相邻 patch 对执行 $\mathcal{L}_{\text{TOC}} = \text{ReLU}(-[(I_{\text{SR}}^p(i) - I_{\text{SR}}^p(j)) \times (I_{\text{HR}}^p(i) - I_{\text{HR}}^p(j))])$。惩罚 SR 和 HR 之间亮度排序不一致的情况。
    - 设计动机：红外图像中更高温度对应更高像素亮度（单调性）。退化（散焦、运动模糊）导致局部温度压缩和峰值偏移，MSE 仅约束绝对值无法保证相对排序。

### 损失函数 / 训练策略

$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_1 \mathcal{L}_{\text{MSE}} + \lambda_2 \mathcal{L}_{\text{TOC}}$。$\lambda_1=0.2, \lambda_2=0.8$。4 × A800，AdamW，10K 次微调。

## 实验关键数据

### 主实验

| 数据集 | 指标 | Real-IISR | DifIISR (之前SOTA) | 提升 |
|--------|------|-----------|-------------------|------|
| FLIR-IISR@Set5 | MUSIQ↑ | 59.90 | 54.79 | +5.11 |
| FLIR-IISR@Set15 | MUSIQ↑ | 57.06 | 53.16 | +3.90 |
| FLIR-IISR@Set5 | LPIPS↓ | 0.1615 | 0.2525 | -0.091 |

### 消融实验

| 配置 | PSNR | MUSIQ | 说明 |
|------|------|-------|------|
| 无 TSG | 下降 | 下降 | 边缘模糊、热分布不准 |
| 无 CAC | 下降 | 下降 | 纹理不稳定 |
| 无 $\mathcal{L}_{\text{TOC}}$ | 下降 | 下降 | 热峰值漂移 |
| VAR vs 扩散基线 | VAR 优 | VAR 优 | 确定性生成更适合红外 |

### 关键发现

- Real-IISR 虽然参数最多（1144.6M）但推理最快（2.45 FPS），因自回归无需多步去噪
- 扩散方法的迭代去噪模糊了高频热细节并破坏结构-热对应
- $\mathcal{L}_{\text{TOC}}$ 有效防止了热峰值漂移，保持了物理一致性

## 亮点与洞察

- **领域特有约束设计**：热序一致性损失巧妙利用红外成像的物理单调性
- **真实数据集构建**：通过自动对焦变化和运动模糊模拟真实退化，填补了真实红外 SR 数据的空白
- **CAC 的低秩扰动**：用低秩结构控制码本调制幅度，平衡灵活性和稳定性

## 局限与展望

- FLIR-IISR 数据集仅 1457 对，规模仍有限
- 仅支持 4× 超分，未验证其他倍率
- 热图和边缘图的质量依赖于 LR 输入，极端退化下可能不可靠
- 未来可扩展到红外视频超分以利用时序信息

## 相关工作与启发

- **vs VARSR**: VARSR 为可见光设计无红外约束，Real-IISR 引入热先验
- **vs DifIISR**: DifIISR 用扩散+梯度对齐但多步去噪慢且模糊红外细节

## 评分

- 新颖性: ⭐⭐⭐⭐ 热-结构引导和热序约束是针对红外的创新设计，填补了真实红外SR的空白
- 实验充分度: ⭐⭐⭐⭐ 两个数据集+多对比方法+全面消融，效率分析完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，热图可视化和灰度波动图直观展示物理一致性
- 价值: ⭐⭐⭐⭐ 填补了真实红外 SR 的数据和方法双重空白，为该领域提供了基准

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FinPercep-RM: A Fine-grained Reward Model and Co-evolutionary Curriculum for RL-based Real-world Super-Resolution](finpercep_rm_a_fine_grained_reward_model_and_co_evolutionary_curriculum_for_rl_ba.md)
- [\[CVPR 2026\] RAR: Restore, Assess, Repeat - A Unified Framework for Iterative Image Restoration](rar_restore_assess_repeat_a_unified_framework_for_iterative_image_restoration.md)
- [\[CVPR 2026\] UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization](unirain_unified_image_deraining_rag_dataset_distillation.md)
- [\[CVPR 2026\] UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution](ucan_unified_convolutional_attention_lightweight_sr.md)
- [\[CVPR 2026\] Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration](beyond_ground-truth_leveraging_image_quality_priors_for_real-world_image_restora.md)

</div>

<!-- RELATED:END -->
