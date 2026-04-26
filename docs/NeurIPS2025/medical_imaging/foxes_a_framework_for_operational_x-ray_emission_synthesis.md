---
title: >-
  [论文解读] FOXES: A Framework For Operational X-ray Emission Synthesis
description: >-
  [NeurIPS 2025][医学图像][Transformer] 提出 FOXES，一个基于 Vision Transformer 的框架，将太阳多通道 EUV 观测图像翻译为软 X 射线（SXR）通量，整体 Pearson 相关达到 0.982，为远端太阳耀斑检测和更完整的耀斑目录构建奠定基础。
tags:
  - NeurIPS 2025
  - 医学图像
  - Transformer
  - 太阳耀斑
  - EUV-SXR翻译
  - 空间天气
  - 注意力图
  - 耀斑分类
---

# FOXES: A Framework For Operational X-ray Emission Synthesis

**会议**: NeurIPS 2025  
**arXiv**: [2510.22801](https://arxiv.org/abs/2510.22801)  
**代码**: 未提供  
**领域**: 太阳物理 / 空间天气预测  
**关键词**: Vision Transformer, 太阳耀斑, EUV-SXR翻译, 空间天气, 注意力图, 耀斑分类

## 一句话总结

提出 FOXES，一个基于 Vision Transformer 的框架，将太阳多通道 EUV 观测图像翻译为软 X 射线（SXR）通量，整体 Pearson 相关达到 0.982，为远端太阳耀斑检测和更完整的耀斑目录构建奠定基础。

## 研究背景与动机

**太阳耀斑与空间天气**：太阳耀斑是日冕中磁能快速释放的辐射表现，是日冕物质抛射（CME）和太阳高能粒子事件的最佳代理指标。当这些事件到达地球时可影响卫星、无线电通信、GPS 和电力网

**现有耀斑分类方法的局限**：
   - 主要依赖 GOES 卫星的积分 SXR 辐照度进行分类（A/B/C/M/X 级），但 GOES 测量的是全日面积分值，无法精确定位单个耀斑的强度和位置
   - 当多个耀斑同时发生时（sympathetic flares），无法区分各自贡献

**远端监测的空白**：目前没有也没有计划在日-地连线之外部署 SXR 监测器，这严重限制了对火星探索任务等远端空间活动的太阳活动保护能力

**EUV 成像的优势**：EUV 图像提供空间分辨信息，STEREO 和 Solar Orbiter 等任务在远端提供 EUV 观测，如果能从 EUV 推断 SXR 通量，就能扩展耀斑检测覆盖范围

## 方法详解

### 数据准备

**输入**：SDO/AIA 的六个 EUV 通道（94, 131, 171, 193, 211, 304 Å），1分钟采样率。通过 ITI 工具链进行预处理：裁剪至 1.1 太阳半径、设备退化校正、图像/曝光归一化、下采样至 512×512。

**输出**：GOES XRS 1-8 Å 波段的 1分钟平均 SXR 通量测量值。

**数据划分**（无时间重叠）：

| 耀斑类别 | 训练集 | 验证集 | 测试集 |
|---------|-------|-------|-------|
| < C | 13,256 | 358 | 13,370 |
| C | 48,140 | 15,722 | 63,503 |
| M | 16,604 | 2,827 | 22,256 |
| X | 1,738 | 81 | 1,075 |

训练期：2013年1月-2022年12月 + 2023年7月1-20日；测试期：2023年8月-2025年9月。

### ViT 架构

基于修改版 PyTorch Lightning ViT 实现：

- **Patch 大小**：16×16 → 生成 1024 个 patch
- **嵌入维度**：512 维，同时编码 6 个 EUV 通道的多波长信息
- **Transformer 层数**：6 层，8 个注意力头
- **前馈层**：512 单元
- **Dropout**：0.1
- **学习率**：1e-4，余弦退火至 1e-5
- **训练**：250 epochs，batch size 64
- **损失函数**：Huber Loss，按耀斑类别的逆频率加权以缓解类别不平衡

### 注意力图的物理解释

ViT 的自注意力图提供了可解释的预测——注意力集中在物理相关区域（活动区/耀斑区），而非随机关联。图 3 左图展示了 131 Å 通道上叠加的注意力热图，清晰指向实际耀斑区域。

## 实验关键数据

### 定量指标（对数空间）

| 耀斑类别 | MSE | RMSE | MAE | Pearson r |
|---------|-----|------|-----|-----------|
| Overall | 1.41e-2 | 1.19e-1 | 8.97e-2 | **0.982** |
| < C | 9.52e-3 | 9.76e-2 | 9.22e-2 | 0.905 |
| C | 1.07e-2 | 1.03e-1 | 7.56e-2 | **0.962** |
| M | 2.56e-2 | 1.60e-1 | 1.26e-1 | 0.857 |
| X | 3.19e-2 | 1.79e-1 | 1.29e-1 | 0.665 |

### 关键观察

1. **C 级耀斑效果最佳**：样本最多，MAE 仅 7.56e-2，相关 0.962
2. **X 级表现最弱**：r = 0.665，样本稀少（训练仅 1,738 条）是主要原因
3. **时间序列跟踪能力**：FOXES 不仅捕获了耀斑的脉冲相和峰值阶段，还准确跟踪了衰减阶段
4. **注意力图物理一致性**：注意力高亮区域与实际观测到的活动区吻合，特别是在 131 Å 通道（对耀斑最敏感）

## 亮点

1. ⭐⭐⭐ **EUV→SXR 翻译的新范式**：首次系统性地用 ViT 将多通道 EUV 图像翻译为 SXR 通量，整体 Pearson r = 0.982 证明了概念可行性
2. ⭐⭐ **远端应用潜力**：可直接应用于 STEREO/Solar Orbiter 等远端 EUV 观测，填补日-地连线外的耀斑检测空白
3. ⭐⭐ **物理可解释性**：自注意力图指向真实活动区域，表明模型学到了物理相关特征而非统计捷径
4. ⭐ **为耀斑目录改进奠基**：更全面的耀斑目录将反哺耀斑预报算法，形成良性循环

## 局限与展望

1. **仅预测积分通量**：当前只能输出全日面积分 SXR 通量，无法定位单个耀斑的位置和强度——这是论文明确提到的未来方向（patch-based SXR 预测）
2. **X 级耀斑性能不足**：r = 0.665 的相关性对高风险事件（最需要准确预测的类别）远不够用，主要受限于样本不平衡
3. **无与其他方法的对比**：论文未与现有 EUV-SXR 翻译方法或 CNN 基线进行系统对比
4. **跨卫星标定未验证**：声称可应用于 STEREO，但需要 ITI 标定，其跨仪器泛化性尚未实际验证
5. **时间建模缺失**：模型独立处理每个时间步的快照，未利用时间序列的连续性信息

## 评分

| 维度 | 评分 |
|------|------|
| 新颖性 | ⭐⭐⭐ |
| 技术深度 | ⭐⭐ |
| 实验充分性 | ⭐⭐ |
| 写作质量 | ⭐⭐⭐ |
| 综合推荐 | ⭐⭐⭐ |

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] Radiative Gaussian Splatting for Efficient X-ray Novel View Synthesis](../../ECCV2024/medical_imaging/radiative_gaussian_splatting_for_efficient_x-ray_novel_view_synthesis.md)
- [\[NeurIPS 2025\] Modeling X-ray Photon Pile-up with a Normalizing Flow](modeling_x-ray_photon_pile-up_with_a_normalizing_flow.md)
- [\[NeurIPS 2025\] Variational Autoencoder with Normalizing Flow for X-ray Spectral Fitting](variational_autoencoder_with_normalizing_flow_for_x-ray_spectral_fitting.md)
- [\[NeurIPS 2025\] SynBrain: Enhancing Visual-to-fMRI Synthesis via Probabilistic Representation Learning](synbrain_enhancing_visual-to-fmri_synthesis_via_probabilistic_representation_lea.md)
- [\[NeurIPS 2025\] Scalable Diffusion Transformer for Conditional 4D fMRI Synthesis](scalable_diffusion_transformer_for_conditional_4d_fmri_synthesis.md)

<!-- RELATED:END -->
