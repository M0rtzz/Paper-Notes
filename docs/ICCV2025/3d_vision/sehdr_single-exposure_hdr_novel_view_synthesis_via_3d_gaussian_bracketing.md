---
title: >-
  [论文解读] SeHDR: Single-Exposure HDR Novel View Synthesis via 3D Gaussian Bracketing
description: >-
  [ICCV 2025][3D视觉][HDR新视角合成] SeHDR 是首个从单曝光多视图 LDR 图像生成 HDR 新视角的3DGS框架，通过在3D空间扩展包围曝光原理（Bracketed 3D Gaussians）并设计可微分神经曝光融合（NeEF）在球谐空间融合，无需 HDR 监督即超越现有方法 14.3dB。
tags:
  - ICCV 2025
  - 3D视觉
  - HDR新视角合成
  - 3D高斯泼溅
  - 单曝光
  - 包围曝光
  - 球谐函数
  - 曝光融合
  - 高动态范围
---

# SeHDR: Single-Exposure HDR Novel View Synthesis via 3D Gaussian Bracketing

**会议**: ICCV 2025  
**arXiv**: [2509.20400](https://arxiv.org/abs/2509.20400)  
**代码**: [GitHub](https://github.com/yiyulics/SeHDR)  
**领域**: 3d_vision  
**关键词**: HDR新视角合成, 3D高斯泼溅, 单曝光, 包围曝光, 球谐函数, 曝光融合, 高动态范围

## 一句话总结

SeHDR 是首个从单曝光多视图 LDR 图像生成 HDR 新视角的3DGS框架，通过在3D空间扩展包围曝光原理（Bracketed 3D Gaussians）并设计可微分神经曝光融合（NeEF）在球谐空间融合，无需 HDR 监督即超越现有方法 14.3dB。

## 研究背景与动机

HDR-NVS 旨在从 LDR 图像重建并渲染 HDR 场景。现有方法（HDR-NeRF, HDRGS）的关键限制：

**需要多曝光输入**：要求多视图 LDR 图像在不同曝光下拍摄

**捕获困难**：同一视角多曝光须完美对齐（需三脚架），跨视图不同曝光需 COLMAP 匹配，容易出现运动模糊和标定错误

**直接组合方案失败**：
   - 多曝光方法用于单曝光输入无效（无跨视图曝光补偿）
   - 单图HDR重建+3DGS会引入多视图不一致（floater和模糊伪影）

## 方法详解

### 核心灵感

来自计算摄影的包围曝光技术：先合成不同曝光的3D高斯，再融合为 HDR 场景表示。

### 步骤1：学习基础3D高斯

- 从单曝光 LDR 输入学习基础高斯集合
- **关键设计**：SH 系数参数化**线性颜色空间**（而非 sRGB）
- 通过估计 CRF 将 LDR sRGB 转换到线性空间
- 线性化使曝光操作在物理上合理（曝光∝辐照度）

### 步骤2：合成包围曝光3D高斯（核心创新）

将2D包围曝光扩展到3D空间：
- 操纵线性颜色值估计额外3D高斯集合
- **相同几何、不同颜色**：共享位置/均值/协方差，仅线性颜色不同
- 条件采样不同曝光值
- 模拟"不同曝光拍摄同一场景的效果"

### 步骤3：可微分神经曝光融合（NeEF）

- **在球谐函数参数空间**融合（非图像空间）
- 保证融合后仍是有效3D高斯，可直接新视角渲染
- 可微分设计支持端到端训练
- 不需要 HDR ground truth

### 训练流程

1. 学习基础高斯 + CRF 线性化
2. 线性空间中模拟不同曝光高斯
3. NeEF 在 SH 空间融合为 HDR 高斯
4. Tone-mapping 回 LDR，与输入计算损失
5. 端到端联合优化

### 关键创新总结

| 组件 | 作用 | 区别 |
|------|------|------|
| 线性化SH | 物理合理曝光操作 | HDR-NeRF直接在网络空间 |
| 3D包围曝光 | 多曝光信息补偿 | 无需多曝光拍摄 |
| SH空间融合 | 3D一致HDR表示 | 2D融合会引入不一致 |
| 无HDR监督 | 自监督训练 | 仅需LDR输入 |

## 实验关键数据

### 主实验

- 超越 SOTA 方法 **14.3 dB**（无 HDR 监督）
- 超越"单图HDR+3DGS"组合：后者有 floater 和模糊
- 超越"HDRGS+单曝光输入"：缺少曝光补偿而失效

### 对比方法

| 方法 | 输入要求 | SeHDR优势 |
|------|---------|----------|
| HDR-NeRF | 多曝光多视图 | 仅需单曝光 |
| HDRGS | 多曝光多视图 | 仅需单曝光 |
| 单图HDR+3DGS | 单曝光 | 无floater伪影 |

### 定性结果

- 过曝区域（阳光直射）恢复细节
- 欠曝区域（暗影）保留纹理
- 色彩准确、无 floater 和模糊

### 适用性

可直接应用于任何标准多视图数据集（COLMAP采集），无需专门多曝光设备。

## 亮点与洞察

1. **核心创意极其优雅**：包围曝光从2D→3D高斯空间，"相同几何、不同颜色"简洁有效
2. **SH空间融合**：在球谐参数空间而非图像空间融合，确保3D一致性，是避免多视图不一致的关键
3. **线性化的重要性**：SH参数化为线性颜色使曝光操作物理合理
4. **无需额外监督**：14.3dB提升通过自监督实现
5. **实用性突破**：消除多曝光需求，HDR-NVS可直接用于任何现有数据集

## 局限性

1. CRF 估计准确性影响线性化质量
2. 严重饱和区域信息可能无法恢复
3. 缓存缺少 NeEF 具体架构和定量消融
4. 合成多组包围曝光高斯可能增加训练时间
5. 单曝光信息量本质少于多曝光，极端动态范围有天花板

## 相关工作

- **HDR-NVS**：HDR-NeRF → HDRGS → **SeHDR**（单曝光）
- **HDR图像重建**：多曝光融合 → 单图LDR-to-HDR → SeHDR 扩展到3D
- **3DGS**：标准3DGS在LDR空间 → SeHDR 在线性/HDR空间
- **计算摄影**：包围曝光 → SeHDR 的3D推广

## 评分

- **新颖性**：9/10 — 首个单曝光HDR-NVS，3D包围曝光创新性强
- **技术深度**：8/10 — 线性化+3D包围+SH融合的技术链条完整且物理合理
- **实验充分性**：6/10 — 14.3dB令人印象深刻但缓存缺完整表格
- **实用性**：9/10 — 消除多曝光需求使HDR-NVS对现有数据集即可用
- **总评**：8/10

## 亮点与洞察

## 局限与展望

## 相关工作与启发

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [High Dynamic Range Novel View Synthesis with Single Exposure](../../ICML2025/3d_vision/high_dynamic_range_novel_view_synthesis_with_single_exposure.md)
- [Physically Inspired Gaussian Splatting for HDR Novel View Synthesis](../../CVPR2026/3d_vision/physically_inspired_gaussian_splatting_for_hdr_novel_view_synthesis.md)
- [Learning Neural Exposure Fields for View Synthesis](../../NeurIPS2025/3d_vision/learning_neural_exposure_fields_for_view_synthesis.md)
- [Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)
- [BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis](billboard_splatting_bbsplat_learnable_textured_primitives_fo.md)

<!-- RELATED:END -->
