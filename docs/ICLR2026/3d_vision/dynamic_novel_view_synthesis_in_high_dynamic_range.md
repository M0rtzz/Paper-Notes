---
title: >-
  [论文解读] Dynamic Novel View Synthesis in High Dynamic Range
description: >-
  [ICLR2026][3D视觉][HDR] 首次提出 HDR 动态新视角合成 (HDR DNVS) 问题，并设计 HDR-4DGS 框架，通过动态色调映射模块在时变场景中实现时序一致的 HDR 辐射场重建，在合成和真实数据集上均超越现有方法。
tags:
  - ICLR2026
  - 3D视觉
  - HDR
  - Dynamic Novel View Synthesis
  - 4D Gaussian Splatting
  - Tone Mapping
  - Radiance Field
---

# Dynamic Novel View Synthesis in High Dynamic Range

**会议**: ICLR2026  
**arXiv**: [2509.21853](https://arxiv.org/abs/2509.21853)  
**代码**: [prinasi/HDR-4DGS](https://github.com/prinasi/HDR-4DGS)  
**领域**: 3d_vision  
**关键词**: HDR, Dynamic Novel View Synthesis, 4D Gaussian Splatting, Tone Mapping, Radiance Field  

## 一句话总结

首次提出 HDR 动态新视角合成 (HDR DNVS) 问题，并设计 HDR-4DGS 框架，通过动态色调映射模块在时变场景中实现时序一致的 HDR 辐射场重建，在合成和真实数据集上均超越现有方法。

## 背景与动机

现有新视角合成方法受限于两个假设：**静态场景** 和 **低动态范围 (LDR) 输入**。

- **动态新视角合成 (DNVS)** 虽然能处理时变场景（如运动物体、变化光照），但仅限于 LDR 图像，在强对比度条件下（直射日光、暗光环境）会丢失过曝/欠曝区域的信息。
- **HDR 新视角合成 (HDR NVS)** 能从多曝光 LDR 图像重建 HDR 场景，但现有方法（如 HDR-NeRF、HDR-GS、GaussHDR）均假设场景完全静态。
- **现实需求**：真实世界的 HDR 场景天然是动态的——包含运动物体、变化光照、瞬态现象。已有方法无法同时处理动态几何和 HDR 辐射重建。

尽管 HDR-HexPlane 初步探索了动态 HDR 重建，但它从未仔细评估 HDR 输出质量，也未在真实场景上验证，留下了大量空白。

## 核心问题

**HDR Dynamic Novel View Synthesis (HDR DNVS)**：从稀疏、时变的多曝光 LDR 输入中，学习一个 HDR 4D 辐射场模型 $\mathcal{F}_h$，使其能在任意时间戳 $t'$ 和任意视角 $V'$ 下渲染时序一致的 HDR 图像。核心挑战在于：

1. 需要联合建模不断演变的场景结构和 HDR 辐射
2. 非刚性运动与时域变化导致复杂的时空不一致性
3. 稀疏 LDR 观测缺乏可靠的亮度先验，导致严重的光度歧义

## 方法详解

### 整体架构：HDR-4DGS

HDR-4DGS 基于 4D Gaussian Splatting 构建，包含两大核心组件：

**1) 动态场景表示 (4DGS)**

- 采用 4D Gaussian Splatting 作为场景表示基础，将时间维度引入 3DGS
- 像素观测 $\mathbf{I}(u,v,t)$ 同时依赖空间坐标和时间戳
- 每个高斯的均值扩展为 $\mu = (\mu_x, \mu_y, \mu_z, \mu_t)$
- 使用 4D 球谐函数 (4DSH) 建模外观的时域演变，构建支持 HDR-LDR 转换的辐射银行 (radiance bank)
- 关键改进：将原始 4DGS 的颜色表示空间从 LDR 扩展到 HDR

**2) 动态色调映射器 (Dynamic Tone Mapper, DTM)**

DTM 是本文的核心创新，受人类视觉适应机制启发：

- **辐射银行 (Radiance Bank)**：存储每个时间戳的平均 HDR 颜色统计 $\mathbf{r}_t^h = \frac{1}{N}\sum_{i=1}^N \mathbf{c}_{i,t}^h$
- **动态辐射上下文学习器 (DRCL)**：使用 GRU 处理滑动窗口 $k$ 帧的辐射签名序列 $\{\mathbf{r}_{t-k:t}^h\}$，生成辐射上下文嵌入 $\mathbf{f}_t \in \mathbb{R}^d$
- **自适应色调映射**：将对数域 HDR 颜色与曝光时间、辐射上下文拼接后，通过逐通道色调映射函数 $g_\theta$ 转换为 LDR 颜色：
$$\mathbf{c}_t^l = g_\theta([\log \mathbf{c}_t^h + \log e_t, \mathbf{f}_t])$$

**3) 模型优化**

- 总损失：$\mathcal{L}_{total} = \mathcal{L}_{ldr} + \alpha \mathcal{L}_{hdr}$
- LDR 损失包含**双重监督**：像素级（2D 色调映射后的 LDR）和光线级（3D 光栅化的 LDR）
- HDR 损失使用 $\mu$-law 压缩对齐 HDR 和 LDR 域
- 图像重建损失 = $(1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$，$\lambda=0.2$

## 实验关键数据

### 数据集（本文构建）

| 数据集 | 场景数 | 类型 | 特点 |
|--------|--------|------|------|
| HDR-4D-Syn | 8 | 合成 | 多曝光视频 + 同步多视角 LDR 流 + HDR 真值 |
| HDR-4D-Real | 4 | 真实 | 6 台 iPhone 14 Pro 同步拍摄，三种曝光 |

### HDR-4D-Syn 上的核心结果（仅 LDR 监督）

| 方法 | HDR PSNR↑ | HDR SSIM↑ | HDR LPIPS↓ | 推理速度 (fps) |
|------|-----------|-----------|------------|---------------|
| HDR-NeRF | 8.54 | 0.062 | 0.552 | 0.061 |
| HDR-GS | 4.64 | 0.158 | 0.645 | 380.38 |
| HDR-HexPlane | 14.70 | 0.649 | 0.287 | 1.61 |
| **HDR-4DGS** | **25.88** | **0.865** | **0.076** | **40.80** |

- HDR PSNR 比次优方法 (HDR-HexPlane) 高 **11.18 dB**
- 推理速度比 HDR-HexPlane 快约 **25×**，比 HDR-NeRF 快约 **669×**
- 使用 LDR+HDR 联合监督时，HDR PSNR 进一步提升至 **30.40 dB**

### 消融实验要点

- 用独立 HDR 重建方法（4DGS + KPNet 等两阶段流水线）最优仅达 PSNR 20.92，远低于联合优化的 25.88
- DTM vs MLP 静态色调映射器：PSNR 25.88 vs 23.92，LPIPS 0.076 vs 0.142
- 像素级监督贡献：去掉后 PSNR 降低 1.03 dB
- 时序上下文长度 $k=20$ 最优，过小（5/10）或过大（30）均有性能下降

## 亮点

1. **问题定义价值高**：首次形式化 HDR DNVS 问题，填补了动态场景 HDR 合成的空白
2. **动态色调映射器设计精巧**：受人类视觉适应机制启发，使用 GRU 建模时序辐射上下文，实现自适应 HDR-LDR 转换，且学到的色调映射曲线可解释（单调递增、随场景亮度变化而动态调整）
3. **完整的基准建设**：构建了合成 + 真实两个新数据集，为后续研究提供了标准化评估平台
4. **双重监督策略**：像素级 + 光线级联合约束有效缓解了 3D 色调映射的过拟合问题
5. **效率优势显著**：在大幅提升质量的同时保持了实时级推理速度

## 局限与展望

1. **运动区域结构退化**：在可移动区域仍存在结构退化，作者归因于底层 4DGS 表示能力的固有限制，未来可探索更强的动态表示
2. **真实场景 HDR 指标不够突出**：HDR-4D-Real 上 HDR PSNR（LDR-only 监督时 14.50）低于 HDR-HexPlane（9.306），但作者指出是 HDR 真值噪声和 PSNR 偏好模糊重建所致，视觉质量实际更优
3. **时序窗口固定**：$k=20$ 是固定超参，未做自适应窗口长度探索
4. **实际部署场景有限**：真实数据集仅 4 个室内场景，未覆盖户外大场景、极端天气等更复杂情况
5. **训练时间相对较长**：HDR-4DGS 训练约 69-99 分钟，比 HDR-GS（14-38 分钟）更慢

## 与相关工作的对比

| 维度 | HDR-NeRF / HDR-GS | HDR-HexPlane | HDR-4DGS (本文) |
|------|-------------------|--------------|----------------|
| 静/动态 | 静态 | 动态 | 动态 |
| 色调映射 | MLP 静态 | Sigmoid 静态 | GRU 动态自适应 |
| 时序一致性 | 不适用 | 弱 | 强（辐射上下文学习） |
| HDR 评估 | 有 | 无 | 有（完整基准） |
| 实时性 | NeRF 慢 / GS 快 | 慢（~1.6 fps） | 快（~41 fps） |

## 启发与关联

- **动态色调映射思路可迁移**：DTM 的"辐射银行 + 序列模型"范式可推广到其他需要时序自适应颜色/辐射转换的任务，如视频 HDR 重建、动态光照下的 relighting
- **双重监督策略通用性强**：像素级 + 光线级联合监督的思路可应用于其他基于 3DGS 的颜色空间转换任务
- **基准数据集价值**：HDR-4D-Syn 和 HDR-4D-Real 可直接用于后续动态 HDR 相关研究的评估

## 评分

- 新颖性: ⭐⭐⭐⭐ — 问题定义新颖，动态色调映射模块设计有创意
- 实验充分度: ⭐⭐⭐⭐ — 合成 + 真实数据集，丰富的消融实验和可视化
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机阐述充分，公式表述规范
- 价值: ⭐⭐⭐⭐ — 开辟了 HDR DNVS 新方向，提供完整基准，代码开源
