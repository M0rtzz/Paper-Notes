---
title: >-
  [论文解读] Points-to-3D: Structure-Aware 3D Generation with Point Cloud Priors
description: >-
  [CVPR 2026][自动驾驶][点云先验] 提出 Points-to-3D，将可见区域点云编码为 TRELLIS 的稀疏结构潜变量（SS latent）并用 mask-aware inpainting 网络补全不可见区域，结合结构补全+边界精炼两阶段采样策略，实现几何可控的高保真 3D 资产/场景生成，在 Toys4K 上 F-Score 达 0.964（可见区域 0.998）。
tags:
  - CVPR 2026
  - 自动驾驶
  - 点云先验
  - 3D生成
  - 扩散模型
  - 结构补全
  - 几何可控
---

# Points-to-3D: Structure-Aware 3D Generation with Point Cloud Priors

**会议**: CVPR 2026  
**arXiv**: [2603.18782](https://arxiv.org/abs/2603.18782)  
**代码**: [项目页面](https://jiatongxia.github.io/points2-3D/)  
**领域**: 自动驾驶 / 3D生成  
**关键词**: 点云先验, 3D生成, 扩散模型, 结构补全, 几何可控

## 一句话总结

提出 Points-to-3D，将可见区域点云编码为 TRELLIS 的稀疏结构潜变量（SS latent）并用 mask-aware inpainting 网络补全不可见区域，结合结构补全+边界精炼两阶段采样策略，实现几何可控的高保真 3D 资产/场景生成，在 Toys4K 上 F-Score 达 0.964（可见区域 0.998）。

## 研究背景与动机

**领域现状**：3D 生成模型（TRELLIS、GaussianAnything 等）已能从图像或文本合成逼真 3D 资产，但都以 2D 图像/文本为条件，缺乏对真实 3D 几何的直接约束，生成结果在几何精度上不可控。

**被忽视的信息源**：在自动驾驶、机器人等场景中，可见区域点云极易获取——来自 LiDAR、结构光、甚至 VGGT 等前馈预测器。这些点云提供了显式的几何约束，但当前生成框架无法利用。

**技术限制**：TRELLIS 的结构生成从纯高斯噪声初始化 SS latent，仅受图像/文本 embedding 引导，无法锚定到真实 3D 观测。简单地将点云作为额外条件注入效果有限——需要将结构先验嵌入潜空间本身。

**核心思路**：将点云引导的 3D 生成重新定义为**潜空间补全（latent inpainting）**问题：可见区域编码为固定约束，不可见区域由补全网络合成。

## 方法详解

### 整体框架

输入可见区域点云 $\mathbf{P}$ → 体素化为 $\mathbf{M}' \in \{0,1\}^{N \times N \times N}$（$N=64$）→ TRELLIS 的 SS VAE 编码为部分观测 SS latent $\mathbf{q}_{\text{vis}} = \mathcal{E}_s(\mathbf{M}')$ → 用 occupancy mask $\mathbf{m}_s$ 标记可见/不可见区域 → 不可见区域填充噪声得到组合输入 $\mathbf{q}_{\text{comb}}$ → Inpainting Flow Transformer 补全 → 两阶段采样输出完整 SS latent → TRELLIS 后续 SLAT 生成 + 渲染。

### 关键设计

1. **点云先验驱动的潜空间初始化**：

    - **功能**：将可见点云编码到 TRELLIS 的 SS latent 空间作为生成起点，替代纯噪声初始化
    - **核心公式**：
    $\mathbf{q}_{\text{comb}} = \mathbf{m}_s \odot \mathbf{q}_{\text{vis}} + (1 - \mathbf{m}_s) \odot \boldsymbol{\epsilon}_s$
      其中 $\mathbf{q}_{\text{vis}} = \mathcal{E}_s(\mathbf{M}')$ 是编码后的可见区域 latent，$\mathbf{m}_s$ 是下采样到 latent 分辨率（$r=16$）的 occupancy mask
    - **设计动机**：直接在潜空间中"锚定"可见几何，使扩散过程受真实 3D 观测约束，而非仅依赖图像/文本的隐式引导

2. **Mask-aware 结构补全网络 $\mathcal{G}_{inp}$**：

    - **功能**：基于 TRELLIS 的 Structure Flow Transformer 微调，学习从可见区域向不可见区域推断几何
    - **输入设计**：将 mask $\mathbf{m}_s$ 沿通道维度拼接到 $\mathbf{q}_{\text{comb}}$，替换掉原始输入层适配新通道数 $(c_s + c_m)$
    - **训练数据构建**：从完整 3D 资产的 $T=24$ 个视角渲染深度图，通过深度一致性检验（阈值 $\tau$）提取每个视角的可见点云，构造 $(\mathbf{q}_{\text{comb}}^t, \mathbf{m}_s^t, \mathbf{I}_t, \mathbf{q}_{\text{gt}})$ 训练对
    - **训练目标**：Conditional Flow Matching 损失
    $\mathcal{L}_{CFM} = \mathbb{E}_{t, \mathbf{q}_{\text{gt}}, \epsilon} \|\mathcal{G}_{inp}(\mathbf{x}_{\text{inp}}, t) - (\epsilon - \mathbf{q}_{\text{gt}})\|_2^2$

3. **两阶段采样策略（Staged Sampling）**：

    - **功能**：将 $t$ 步采样分为结构补全阶段（$s$ 步）和边界精炼阶段（$t-s$ 步）
    - **结构补全阶段**：每步重建 $\mathbf{q}_{\text{pred}}$ 后与 mask $\mathbf{m}_s$ 重新拼接，循环 $s$ 步，保持可见区域锚定
    - **边界精炼阶段**：将 mask 替换为全 1（$\mathbf{m}_1$），转为标准去噪，修复补全边界处的"空洞"伪影
    - **设计动机**：纯 inpainting 会在可见/不可见区域交界处因下采样信息损失产生几何空洞；后续精炼步骤可在不破坏全局结构的前提下修复边界
    - **最佳配置**：$s=25$, $t-s=25$（总 50 步）

### 损失函数 / 训练策略

- **损失**：Conditional Flow Matching Loss，仅在 SS latent 空间上计算
- **训练**：在 3D-FUTURE + HSSD + ABO 三数据集上训练 20k 迭代，batch size 8，4×A100
- **每物体采样** $S=50{,}000$ 个点，$T=24$ 视角，构造可见/完整 latent 对
- 推理时支持两种输入：(a) 真实点云先验；(b) VGGT 从单张图像估计的点云

## 实验关键数据

### 主实验

**单物体生成（Toys4K）**：

| 方法 | PSNR↑ | SSIM(%)↑ | LPIPS↓ | CD↓ | F-Score↑ |
|---|---|---|---|---|---|
| TRELLIS | 21.94 | 91.46 | 0.105 | 0.034 | 0.832 |
| SAM3D | 22.42 | 91.45 | 0.111 | 0.033 | 0.835 |
| Points-to-3D (VGGT) | 22.55 | 92.09 | 0.088 | 0.024 | 0.881 |
| **Points-to-3D (P.C.)** | **22.91** | **92.83** | **0.070** | **0.013** | **0.964** |

**场景级生成（3D-FRONT）**：

| 方法 | PSNR↑ | LPIPS↓ | CD↓ | F-Score↑ |
|---|---|---|---|---|
| TRELLIS | 18.21 | 0.239 | 0.094 | 0.478 |
| MIDI | 19.23 | 0.166 | 0.075 | 0.513 |
| **Points-to-3D (P.C.)** | **21.63** | **0.124** | **0.025** | **0.886** |

### 消融实验

| 配置 (Inp./Ref. 步数) | CD↓ | F-Score↑ | PSNR-N↑ | 说明 |
|---|---|---|---|---|
| 50/0 (纯 inpainting) | 0.014 | 0.960 | 25.88 | 边界有空洞 |
| 25/25 (最佳) | **0.013** | **0.963** | **27.10** | 空洞消除 |
| 10/40 | 0.014 | 0.961 | 26.72 | inpainting 不足 |

### 关键发现

- 可见区域 F-Score 达 **0.998**，CD 仅 **0.007**，几乎完美保留输入先验
- 即使使用 VGGT 估计的（带噪声的）点云，仍显著优于所有基线，验证了框架的鲁棒性
- SAM3D 虽也用点云但通过 attention 间接融合，无法实现显式几何控制
- 场景级生成提升尤其显著（F-Score: 0.513 → 0.886），显示点云先验对复杂多物体场景的指导价值

## 亮点与洞察

1. **关键创新**：将点云条件生成重新定义为潜空间 inpainting 问题，是一个简洁优雅且有效的范式
2. **两阶段采样**：结构补全+边界精炼的分离采样策略巧妙解决了 inpainting 边界伪影
3. **灵活输入**：支持真实传感器点云和 VGGT 预测点云，覆盖有/无真实 3D 先验的场景
4. **即插即用**：基于 TRELLIS 框架，仅修改输入层和训练数据，架构改动最小

## 局限与展望

1. VGGT 预测点云与真实点云仍有差距（F-Score 0.881 vs 0.964），受限于前馈预测精度
2. 体素化分辨率 $N=64$ 可能限制细粒度几何表达
3. 仅在物体和室内场景评估，未验证大规模户外场景（如自动驾驶点云）

## 相关工作与启发

- **TRELLIS**：提供了 SS latent 框架，使点云注入成为可能
- **VGGT**：前馈 3D 预测器，为没有主动传感器的场景提供点云输入
- **VoxHammer**：同样使用 3D 先验但采用 3D 反演策略，在补全未知区域时效果不佳
- **启发**：这种"将先验编码为 latent 初始化 + inpainting 补全"的范式有望推广到其他条件生成任务

## 评分

- 新颖性: ⭐⭐⭐⭐ (潜空间 inpainting 视角新颖，范式清晰)
- 实验充分度: ⭐⭐⭐⭐ (物体+场景+真实图像+消融，比较全面)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，问题-方案-结果逻辑线流畅)
- 价值: ⭐⭐⭐⭐⭐ (为3D生成引入显式3D先验的范式，LiDAR/RGBD+生成的应用前景广阔)
