---
title: >-
  [论文解读] ZeroStereo: Zero-shot Stereo Matching from Single Images
description: >-
  [ICCV2025][3D视觉][stereo matching] 提出 ZeroStereo 管线：从任意单张图像出发，利用单目深度估计生成伪视差，再用微调的扩散修复模型合成高质量右视图，实现只需 35K 合成数据即达到 SOTA 零样本立体匹配泛化性能。
tags:
  - ICCV2025
  - 3D视觉
  - stereo matching
  - zero-shot generalization
  - 扩散模型
  - 深度估计
  - view synthesis
---

# ZeroStereo: Zero-shot Stereo Matching from Single Images

**会议**: ICCV2025  
**arXiv**: [2501.08654](https://arxiv.org/abs/2501.08654)  
**代码**: [GitHub](https://github.com/Windsrain/ZeroStereo)  
**领域**: 3d_vision  
**关键词**: stereo matching, zero-shot generalization, diffusion inpainting, monocular depth estimation, view synthesis

## 一句话总结

提出 ZeroStereo 管线：从任意单张图像出发，利用单目深度估计生成伪视差，再用微调的扩散修复模型合成高质量右视图，实现只需 35K 合成数据即达到 SOTA 零样本立体匹配泛化性能。

## 研究背景与动机

- **核心痛点**：有监督立体匹配模型在标准基准上表现优异，但因真实世界标注立体数据极度稀缺，泛化到真实场景时性能大幅下降。
- **现有方案的局限**：
    1. **域不变特征学习**（DSMNet、GraftNet、ITSA 等）：合成数据与真实数据之间仍存在不可忽视的域差距。
    2. **自监督学习**（光度损失）：受遮挡、鬼影伪影和病态区域影响严重，且大规模采集高质量立体图对本身也不简单。
    3. **视图合成**：
        - 早期方法（Luo et al., MfS-Stereo）用单目深度 + 前向扭曲生成右图，但遮挡区域只能用邻近像素或随机背景填充，导致结构不一致。
        - NeRF-Stereo 需要多视角输入做场景重建，灵活性差；NeRF 对远景重建质量不佳，不适合大规模户外场景。
- **动机**：能否只用单张图像 + 预训练扩散模型来高质量补全遮挡区域，同时无需额外训练即可评估伪标签的可靠性？

## 方法详解

### 总体流水线

给定单张左图 $\mathbf{I}_l$：

1. **单目深度估计**：用 Depth Anything V2 (DAv2) 得到归一化逆深度图 $\mathbf{D}$。
2. **自适应视差选择 (ADS)**：将 $\mathbf{D}$ 乘以缩放因子 $s \cdot w$ 得到伪视差图 $\mathbf{d}$，$s$ 从分段均匀分布中采样。
3. **前向扭曲**：按照 MfS-Stereo 的方法生成扭曲右图 $\tilde{\mathbf{I}}_r$、非遮挡掩码 $\mathbf{M}_{noc}$ 和修复掩码 $\mathbf{M}_{inp}$。
4. **扩散修复**：微调的 SDv2I 对遮挡区域做语义一致的补全，得到最终右图 $\mathbf{I}_r$。
5. **无训练置信度生成 (TCG)**：利用水平翻转对称性计算深度估计的置信度 $\mathbf{C}$。
6. **立体训练**：用 ZeroStereo Loss 训练 RAFT-Stereo / IGEV-Stereo。

### 扩散修复模型 (Diffusion Inpainting)

- **基础模型**：Stable Diffusion V2 Inpainting (SDv2I)。
- **为什么需要微调**：
    - 标准文本引导修复没有适用于立体遮挡区域的文本提示。
    - 立体遮挡掩码形状多样且不规则，预训练模型直接使用效果欠佳。
- **微调协议**：
    - 用 Scene Flow、Tartan Air、CREStereo Dataset、VKITTI 2 等合成数据集（有精确稠密视差 GT）。
    - 冻结 VAE，仅微调 U-Net；禁用文本条件；DDPM 噪声调度器 1000 步。
    - 50K 步训练，batch size 32（梯度累积 4 步），AdamW + One-cycle LR 2e-5，裁剪 512×512。
- **推理协议**：DDIM 调度器，50 步采样。最终输出通过掩码混合：$\mathbf{I}_r = \mathbf{M}_{inp} \odot \mathbf{I}_d + (1-\mathbf{M}_{inp}) \odot \tilde{\mathbf{I}}_r$。
- **优势**：相比 StereoDiffusion（在潜空间扭曲导致结构失真），内存仅需 5.8G vs 14.6G，速度 1.9s vs 31.2s。

### 无训练置信度生成 (Training-Free Confidence Generation, TCG)

- **核心思想**：现代单目深度模型预测相对深度（逆深度），水平翻转后像素间相对深度应保持一致。
- **具体做法**：
    1. 对左图做水平翻转，分别送入 DAv2 得到 $\mathbf{D}$ 和 $\mathbf{D}'$。
    2. 将翻转后的深度反翻转对齐，计算逐像素差异：$\mathbf{u} = 1 - |\mathbf{D} - \mathbf{H}^{-1}(\mathbf{D}')|$。
    3. 归一化得到置信度 $\mathbf{C}$。
- **效果**：低置信度区域集中在边缘、无纹理区域和细小物体上——这些恰恰是立体匹配中最具歧义的位置。

### 自适应视差选择 (Adaptive Disparity Selection, ADS)

- **问题**：MfS-Stereo 固定范围均匀采样最大视差，低分辨率时视差/宽度比过大引起前景畸变和过度遮挡；高分辨率时比值过小导致左右图差异不明显。
- **解决**：视差 $\mathbf{d} = \mathbf{D} \cdot s \cdot w$，$s$ 从三段式分布中采样：
    - 中心段 $(c-r, c+r)$，概率 $p_c = 0.8$（主要工作区间）。
    - 小视差段 $(c-2r, c-r)$，概率 $p_s = 0.1$。
    - 大视差段 $(c+r, c+2r)$，概率 $p_l = 0.1$。
    - 默认 $c=0.1, r=0.05$。
- **效果**：按图像宽度自适应调节视差范围，既保证多样性又避免极端情况。

### ZeroStereo Loss

综合三项损失：

1. **视差损失** $\mathcal{L}_d = \|\tilde{\mathbf{d}} - \mathbf{d}\|_1$。
2. **非遮挡光度损失** $\mathcal{L}_{np}$：对估计视差反向扭曲右图后与左图比较，用 SSIM + L1 组合（$\beta=0.85$），并用非遮挡掩码和反扭曲修复掩码排除不可靠像素。
3. **ZeroStereo Loss**：$\mathcal{L}_{Zero} = \mathbf{C} \odot \mathcal{L}_d + \mu \cdot (1-\mathbf{C}) \odot \mathcal{L}_{np}$，$\mu=0.1$。**高置信度区域以伪 GT 监督为主，低置信度区域依赖光度一致性自监督**。

## 实验关键数据

### 消融实验（IGEV-Stereo，Tab.1）

| 模块组合 | KITTI-15 EPE | KITTI-15 >3px | Midd-T EPE | ETH3D >1px |
|----------|-------------|---------------|------------|------------|
| Baseline | 1.52 | 4.89 | 2.71 | 2.38 |
| +ADS | 1.24 | 4.84 | 2.28 | 2.27 |
| +Inpainting | 1.44 | 4.85 | 2.34 | 1.92 |
| +ADS+Inpainting | 1.06 | 4.74 | 2.26 | 2.05 |
| +ADS+Inp+TCG | 1.05 | 4.71 | 2.18 | 2.01 |
| **全部 + ZeroStereo Loss** | **1.04** | **4.73** | **2.09** | **1.90** |

### 零样本泛化基准（Tab.8, Zero-RAFT-Stereo）

| 数据集 | KITTI-15 >3px (All) | Midd-T H >2px (Noc) | ETH3D >1px (Noc) |
|--------|--------------------|--------------------|-------------------|
| Zero-RAFT-Stereo | **4.53** | **4.45** | **2.13** |
| NS-RAFT-Stereo (NeRF-Stereo 官方) | 5.41 | 6.45 | 2.55 |
| RAFT-Stereo (SceneFlow GT) | 5.47 | 8.66 | 2.29 |

### 数据集规模对比（Tab.5, Zero-RAFT-Stereo）

仅用 35K 合成数据（MfS35K），性能优于 FoundationStereo 的 1106K 数据。数据多样性比绝对数量更重要。

### 合成效率（Tab.4）

| 方法 | 分辨率 | 显存 | 单图时间 |
|------|--------|------|----------|
| RePaint | 256×256 | 2.7G | 156.5s |
| StereoDiffusion | 512×512 | 14.6G | 31.2s |
| **ZeroStereo (Ours)** | 512×512 | **5.8G** | **1.9s** |

## 亮点与洞察

1. **范式巧妙**：将"立体数据生成"拆解为单目深度 → 前向扭曲 → 扩散修复三个模块化阶段，每个阶段利用各自领域的最强预训练模型，避免端到端从头训练。
2. **TCG 无需额外训练**：利用单目深度模型的翻转对称性这一免费信号评估置信度，思路简洁高效，可迁移到其他伪标签场景。
3. **ADS 自适应分辨率**：解决了以往固定视差范围与图像分辨率不匹配的实际工程问题。
4. **35K 数据胜过百万级**：证明了基于真实单图合成的训练数据因场景多样性天然优于规模更大的纯合成数据集。
5. **边缘误差分析**（Tab.7）：MfS35K 训练的模型在边缘区域误差最低，说明扩散修复在遮挡边界的补全质量确实优于其他方法。

## 局限与展望

1. **微调的 SDv2I 在复杂场景仍会失败**：如透明物体、网状结构等病态区域，前向扭曲本身就无法正确处理。
2. **偶尔出现颜色不一致**：因微调使用的是合成数据集，合成图与真实图的颜色分布有差异。
3. **依赖单目深度模型质量**：整个管线的天花板由 DAv2 的精度决定，若 DAv2 在某些场景失败则所有下游都会受影响。
4. **可能的改进**：
    - 替换更先进的扩散模型（如 SDXL Inpainting 或 Flux）以提升修复质量。
    - 引入多尺度视差生成策略。
    - 结合视频扩散模型实现时间一致的立体视频数据生成。
    - TCG 的翻转不变性假设在包含文字、方向性强的场景中可能不成立，可以考虑旋转等更多几何变换。

## 相关工作与启发

- **MfS-Stereo** [Watson et al.]：本文的直接前驱，用单目深度 + 前向扭曲生成立体对，但遮挡区域用随机背景填充。
- **NeRF-Stereo** [Tosi et al.]：用 NeRF 重建 3D 场景再渲染立体对，引入 Ambient Occlusion 做置信度，但需多视角输入且远景重建差。
- **Marigold** [Ke et al.]：证明扩散模型可从合成数据微调到单目深度估计，启发本文将类似思路用于立体图合成。
- **Depth Anything V2**：本文使用的单目深度模型，为伪视差提供基础。
- **Stable Diffusion V2 Inpainting**：本文微调的基础修复模型。
- **启发**：这种"用一个领域的强预训练模型为另一个领域自动生成训练数据"的范式，有望推广到光流估计、场景流估计等更多任务。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 各模块单独看并不全新，但整体管线设计和 TCG 的翻转对称性思路较为精巧
- 实验充分度: ⭐⭐⭐⭐⭐ — 消融实验完整，多数据集多模型评估，与 NeRF-Stereo 公平对比，还有边缘误差和合成效率分析
- 写作质量: ⭐⭐⭐⭐ — 论文结构清晰，图表信息量大，符号使用规范
- 价值: ⭐⭐⭐⭐ — 实用性强，35K 数据即可达 SOTA，对计算资源有限的场景意义显著

<!-- RELATED:START -->

## 相关论文

- [RobuSTereo: Robust Zero-Shot Stereo Matching under Adverse Weather](robustereo_robust_zero-shot_stereo_matching_under_adverse_weather.md)
- [FoundationStereo: Zero-Shot Stereo Matching](../../CVPR2025/3d_vision/foundationstereo_zero-shot_stereo_matching.md)
- [Stereo Any Video: Temporally Consistent Stereo Matching](stereo_any_video_temporally_consistent_stereo_matching.md)
- [MVSAnywhere: Zero-Shot Multi-View Stereo](../../CVPR2025/3d_vision/mvsanywhere_zero-shot_multi-view_stereo.md)
- [Lite Any Stereo: Efficient Zero-Shot Stereo Matching](../../CVPR2026/3d_vision/lite_any_stereo_efficient_zero-shot_stereo_matching.md)

<!-- RELATED:END -->
