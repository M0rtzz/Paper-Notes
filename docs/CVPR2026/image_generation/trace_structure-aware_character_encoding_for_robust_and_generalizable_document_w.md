---
title: >-
  [论文解读] TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking
description: >-
  [CVPR 2026][图像生成][document watermarking] 提出 TRACE——基于字符结构编码的文档水印框架，利用扩散模型（DragDiffusion）精确移动字符骨架关键点来嵌入信息，通过自适应扩散初始化（ADI）、引导扩散编码（GDE）和掩码区域替换（MRR）三大组件，同时实现跨介质传输鲁棒性、多语言/多字体泛化性和高隐蔽性。
tags:
  - CVPR 2026
  - 图像生成
  - document watermarking
  - data hiding
  - 扩散模型
  - character structure
  - cross-media robustness
---

# TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking

**会议**: CVPR 2026  
**arXiv**: [2603.12873](https://arxiv.org/abs/2603.12873)  
**代码**: 待确认  
**领域**: image_generation  
**关键词**: document watermarking, data hiding, diffusion model, character structure, cross-media robustness

## 一句话总结

提出 TRACE——基于字符结构编码的文档水印框架，利用扩散模型（DragDiffusion）精确移动字符骨架关键点来嵌入信息，通过自适应扩散初始化（ADI）、引导扩散编码（GDE）和掩码区域替换（MRR）三大组件，同时实现跨介质传输鲁棒性、多语言/多字体泛化性和高隐蔽性。

## 研究背景与动机

**文档水印的三难困境**：现有文档隐写方法难以同时满足鲁棒性、泛化性和隐蔽性——
   - **基于图像的方法**（像素翻转）：通过调整黑白像素比嵌入数据，但跨介质传输（打印-扫描-拍照）引入的噪声会严重破坏像素特征
   - **基于字体的方法**（预定义码本）：将原字符替换为码本设计的变体，鲁棒性较好但泛化性受限——码本无法覆盖所有可能出现的字符，遇到手写体或艺术字体即失效
   - **基于格式的方法**（行间距/词间距）：嵌入容量低、鲁棒性差

**字符结构的优势**：字符结构（骨架+关键点+连接线）具有三重天然优势：(a) 对噪声干扰稳定，跨介质传输后结构基本保持不变；(b) 提供跨语言/字体的统一表征——无论何种字符都可提取骨架；(c) 修改结构附近少量像素不会改变外观，保证隐蔽性。

**扩散模型图像编辑的新能力**：DragDiffusion 等点对点图像编辑方法提供了精确的局部像素操控能力，为基于结构的字符水印提供了技术基础。

## 方法详解

### 整体框架

TRACE 包含**数据嵌入**和**数据提取**两个阶段。嵌入阶段由三步组成：自适应扩散初始化（ADI）→ 引导扩散编码（GDE）→ 掩码区域替换（MRR）。

### 关键设计 1：自适应扩散初始化（ADI）

给定文本图像 $I_{\text{cover}}$，ADI 确定三个引导扩散过程的关键元素：

**关键点检测**：基于轻量 OpenPose 架构提取端点集 $E$ 和交叉点集 $C$，输出三通道热力图（端点/交叉点/背景）。

**Movement Probability Evaluator（MPE）**：自动选择最优 handle point $P_h$ 和 reference point $P_r$。
- 仅考虑端点作为候选 $P_h$（交叉点连接多笔画，移动会破坏结构）
- 对每个端点 $p_i^e$，在 $\tau$ 邻域内寻找参考点集 $R_i$
- 评分规则：初始分 1；若 $p_i^e$ 与 $p_{i,j}^r$ 不在同一笔画上加 1 分；多个满分时 y 坐标最小者再加 1 分
- 最高分的端点成为 $P_h$，对应参考点为 $P_r$

**Target Point Estimation（TPE）**：根据嵌入比特值确定目标点 $P_t$。

定义方向轴 $\lambda$：

$$\lambda\text{-axis} = \begin{cases} X\text{-axis}, & d_x \leq d_y \\ Y\text{-axis}, & d_x > d_y \end{cases}$$

其中 $d_x = |x_h - x_r|, d_y = |y_h - y_r|$，$\Delta(P_h, P_r) = \min\{d_x, d_y\}$。

嵌入规则：
- 比特 0：若 $\Delta(P_h, P_r) > T_{\text{embed}}$，移动 $P_h$ 使 $\Delta(P_t, P_r) \leq T_{\text{embed}}$
- 比特 1：若 $\Delta(P_h, P_r) \leq T_{\text{embed}}$，移动 $P_h$ 使 $\Delta(P_t, P_r) > T_{\text{embed}}$

移动方向由笔画方向向量 $\vec{\mathcal{V}}$ 和 $P_h$ 到 $P_r$ 的方向向量 $\vec{\mathcal{H}}$ 联合确定：

$$x_t = x_h + \mathcal{D} \times \frac{\mathcal{V}_x}{\|\vec{\mathcal{V}}\|} \times \text{sgn}(\mathcal{H}_x)$$

**Mask Drawing Module（MDM）**：基于 $P_h$ 和 $P_t$ 构造最小编辑区域矩形掩码 $\mathcal{M}$，并扩展边界 $\sigma$ 确保扩散质量。

### 关键设计 2：引导扩散编码（GDE）

利用 DragDiffusion 将 $P_h$ 移动到 $P_t$：
1. LoRA 微调 UNet 以捕获原始图像特征
2. DDIM 反演生成初始扩散潜变量
3. 运动监督 + 点跟踪迭代优化，直到 handle point 对齐目标点
4. 参考潜变量控制：在 self-attention 中用初始潜变量的 key/value 替换编辑潜变量的，保持一致性

引入**局部一致性损失** $L_{lc}$ 确保掩码区域内编辑前后特征一致：

$$L_{lc}(\hat{z}_t^k) = \sum_{q \in \Omega} \|G_{q+d}(\hat{z}_{t-1}^k) - \text{sg}(G_q(\hat{z}_{t-1}^0))\|_1$$

总损失：$L(\hat{z}_t^k) = L_{ms}(\hat{z}_t^k) + \eta L_{lc}(\hat{z}_t^k)$，其中 $\eta = 0.003$。

### 关键设计 3：掩码区域替换（MRR）

将扩散编辑后图像的掩码区域内容替换到原始图像对应区域，仅在目标区域嵌入数据，最小化对其他区域的影响。

### 数据提取

1. 使用 CRAFT 算法分割单个字符
2. 对每个字符运行 MPE 识别 $P_h$ 和 $P_r$
3. 计算 $\Delta(P_h, P_r)'$：若 $> T_{\text{embed}}$ 提取 1，否则提取 0

## 实验结果

### 截图鲁棒性（ACC, %）

| 字体 | 方法 | 12pt | 16pt | 20pt | 24pt | 28pt | 36pt |
|------|------|------|------|------|------|------|------|
| Arial | ASF | 85.83 | 91.67 | 90.00 | 87.50 | 88.33 | 82.50 |
| Arial | **TRACE** | **96.67** | **97.50** | **99.17** | **100** | **100** | **100** |
| Calibri | ASF | 95.00 | 95.83 | 96.67 | 98.33 | 97.50 | 100 |
| Calibri | **TRACE** | **97.50** | **99.17** | **99.17** | **100** | **100** | **100** |

### 打印-扫描鲁棒性

| 字体 | 方法 | 12pt | 16pt | 20pt | 24pt | 28pt | 36pt |
|------|------|------|------|------|------|------|------|
| Arial | ASF | 80.83 | 68.33 | 70.83 | 72.50 | 70.00 | 76.67 |
| Arial | **TRACE** | **95.83** | **97.50** | **99.17** | **99.17** | **100** | **100** |
| TNR | ASF | 64.17 | 72.50 | 79.17 | 68.33 | 79.17 | 72.50 |
| TNR | **TRACE** | **92.50** | **94.17** | **95.83** | **97.50** | **99.17** | **99.17** |

### 隐蔽性对比

| 指标 | StegaStamp | IHA | **TRACE** |
|------|-----------|-----|-----------|
| 截图 ACC | 100 | 84.58 | **100** |
| 打印扫描 ACC | 98.54 | 84.29 | **99.05** |
| 拍照 ACC | 98.12 | 83.94 | **98.75** |
| PSNR↑ | 27.19 | 29.60 | **33.34** |
| SSIM↑ | 0.8986 | 0.9910 | **0.9962** |

TRACE 在鲁棒性和隐蔽性上同时达到最优，PSNR 比 StegaStamp 高 6+ dB。

### 泛化性

- 手写体：截图 94.43%、打印扫描 93.17%、拍照 91.67%，PSNR 38.20
- 艺术字体：截图 97.27%、打印扫描 94.77%、拍照 92.93%，PSNR 41.37
- 成功扩展到中文、日文等多语言和数学公式

### 消融实验

| 设计 | MPE | TPE | ACC |
|------|-----|-----|-----|
| Setting 1 | ✗ | ✗ | 49.95% |
| Setting 2 | ✓ | ✗ | 68.75% |
| Setting 3 | ✗ | ✓ | 53.50% |
| **Setting 4 (Ours)** | **✓** | **✓** | **100%** |

- MPE+TPE 协同工作才能实现无误提取
- $L_{lc}$ 损失显著改善掩码区域形状保持
- MRR 使 PSNR/SSIM 在不同字体上均获提升

## 优点与局限

**优点**：
- 首次将字符结构编码引入文档水印，开辟全新范式
- 同时解决鲁棒性-泛化性-隐蔽性三难困境，在所有维度上超越 SOTA
- ADI（MPE+TPE+MDM）的自动化设计精巧，确保编码-解码同步
- 支持预计算码本（常见字符）和动态生成（未知字符）的混合模式
- 对抗结构形变攻击仍保持 96%+ 提取精度

**局限**：
- 每字符仅嵌入 1 bit，嵌入容量较低
- 基于 DragDiffusion 的编码过程需要 LoRA 微调和 DDIM 反演，计算成本较高
- 对于笔画极少的字符（如"一"、"I"），可用关键点有限

## 个人评价

⭐⭐⭐⭐

这是一篇思路非常巧妙的工作——利用字符骨架结构的天然稳定性作为水印载体，完美匹配了文档水印的核心需求。MPE 的自动化评分机制设计精细，TPE 基于笔画方向的移动策略保证了编码-解码的同步性。实验覆盖中英文 × 多字体 × 多尺寸 × 多传输通道（截图/打印扫描/拍照），验证非常充分。在已有方法难以兼顾的鲁棒性-泛化性-隐蔽性三维空间中实现了全面领先。1 bit/字符的容量限制和扩散模型的计算开销是主要瓶颈，但作为范式创新的首作，贡献值得肯定。

<!-- RELATED:START -->

## 相关论文

- [Towards Robust Content Watermarking Against Removal and Forgery Attacks](towards_robust_content_watermarking_against_removal_and_forgery_attacks.md)
- [SPDMark: Selective Parameter Displacement for Robust Video Watermarking](spdmark_selective_parameter_displacement_for_robust_video_watermarking.md)
- [Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](editing_away_the_evidence_diffusionbased_image_man.md)
- [Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing](rel-zero_harnessing_patch-pair_invariance_for_robust_zero-watermarking_against_a.md)
- [SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models](segquant_diffusion_model_quantization.md)

<!-- RELATED:END -->
