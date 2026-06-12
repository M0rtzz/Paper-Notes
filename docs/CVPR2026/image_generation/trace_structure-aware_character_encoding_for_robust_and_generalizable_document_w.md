---
title: >-
  [论文解读] TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking
description: >-
  [CVPR 2026][图像生成][document watermarking] 提出 TRACE——基于字符结构编码的文档水印框架，利用扩散模型（DragDiffusion）精确移动字符骨架关键点来嵌入信息，通过自适应扩散初始化（ADI）、引导扩散编码（GDE）和掩码区域替换（MRR）三大组件…
tags:
  - "CVPR 2026"
  - "图像生成"
  - "document watermarking"
  - "data hiding"
  - "扩散模型"
  - "character structure"
  - "cross-media robustness"
---

# TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking

**会议**: CVPR 2026  
**arXiv**: [2603.12873](https://arxiv.org/abs/2603.12873)  
**代码**: 待确认  
**领域**: 图像生成  
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

TRACE 想同时啃下文档水印的「鲁棒性—泛化性—隐蔽性」三难：既要扛得住打印-扫描-拍照的跨介质噪声，又要不挑语言字体，还得肉眼看不出改动。它的突破口是改字符的**结构**而非像素——字符骨架的关键点对噪声稳定、跨语言统一、微调附近像素又不改外观。整条流水线分嵌入与提取两段：嵌入端依次做自适应扩散初始化（ADI）→ 引导扩散编码（GDE）→ 掩码区域替换（MRR），提取端则反过来从字符结构读回比特。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入文档字符 + 待嵌比特"] --> B["自适应扩散初始化 ADI<br/>骨架检测 → MPE 选点 → TPE 定目标点 → MDM 画掩码"]
    B --> C["引导扩散编码 GDE<br/>DragDiffusion 把 handle point 拖到目标点"]
    C --> D["掩码区域替换 MRR<br/>仅替换掩码内像素，区域外原样保留"]
    D --> E["含水印文档"]
    E -->|打印-扫描-拍照后的可疑文档| F["提取端：CRAFT 分割单字符<br/>复跑 MPE 定位 handle/reference point"]
    F --> G["按阈值 T_embed 比较 Δ 判 0/1<br/>读回比特"]
```

### 关键设计

**1. 自适应扩散初始化（ADI）：决定动哪个点、往哪动、动多远**

要把比特编码进结构，先得确定「移动哪个关键点、移到哪、只在多大范围内动」，这一步直接决定了编码-解码能否对齐。ADI 先用一个轻量 OpenPose 架构检测端点集 $E$ 和交叉点集 $C$（输出端点/交叉点/背景三通道热力图），再由三个子模块定下编辑方案。

**Movement Probability Evaluator（MPE）** 自动挑选 handle point $P_h$ 和 reference point $P_r$。只把端点当 $P_h$ 候选（交叉点连多笔画、移动会破坏结构），对每个端点 $p_i^e$ 在 $\tau$ 邻域内找参考点集 $R_i$；评分规则为初始 1 分，$p_i^e$ 与 $p_{i,j}^r$ 不在同一笔画上加 1 分，多个满分时 y 坐标最小者再加 1 分，最高分端点即 $P_h$、对应参考点为 $P_r$。

**Target Point Estimation（TPE）** 按要嵌入的比特定目标点 $P_t$。先定方向轴

$$\lambda\text{-axis} = \begin{cases} X\text{-axis}, & d_x \leq d_y \\ Y\text{-axis}, & d_x > d_y \end{cases}$$

其中 $d_x = |x_h - x_r|, d_y = |y_h - y_r|$，$\Delta(P_h, P_r) = \min\{d_x, d_y\}$。嵌入规则是：比特 0 时若 $\Delta(P_h, P_r) > T_{\text{embed}}$ 就移动 $P_h$ 使 $\Delta(P_t, P_r) \leq T_{\text{embed}}$；比特 1 时若 $\Delta(P_h, P_r) \leq T_{\text{embed}}$ 就反向移动使 $\Delta(P_t, P_r) > T_{\text{embed}}$。移动方向由笔画方向向量 $\vec{\mathcal{V}}$ 与 $P_h \to P_r$ 方向 $\vec{\mathcal{H}}$ 联合确定：

$$x_t = x_h + \mathcal{D} \times \frac{\mathcal{V}_x}{\|\vec{\mathcal{V}}\|} \times \text{sgn}(\mathcal{H}_x)$$

最后 **Mask Drawing Module（MDM）** 基于 $P_h$、$P_t$ 画出最小编辑矩形掩码 $\mathcal{M}$，并外扩边界 $\sigma$ 保证扩散质量。

**2. 引导扩散编码（GDE）：用 DragDiffusion 把关键点精确拖到目标位置**

定好方案后，真正「动点」交给 DragDiffusion：先 LoRA 微调 UNet 捕获原图特征、DDIM 反演生成初始扩散潜变量，再用运动监督 + 点跟踪迭代优化，直到 handle point 对齐 $P_t$；self-attention 里用初始潜变量的 key/value 替换编辑潜变量的，维持一致性。为了让掩码内编辑前后特征一致、不破坏字形，引入局部一致性损失

$$L_{lc}(\hat{z}_t^k) = \sum_{q \in \Omega} \|G_{q+d}(\hat{z}_{t-1}^k) - \text{sg}(G_q(\hat{z}_{t-1}^0))\|_1$$

总损失 $L(\hat{z}_t^k) = L_{ms}(\hat{z}_t^k) + \eta L_{lc}(\hat{z}_t^k)$，其中 $\eta = 0.003$。

**3. 掩码区域替换（MRR）：只把改动落回目标区域、其余像素原封不动**

扩散编辑难免对掩码外区域有轻微扰动，影响隐蔽性。MRR 的做法很直接：只把编辑后图像掩码区域内的内容替换回原图对应位置，其余区域保持原样，从而把改动严格限制在目标区域、最大化对外观的隐蔽性。

### 一个完整示例：从图像读回比特

提取端不需要原图，对每个字符复跑一遍 MPE 即可：先用 CRAFT 算法分割出单个字符，对每个字符运行 MPE 识别出 $P_h$ 和 $P_r$，再算 $\Delta(P_h, P_r)'$——若 $> T_{\text{embed}}$ 提取为 1，否则为 0。正因为编码端（TPE）和解码端用的是同一套 MPE 关键点定位 + 同一个阈值 $T_{\text{embed}}$，编码-解码才能严格同步、实现无误提取（消融里 MPE+TPE 齐备时 ACC 达 100%）。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Towards Robust Content Watermarking Against Removal and Forgery Attacks](towards_robust_content_watermarking_against_removal_and_forgery_attacks.md)
- [\[CVPR 2026\] SPDMark: Selective Parameter Displacement for Robust Video Watermarking](spdmark_selective_parameter_displacement_for_robust_video_watermarking.md)
- [\[ICML 2026\] Semantic-Aware Motion Encoding for Topology-Agnostic Character Animation](../../ICML2026/image_generation/semantic-aware_motion_encoding_for_topology-agnostic_character_animation.md)
- [\[CVPR 2026\] Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing](rel-zero_harnessing_patch-pair_invariance_for_robust_zero-watermarking_against_a.md)
- [\[CVPR 2026\] Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](editing_away_the_evidence_diffusion-based_image_manipulation_and_the_failure_mod.md)

</div>

<!-- RELATED:END -->
