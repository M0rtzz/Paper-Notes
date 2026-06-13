---
title: >-
  [论文解读] Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking
description: >-
  [CVPR 2026][图像生成][扩散模型图像编辑] 本文从理论和实验两方面统一分析了非对抗性扩散编辑如何无意中破坏鲁棒隐形水印，推导了水印 SNR 衰减和互信息衰减的界，并在指令编辑、拖拽编辑、无训练合成等场景下验证了水印恢复的系统性失效。 水印鲁棒性假设面临挑战：现有深度学习水印（StegaStamp、TrustMar…
tags:
  - "CVPR 2026"
  - "图像生成"
  - "扩散模型图像编辑"
  - "鲁棒水印"
  - "水印退化"
  - "信息论分析"
  - "数字水印安全"
---

# Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking

**会议**: CVPR 2026  
**arXiv**: [2603.12949](https://arxiv.org/abs/2603.12949)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 扩散模型图像编辑, 鲁棒水印, 水印退化, 信息论分析, 数字水印安全

## 一句话总结

本文从理论和实验两方面统一分析了非对抗性扩散编辑如何无意中破坏鲁棒隐形水印，推导了水印 SNR 衰减和互信息衰减的界，并在指令编辑、拖拽编辑、无训练合成等场景下验证了水印恢复的系统性失效。

## 研究背景与动机

**水印鲁棒性假设面临挑战**：现有深度学习水印（StegaStamp、TrustMark、VINE 等）通过端到端训练在 JPEG 压缩、缩放、裁剪等常规后处理下保持高恢复率，但其训练分布并未覆盖扩散编辑这一全新变换族。

**扩散编辑的本质不同于传统攻击**：扩散编辑通过大幅加噪再去噪的方式重建图像，依赖强大的生成先验。水印作为低幅度结构化扰动，会被去噪器当作"不自然残余"而移除——即使用户完全没有移除水印的意图。

**编辑方式日益多样化**：从文本指令编辑（InstructPix2Pix、UltraEdit）到交互式拖拽编辑（DragDiffusion、DragFlow）再到无训练合成（TF-ICON、SHINE），扩散编辑生态系统持续扩展，对水印构成系统性威胁。

**已有工作缺乏统一分析**：此前关于扩散重生成攻击的研究只聚焦特定水印或特定攻击，缺少将常规编辑工作流作为系统性压力测试的综合理论框架。

**水印与内容溯源基础设施的可靠性存疑**：水印正被部署为版权保护和内容溯源基础设施，但若常规编辑即可无意破坏水印，则下游溯源声明的可靠性将受到根本质疑。

**核心研究问题**：在何种条件下，扩散图像编辑会无意损害鲁棒水印恢复？哪些理论原则解释了观察到的失效？

## 方法详解

### 整体框架

论文将扩散编辑形式化为作用于水印图像的 **Markov 核**：

$$K_{\mathcal{T}}(\tilde{\mathbf{x}}|\mathbf{x}_w, \mathbf{y}) = \int p(\mathbf{x}_{t^\star}|\mathbf{x}_w) \, p_\theta(\tilde{\mathbf{x}}|\mathbf{x}_{t^\star}, \mathbf{y}) \, d\mathbf{x}_{t^\star}$$

其中 $p(\mathbf{x}_{t^\star}|\mathbf{x}_w)$ 为前向加噪过程（加噪到强度 $t^\star$），$p_\theta$ 为条件反向去噪过程。不同编辑器对应 $p_\theta$ 的不同参数化：指令编辑学习条件去噪器，拖拽编辑在潜空间优化后重采样，合成框架通过注意力/适配器引导去噪。

水印信号被建模为加性残差：$\mathbf{x}_w = \mathbf{x} + \gamma \mathbf{s}(\mathbf{m}, \mathbf{k}, \mathbf{x})$，其中 $\mathbf{s}$ 为有界能量嵌入信号，$\gamma$ 控制强度。

### 关键设计

**1. SNR 衰减分析**：前向加噪过程将水印图像映射为 $\mathbf{x}_{t^\star} = \sqrt{\bar\alpha_{t^\star}} \mathbf{x}_w + \sqrt{1-\bar\alpha_{t^\star}} \epsilon$，水印 SNR 随 $t^\star$ 增大而单调递减。当 $\bar\alpha_{t^\star}$ 足够小时，水印信号被噪声完全淹没。

**2. 互信息衰减界**：论文推导了去噪后水印载荷与观测图像间互信息的上界，并连接到 Fano 不等式，得出比特错误率的下界——当编辑强度超过阈值时，可靠恢复在信息论意义上不可能。

**3. 频域分析**：定义频谱保留比 $\rho_\Omega$ 来量化水印能量在低/中/高频段的存活率。扩散去噪在高频段的抑制最强，而多数水印恰恰将能量集中在中高频以保持不可见性，形成结构性矛盾。

**4. DEW-ST 评估协议**：提出标准化的扩散编辑水印压力测试（Algorithm 1），覆盖指令编辑、区域编辑、拖拽编辑、合成编辑四大类，每类在多个强度 $t^\star \in \{0.2, 0.4, 0.6, 0.8\}$ 下测试。

### 损失函数/训练策略

论文提出**扩散增强水印训练**的概念性框架（Algorithm 2）：

$$\min_{E,D} \mathbb{E}_{\mathbf{x},\mathbf{m},j,\xi} [\ell_{\mathrm{rec}}(D(\mathcal{T}_j(E(\mathbf{x},\mathbf{m}));\xi), \mathbf{m})] + \lambda \mathbb{E}_{\mathbf{x},\mathbf{m}} [\ell_{\mathrm{qual}}(E(\mathbf{x},\mathbf{m}), \mathbf{x})]$$

在训练中随机采样扩散编辑器 $\mathcal{T}_j$ 和强度 $s$ 作为数据增强，使水印学会在生成式变换下存活。但论文指出这仅是防御模板，实际部署需轻量化代理以降低计算成本。

## 实验关键数据

### 主实验

**表4：不同变换下水印比特准确率（%），随机猜测 ≈50%**

| 变换 | 强度 | StegaStamp | TrustMark | VINE |
|------|------|-----------|-----------|------|
| 无编辑 | – | 99.4 | 99.7 | 99.8 |
| JPEG (Q=50) | – | 96.1 | 98.2 | 98.9 |
| InstructPix2Pix | $t^\star$=0.4 | 71.5 | 76.1 | 85.4 |
| InstructPix2Pix | $t^\star$=0.8 | 53.2 | 55.0 | 60.7 |
| DragDiffusion | 中等 | 63.4 | 67.9 | 78.6 |
| DragFlow | 中等 | 60.8 | 65.1 | 76.9 |
| TF-ICON 合成 | – | 58.9 | 63.2 | 74.8 |
| SHINE 插入 | – | 55.6 | 60.4 | 72.2 |

**表5：按编辑类型分解（中等强度）**

| 编辑类型 | StegaStamp | TrustMark | VINE |
|----------|-----------|-----------|------|
| 风格迁移 | 54.0 | 56.8 | 62.5 |
| 光照变化 | 60.7 | 65.2 | 74.6 |
| 物体替换 | 58.3 | 63.9 | 73.1 |
| 局部修补 | 74.6 | 79.2 | 88.1 |
| 拖拽编辑 | 63.4 | 67.9 | 78.6 |

### 消融实验

**编辑强度 $t^\star$ 的影响（InstructPix2Pix）**：所有方法的比特准确率随 $t^\star$ 单调下降。StegaStamp 从 $t^\star$=0.2 的 86.7% 降至 $t^\star$=0.8 的 53.2%；VINE 从 93.5% 降至 60.7%。多种子投票仅带来微弱改善（~1%），说明失效是系统性信号收缩而非随机损坏。

**分辨率影响**：256 嵌入后上采样 vs 512 直接嵌入对常规后处理差异不大，但在强编辑下均接近随机猜测。

**频谱保留比**：高频段 $\rho_{\mathrm{high}}$ 在所有编辑器下均低于 0.22（VINE）或 0.15（StegaStamp），确认扩散去噪是高频水印残余的强抑制器。

**ECC 解码**：纠错码在弱编辑下可提升消息恢复（VINE 从 85.4% BA→55.6% MsgAcc），但在强编辑下完全失效（60.7% BA→2.1% MsgAcc），因错误已接近随机。

### 关键发现

1. **扩散编辑与传统后处理存在质的差异**：JPEG/缩放等传统变换下三种水印均保持 >92% 准确率，但中等强度扩散编辑即可降至 60-85%，强编辑接近随机猜测。
2. **"局部"编辑不等于"水印安全"**：由于扩散去噪在潜空间耦合像素，即使只编辑小区域也可能影响全局分布的水印信号。
3. **扩散原生水印（Tree-Ring、Stable Signature）在跨模型编辑下同样脆弱**：同模型编辑 AUC 仍可达 0.89-0.92，但跨模型编辑降至 0.58-0.65。
4. **高视觉保真度不等于水印保留**：编辑后 PSNR/SSIM 与水印恢复率之间无正相关。

## 亮点与洞察

- **理论与实验的统一性**出色：SNR 衰减→互信息衰减→Fano 界的理论链条清晰，且与实验中比特准确率随编辑强度下降的趋势高度吻合
- **评估覆盖面广**：横跨指令/拖拽/合成三大编辑范式，三种代表性水印，四个编辑强度，构成目前最全面的扩散编辑-水印交互基准
- **频域分析提供机制性解释**：$\rho_\Omega$ 指标清楚揭示了水印高频能量被去噪器优先清除的结构性原因
- **防御方向清晰**：指出扩散韧性水印应 (i) 集成到生成过程或 (ii) 优化语义不变性，而非仅在传统噪声层上增强

## 局限性

- 实验数据为"illustrative/hypothetical"值，虽声称与文献趋势一致，但未提供真实实验验证，降低了说服力
- 理论分析依赖水印的加性残差近似（Assumption 3.1），对非线性嵌入（如基于注意力或 VAE 潜空间的方法）适用性待验证
- DEW-ST 协议开销大（每张图×16条指令×4强度×3种子），实际部署可行性存疑
- 未深入讨论视频水印和多模态水印在扩散编辑下的表现
- 防御方案（Algorithm 2）仅为概念性框架，缺乏实际训练和验证

## 相关工作

- **鲁棒水印**：HiDDeN、StegaStamp、TrustMark、VINE、RoSteALS、Watermark Anything——本文选取后三者作为代表性基线
- **扩散编辑**：SDEdit、Prompt-to-Prompt、InstructPix2Pix、UltraEdit、DragDiffusion、DragFlow、TF-ICON、SHINE——构成本文评估的编辑器生态
- **扩散原生水印**：Tree-Ring、Stable Signature、SynthID——作为对比，说明生成器集成方案在跨模型场景下同样脆弱
- **水印攻击与移除**：Zhao et al. 的重生成攻击可证明性分析、Ni et al. 的扩散攻击——本文区别在于聚焦**无意**移除而非对抗性攻击
- **概念擦除**：MACE、ANT、EraseAnything——说明扩散模型可选择性抑制特定信号，暗示水印面临的结构性风险

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将扩散编辑统一建模为 Markov 核并推导水印信息论失效条件
- 实验充分度: ⭐⭐⭐ — 覆盖面广但数据为假设性值，未提供真实实验
- 写作质量: ⭐⭐⭐⭐ — 理论推导严谨清晰，符号体系统一，叙事结构合理
- 价值: ⭐⭐⭐⭐ — 对水印安全社区有重要警示，评估协议有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing](rel-zero_harnessing_patch-pair_invariance_for_robust_zero-watermarking_against_a.md)
- [\[CVPR 2026\] Towards Robust Content Watermarking Against Removal and Forgery Attacks](towards_robust_content_watermarking_against_removal_and_forgery_attacks.md)
- [\[CVPR 2026\] SPDMark: Selective Parameter Displacement for Robust Video Watermarking](spdmark_selective_parameter_displacement_for_robust_video_watermarking.md)
- [\[CVPR 2026\] TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking](trace_structure-aware_character_encoding_for_robust_and_generalizable_document_w.md)
- [\[CVPR 2026\] Image Generation as a Visual Planner for Robotic Manipulation](image_generation_as_a_visual_planner_for_robotic_manipulation.md)

</div>

<!-- RELATED:END -->
