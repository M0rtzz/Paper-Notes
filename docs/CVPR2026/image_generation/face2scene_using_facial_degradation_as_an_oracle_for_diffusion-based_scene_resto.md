---
title: >-
  [论文解读] Face2Scene: Using Facial Degradation as an Oracle for Diffusion-Based Scene Restoration
description: >-
  [CVPR2026][图像生成][图像复原] 提出 Face2Scene 两阶段框架：先用参考人脸复原模型(Ref-FR)获得 HQ-LQ 人脸对，从中提取退化编码作为"oracle"，再以此条件化单步扩散模型完成包含身体与背景的全场景图像复原。
tags:
  - CVPR2026
  - 图像生成
  - 图像复原
  - 扩散模型
  - 人脸参考
  - 退化估计
  - 全场景恢复
  - 单步推理
---

# Face2Scene: Using Facial Degradation as an Oracle for Diffusion-Based Scene Restoration

**会议**: CVPR2026  
**arXiv**: [2603.16570](https://arxiv.org/abs/2603.16570)  
**代码**: [项目主页](https://amirhossein-kz.github.io/face2scene/)  
**领域**: 图像生成  
**关键词**: 图像复原, 扩散模型, 人脸参考, 退化估计, 全场景恢复, 单步推理

## 一句话总结

提出 Face2Scene 两阶段框架：先用参考人脸复原模型(Ref-FR)获得 HQ-LQ 人脸对，从中提取退化编码作为"oracle"，再以此条件化单步扩散模型完成包含身体与背景的全场景图像复原。

## 研究背景与动机

**参考人脸复原局限于面部区域**：现有 Ref-FR 方法仅复原人脸裁剪区域，忽视了真实退化(噪声、模糊、压缩)往往均匀影响整个场景的事实，实际可用性有限。

**全场景复原器缺乏退化感知**：大多数全图复原方法仅以 LQ 图作为输入，问题不确定性高，容易产生视觉伪影；对人脸真实感和身份一致性也难以保持。

**退化估计信号粗糙**：S3Diff 仅从 LQ 图预测噪声和模糊两个全局标量；DeeDSR 通过无监督对比学习从 LQ 编码退化，但在复杂退化下学习效果受限，容易将纹理/光照误判为退化。

**人脸具有天然的退化探测优势**：人脸具有稳定的几何结构与可靠的关键点；身份参考图通常可获得；HQ-LQ 对差异可有效隔离退化与内容。

**场景级身份参考数据集匮乏**：公开数据集缺少同时包含身份参考和全场景 HQ 图像的配对数据，制约了研究可复现性。

**单步高效推理需求**：多步扩散模型(DiffBIR 34.6s, SUPIR 19.9s)推理开销大，实用场景需要兼顾质量与速度的单步方案。

## 方法详解

### 整体框架

Face2Scene 分两阶段：

- **Stage 1 — 参考人脸复原**：检测退化图 $I^{LQ}$ 中的人脸 → 对齐裁剪到 512×512 → Ref-FR 模型 $F_\theta$ 利用同身份参考图集 $\{x_k^{ref}\}$ 重建 HQ 人脸 → 反变换回原始坐标，获得空间对齐的 $(x^{HQ}, x^{LQ})$ 对。
- **Stage 2 — 退化条件化全场景复原**：FaDeX 从 HQ-LQ 人脸对提取退化编码 $Z_{face}$ → MapNet 将其映射为多尺度 token → token 与正文本嵌入拼接，通过 cross-attention 条件化 SD-Turbo 单步扩散模型，一步完成全场景复原。

### 关键设计

**FaDeX (Face-derived Degradation eXtractor)**

- 轻量卷积编码器 $E_\phi$ 接收 6 通道输入(HQ 与 LQ 通道拼接)，输出空间特征 $Z_{face} \in \mathbb{R}^{H' \times W' \times C}$。
- 经全局平均池化 + MLP 投射头得到单位归一化向量 $q$，采用 **对比学习** 训练：同一退化算子 $\mathcal{G}$ 生成的样本为正对，不同退化为负对，损失函数为 SupCon 形式。
- 训练时额外使用 GT 人脸代替 Ref-FR 输出作为 HQ，增强鲁棒性。训练完成后冻结。

**MapNet (Degradation Mapping Network)**

- 对 $Z_{face}$ 做 overlap patch embedding (3×3 Conv stride=2 + LayerNorm)。
- 双分支残差注意力：$DegAttn(Z_{face}) = (A_1 - \lambda A_2)[V_1; V_2]$，$\lambda$ 可学习。
- 三尺度网格平均池化(4×4, 2×2, 1×1) + MLP + LayerNorm → 生成 **21 个退化 token**(16+4+1)。
- Token 与文本 token 拼接后送入 SD-Turbo 的 cross-attention 层。

### 损失函数

$$
\mathcal{L}(\theta) = \underbrace{\lambda_2 \|{\hat{I} - I^{HQ}}\|_2^2 + \lambda_{LPIPS} \cdot LPIPS(\hat{I}, I^{HQ})}_{\mathcal{L}_{rec}} + \lambda_{GAN} \cdot \mathcal{L}_{GAN}
$$

- 重建损失：$\ell_2$ + LPIPS 感知损失。
- 对抗损失：判别器使用冻结 DINO 骨干 + 多级独立分类器，**无需扩散教师**，单步直出。
- 超参：$\lambda_{L2}=2.0$，$\lambda_{LPIPS}=5.0$，$\lambda_{GAN}=0.5$。

## 实验

### 数据集

作者构建了 **InScene** 基准(约 57.5K 训练图)：

| 类别 | 数量 | 身份数 | 来源 |
|------|------|--------|------|
| 合成参考集(CelebRef-HQ + InfiniteYou) | 11,266 | 905 | 合成 |
| 真实参考集(gallery) | 16,486 | 914 | 真实 |
| 非参考集(CC12M 过滤) | 29,697 | — | 真实 |

测试集包含三星 S25 Edge 实拍 100 张(10 身份)，涵盖 ISO 3200/1600/800 及运动模糊等真实退化。

### 定量结果 (InScene Synthetic Validation)

| 方法 | 步数 | DISTS↓ | LPIPS↓ | FID↓ | MUSIQ↑ | CLIP-IQA↑ |
|------|------|--------|--------|------|--------|-----------|
| SUPIR | 50 | 0.1361 | 0.3123 | 24.85 | 70.20 | 0.6015 |
| DiffBIR | 50 | 0.1831 | 0.3958 | 36.15 | 68.51 | 0.6975 |
| S3Diff | 1 | 0.1131 | 0.2557 | 18.06 | 72.18 | 0.6980 |
| **Face2Scene** | **1+N** | **0.1007** | **0.2421** | **15.26** | **74.76** | **0.7640** |

在真实验证集上，Face2Scene 同样全面领先：DISTS 0.1178 vs S3Diff 0.2231，LPIPS 0.2502 vs 0.5149，FID 42.21 vs 38.64(S3Diff 略优)，MUSIQ/CLIP-IQA/MANIQA/LIQE/TOPIQ 均为最佳。

### 消融实验

| 配置 | DISTS↓ | LPIPS↓ | FID↓ | MUSIQ↑ | Wins |
|------|--------|--------|------|--------|------|
| Face2Scene **w/** 退化估计 | 0.1007 | 0.2421 | 15.26 | 74.76 | **10/10** |
| Face2Scene **w/o** 退化估计 | 0.1293 | 0.2711 | 17.81 | 71.78 | 0/10 |

- 移除退化估计后所有指标均下降，验证 FaDeX+MapNet 的核心作用。
- GT Face Inserted 实验：将 GT 人脸贴回复原结果后 Face2Scene 仍显著优于 S3Diff(8/10 赢)，说明提升来自全场景而非仅人脸。
- FaDeX 余弦相似度分析：同退化不同图像的嵌入高度相似，不同退化同图像的嵌入差异明显——证明退化编码有效去耦内容。

### 关键发现

1. 从人脸 HQ-LQ 对提取退化信息比仅从 LQ 盲估退化有效得多，FID 改善 15%+。
2. 单步推理 3.3s (1024×1024)，速度与 S3Diff(1.4s) 同量级，远快于 DiffBIR(34.6s) / SUPIR(19.9s)。
3. 退化编码在内容维度高度解耦，具备跨图像/跨场景的泛化能力。

## 亮点

- **巧妙的 oracle 思想**：将人脸复原视为退化探测器，从信号最强处(人脸)估计全局退化，再反哺整个场景——动机自然、逻辑清晰。
- **FaDeX 对比学习方案**：无需退化标签即可学习退化嵌入，且与图像内容有效解耦。
- **MapNet 三尺度 token 设计**：21 个多尺度 token 提供丰富的退化条件信号，与 SD-Turbo cross-attention 无缝集成。
- **自建 InScene 基准**：综合合成+真实+实拍三类数据，填补了身份参考场景级数据的空白。
- **单步推理**：SD-Turbo + LoRA + 无教师对抗训练，兼顾质量与速度。

## 局限性

- **假设全图退化空间均匀**：对景深虚化、局部运动模糊等空间变化型退化不适用。
- **依赖人脸检测与 Ref-FR 质量**：若人脸检测失败或参考图质量差，退化估计将不可靠。
- **Stage 1 增加推理开销**：Ref-FR(约 2.0s)占总推理时间 60%，限制了实时应用。
- **仅支持含人脸场景**：无法泛化到不含人脸的通用图像复原。

## 相关工作

- **参考人脸复原**: FaceMe、RestorerID、MGFR — 仅复原面部区域，不处理场景。
- **场景级人像复原**: DiffBody、OSDHuman、HAODiff — 扩展到全身但不利用身份参考做退化估计。
- **退化感知扩散**: S3Diff(两标量)、DeeDSR(LQ 对比学习) — 退化信号粗糙，不如 HQ-LQ 对直接估计。
- **通用盲复原**: DiffBIR、SUPIR、PASD — 强通用性但人脸/身份保持差，多步推理慢。

## 评分

- 新颖性: ⭐⭐⭐⭐ — "人脸作为退化 oracle" 的切入角度新颖，将 Ref-FR 与退化感知扩散有机结合
- 实验充分度: ⭐⭐⭐⭐ — 自建三类数据集、10 个指标、详尽消融、FaDeX 解耦验证；缺少更多空间变化退化测试
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表丰富，动机阐述充分
- 价值: ⭐⭐⭐⭐ — 实用场景明确(手机拍摄低质照片+个人相册参考)，方法可扩展性强

<!-- RELATED:START -->

## 相关论文

- [Bridging Degradation Discrimination and Generation for Universal Image Restoration](../../ICLR2026/image_generation/bridging_degradation_discrimination_and_generation_for_universal_image_restorati.md)
- [GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration](../../CVPR2025/image_generation/gendeg_diffusion-based_degradation_synthesis_for_generalizable_all-in-one_image_.md)
- [ExpressEdit: Fast Editing of Stylized Facial Expressions with Diffusion Models in Photoshop](expressedit_fast_editing_of_stylized_facial_expressions_with_diffusion_models_in.md)
- [High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning](high-fidelity_diffusion_face_swapping_with_id-constrained_facial_conditioning.md)
- [Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models](erasure_or_erosion_evaluating_compositional_degradation_in_unlearned_text-to-ima.md)

<!-- RELATED:END -->
