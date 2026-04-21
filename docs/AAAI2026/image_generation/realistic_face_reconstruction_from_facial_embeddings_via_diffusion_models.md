---
title: >-
  [论文解读] Realistic Face Reconstruction from Facial Embeddings via Diffusion Models
description: >-
  [AAAI 2026][图像生成][人脸重建] 提出 FEM（Face Embedding Mapping）框架，利用 KAN 网络将任意人脸识别/隐私保护人脸识别系统的嵌入向量映射到预训练身份保持（ID-Preserving）扩散模型的嵌入空间，实现高分辨率真实人脸重建，可用于评估人脸识别系统的隐私泄露风险。
tags:
  - AAAI 2026
  - 图像生成
  - 人脸重建
  - 人脸嵌入
  - 隐私攻击
  - Kolmogorov-Arnold Network
  - 扩散模型
---

# Realistic Face Reconstruction from Facial Embeddings via Diffusion Models

**会议**: AAAI 2026  
**arXiv**: [2602.13168](https://arxiv.org/abs/2602.13168)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 人脸重建, 人脸嵌入, 隐私攻击, Kolmogorov-Arnold Network, 扩散模型

## 一句话总结

提出 FEM（Face Embedding Mapping）框架，利用 KAN 网络将任意人脸识别/隐私保护人脸识别系统的嵌入向量映射到预训练身份保持（ID-Preserving）扩散模型的嵌入空间，实现高分辨率真实人脸重建，可用于评估人脸识别系统的隐私泄露风险。

## 研究背景与动机

### 领域现状

人脸识别（FR）系统通过黑盒模型（CNN/DNN）生成人脸嵌入向量作为身份模板。为增强隐私保护，隐私保护人脸识别（PPFR）系统被提出（如 DCTDP、HFCF、PartialFace、MinusFace），此外还有嵌入保护算法（如 PolyProtect、MLP-Hash、SlerpFace）。

### 现有痛点

**CNN 方法重建质量差**：NbNet、端到端 CNN 方法重建的人脸图像模糊、带噪声伪影，且通常只能生成低分辨率图像

**GAN 方法局限性大**：FaceTI 基于 StyleGAN3，训练过程资源密集（51 小时/epoch，25GB 显存）；MAP2V 无需训练但推理极慢（111 秒/张）

**针对 PPFR 的研究不足**：现有方法主要关注正常 FR 系统，对隐私保护系统的嵌入攻击研究有限

**缺乏通用性**：现有方法难以处理部分泄露嵌入、受保护嵌入等真实攻击场景

### 核心矛盾

如何用轻量级的映射网络将不同 FR/PPFR 系统的嵌入空间（各不相同）统一映射到一个高质量人脸生成模型的嵌入空间，实现高效、高质量的人脸重建？

### 切入角度

利用预训练的 ID-Preserving 扩散模型 IPA-FaceID（已具备从嵌入生成高质量人脸的能力），核心问题转化为**嵌入空间之间的映射学习**。引入 KAN（Kolmogorov-Arnold Network）来捕捉嵌入空间间的复杂非线性关系。

## 方法详解

### 整体框架

FEM 框架分为训练和推理两个阶段：

**训练阶段**：
1. 将公开人脸数据集分别送入目标 FR/PPFR 模型 $\Gamma'(\cdot)$ 和 IPA-FaceID 的默认 FR 模型 $\Gamma(\cdot)$
2. 获取两个嵌入分布 $\mathcal{D}'(e'_i)$ 和 $\mathcal{D}(e_i)$
3. 训练 FEM 模型 $\mathcal{M}(\cdot)$ 使映射后的嵌入 $\hat{e}_i = \mathcal{M}(e'_i)$ 尽可能接近对应的 $e_i$

**推理阶段**：
1. 将泄露的目标系统嵌入 $e'$ 输入训练好的 FEM
2. FEM 映射得到 $\hat{e}$
3. 直接用 IPA-FaceID 生成高分辨率真实人脸图像

### 关键设计

#### 1. **FEM-KAN：基于 KAN 的嵌入映射**

**核心思路**：利用 Kolmogorov-Arnold 定理——任何连续函数都可以表示为有限个单变量连续函数的组合。人脸嵌入间的映射关系本质上可以被分解为单变量函数操作的组合。

$$f(x) = \sum_q \Phi_q\left(\sum_i \phi_{q,i}(x_i)\right)$$

**与 FEM-MLP 的区别**：
- FEM-MLP 使用固定激活函数（GELU），3 层 MLP + 1D 批归一化
- FEM-KAN 使用**可学习的激活函数**位于边上，3 层 KAN，能更准确地捕捉非线性映射

**设计动机**：人脸嵌入虽然高维但具有结构性，KAN 的单变量函数分解能更好地捕捉嵌入空间间的复杂非线性关系。UMAP 可视化表明 FEM 能有效将目标域嵌入映射到 IPA-FR 的目标域或边界区域。

#### 2. **损失函数设计**

使用均方误差（MSE）作为重建损失：

$$\mathcal{L}_{MSE}(e_i, \hat{e}_i) = \frac{\sum_{i=0}^{N-1}(e_i - \hat{e}_i)^2}{N}$$

其中 $e_i$ 是目标嵌入（IPA-FR 输出），$\hat{e}_i = \mathcal{M}(e'_i)$ 是 FEM 映射后的嵌入。

#### 3. **利用 IPA-FaceID 的 ID 保持能力**

IPA-FaceID 使用解耦交叉注意力将人脸嵌入注入预训练 T2I 扩散模型。固定文本提示为 "front portrait of a person" 以生成正面肖像。FEM 将映射后的嵌入落入目标域后，IPA-FaceID 可直接生成保持身份的人脸图像。

### 训练策略

- 使用 90% FFHQ 数据集训练，测试在 CelebA-HQ 的 1000 张从未见过的身份
- AdamW 优化器，初始学习率 $10^{-2}$，指数衰减率 0.8
- Batch size 128，训练 20 个 epoch
- 在 Tesla V100 32GB GPU 上训练

## 实验关键数据

### 主实验

CelebA-HQ 数据集上的攻击成功率（ASR）：

| 目标模型 | 方法 | MF | EF | GF | AF | 平均 |
|---------|------|-----|-----|-----|-----|------|
| IRSE50 (FR) | FaceTI | 93.4 | 80.8 | 49.6 | 66.8 | 72.7 |
| | MAP2V | 94.0 | 86.2 | 59.3 | 72.0 | 77.9 |
| | FEM-MLP | 98.0 | 91.8 | 62.6 | 73.4 | 81.5 |
| | **FEM-KAN** | **99.2** | **93.8** | **65.7** | **76.1** | **83.7** |
| HFCF (PPFR) | MAP2V | 76.3 | 15.4 | 5.3 | 14.8 | 28.0 |
| | **FEM-KAN** | **98.3** | **90.7** | **66.5** | **76.9** | **83.1** |
| MinusFace (PPFR) | MAP2V | 68.0 | 4.8 | 2.3 | 5.6 | 20.2 |
| | **FEM-KAN** | **96.5** | **71.3** | **44.5** | **58.1** | **67.6** |

FEM-KAN 在所有 FR 和 PPFR 目标模型上均取得最高平均 ASR，特别是在 HFCF 和 MinusFace 等 PPFR 模型上远超 MAP2V（83.1 vs 28.0，67.6 vs 20.2）。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| FEM 训练时间 | 3 小时 vs FaceTI 51 小时 | **17 倍**更快 |
| FEM GPU 内存 | 4325 MiB vs FaceTI 25383 MiB | **5.8 倍**更省 |
| FEM 推理时间 | 2.6s vs MAP2V 111s | **42 倍**更快 |
| 50% 嵌入泄露 | FEM-KAN ASR 53.2% vs FaceTI 50.8% | FEM 更鲁棒 |
| 30% 嵌入泄露 | FEM-KAN ASR 32.5% | 极低泄露仍可攻击 |
| 化妆场景 (LADN-M) | FEM-KAN 平均 ASR 85.1% vs FaceTI 56.4% | FEM 对化妆更鲁棒 |

### 受保护嵌入攻击

| 保护算法 | FEM-KAN MF/EF/GF/AF | MAP2V MF/EF/GF/AF |
|---------|---------------------|-------------------|
| MLP-Hash | 82.1/54.7/56.5/71.6 | 48.1/0.6/0.3/1.5 |
| SlerpFace | 79.4/9.3/7.8/15.4 | 11.4/0.0/0.1/0.1 |
| PolyProtect | 50.3/7.1/5.6/15.4 | 28.6/4.4/3.6/4.3 |

FEM 在 MLP-Hash 保护下的 ASR 接近无保护水平，说明该嵌入保护算法存在严重安全隐患。

### 关键发现

- **KAN 优于 MLP**：FEM-KAN 在几乎所有场景下均优于 FEM-MLP，证明了可学习激活函数在嵌入映射中的优势
- **PPFR 系统并不安全**：即使图像经过频域变换等隐私保护处理，嵌入中仍保留了足够的身份信息可被用于高质量人脸重建
- **部分嵌入也能攻击**：即使只获得 30% 的嵌入向量，FEM-KAN 仍能实现 32.5% 的 ASR
- **化妆对 FEM 影响小**：化妆导致 FaceTI ASR 下降 18.1%，但 FEM-KAN 仅下降 6.4%
- **人脸反欺骗系统可被绕过**：重建的高质量人脸可通过 FASNet 检测

## 亮点与洞察

1. **KAN 在嵌入映射中的首次应用**：证明了 KAN 的可学习激活函数对于高维结构化数据（人脸嵌入）的非线性映射确实优于传统 MLP
2. **框架思维**：将人脸重建问题巧妙转化为"嵌入空间映射 + 预训练生成"的两阶段方案，使得映射网络可以极其轻量
3. **全面的安全评估**：覆盖了正常 FR、PPFR、部分泄露、受保护嵌入、受保护图像等多种场景，具有实际安全评估价值
4. **训练效率极高**：3 层网络、3 小时训练、2.6 秒推理，远优于现有方法

## 局限与展望

- 依赖 IPA-FaceID 作为生成后端，如果该模型更新或下架则失效
- 文本提示固定为 "front portrait of a person"，对非正面角度的人脸重建可能不够准确
- FEM 需要访问目标 FR 系统的黑盒查询能力来构建训练数据
- 低分辨率人脸（LFW 112×112）的 ASR 明显下降
- 当嵌入泄露率低于 30% 时，重建人脸出现明显伪影
- 仅使用了 MSE 损失，未探索其他距离度量（如余弦距离、对比损失）

## 相关工作与启发

- FEM 可作为 FR/PPFR 系统隐私安全的**评估工具**，量化不同保护算法的隐私泄露风险
- 启发安全研究：嵌入保护算法（特别是 MLP-Hash）需要重新设计以抵抗映射攻击
- KAN 在其他嵌入空间对齐任务（如跨模态检索、领域适应）中可能也有应用潜力
- 提示了"嵌入即隐私"的安全范式：即使图像被保护，嵌入向量仍可能泄露身份

## 评分

- 新颖性: ⭐⭐⭐⭐ （KAN 用于嵌入映射有新意，框架设计优雅简洁）
- 实验充分度: ⭐⭐⭐⭐⭐ （覆盖多种 FR/PPFR 系统、多种攻击场景、资源对比）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，实验设置详细）
- 价值: ⭐⭐⭐⭐ （对隐私安全研究有实际意义，暴露了 PPFR 的安全风险）

<!-- RELATED:START -->

## 相关论文

- [High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning](../../CVPR2026/image_generation/high-fidelity_diffusion_face_swapping_with_id-constrained_facial_conditioning.md)
- [OFER: Occluded Face Expression Reconstruction](../../CVPR2025/image_generation/ofer_occluded_face_expression_reconstruction.md)
- [Diffusion Reconstruction-Based Data Likelihood Estimation for Core-Set Selection](diffusion_reconstruction-based_data_likelihood_estimation_for_core-set_selection.md)
- [FreeUV: Ground-Truth-Free Realistic Facial UV Texture Recovery via Cross-Assembly](../../CVPR2025/image_generation/freeuv_ground-truth-free_realistic_facial_uv_texture_recovery_via_cross-assembly.md)
- [ExpressEdit: Fast Editing of Stylized Facial Expressions with Diffusion Models in Photoshop](../../CVPR2026/image_generation/expressedit_fast_editing_of_stylized_facial_expressions_with_diffusion_models_in.md)

<!-- RELATED:END -->
