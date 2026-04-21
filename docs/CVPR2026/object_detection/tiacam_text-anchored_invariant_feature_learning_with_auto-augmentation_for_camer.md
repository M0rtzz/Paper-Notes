---
title: >-
  [论文解读] TIACam: Text-Anchored Invariant Feature Learning with Auto-Augmentation for Camera-Robust Zero-Watermarking
description: >-
  [CVPR2026][目标检测][零水印] 提出 TIACam 框架，通过可学习自动增强器模拟相机失真、文本锚定跨模态对抗训练学习不变特征、零水印头在特征空间绑定消息，实现无需修改图像像素的相机鲁棒零水印方案，在屏幕翻拍/打印翻拍/截图三种真实场景下均达到 SOTA 提取精度。
tags:
  - CVPR2026
  - 目标检测
  - 零水印
  - 跨模态对齐
  - 可学习数据增强
  - 相机鲁棒性
  - CLIP
  - 对抗训练
  - 不变特征学习
---

# TIACam: Text-Anchored Invariant Feature Learning with Auto-Augmentation for Camera-Robust Zero-Watermarking

**会议**: CVPR2026  
**arXiv**: [2602.18863](https://arxiv.org/abs/2602.18863)  
**代码**: 待确认  
**领域**: 目标检测（实际为多媒体安全/水印）  
**关键词**: 零水印, 跨模态对齐, 可学习数据增强, 相机鲁棒性, CLIP, 对抗训练, 不变特征学习

## 一句话总结

提出 TIACam 框架，通过可学习自动增强器模拟相机失真、文本锚定跨模态对抗训练学习不变特征、零水印头在特征空间绑定消息，实现无需修改图像像素的相机鲁棒零水印方案，在屏幕翻拍/打印翻拍/截图三种真实场景下均达到 SOTA 提取精度。

## 背景与动机

1. **零水印范式**：传统水印在空间域或变换域修改图像，零水印则不修改像素，而将水印与图像固有特征关联，兼顾不可见性与验证可靠性
2. **相机翻拍挑战**：相机重新拍摄引入透视畸变、光照变化、传感器噪声、摩尔纹等复合且空间耦合的退化，是水印提取最困难的场景之一
3. **手工噪声层局限**：StegaStamp、PIMoG 等方法手动设计相机噪声层，但真实光学失真因环境而异、非线性耦合，固定增强难以覆盖
4. **预训练特征非最优**：DINO 等自监督模型的特征鲁棒性是副产品，并非针对水印任务显式优化
5. **单一不变性不足**：仅靠文本引导或仅靠失真对抗的不变性学习，无法同时保证语义一致性和失真鲁棒性
6. **缺乏统一框架**：现有方法将增强、特征学习和水印绑定分离处理，缺少端到端联合优化机制

## 方法详解

### 整体框架

TIACam 由三个模块组成，以三方对抗循环联合训练：

- **Auto-Augmentor（自动增强器）**：可微分的相机失真模拟管线
- **Text-Anchored Invariant Feature Learner（文本锚定不变特征学习器）**：基于 CLIP 的跨模态对抗对齐
- **Zero-Watermarking Head（零水印头）**：在不变特征空间绑定二进制消息

### 可学习自动增强器

由 6 个可微分模块串联：

| 模块 | 功能 | 关键参数 |
|------|------|----------|
| 几何模块 | 透视/旋转/缩放变换 | 可学习 3×3 透视矩阵 A |
| 光度模块 | 亮度/对比度/伽马 | 可学习 α, γ, β |
| 加性噪声模块 | 传感器噪声 | 重参数化 σ·z, z~N(0,1) |
| 滤波模块 | 光学模糊/镜头涂抹 | 可学习卷积核 K |
| 压缩模块 | JPEG 量化 & 频域掩码 | 平滑量化 + 可训练掩码 M |
| 摩尔纹模块 | 传感器-显示器干涉条纹 | 可学习频率 (fx,fy) 和振幅 α |

组合公式：$\hat{x} = \mathcal{T}_{\text{aug}}(x;\Theta) = \mathcal{T}_{\text{comp}} \circ \mathcal{T}_{\text{filter}} \circ \mathcal{T}_{\text{add}} \circ \mathcal{T}_{\text{photo}} \circ \mathcal{T}_{\text{geo}} \circ \mathcal{T}_{\text{moire}}(x)$

每个模块先在对应失真类型上用 MSE+SSIM 预训练 10k 样本，再在整体对抗训练中微调。

### 文本锚定不变特征学习

- **特征提取器 fθ**：冻结的 CLIP 图像编码器 + 可训练的不变特征提取器（3 个残差块 + 投影头 → 1024 维）
- **判别器 Dψ**：4 层 Transformer（8 头注意力，隐藏维 512），接收图像-文本特征对，判断语义是否匹配
- **训练目标**：图像 x 及其增强版 x̂ 与正负文本锚 T⁺/T⁻ 组成真假对，判别器 loss 为 $\mathcal{L}_{\text{disc}}$，生成器 loss 为 $\mathcal{L}_{\text{adv}}$
- **增强器-提取器对抗**：增强器最大化 $\mathcal{L}_{\text{inv}} - \lambda_{\text{sem}}\mathcal{L}_{\text{sem}}$，提取器最小化 $\mathcal{L}_{\text{inv}}$，其中语义保真度用冻结 ViT 的余弦相似度度量

三方交替更新：① 更新 Dψ 提升配对判别 → ② 更新 Θ 生成更强失真 → ③ 更新 fθ 对齐正文本锚并抵抗失真。

### 零水印头

- 提取不变特征 $\tilde{F} = \Psi(f_\theta(x))$，Ψ 为全局平均池化 + 线性投影
- 维护可学习参考矩阵 $C \in \mathbb{R}^{k \times d}$，第 i 行为第 i 比特方向码
- 预测：$\hat{W}_i = \sigma(\tilde{F} \cdot C_i)$
- 注册阶段：对每个图像-消息对优化 C 和 Ψ（BCE + L2 正则），fθ 冻结
- 提取阶段：对失真图像 x' 计算 $\tilde{F}' = \Psi(f_\theta(x'))$，阈值 0.5 恢复二进制消息

## 实验关键数据

### 特征不变性（余弦相似度，原始 vs 失真图像）

| 失真类型 | SimCLR | BYOL | Barlow | VICReg | VIbCReg | **TIACam** |
|----------|--------|------|--------|--------|---------|------------|
| 加性噪声 | 0.82 | 0.88 | 0.79 | 0.83 | 0.89 | **0.97** |
| 光度变化 | 0.84 | 0.84 | 0.81 | 0.76 | 0.88 | **0.93** |
| 透视变换 | 0.87 | 0.85 | 0.87 | 0.83 | 0.88 | **0.95** |
| JPEG 压缩 | 0.79 | 0.80 | 0.87 | 0.81 | 0.73 | **0.98** |
| 摩尔纹 | 0.85 | 0.83 | 0.84 | 0.89 | 0.87 | **0.97** |
| 滤波模糊 | 0.88 | 0.88 | 0.89 | 0.87 | 0.88 | **0.98** |
| 全部组合 | 0.74 | 0.71 | 0.74 | 0.77 | 0.77 | **0.94** |

### 真实场景水印提取精度（Bit Accuracy %）

| 方法 | 屏幕翻拍 30b | 屏幕翻拍 100b | 打印翻拍 30b | 打印翻拍 100b | 截图 30b | 截图 100b |
|------|:-----------:|:------------:|:-----------:|:------------:|:-------:|:--------:|
| HiDDeN | 70.6 | 68.8 | 67.1 | 65.7 | 74.5 | 70.6 |
| PIMoG | 82.3 | 80.1 | 75.7 | 72.3 | 79.7 | 78.6 |
| StegaStamp | 93.8 | 91.2 | 92.2 | 91.3 | 93.7 | 93.9 |
| **TIACam** | **99.1** | **98.2** | **96.6** | **95.1** | **97.4** | **95.2** |

### 消融实验：不变特征提取器的贡献

| 数据集 | CLIP Only | CLIP + TIACam |
|--------|:---------:|:-------------:|
| Visual Genome | 0.78 | **0.92** |
| Flickr | 0.84 | **0.93** |
| MSCOCO | 0.76 | **0.89** |
| ImageNet | 0.82 | **0.93** |

特征提取器使余弦相似度提升约 13–15%，证明鲁棒性来自框架而非 CLIP 预训练本身。

### 特征区分度测试

200 对相同 caption 生成的不同图像：仅注册图像可 100% 恢复水印，另一图像和文本特征提取精度降至 ~84%，平均余弦相似度 0.73，说明框架在保持不变性的同时维持了视觉个体区分性。

## 亮点

- **三方对抗统一框架**：增强器、特征提取器、判别器联合优化，首次将失真模拟和跨模态对齐统一为一个训练循环
- **全可微增强管线**：6 个可微分模块覆盖几何/光度/噪声/滤波/压缩/摩尔纹，梯度可回传优化增强策略
- **无需像素修改**：零水印范式完全不改变图像，通过特征空间点积 + 阈值即可提取消息
- **真实场景验证充分**：在屏幕翻拍、打印翻拍、截图三种真实物理退化下均大幅领先 SOTA
- **无需定位步骤**：利用不变特征空间的强鲁棒性，直接在整幅图像上提取水印，无需先检测水印区域

## 局限与展望

- 领域标注为 object_detection 但实际为多媒体安全/水印领域，分类待修正
- 图像统一缩放到 128×128，对高分辨率图像的局部特征保持能力未充分讨论
- 零水印注册需按每个图像-消息对单独优化 C 和 Ψ，批量注册效率可能是实际部署瓶颈
- 仅在 RTX 4090 上实验，推理延迟和部署到移动端/嵌入式的可行性未讨论
- 语义相似但视觉不同的图像仍有 84% 准确率（理想应更低），特征空间的跨实例泄露值得关注
- 文本锚（caption）的获取在实际使用中需要额外模块或人工提供

## 与相关工作的对比

| 方法 | 类型 | 增强策略 | 特征来源 | 相机鲁棒性 |
|------|------|----------|----------|:----------:|
| HiDDeN | 嵌入式 | 固定噪声层 | 自训练 CNN | 低 |
| StegaStamp | 嵌入式 | 手工相机噪声层 | 自训练 CNN | 中高 |
| PIMoG | 嵌入式 | 手工投影噪声 | 自训练 CNN | 中 |
| InvZW | 零水印 | 失真对抗 | 对抗训练 | 中 |
| DINO-based | 零水印 | 无 | 预训练 SSL | 中 |
| **TIACam** | **零水印** | **可学习自动增强** | **CLIP+对抗训练** | **高** |

核心区别：TIACam 是首个将可学习增强器、跨模态文本锚定和零水印统一到对抗训练框架中的方法。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 三方对抗训练框架和可微分增强管线设计新颖
- 实验充分度: ⭐⭐⭐⭐ — 合成+真实场景、消融、区分度测试较完整，但缺少运行效率分析
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式推导完整，图示直观
- 价值: ⭐⭐⭐⭐ — 在相机鲁棒零水印方向有显著进展，但实际部署可行性需进一步验证

<!-- RELATED:START -->

## 相关论文

- [Learning by Neighbor-Aware Semantics, Deciding by Open-form Flows: Towards Robust Zero-Shot Skeleton Action Recognition](learning_by_neighbor-aware_semantics_deciding_by_open-form_flows_towards_robust_.md)
- [SDF-Net: Structure-Aware Disentangled Feature Learning for Optical–SAR Ship Re-Identification](sdf-net_structure-aware_disentangled_feature_learning_for_opticall-sar_ship_re-i.md)
- [Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](mitigating_memorization_in_text-to-image_diffusion_via_region-aware_prompt_augme.md)
- [Advancing Textual Prompt Learning with Anchored Attributes](../../ICCV2025/object_detection/advancing_textual_prompt_learning_with_anchored_attributes.md)
- [SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking](slice_semantic_latent_injection_via_compartmentali.md)

<!-- RELATED:END -->
