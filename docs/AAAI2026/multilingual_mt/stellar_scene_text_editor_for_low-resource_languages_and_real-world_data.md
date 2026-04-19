---
title: >-
  [论文解读] STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data
description: >-
  [AAAI 2026][场景文本编辑] 提出 STELLAR 框架，通过语言自适应字形编码器和合成预训练+真实微调的两阶段训练策略，实现韩语/阿拉伯语/日语等低资源语言的场景文本编辑，并提出可解释的 TAS 指标无需 ground truth 评估字体/颜色/背景风格保持，韩语识别准确率从基线最高 22.1% 飙升至 80.4%。
tags:
  - AAAI 2026
  - 场景文本编辑
  - 低资源语言
  - 扩散模型
  - 域适应
  - 文本外观相似性
---

# STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data

**会议**: AAAI 2026  
**arXiv**: [2511.09977](https://arxiv.org/abs/2511.09977)  
**代码**: [github.com/yongchoooon/stellar](https://github.com/yongchoooon/stellar)  
**领域**: 文本图像编辑 / OCR  
**关键词**: 场景文本编辑, 低资源语言, 扩散模型, 域适应, 文本外观相似性

## 一句话总结

提出 STELLAR 框架，通过语言自适应字形编码器和合成预训练+真实微调的两阶段训练策略，实现韩语/阿拉伯语/日语等低资源语言的场景文本编辑，并提出可解释的 TAS 指标无需 ground truth 评估字体/颜色/背景风格保持，韩语识别准确率从基线最高 22.1% 飙升至 80.4%。

## 研究背景与动机

**领域现状**：场景文本编辑 (STE) 旨在修改图像中的文本内容同时保持字体、颜色和背景等视觉风格。随着全球内容产业的扩展——广告横幅、产品包装、游戏与影视本地化、AR 标牌翻译——对多语言 STE 的需求日益迫切。技术路线从早期的 GAN 发展到当前的扩散模型，主流有两种范式：mask-and-inpaint（遮盖文本区域后修复）和 direct substitution（解耦风格与内容后直接替换）。

**现有痛点**：
   - **(1) 低资源语言支持不足**：AnyWord-3M 数据集中，中英文各有百万级样本，而韩/阿/日语每种仅约 2K 样本。复杂书写系统（阿拉伯语从右到左+上下文依赖字形变化、韩语组合字形 jamo 结构）难以由英文预训练模型正确处理。
   - **(2) 合成-真实域差距**：大多数 STE 模型仅在合成渲染引擎生成的数据上训练（如 SynthText、SynthTIGER），无法捕获真实场景的光照、纹理与噪声。推理时出现颜色失真、纹理伪影等明显退化。
   - **(3) 评估指标不适用**：SSIM、PSNR、MSE 和 FID 等指标在文本内容发生变化时仍然惩罚差异（即使风格完全保持），且在无 ground truth 时无法使用，严重限制了真实场景的评估。

**核心矛盾**：低资源语言的字形结构复杂且数据稀缺，合成数据训练的模型无法泛化到真实场景，而现有评估指标无法区分"文本内容变化"和"风格变化"。

**切入角度**：用语言特定的预训练 OCR 识别器（PPOCRv4）监督字形编码器学习语言相关结构特征；收集首个真实多语言文本图像对数据集 STIPLAR 做域适应微调；设计基于风格分解的 TAS 评估指标。

## 方法详解

### 整体框架

STELLAR 基于 TextCtrl（直接替换范式）构建，采用"两编码器 + 一生成器"架构：

- **文本风格编码器 S**（ViT-B backbone）：从源文本图像提取风格特征 C_style，通过线性投影分为纹理特征 c_tex（驱动颜色/字体子任务）和空间特征 c_spa（驱动背景去除/文本分割子任务）
- **语言自适应字形编码器 T**（轻量 Transformer）：从目标文本提取语言特化的字形特征 C_glyph，以语言特定 OCR 识别器的视觉特征为监督信号
- **扩散生成器 G**（Stable Diffusion v1.5）：C_glyph 通过 cross-attention 注入引导文本渲染，C_style 通过 skip-connection 和 middle block 注入引导风格保持

### 关键设计

**1. 语言自适应字形编码器 (Language-Adaptive Glyph Encoder)**

基线方法 TextCtrl 的字形编码器使用仅在英文上预训练的 ABINet 识别器提取字形特征，对非拉丁语系效果极差。STELLAR 以模块化设计替换：为每种目标语言配备对应的 PPOCRv4 预训练识别器（korean_PP-OCRv4、arabic_PP-OCRv4、japan_PP-OCRv4），字形编码器在 CLIP Loss 监督下对齐字符级特征与语言特定 OCR 视觉特征。这种设计天然适应各语言的结构特性——阿拉伯语的右到左书写与上下文依赖字形、韩语的 jamo 组合结构、日语的汉字/假名混合体系。扩展新语言只需接入对应 OCR 识别器，无需修改架构。

**2. 文本风格编码器的多任务学习**

风格编码器 S 通过四个子任务训练以学习可解释的风格表示：
- **颜色迁移**：将灰度文本着色（ResNet34 + AdaIN），学习纹理特征中的颜色信息
- **字体迁移**：将模板字体转换为源风格（ResNet34 + PPM），捕获字体结构特征
- **文本去除**：利用空间特征重建背景图像
- **文本分割**：生成二值文本区域掩码

这种多任务学习让编码器学到结构化、可迁移的风格特征，也为后续 TAS 指标提供了计算基础。

**3. 多阶段训练策略**

- **Stage 1（合成预训练）**：每种语言 200K 合成文本图像对，训练 100 epochs（66h，2xH100）。关键细节：使用 PPOCRv4 过滤仅保留 OCR 识别正确的高质量样本，每种语言独立组织和训练数据以捕获语言特定字体风格。
- **Stage 2（真实微调）**：在 STIPLAR 数据集上微调仅 10 epochs（0.3h）。使用不到 5% 的 Stage 1 数据量和约 10% 的训练 epoch 数，即可快速适应真实域，无需任何 post-hoc 推理技术。

**4. STIPLAR 数据集**

首个面向低资源语言的真实场景文本图像对 (I2I) 数据集。两个数据来源：
- **开源数据**：从 MLT-2019 训练集裁剪文本区域，每种语言选 1000 张，人工标注修正标签错误并配对（韩语 1818 对、阿语 2328 对、日语 453 对）
- **网络爬取**：用 GPT-4o 生成多语言+英语检索查询，搜索 CC 协议图像，经 OCR 检测、裁剪、质量过滤、安全检查和隐私打码（韩语 7946 对、阿语 3988 对、日语 1570 对）
- **总计**：韩语 9764 对、阿拉伯语 6316 对、日语 2023 对，按 8:2 划分训练/评估集

**5. Text Appearance Similarity (TAS) 指标**

利用风格编码器 S 将两张图像的风格解耦为三个独立维度评估：TAS = (s_clr + s_fnt + s_bg) / 3

- s_clr：颜色相似度，用 CIEDE2000 色差归一化
- s_fnt：字体相似度，用 FSIM 特征相似度
- s_bg：背景相似度，用 MS-SSIM

关键优势：不受文本内容变化影响，无需 ground truth 即可评估。

### 损失函数 / 训练策略

- **风格编码器**：L = L_clr(MSE) + L_fnt(Dice) + L_bg(MAE) + L_seg(Dice)
- **字形编码器**：CLIP Loss 对齐字符级特征与语言特定 OCR 视觉特征
- **扩散生成器**：标准扩散去噪损失，推理时 50 步去噪，CFG scale 2.0
- **优化器**：AdamW，LR 1e-5，weight decay 0.01

## 实验关键数据

### 主实验：STIPLAR 评估集上与多语言基线对比

| 语言 | 指标 | STELLAR | TextFlux | AnyText2 | AnyText |
|------|------|---------|----------|----------|---------|
| 韩语 | SSIM | **0.5061** | 0.3409 | 0.2626 | 0.2822 |
| 韩语 | TAS | **0.8596** | 0.8464 | 0.7173 | 0.7227 |
| 韩语 | Rec.Acc | **0.8042** | 0.2213 | 0.0899 | 0.0010 |
| 韩语 | NED | **0.9115** | 0.4836 | 0.2796 | 0.0116 |
| 阿拉伯语 | TAS | **0.8601** | 0.8049 | 0.6671 | 0.6908 |
| 阿拉伯语 | Rec.Acc | **0.6840** | 0.0714 | 0.0082 | 0.0000 |
| 阿拉伯语 | NED | **0.8985** | 0.4449 | 0.0576 | 0.0054 |
| 日语 | TAS | 0.7714 | **0.7857** | 0.5833 | 0.6393 |
| 日语 | Rec.Acc | **0.4338** | 0.4156 | 0.1013 | 0.0000 |

### 消融实验：多阶段训练策略分析

| 配置 | 韩语 TAS | 韩语 Rec.Acc | 阿拉伯语 TAS | 阿拉伯语 Rec.Acc | 日语 TAS | 日语 Rec.Acc |
|------|---------|-------------|-------------|-----------------|---------|-------------|
| S1 (仅合成预训练) | 0.8362 | 0.6676 | 0.8308 | 0.6412 | 0.7657 | 0.2987 |
| S1 + Post-hoc | 0.8375 | 0.6710 | 0.8333 | 0.6290 | 0.7660 | 0.2961 |
| S1 + S2 下采样数据 | 0.8543 | 0.7710 | 0.8562 | 0.6799 | - | - |
| **STELLAR (完整)** | **0.8596** | **0.8042** | **0.8601** | **0.6840** | **0.7714** | **0.4338** |

### TAS 指标验证：合成可控变化实验（韩语）

| 变化类型 | SSIM | PSNR | FID | TAS |
|---------|------|------|-----|-----|
| 仅改文本内容（风格不变） | 0.5768 | 17.29 | 27.28 | **0.8933** |
| 改字体（文本不变） | 0.6379 | 18.09 | 24.92 | 0.8374 |
| 改颜色（文本不变） | 0.7783 | 21.04 | 21.25 | 0.8379 |
| 改背景（文本不变） | 0.5653 | 10.25 | 41.77 | 0.7076 |
| 改全部风格（文本不变） | 0.3859 | 8.61 | 42.61 | 0.5555 |

### 关键发现

- **识别准确率碾压式提升**：韩语 Rec.Acc 从基线最高的 0.2213 (TextFlux) 提升至 0.8042，绝对提升 +58.3%，证明语言自适应字形编码的根本性作用
- **Stage 2 微调极高效**：仅 0.3 小时真实数据微调，韩语 Rec.Acc 0.6676 到 0.8042（+13.7%），背景保持 s_bg 0.7748 到 0.8325
- **Post-hoc 技术几乎无效**：S1 vs S1+Post-hoc，TAS 仅 +0.0013，Rec.Acc 甚至在阿拉伯语上下降（0.6412 到 0.6290），真实数据微调远优于推理阶段修补
- **TAS 正确反映风格保持**：仅改文本内容时传统 SSIM(0.58) 错误给出低分，TAS(0.89) 正确识别风格未变；改颜色时 SSIM(0.78) 高估相似度，TAS(0.84) 更灵敏
- **日语是特例**：TextFlux 在日语 TAS 上略优，因日语汉字与中文字符视觉相似，基线大量中文训练数据可迁移；但 STELLAR 在 Rec.Acc 上仍占优

## 亮点与洞察

- **TAS 指标设计精巧且可复用**：将风格评估解耦为颜色/字体/背景三个独立维度，不依赖 ground truth。该思路可推广到图像风格迁移、字体生成等相关任务的评估
- **极高效的域适应范式**：Stage 2 仅用不到 5% 数据量和约 0.5% 训练时间即实现显著提升，展示了"大规模合成预训练 + 少量真实微调"在数据稀缺场景的强大实用性
- **模块化语言扩展设计**：新增语言只需接入对应 PPOCRv4 识别器，不修改架构。即插即用的设计理念值得在多语言/多模态系统中借鉴
- **数据质量优于数据数量**：Stage 1 使用 OCR 过滤确保训练数据质量，消融实验证实高质量数据比单纯增加数据量更有效

## 局限与展望

- 仅支持韩/阿/日三种语言，未覆盖印地语、泰语、孟加拉语等更多低资源语言
- STIPLAR 数据集总量仅 1.8 万对，在极复杂场景和罕见字体下泛化受限
- 长文本编辑性能下降（训练数据中长文本样本稀少）
- 基于 SD v1.5，生成分辨率和质量不如新架构（如 SDXL、Flux）
- TAS 指标依赖风格编码器的质量，在编码器未专门训练的语言上可能不准
- 未处理弯曲/透视变形文本的编辑场景
- 可探索无监督域适应和零样本泛化到未见语言

## 相关工作与启发

- **vs TextCtrl**：STELLAR 的直接基线，仅支持英文合成数据训练。本文通过语言自适应编码器+真实数据微调两步扩展，证明 direct substitution 范式的多语言可扩展性
- **vs AnyText/AnyText2**：Mask-and-inpaint 范式代表，在低资源语言上近乎完全失败（韩语 Rec.Acc 0.001-0.09），说明遮盖-修复方式在缺乏对应语言训练数据时无法准确渲染目标文本
- **vs TextFlux**：基于 DiT/Flux 的最新方法，背景/风格一致性不错但文本准确率远不及 STELLAR，暗示强大的生成模型也需要语言特化的字形引导
- **启发**：真实域适应不需要海量数据，关键在于多阶段训练策略和数据质量控制；评估指标应与任务目标对齐（风格保持 vs 像素级相似），TAS 的解耦思路值得推广

## 评分

4/5 星

- **新颖性** 4/5：语言自适应字形编码、两阶段域适应和 TAS 指标三个贡献互相配合，形成完整的低资源语言 STE 方案
- **实验充分度** 4/5：三种语言四个基线、完整消融、TAS 可控验证实验全面，数据集和代码均开源
- **写作质量** 4/5：问题-方案对应清晰，图表设计合理
- **实用价值** 4/5：解决工业界亟需的低资源语言文本编辑问题，数据集和指标可直接复用

<!-- RELATED:START -->

## 相关论文

- [ViDia2Std: A Parallel Corpus and Methods for Low-Resource Vietnamese Dialect-to-Standard Translation](vidia2std_a_parallel_corpus_and_methods_for_low-resource_vietnamese_dialect-to-s.md)
- [SEA-Vision: A Multilingual Benchmark for Document and Scene Text Understanding in Southeast Asia](../../CVPR2026/multilingual_mt/sea-vision_a_multilingual_benchmark_for_comprehensive_document_and_scene_text_un.md)
- [The Esethu Framework: Reimagining Sustainable Dataset Governance and Curation for Low-Resource Languages](../../ACL2025/multilingual_mt/the_esethu_framework_reimagining_sustainable_dataset_governance_and_curation_for.md)
- [Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for Low-Resource Languages Using Bilingual Dictionaries](../../ACL2025/multilingual_mt/dictionaries_to_the_rescue_cross-lingual_vocabulary_transfer_for_low-resource_la.md)
- [Exploring In-Image Machine Translation with Real-World Background](../../ACL2025/multilingual_mt/exploring_in-image_machine_translation_with_real-world_background.md)

<!-- RELATED:END -->
