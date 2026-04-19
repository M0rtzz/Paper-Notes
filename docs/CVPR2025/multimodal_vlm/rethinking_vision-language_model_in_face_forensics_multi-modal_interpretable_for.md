---
title: >-
  [论文解读] Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector
description: >-
  [CVPR 2025][多模态][深度伪造检测] 提出 M2F2-Det，首个同时输出深度伪造检测得分和文本解释的多模态人脸伪造检测器，通过 Forgery Prompt Learning 适配 CLIP 学习伪造特征、Bridge Adapter 融合 CLIP 与 deepfake 编码器特征、频域 token 引导 LLM 生成可信解释。
tags:
  - CVPR 2025
  - 多模态
  - 深度伪造检测
  - CLIP适配
  - 提示学习
  - 可解释性
---

# Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector

**会议**: CVPR 2025  
**arXiv**: [2503.20188](https://arxiv.org/abs/2503.20188)  
**代码**: 有（论文提供链接）  
**领域**: 多模态VLM  
**关键词**: 深度伪造检测, CLIP适配, Prompt学习, 可解释性, 多模态

## 一句话总结

提出 M2F2-Det，首个同时输出深度伪造检测得分和文本解释的多模态人脸伪造检测器，通过 Forgery Prompt Learning 适配 CLIP 学习伪造特征、Bridge Adapter 融合 CLIP 与 deepfake 编码器特征、频域 token 引导 LLM 生成可信解释。

## 研究背景与动机

深度伪造检测领域存在三大局限：

1. **检测与解释割裂**：传统方法（如 EfficientNet、Multi-Attention 等）只输出二元分类分数，缺乏可解释性；而 DDVQA-BLIP 只生成文本解释但检测精度低（比传统方法低 18%）
2. **CLIP 适配不充分**：现有 CLIP-based 检测器（UniFake、DEFAKE）仅使用简单线性层或 ResNet-18 对接 CLIP，缺乏针对人脸伪造的专门 prompt 设计，无法充分利用 CLIP 的多模态学习能力
3. **CLIP+LLM 整合空白**：在文档解析、医学诊断等领域 CLIP+LLM 组合已有成功案例，但在深度伪造检测中 CLIP 的开放集识别能力与 LLM 的文本生成能力的结合尚未被探索

## 方法详解

### 整体框架

M2F2-Det 由四个组件构成：(1) 冻结的 CLIP 图像编码器 $\mathcal{E}_I$ 和文本编码器 $\mathcal{E}_T$；(2) Forgery Prompt Learning (FPL) 生成伪造注意力图；(3) Bridge Adapter 融合 CLIP 和 deepfake 编码器 $\mathcal{E}_D$ 的特征进行二分类检测；(4) Forgery Explanation Module 结合LLM生成文本解释。训练分三阶段进行。

### 关键设计

1. **Forgery Prompt Learning (FPL)**:
    - 功能：将 CLIP 的通用视觉语言能力适配到深度伪造检测，生成像素级伪造注意力图
    - 核心思路：设计 Universal Forgery Prompts (UF-prompts)，包含两种可训练token——通用伪造token $[\mathbf{v}^G]$ 捕捉跨伪造方法的共性模式，特定伪造token $[\mathbf{v}^S]$ 由 CLIP 图像全局特征 $\mathbf{g}^I$ 通过 MLP 生成，编码图像特定的伪造线索。UF-prompt 格式为 $\mathbf{S} = [\mathbf{v}_1^G]..[\mathbf{v}_m^G][\mathbf{v}_1^S]..[\mathbf{v}_u^S]\text{[forged][face]}$。同时在冻结的 CLIP 文本编码器每层引入可训练的层级伪造 token (LF-tokens)。最终输出全局文本嵌入 $\mathbf{g}^T$，与 CLIP 图像编码器的 patch 特征 $\mathbf{F}_I$ 计算逐 patch 余弦相似度，得到伪造注意力图 $\mathbf{M}_b$
    - 设计动机：(a) 通用 vs 特定 token 的分工让模型同时学到跨方法共性和逐图像特性；(b) 使用固定文字"forged face"而非变化的类名来稳定训练；(c) 不同于 CoOp/CoCoOp 做全局分类，FPL 创新性地做像素级伪造定位任务；(d) LF-tokens 保护预训练权重同时增强任务适配

2. **Bridge Adapter (Bri-Ada)**:
    - 功能：桥接 CLIP 图像编码器与 deepfake 编码器，融合通用识别能力和领域伪造知识
    - 核心思路：由 Transformer 编码器块组成，接收 CLIP 图像编码器 $\mathcal{E}_I$ 和 deepfake 编码器 $\mathcal{E}_D$ 的中间层特征作为输入。将两者输出特征图拼接为 $\mathbf{F}^0 \in \mathbb{R}^{w \times h \times c}$，然后用 FPL 生成的伪造注意力图 $\mathbf{M}_b$ 对其加权：$\mathbf{f}^0 = \text{AVGPOOL}(\text{CONV}(\mathbf{F}^0 \odot \mathbf{M}_b))$，得到最终检测表示
    - 设计动机：(a) CLIP 图像编码器在大规模互联网数据上预训练，天然具备泛化性，但缺乏伪造领域知识；deepfake 编码器有领域知识但泛化性弱。Bri-Ada 取两者之长；(b) 用 $\mathbf{M}_b$ 作为空间先验，引导模型关注伪造区域，形成 FPL 和 Bri-Ada 的互利循环

3. **Forgery Explanation Module (频域 token + LLM)**:
    - 功能：将检测结果转化为人类可读的文本解释
    - 核心思路：将 Bri-Ada 输出的检测表示 $\mathbf{F}^0$ 转化为频域 token $\mathbf{H}_F \in \mathbb{R}^{N \times D}$，同时将 CLIP 图像编码器输出转化为视觉 token $\mathbf{H}_V$。两者与文本 token $\mathbf{H}_T$ 拼接输入 LLM，自回归生成文本解释：$p(\mathbf{X}_A | \mathbf{H}_V, \mathbf{H}_F, \mathbf{H}_T) = \prod_{z=1}^{Z} p_\theta(\mathbf{x}_z | \mathbf{H}_V, \mathbf{H}_F, \mathbf{H}_{T,<z}, \mathbf{x}_{A,<z})$
    - 设计动机：(a) 频域 token 携带深度伪造领域知识（真假图像在频域差异显著），告诉 LLM 图像是否为伪造；(b) 视觉 token 提供面部外观信息帮助描述；(c) 不同于直接用 MLLM 做检测（如 DDVQA-BLIP），M2F2-Det 先通过专门机制检测，再用 LLM 解释，检测和解释互相增强

### 损失函数 / 训练策略

三阶段训练：
1. **阶段一**：训练 deepfake 编码器 $\mathcal{E}_D$ + FPL (UF-prompts + LF-tokens)，最小化二分类交叉熵。CLIP 编码器冻结
2. **阶段二**：对齐视觉 token $\mathbf{H}_V$ 和频域 token $\mathbf{H}_F$ 与 LLM 输入空间，仅训练 MLP 投影层，冻结其他组件
3. **阶段三**：训练 MLP 层 + LLM（使用 LoRA），最大化文本生成似然

使用 EfficientNet-B4 作为 deepfake 编码器，CLIP ViT-L/14-336 作为 CLIP 编码器，Vicuna-7B 作为 LLM。DD-VQA 数据集（14,782 QA 对）用于二三阶段训练。

## 实验关键数据

### 主实验（域内检测）

| 数据集 | 指标 | M2F2-Det | 之前SOTA | 提升 |
|--------|------|----------|----------|------|
| FF++ (c23) | Acc/AUC | 98.79/99.34 | 98.65/99.87 (TALL) | Acc+0.14 |
| FF++ (c40) | Acc/AUC | 93.83/96.58 | 92.82/94.57 (TALL) | Acc+1.01, AUC+2.01 |
| Celeb-DF | Acc/AUC | 98.98/99.92 | 98.59/99.94 (RECCE) | Acc+0.39 |
| WDF | Acc/AUC | 86.05/93.14 | 83.25/92.02 (RECCE) | Acc+2.80, AUC+1.12 |

跨数据集泛化（训练于 FF++，测试于其他数据集）：

| 数据集 | AUC | 之前SOTA | 提升 |
|--------|-----|----------|------|
| DFDC | 87.80 | 87.56 (FreqBlender) | +0.24 |
| FFIW | 88.70 | 86.14 (FreqBlender) | +2.56 |
| Celeb-DF | 95.10 | 95.40 (LAA-Net) | -0.30 |

文本解释生成（DD-VQA 数据集）：

| 方法 | 判断Acc | 判断F1 |
|------|--------|--------|
| DDVQA-BLIP | 87.49 | 90.07 |
| Fine-tuned LLaVA | 86.41 | 92.10 |
| M2F2-Det | **95.23** | **96.61** |

### 消融实验

| 配置 | FF++(c40) AUC | Celeb-DF AUC | 说明 |
|------|-------------|-------------|------|
| 基线 (EfficientNet-B4) | 91.03 | 65.78 | 无CLIP |
| +LF-tokens | 92.57 | 67.37 | +1.54 |
| +UF-prompts | 92.66 | 66.08 | +1.63 |
| +完整FPL | 93.65 | 68.68 | +2.62 |
| +Bri-Ada (无FPL) | 93.80 | 70.71 | +4.93 泛化提升 |
| +UF-prompts+Bri-Ada | 94.20 | 71.08 | 互利效果 |
| 完整 M2F2-Det | **96.58** | **74.82** | 全部组件协同 |

频域 token 消融：去掉 $\mathbf{H}_F$ 后判断 Acc 从 95.23% 降至 85.11%（-10.12%）

### 关键发现

- **FPL 显著优于通用 prompt learning**：FPL 的 AUC 比 CoOp 高 9.31%，比 CoCoOp 高 8.13%，因为通用 prompt learning 旨在识别语义而非伪造线索
- **CLIP 提供的泛化性是跨数据集性能的关键**：Bri-Ada 在 Celeb-DF 上带来 +4.93% AUC 提升，因为 CLIP 在互联网数据上的预训练减少了对特定伪造模式的过拟合
- **检测和解释互相增强**：M2F2-Det 的判断准确率 95.23% 远超仅做解释的 DDVQA-BLIP (87.49%)，因为检测机制提供了深度伪造领域知识
- 伪造注意力图 $\mathbf{M}_b$ 是无监督学到的（仅二分类标签），但能精确定位伪造区域

## 亮点与洞察

- **首个同时做好检测+解释的统一框架**：之前只能"检测准但不可解释"或"可解释但检测差"二选一
- **FPL 的通用/特定 token 设计精巧**：通用 token 抓共性（如所有伪造都有的边界不连续性），特定 token 抓个性（如特定伪造方法造成的眼部模糊），这种分工对应了伪造检测需要同时具备泛化性和识别力的需求
- **频域 token 的巧妙引入**：利用伪造人脸在频域高频信号上与真实人脸的显著差异，为 LLM 提供了超越 RGB 视觉信息的判断依据
- 三阶段训练策略确保各组件稳定学习，避免端到端训练的不稳定

## 局限与展望

- DFD 数据集上跨域泛化不如 AUNet，可能因为 AUNet 利用了面部动作单元（AU）的先验知识
- LLM 使用 Vicuna-7B 较小，升级到更大模型可能提升解释质量
- 解释训练数据仅来自 DD-VQA (~14K 对)，数据量有限
- 未探索视频层面的时序一致性线索
- 伪造注意力图是无监督的，性能上界可能受限

## 相关工作与启发

- FPL 可视为 CoOp/CoCoOp 在像素级任务上的推广，为其他需要空间精确性的 CLIP 适配任务提供了范式
- Bridge Adapter 的"冻结通用编码器 + 领域特定编码器"的融合策略可推广到其他需要平衡泛化与特化的任务
- 频域 token 的思路可启发其他领域将非 RGB 特征引入 LLM 进行多模态推理

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个检测+解释统一框架，FPL和频域token设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 6个检测数据集+1个解释数据集，域内域外全覆盖，消融详尽
- 写作质量: ⭐⭐⭐⭐ 结构清晰，组件关系说明完善
- 价值: ⭐⭐⭐⭐ 为深度伪造检测引入了可解释性维度，实用价值高

<!-- RELATED:START -->

## 相关论文

- [MMRL: Multi-Modal Representation Learning for Vision-Language Models](mmrl_multi-modal_representation_learning_for_vision-language_models.md)
- [Can Vision-Language Models Answer Face to Face Questions in the Real-World?](../../ICLR2026/multimodal_vlm/can_vision-language_models_answer_face_to_face_questions_in_the_real-world.md)
- [EgoLM: Multi-Modal Language Model of Egocentric Motions](egolm_multi-modal_language_model_of_egocentric_motions.md)
- [Cropper: Vision-Language Model for Image Cropping through In-Context Learning](cropper_vision-language_model_for_image_cropping_through_in-context_learning.md)
- [Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model](../../ICCV2025/multimodal_vlm/interpretable_zero-shot_learning_with_locally-aligned_vision-language_model.md)

<!-- RELATED:END -->
