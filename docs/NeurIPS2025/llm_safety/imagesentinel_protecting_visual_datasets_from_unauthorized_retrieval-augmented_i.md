---
title: >-
  [论文解读] ImageSentinel: Protecting Visual Datasets from Unauthorized Retrieval-Augmented Image Generation
description: >-
  [NeurIPS 2025][LLM安全][检索增强图像生成] 提出 ImageSentinel 框架，通过合成与私有数据集视觉一致的哨兵图像（sentinel images）并绑定随机字符检索键，实现对检索增强图像生成（RAIG）系统未授权使用私有数据集的可靠检测——仅需 3–10 次查询即可达到接近 100% 的 AUC。
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "检索增强图像生成"
  - "数据集保护"
  - "哨兵图像"
  - "水印"
  - "版权检测"
  - "RAIG"
---

# ImageSentinel: Protecting Visual Datasets from Unauthorized Retrieval-Augmented Image Generation

**会议**: NeurIPS 2025  
**arXiv**: [2510.12119](https://arxiv.org/abs/2510.12119)  
**代码**: [GitHub](https://github.com/luo-ziyuan/ImageSentinel)  
**领域**: 图像生成 / 数据版权保护  
**关键词**: 检索增强图像生成, 数据集保护, 哨兵图像, 水印, 版权检测, RAIG

## 一句话总结
提出 ImageSentinel 框架，通过合成与私有数据集视觉一致的哨兵图像（sentinel images）并绑定随机字符检索键，实现对检索增强图像生成（RAIG）系统未授权使用私有数据集的可靠检测——仅需 3–10 次查询即可达到接近 100% 的 AUC。

## 研究背景与动机
**领域现状**：检索增强图像生成（RAIG）通过从外部参考图像数据库检索相关图片来增强生成质量，已在稀有概念生成、细粒度图像合成等任务上展现出卓越性能。典型系统如 ImageRAG 已证明 RAIG 可直接应用于现有文生图模型（如 SDXL + IP-adapter、OmniGen）。

**现有痛点**：RAIG 系统严重依赖高质量参考图像数据库，恶意用户可能未经授权就将私有数据集纳入其检索系统。这不仅侵犯知识产权，还带来法律和商业风险。然而，目前缺乏有效机制保护视觉数据集免受 RAIG 系统的未授权使用。

**传统方案失效**：数字水印是文本 RAG 保护的常用手段，但在视觉 RAIG 中失效——图像生成涉及复杂的特征提取和重组过程，会破坏嵌入的水印信号，使得水印无法在生成输出中保留。

**语义检索的局限**：基于语义的检索保护方法面临两个困难：(a) 大规模数据库中存在大量语义相似图像，难以精确定位目标；(b) 某些 RAIG 系统在生成器能直接产生满意结果时会跳过检索过程，导致语义触发失效。

**核心idea**：向私有数据集注入精心合成的"哨兵图像"，这些图像视觉上与原始数据集一致，但绑定了唯一的随机字符检索键。通过查询 RAIG 系统并检测输出是否匹配哨兵图像，即可判断数据集是否被未授权使用。

## 核心问题
如何在不修改原始私有图像、不影响授权用户正常使用的前提下，可靠检测 RAIG 系统是否未经授权使用了私有视觉数据集？

## 方法详解

### 整体框架
ImageSentinel 包含三个核心组件：**密钥生成**、**哨兵图像合成**和**未授权使用检测**。数据集所有者在发布前将哨兵图像混入私有数据集形成受保护数据集；检测时通过预定义密钥查询可疑 RAIG 系统，分析输出是否匹配哨兵图像。

### 关键设计

1. **密钥生成（Key Generation）**

    - 生成随机大小写字母组合（如 "VasWiW"），长度默认 6 个字符
    - 这些随机字符串极不可能出现在正常用户提示中，确保对 RAIG 系统日常运行无干扰
    - 同时作为唯一触发器用于检测未授权数据集使用

2. **哨兵图像合成（Sentinel Image Synthesis）**

    - **阶段一：语义属性提取**——从私有数据集 $\mathcal{D}_p$ 中随机选取参考图像 $I_r$，利用视觉语言模型（GPT-4o）提取语义属性集合 $\mathcal{A}$（主题、风格、色调等）和详细描述 $d_r$
    - **阶段二：密钥引导合成**——将提取的属性、描述与随机字符密钥 $k$ 通过模板化提示词输入文生图模型 $\mathcal{T}$，生成哨兵图像 $I_s$：
    $I_s \leftarrow \mathcal{T}(\mathcal{A}, d_r, p_k)$
    - 合成的哨兵图像同时满足三个性质：
        - **隐蔽性（Stealthiness）**：与原始数据集视觉语义一致，难以被区分
        - **透明性（Transparency）**：不影响 RAIG 系统对授权用户的正常生成质量
        - **可触发性（Triggerability）**：能被预定义密钥可靠检索触发
    - 哨兵数据集 $\mathcal{D}_s$ 远小于原始数据集：$|\mathcal{D}_s| \ll |\mathcal{D}_p|$
    - 受保护数据集 $\hat{\mathcal{D}}_p = \mathcal{D}_p \cup \mathcal{D}_s$（添加而非替换）

3. **未授权使用检测（Detection）**

    - 用预定义密钥 $k \in \mathcal{K}$ 构造提示词查询可疑 RAIG 系统，获得生成图像 $I_{\text{out}}^k$
    - 利用 DINO ViT-S/16 提取特征，计算生成图像与对应哨兵图像的余弦相似度：
    $\phi(I_{\text{out}}^k, I_s^k) = \cos(f_{\text{DINO}}(I_{\text{out}}^k), f_{\text{DINO}}(I_s^k))$
    - 聚合多次查询的相似度得分：
    $s = \frac{1}{|\mathcal{K}|} \sum_{k \in \mathcal{K}} \phi(I_{\text{out}}^k, I_s^k)$
    - 当 $s > \eta$（预定义阈值）时，判定 RAIG 系统未授权使用了私有数据集

### 威胁模型
- 数据集所有者可以在发布前预处理数据集，可以查询 RAIG 系统并分析输出
- 黑盒设定：所有者无法直接访问 RAIG 系统的参考数据库或生成模块参数
- 目标：检测未授权使用同时保持授权应用的数据集效用

## 推理/检测流程（无需训练）

本方法**不涉及任何模型训练**，完全基于现有预训练模型的推理能力。

### 保护阶段（离线，一次性）
1. **密钥采样**：随机生成 $|\mathcal{K}|$ 个长度为 6 的大小写字母字符串
2. **参考图采样**：从 $\mathcal{D}_p$ 中随机选取与密钥等量的参考图像
3. **属性提取**：对每张参考图调用 VLM（GPT-4o）提取语义属性（主题、风格、色调、构图、调色板等）和详细描述
4. **哨兵合成**：将属性、描述与对应密钥拼接为模板化 prompt，送入文生图模型（GPT-4o）生成哨兵图像
5. **数据集合并**：$\hat{\mathcal{D}}_p = \mathcal{D}_p \cup \mathcal{D}_s$，发布受保护数据集

### 检测阶段（在线，按需）
1. 用预定义密钥构造查询 prompt（如 "Generate an image of VasWiW"）
2. 将 prompt 输入可疑 RAIG 系统，收集生成图像 $I_{\text{out}}^k$
3. 用 DINO ViT-S/16 分别提取生成图像和对应哨兵图像的特征
4. 计算余弦相似度并跨所有密钥聚合为检测分数 $s$
5. $s > \eta$ 则判定侵权

整个流程的计算瓶颈在哨兵图像合成（需调用 VLM + 文生图模型各 $|\mathcal{K}|$ 次），但这是一次性开销；检测阶段仅需 3–10 次 RAIG 查询 + DINO 前向传播，开销极低。

## 实验关键数据

### 实验设置
- **数据集**：LLaVA-Pretrain（10,000 张）、Product-10K（30,000 张）
- **RAIG 系统**：SDXL + IP-adapter、OmniGen、GPT-4o
- **检索器**：CLIP ViT-B/32、SigLIP ViT-B/16
- **检测特征**：DINO ViT-S/16
- **基线方法**：Ward-HiDDeN、Ward-FIN（文本 RAG 水印方法适配到图像）

### 检测性能（LLaVA-Pretrain, 核心结果）

| RAIG 系统 | 查询数 | AUC | TPR@1%FPR | TPR@10%FPR |
|-----------|--------|-----|-----------|------------|
| SDXL | 3 | 0.974 | 0.934 | 0.958 |
| SDXL | 10 | **1.000** | **1.000** | **1.000** |
| OmniGen | 3 | 0.873 | 0.584 | 0.744 |
| OmniGen | 20 | **1.000** | 0.996 | **1.000** |
| GPT-4o | 3 | 0.983 | 0.954 | 0.974 |
| GPT-4o | 10 | **1.000** | **1.000** | **1.000** |

- 基线方法 Ward-HiDDeN 和 Ward-FIN 的 AUC 始终在 0.5–0.6 左右，接近随机猜测

### 触发率与检索精度
- ImageSentinel 在 SDXL 和 OmniGen 上触发率均为 **100%**，语义方法仅 21.3%/39.0%
- Hit@1 检索准确率：ImageSentinel 69.7% vs 语义方法 58.3%

### Product-10K 检测性能
- 仅 1 次查询 AUC = 0.870；5 次查询 AUC = 0.999；8 次及以上达到 1.000

### 生成质量保持
- ImageSentinel 对正常生成质量影响极小（CLIP/SigLIP/DINO 相似度仅比原始下降 0.004–0.06）
- 而"哨兵替换"策略（替换原图而非添加）会导致显著质量下降

### 消融实验
- **密钥长度**：6 字符最优（AUC 0.997），4 字符和 8 字符略低（0.965/0.972）
- **文生图模型**：GPT-4o 生成的哨兵图像视觉一致性远优于 SDXL
- **保护策略**：添加优于替换，保持了原始数据集的生成能力

### 自适应攻击
- **Detect-and-Inpaint 攻击**：攻击者用 EasyOCR 检测数据库中所有图像的文字区域，再用 Stable Diffusion 2.0 Inpainting 将检测到的文字区域修复，试图在入库前移除哨兵中的字符触发器
- 攻击效果随查询数变化显著：
    - 5 次查询：AUC 从 0.99 骤降至 0.62，TPR@1%FPR 从 0.98 降至 0.15
    - 50 次查询：AUC 恢复至 0.91，TPR@1%FPR = 0.65
    - 100 次查询：AUC = 0.98，TPR@1%FPR = 0.94，接近无攻击水平
- 攻击对生成质量影响极小（CLIP 从 0.772→0.769，DINO 从 0.605→0.597），说明攻击者可低代价执行
- **但攻击存在固有矛盾**：无差别移除所有检测到的文字会误删品牌名、产品标签、说明文字等原始数据集中的重要语义信息，降低数据集在正常检索生成任务中的实用性

## 亮点
1. **问题新颖性高**：首次系统化地定义并解决 RAIG 系统中视觉数据集未授权使用检测问题
2. **方案设计巧妙**：利用随机字符作为检索键，绕过了语义检索的模糊性和 RAIG 系统跳过检索的问题
3. **非侵入式保护**：不修改原始图像，仅在数据集中添加少量哨兵图像即可实现保护
4. **实用性强**：黑盒检测设定，仅需 3–10 次查询即可达到近乎完美的检测效果
5. **跨系统泛化**：在三种不同 RAIG 系统（SDXL、OmniGen、GPT-4o）上均有效

## 局限与展望
1. **依赖文生图模型的文字嵌入能力**：SDXL 在图像中嵌入字符的能力较弱，当前主要依赖 GPT-4o，未来更强的文生图模型可进一步提升效果
2. **对自适应攻击的鲁棒性有限**：Detect-and-Inpaint 攻击在少量查询时能显著削弱检测能力，需要更鲁棒的保护策略
3. **检测指标单一**：目前仅使用 DINO 余弦相似度，更精确的相似度度量可能进一步提升检测
4. **可扩展性未验证**：最大数据库仅 30,000 张，更大规模数据库上的效果尚未验证
5. **哨兵图像的不可见性**：哨兵图像虽然与数据集语义一致，但包含随机字符文本，攻击者通过简单 OCR 筛查即可定位并移除

## 与相关工作的对比

| 方法 | 适用场景 | 保护机制 | 检测方式 | 效果 |
|------|---------|---------|---------|------|
| Ward-HiDDeN | 文本 RAG → 图像适配 | 深度水印嵌入原图 | 提取生成图中水印 | AUC ≈ 0.55，基本失效 |
| Ward-FIN | 文本 RAG → 图像适配 | 流模型水印嵌入原图 | 提取生成图中水印 | AUC ≈ 0.53，基本失效 |
| 语义检索方法 | 图像 RAIG | 语义描述匹配 | 语义相似度检测 | 触发率低（21–39%）|
| **ImageSentinel** | **图像 RAIG** | **合成哨兵图像 + 随机键** | **DINO 特征相似度** | **AUC ≈ 1.0（10 次查询）** |

核心优势在于：(1) 避免了水印在图像生成过程中被破坏的根本问题；(2) 随机字符键确保了精准触发而非模糊的语义匹配。

## My Notes

### 核心启发
1. **思路可迁移**：哨兵图像 + 密钥绑定的保护范式可推广到视频生成、3D 生成等其他检索增强生成场景
2. **与数据投毒的关联**：哨兵图像本质上是一种"良性投毒"——在数据集中植入可控触发器用于检测而非攻击，方法论上与 backdoor attack 共享相同的技术管道（trigger injection → trigger activation → detection），但目的相反
3. **版权保护的新范式**：从"在内容中嵌入信息"转向"在数据集中植入可检测探针"，适应了生成式 AI 对原始内容大幅变换的特点
4. **防御提升方向**：可结合频域/潜空间嵌入替代显式文字嵌入，提升对 OCR+修复攻击的鲁棒性

### 批判性思考
- **最大弱点是显式文字嵌入**：哨兵图像中包含可见的随机字符文本，攻击者无需复杂分析，仅用 OCR 扫描即可定位哨兵图像并将其从数据库中移除（而非 inpainting 修复），这比论文中讨论的 Detect-and-Inpaint 攻击更直接更致命，因为直接删除哨兵图像而非修复文字区域，可以完全消除保护机制
- **检测的非对称性**：方法假设数据集所有者能向 RAIG 系统发送查询并获得输出，但现实中许多 RAIG 系统可能是封闭的内部系统，外部无法查询
- **与 model watermarking 的根本区别**：本方法保护的是「数据集」而非「模型」，当同一私有数据集的子集被多个 RAIG 系统部分使用时，检测可能出现假阴性
- **Sentinel 图像数量 vs 检测可靠性的 trade-off 未充分讨论**：论文只提到 $|\mathcal{D}_s| \ll |\mathcal{D}_p|$，但未给出具体比例建议和灵敏度分析

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 首次针对 RAIG 视觉数据集保护提出完整框架，问题定义和解决思路均具原创性
- 实验充分度: ⭐⭐⭐⭐ — 多数据集、多 RAIG 系统、多基线对比、消融实验和自适应攻击评估，但缺少更大规模验证
- 写作质量: ⭐⭐⭐⭐ — 问题形式化清晰，方法阐述系统，图示直观
- 价值: ⭐⭐⭐⭐ — 解决了 RAIG 领域的实际痛点，但哨兵图像中的显式文字嵌入是明显弱点，限制了实际部署的安全性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Watermarking Autoregressive Image Generation](watermarking_autoregressive_image_generation.md)
- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](../../ACL2026/llm_safety/differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)
- [\[AAAI 2026\] Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering](../../AAAI2026/llm_safety/privacy-protected_retrieval-augmented_generation_for_knowledge_graph_question_an.md)
- [\[ACL 2026\] Beyond Explicit Refusals: Soft-Failure Attacks on Retrieval-Augmented Generation](../../ACL2026/llm_safety/beyond_explicit_refusals_soft-failure_attacks_on_retrieval-augmented_generation.md)
- [\[ACL 2026\] Knowledge Poisoning Attacks on Medical Multi-Modal Retrieval-Augmented Generation](../../ACL2026/llm_safety/knowledge_poisoning_attacks_on_medical_multi-modal_retrieval-augmented_generatio.md)

</div>

<!-- RELATED:END -->
