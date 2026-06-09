---
title: >-
  [论文解读] ExpressEdit: Fast Editing of Stylized Facial Expressions with Diffusion Models in Photoshop
description: >-
  [CVPR 2026][图像生成][表情编辑] 本文提出 ExpressEdit，一个完全开源的 Photoshop 插件，通过基于 SPICE 的扩散模型后端结合 Danbooru 表情标签数据库和 RAG 系统，在单个消费级 GPU 上 3 秒内完成风格化面部表情的无噪声编辑…
tags:
  - "CVPR 2026"
  - "图像生成"
  - "表情编辑"
  - "Photoshop插件"
  - "扩散模型"
  - "风格化角色"
  - "检索增强生成"
---

# ExpressEdit: Fast Editing of Stylized Facial Expressions with Diffusion Models in Photoshop

**会议**: CVPR 2026  
**arXiv**: [2604.03448](https://arxiv.org/abs/2604.03448)  
**代码**: [https://github.com/kenantang/ExpressEdit](https://github.com/kenantang/ExpressEdit)  
**领域**: 扩散模型  
**关键词**: 表情编辑, Photoshop插件, 扩散模型, 风格化角色, 检索增强生成

## 一句话总结

本文提出 ExpressEdit，一个完全开源的 Photoshop 插件，通过基于 SPICE 的扩散模型后端结合 Danbooru 表情标签数据库和 RAG 系统，在单个消费级 GPU 上 3 秒内完成风格化面部表情的无噪声编辑，显著优于 GPT/Grok/Nano Banana 2 等商业模型。

## 研究背景与动机

1. **领域现状**：面部表情是视觉叙事的核心元素。当前 AI 图像编辑模型（如 FLUX.2、GPT、Grok、Nano Banana 2）已能辅助表情生成和编辑，但主要针对写实表情，对 2D/3D 动画角色的风格化表情支持不足。
2. **现有痛点**：
    - 文本描述要求高，用户需要写详细的表情描述，否则生成结果缺乏多样性，认知负担大；
    - 商业模型编辑时引入全局噪声和水印，多步编辑后噪声累积导致图像严重退化；
    - 与 Photoshop 等专业软件集成不佳，产生分辨率变化和像素漂移。
3. **核心矛盾**：现有模型无法在保持图像质量的同时精确控制面部元素的大小和位置，且推理速度慢（大多需要 7-50 秒）。
4. **本文目标**：如何在专业编辑软件中实现快速、无噪声、可迭代的风格化表情编辑？
5. **切入角度**：利用开源扩散模型 SPICE 作为后端，结合 Photoshop 原生操作（Liquify、Selection、Scale）提供精确的空间控制，同时构建 135 个表情标签数据库降低使用门槛。
6. **核心 idea**：将开源扩散模型后端与 Photoshop 原生操作深度融合，通过 Canny 边缘控制消除像素漂移，配合 RAG 表情标签系统实现快速无损的风格化表情编辑。

## 方法详解

### 整体框架

ExpressEdit 想解决的是「在专业绘图软件里快速、无损地改风格化角色的表情」。它把这件事拆成两条串起来的流水线：前半段是一个检索增强的提示生成器，负责把用户脑子里的故事或指令翻译成扩散模型听得懂的表情标签；后半段是表情编辑器，拿着这段标签提示，再加上用户在 Photoshop 里给出的图像、空间变换和选区，调用 SPICE 扩散后端只在选区内重绘，最后把结果作为一个新图层贴回 Photoshop。整条链路的关键在于：文本不直接控制空间，空间交给 Photoshop 的原生操作（Liquify、Scale、选区）来给，扩散模型只负责把这些「粗糙的人手操作」收拾成自然结果。

### 关键设计

**1. 检索增强提示生成器：让不会写标签的人也能驱动扩散模型**

风格化角色生成里常用的是 Danbooru 那套标签格式提示（如 `+_+`、`:O`），和自然语言写法完全不同，新用户根本不知道该填什么，写得不全又会导致结果单调、认知负担大。作者为此构建了一个覆盖 135 个表情标签的多模态数据库：每个标签都配了定义、3375 张示例图、332 个跨中日韩英的替代标签、以及 2700 个短故事。用户只要用自然语言描述一个故事，VLM 就从这个库里检索出最相关的标签，再填进一个「前缀（图像内容）+ 后缀（风格控制）」的提示模板里。本质上这是用一个大模型（VLM + 检索库）当翻译层，把用户意图桥接到只需少量算力就能跑的小扩散模型上，算力不足的用户也能用。

**2. 基于 SPICE 的无噪声编辑后端：用 Canny 边缘把编辑锁死在选区里**

商业模型最致命的问题是「全局重绘」——哪怕给了选区，重绘后的边缘也对不上原图，Nano Banana Pro 就会在耳垂、下巴这种地方留伪影，多步编辑后噪声不断累积，图像很快退化。ExpressEdit 以 WAI-illustrious-SDXL 为基座，叠一层 SDXL Canny ControlNet：用户用 Selection Brush 以全硬度画出选区，模型只动选区内的像素，而 Canny 边缘控制强制编辑区的轮廓和原图严格对齐，从源头掐掉了像素漂移。于是「选区」在这里不只是一个遮罩，而是一个硬约束，比那种「在提示里描述哪些地方别改」的软办法可靠得多。

**3. Photoshop 原生操作协同：把人手的变换当成扩散模型的空间提示**

实验里所有基线都读不懂精确的数值/方向指令——你说「把虹膜直径缩小 50%」或「视线往左移」，多模态模型基本失败。ExpressEdit 干脆不让文本去管空间：用户先用 Liquify 把面部元素粗略推到想要的位置，或用 Scale 把虹膜缩小，得到一个带变形伪影的中间结果，再让扩散模型把这个「粗糙稿」修成自然表情。这样方向和数值都由人手的可视化操作给定，扩散模型只需要识图修补，绕开了多模态模型对空间指令不敏感这个根本短板。论文统计 35/135 个标签需要这类变换辅助才能可靠编辑，其余 100 个可直接无变换编辑。

### 一个完整示例：把角色虹膜缩小并换成 `+_+` 表情

以「让角色露出闪亮大眼的兴奋表情、但眼睛要更聚焦」为例走一遍：用户先在提示生成器里写下故事「她看到礼物时眼睛一亮」，VLM 从 135 标签库里检索出 `+_+`（星星眼）并填进模板，得到形如 `[图像内容前缀], +_+, [风格后缀]` 的提示；接着用户嫌虹膜太大，直接在 Photoshop 里用 Scale 把虹膜缩到 50%，这一步会留下生硬的缩放痕迹；然后用 Selection Brush 全硬度框住眼部选区，SPICE 后端在 Canny 边缘约束下只重绘选区，把缩放伪影补成自然的星星眼，结果作为新图层贴回。整套操作在单张消费级 GPU 上约 3 秒完成、选区外像素零改动，需要再调时重复这一步也不会累积噪声——这正是它能扛住 100 步压力测试的原因。

### 损失函数 / 训练策略

ExpressEdit 直接复用预训练的 SPICE 后端做推理，本身不引入额外训练。为了进一步提速，它支持 Speed-Up LoRA，把采样步数从 30 步降到 8 步，API 延迟从 4.06 秒压到 2.18 秒（降约 46%），代价只是睫毛等微细节上的微小差异。

## 实验关键数据

### 主实验

| 方法 | 延迟 (s) | 噪声引入 | 选区支持 | 多步编辑 |
|------|----------|----------|----------|----------|
| FLUX.2 [max] | 49.94±13.39 | 严重 | 无 | 退化 |
| GPT | 46.01±11.74 | 严重 | 无 | 退化 |
| Grok | 7.11±0.50 | 中等 | 无 | 退化 |
| Nano Banana 2 Fast | 23.18±3.92 | 严重(对角线模式) | 有(边缘问题) | 退化 |
| Nano Banana 2 Pro | 41.92±22.08 | 严重(对角线模式) | 有(边缘问题) | 8步后损坏 |
| **ExpressEdit (30步)** | **4.06±0.02** | **零** | **原生** | **100步稳定** |
| **ExpressEdit (8步+LoRA)** | **2.18±0.02** | **零** | **原生** | **100步稳定** |

### 功能对比

| 功能 | ExpressEdit | 基线模型 |
|------|-------------|----------|
| 精确数值控制（虹膜缩放50%） | ✓（通过Scale变换） | ✗（全部失败） |
| 方向控制（视线移动） | ✓（通过Liquify） | 需文本描述，不可靠 |
| 高分辨率支持（1664×2432） | ✓ | Nano Banana 2 Pro有退化 |
| 多标签组合 | ✓（"+_+" + ":O"） | 需复杂文本描述 |
| 开源免费 | ✓ | ✗ |

### 关键发现

- Nano Banana 2 Pro 在 8 步连续编辑后图像被噪声完全损坏，而 ExpressEdit 即使在 100 步选区严格重叠的压力测试中也只在选区边缘有轻微噪声，且一步即可修复
- 35/135 个表情标签需要 Photoshop 变换辅助才能可靠编辑，其余 100 个可直接无变换编辑
- Speed-Up LoRA 将延迟降低 46%，仅在睫毛等微细节上有微小差异

## 亮点与洞察

- **Photoshop 原生操作作为视觉提示**：巧妙地将 Liquify/Scale 变换的"粗糙结果"作为扩散模型的空间提示，绕开了文本指令对空间方向不敏感的根本问题。这个思路可以迁移到任何需要精确空间控制的图像编辑任务
- **选区即控制**：通过限制扩散模型只编辑选区内容，配合 Canny 边缘控制消除像素漂移，实现了真正的非破坏性编辑。这比"在提示中描述不要修改的区域"的方式可靠得多
- **表情标签数据库的系统化构建**：135 个标签 + 3375 张示例图 + 2700 个故事 + 多语言替代标签，形成了一个完整的表情编辑知识库

## 局限与展望

- 当前仅支持 Photoshop，虽然代码开源但普通用户可能没有 Photoshop 授权
- 没有定量评估 RAG 系统的检索准确率，缺少用户研究中关于标签检索质量的数据
- 对极端风格（如水墨画、像素艺术）的泛化能力未验证
- 表情标签来自 Danbooru，可能存在数据集偏差，对非东亚风格角色的表现未充分测试

## 相关工作与启发

- **vs FLUX.2/GPT/Grok**: 这些通用编辑模型能力强但引入全局噪声，无法用于迭代编辑工作流。ExpressEdit 通过选区+Canny控制解决了此问题
- **vs Nano Banana 2**: 最接近的竞争者，速度较快但仍有对角线噪声模式和像素漂移。ExpressEdit 速度更快且无噪声
- **vs SPICE**: ExpressEdit 的核心后端，本文的贡献在于将其工程化为完整的 Photoshop 插件并构建了表情标签生态

## 评分

- 新颖性: ⭐⭐⭐ 方法本身无重大技术创新，主要是工程集成和系统设计
- 实验充分度: ⭐⭐⭐⭐ 与多个商业模型的全面对比，但缺少定量用户研究
- 写作质量: ⭐⭐⭐⭐ 论述清晰，图表丰富，但篇幅较长
- 价值: ⭐⭐⭐⭐ 开源+实用，对动画/游戏行业有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning](high-fidelity_diffusion_face_swapping_with_id-constrained_facial_conditioning.md)
- [\[ECCV 2024\] RegionDrag: Fast Region-Based Image Editing with Diffusion Models](../../ECCV2024/image_generation/regiondrag_fast_region-based_image_editing_with_diffusion_models.md)
- [\[AAAI 2026\] Realistic Face Reconstruction from Facial Embeddings via Diffusion Models](../../AAAI2026/image_generation/realistic_face_reconstruction_from_facial_embeddings_via_diffusion_models.md)
- [\[CVPR 2026\] Face2Scene: Using Facial Degradation as an Oracle for Diffusion-Based Scene Restoration](face2scene_using_facial_degradation_as_an_oracle_for_diffusion-based_scene_resto.md)
- [\[CVPR 2026\] Low-Resolution Editing is All You Need for High-Resolution Editing](low-resolution_editing_is_all_you_need_for_high-resolution_editing.md)

</div>

<!-- RELATED:END -->
