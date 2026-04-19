---
title: >-
  [论文解读] Inpaint4Drag: Repurposing Inpainting Models for Drag-Based Image Editing via Bidirectional Warping
description: >-
  [ICCV 2025][图像生成][拖拽编辑] 提出Inpaint4Drag，将拖拽式图像编辑分解为像素空间双向warp和图像修复两个阶段，受弹性物体变形启发设计双向warping算法实现实时预览（0.01s）和高效生成（0.3s），比现有方法快600倍，且可作为任意修复模型的通用适配器。
tags:
  - ICCV 2025
  - 图像生成
  - 拖拽编辑
  - 图像修复
  - 双向warp
  - 实时预览
  - 像素空间变形
---

# Inpaint4Drag: Repurposing Inpainting Models for Drag-Based Image Editing via Bidirectional Warping

**会议**: ICCV 2025  
**arXiv**: [2509.04582](https://arxiv.org/abs/2509.04582)  
**代码**: [项目主页](https://visual-ai.github.io/inpaint4drag)  
**领域**: 图像生成 / 图像编辑  
**关键词**: 拖拽编辑, 图像修复, 双向warp, 实时预览, 像素空间变形

## 一句话总结

提出Inpaint4Drag，将拖拽式图像编辑分解为像素空间双向warp和图像修复两个阶段，受弹性物体变形启发设计双向warping算法实现实时预览（0.01s）和高效生成（0.3s），比现有方法快600倍，且可作为任意修复模型的通用适配器。

## 研究背景与动机

- **拖拽编辑的兴起**：DragGAN/DragDiffusion等方法通过鼠标拖拽实现直觉式图像操控
- **现有方法的三大根本局限**：
  1. **精度不足**：在latent空间操控时控制点从512×512下采样到32×32，精度大幅损失
  2. **交互性差**：生成过程无法提供即时视觉反馈，用户被迫反复试错
  3. **能力有限**：使用通用文生图模型处理遮挡区域效果差（如转头、张嘴等大面积遮挡）
- **核心洞察**：
    - 拖拽编辑本质上可分解为两步：几何变换（warping）+ 内容生成（inpainting）
    - 像素空间的变形估计可提供精确控制和实时预览
    - 专业的修复模型比通用生成模型更擅长填充新暴露区域

## 方法详解

### 整体框架

Inpaint4Drag包含三个模块：
1. **区域指定与边界精化**（可选，基于SAM）
2. **双向warping算法**（实时像素空间变形）
3. **修复模型集成**（标准修复格式输入）

### 模块1：区域指定与SAM边界精化

用户绘制mask指定可变形区域，可选SAM精化模块改善边界。

**问题**：直接使用SAM可能生成断开区域或捕获非预期对象。

**解决方案**：用膨胀/腐蚀mask约束SAM预测：
$$M = (M_{\text{pred}} \cap M_{\text{dilated}}) \cup M_{\text{eroded}}$$

其中 $M_{\text{dilated}}$ 和 $M_{\text{eroded}}$ 分别通过半径 $r_1=10$ 的核对用户mask进行膨胀和腐蚀得到。

### 模块2：双向Warping算法

受弹性物体变形启发，将图像区域视为可变形材料。算法包含四步：

**Step 1 - 轮廓提取与控制点关联**：
$$\mathcal{C} = \text{findContours}(M)$$
每个轮廓仅响应其内部的控制点。

**Step 2 - 前向warping**（定义目标区域边界+建立初始映射）：
$$p_t = p + \sum_{i=1}^{N_\mathcal{C}} w_i(t_i - h_i)$$

权重通过反距离加权计算：
$$w_i = \frac{1/(\|p - h_i\| + \epsilon)}{\sum_{j=1}^{N_C} 1/(\|p - h_j\| + \epsilon)}$$

**Step 3 - 反向映射**（填补前向warp的采样缺口）：
$$p_s = p_t + \sum_{i=1}^{N_n} w_i(p_i^{\text{src}} - p_i^{\text{tgt}})$$

使用 $N_n=4$ 个最近邻参考点，基于局部邻域的逆映射确保完整像素覆盖。

**Step 4 - 生成warp结果与修复mask**：
$$I_{\text{warped}}(p_t) = I(p_s) \quad \text{for all valid pairs}$$
$$M_{\text{inpaint}} = \text{dilate}(M_{\text{temp}} \cup \partial M_{\text{warped}}, K_2)$$

修复mask包含：（1）因变形暴露的未映射区域；（2）warp边界的窄带缓冲区。膨胀核半径 $r_2=5$。

### 模块3：修复模型集成

将 $I_{\text{warped}}$ 和 $M_{\text{inpaint}}$ 作为标准修复输入：
$$I_{\text{edit}} = \text{Inpaint}(I_{\text{warped}}, M_{\text{inpaint}})$$

使用SD 1.5 Inpainting Checkpoint，配合TinyAutoencoder SD（TAESD）、LCM LoRA（减少采样步数至8步）、空文本提示（消除CFG计算），以及缓存的空prompt嵌入。

**关键设计**：修复前提供实时预览——用户可调整mask和控制点直到满意再执行修复。

## 实验关键数据

### 主实验：拖拽编辑方法对比

| 方法 | DragBench-S MD↓ | DragBench-S LPIPS↓ | DragBench-D MD↓ | DragBench-D LPIPS↓ | 显存(GB)↓ | 时间(s)↓ |
|------|----------------|-------------------|----------------|-------------------|----------|---------|
| DragDiffusion | 7.0 | 18.0 | 6.7 | 10.2 | 11.6 | 177.7 |
| DiffEditor | 23.6 | 17.6 | 22.1 | 10.9 | 6.6 | 43.1 |
| SDE-Drag | 7.5 | 11.4 | 8.1 | 14.9 | 6.9 | 126.1 |
| FastDrag | 4.1 | 24.1 | 5.1 | 13.5 | 5.0 | 4.2 |
| **Inpaint4Drag** | **3.6** | **11.4** | **3.9** | **9.1** | **2.7** | **0.3** |

（MD和LPIPS值均×100）

关键发现：
- **最精确**：MD最低（3.6/3.9），拖拽精度最高
- **最快**：比FastDrag快14倍，比DragDiffusion快**600倍**
- **最省内存**：仅需2.7GB显存
- **最佳图像一致性**：LPIPS在两个benchmark上均最优或可比

### 各阶段耗时分解

| 阶段 | 耗时 |
|------|------|
| SAM边界精化 | 0.02s |
| 双向warping预览 | 0.01s |
| SD修复生成 | 0.29s |

### 消融实验：单向vs双向warping

定性对比显示：
- **单向（仅前向）warping**：在拉伸区域产生明显采样伪影，目标位置出现未映射缝隙
- **双向warping**：通过先前向确定目标轮廓、再反向逐像素映射填补缝隙，产生平滑无伪影的变换结果

### Mask精化模块效果

定性展示了三类情况的改善：
- 原始粗糙mask → SAM原始预测（可能包含断开区域/非预期对象）→ 精化结果（保留用户意图+改善边界精度）

## 亮点与洞察

1. **范式创新**：将拖拽编辑从"在latent空间中引导生成"转变为"像素空间warp + 修复"，实现了几何变换与内容生成的清晰分离
2. **物理启发**：将图像区域视为弹性材料的思路自然且直觉化，符合用户对物理世界的认知
3. **通用适配器**：由于输出标准修复格式，可无缝接入任何修复模型，自动继承修复技术的未来进步
4. **实时交互**：0.01s的warp预览让用户在执行昂贵的修复前就能看到效果，大幅改善编辑体验
5. **效率极致**：2.7GB显存+0.3s推理，使得拖拽编辑首次可在消费级GPU上流畅运行

## 局限性

- 双向warping基于反距离插值，对于复杂非刚性变形（如面部表情变化）可能产生不自然结果
- 依赖修复模型的质量，当暴露区域较大时修复质量取决于下游模型
- 重新标注了DragBench的deformable region（与原始editable region理念不同），评估可能不完全公平
- 对于需要改变纹理/风格（而非几何变形）的拖拽编辑场景不适用

## 相关工作与启发

- **与DragGAN/DragDiffusion的根本区别**：前者在latent空间做迭代优化，本文在像素空间做解析变换
- **与FastDrag的对比**：FastDrag也有"拉伸"直觉但仅在latent空间顺序处理，缺乏像素级精度和修复能力
- **启发**：很多生成式任务可以通过分解为"确定性几何变换 + 局部生成"来大幅提升效率和精度

## 评分 ⭐⭐⭐⭐

方法简洁优雅，解决了拖拽编辑中精度和效率的核心痛点。600倍加速和实时预览使其具有极强的实用价值。将拖拽编辑转化为修复问题的视角新颖且有效。

<!-- RELATED:START -->

## 相关论文

- [From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition](../../CVPR2026/image_generation/from_inpainting_to_layer_decomposition_repurposing_generative_inpainting_models_.md)
- [BVINet: Unlocking Blind Video Inpainting with Zero Annotations](bvinet_unlocking_blind_video_inpainting_with_zero_annotations.md)
- [OmniPaint: Mastering Object-Oriented Editing via Disentangled Insertion-Removal Inpainting](omnipaint_mastering_object-oriented_editing_via_disentangled_insertion-removal_i.md)
- [Dense Policy: Bidirectional Autoregressive Learning of Actions](dense_policy_bidirectional_autoregressive_learning_of_actions.md)
- [Trans-Adapter: A Plug-and-Play Framework for Transparent Image Inpainting](trans-adapter_a_plug-and-play_framework_for_transparent_image_inpainting.md)

<!-- RELATED:END -->
