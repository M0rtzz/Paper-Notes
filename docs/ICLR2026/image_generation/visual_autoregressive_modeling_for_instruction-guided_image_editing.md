---
title: >-
  [论文解读] Visual Autoregressive Modeling for Instruction-Guided Image Editing
description: >-
  [ICLR 2026][图像生成][图像编辑] 提出VAREdit，将指令引导的图像编辑重新定义为多尺度预测问题，通过Scale-Aligned Reference模块解决最细尺度条件化的尺度失配问题，在编辑遵循度和效率上大幅超越扩散模型方法。
tags:
  - ICLR 2026
  - 图像生成
  - 图像编辑
  - 视觉自回归
  - 多尺度预测
  - 指令引导
  - 尺度对齐
---

# Visual Autoregressive Modeling for Instruction-Guided Image Editing

**会议**: ICLR 2026  
**arXiv**: [2508.15772](https://arxiv.org/abs/2508.15772)  
**代码**: [GitHub](https://github.com/HiDream-ai/VAREdit)  
**领域**: image_generation  
**关键词**: 图像编辑, 视觉自回归, 多尺度预测, 指令引导, 尺度对齐

## 一句话总结

提出VAREdit，将指令引导的图像编辑重新定义为多尺度预测问题，通过Scale-Aligned Reference模块解决最细尺度条件化的尺度失配问题，在编辑遵循度和效率上大幅超越扩散模型方法。

## 研究背景与动机

1. **领域现状**: 指令引导图像编辑主要由扩散模型主导（InstructPix2Pix等），通过channel-wise拼接源图和目标图进行联合去噪。

2. **现有痛点**: 扩散模型的全局去噪过程天然将编辑区域与整幅图像耦合，导致：(1) 非编辑区域出现虚假修改（"bleeding"问题）；(2) 编辑指令遵循度不足；(3) 多步迭代去噪的计算代价高。

3. **核心矛盾**: 扩散模型的优势（全局一致性建模）恰恰是编辑任务的劣势——编辑需要精准的局部修改与全局保留的分离。自回归模型的因果性和组合性天然适合编辑，但VAR范式在图像编辑中尚未被探索。

4. **本文目标**: 将视觉自回归（VAR）的多尺度预测范式引入指令引导图像编辑。

5. **切入角度**: 发现VAR编辑的核心挑战在于源图像条件化策略——全尺度条件太贵（$O(n^2)$），最细尺度条件高效但存在尺度失配。通过分析注意力热力图发现失配仅影响第一个自注意力层。

6. **核心 idea**: 仅在第一个自注意力层注入尺度对齐的参考特征，其余层使用最细尺度条件，兼顾效率和编辑质量。

## 方法详解

### 整体框架

VAREdit基于预训练Infinity模型，将图像编辑建模为条件多尺度预测：给定源图像 $\mathbf{I}^{(src)}$ 和文本指令 $\mathbf{t}$，自回归生成目标图像的K层残差图 $\mathbf{R}_{1:K}^{(tgt)}$。使用最细尺度源特征 $\mathbf{F}_K^{(src)}$ 作为主要条件，在第一个自注意力层额外注入尺度对齐参考特征。训练使用3.92M配对样本。

### 关键设计

**1. Scale-Aligned Reference (SAR) 模块**

- **功能**: 在保持最细尺度条件化效率的同时解决尺度失配问题
- **核心思路**: 通过下采样最细尺度特征生成与每个目标尺度空间维度匹配的参考特征 $\mathbf{F}_k^{(ref)} = \text{Down}(\mathbf{F}_K^{(src)}, (h_k, w_k))$。在第一个自注意力层中，目标尺度k的query同时attend到尺度对齐的参考特征和先前生成的目标历史。仅在第一层应用SAR，后续层仍只使用最细尺度条件
- **设计动机**: 注意力热力图分析发现：第一层负责建立全局布局和长距离依赖（需要尺度匹配的条件），深层则负责局部细化（最细尺度条件即可）

**2. 最细尺度条件化策略**

- **功能**: 大幅减少计算开销
- **核心思路**: 只将最细尺度（最高分辨率）特征 $\mathbf{F}_K^{(src)}$ 前置到目标序列，而非全部K个尺度的特征。相比全尺度方法大幅减少序列长度（自注意力为 $O(n^2)$ 复杂度）
- **设计动机**: 最细尺度包含最丰富的高频细节信息，对编辑引导最关键

**3. 自回归多尺度编辑建模**

- **功能**: 利用VAR的因果组合性实现精准编辑
- **核心思路**: 编辑被分解为K个尺度的残差预测 $p(\mathbf{R}_{1:K}^{(tgt)}|\mathbf{I}^{(src)}, \mathbf{t}) = \prod_{k=1}^K p(\mathbf{R}_k^{(tgt)}|\mathbf{F}_{1:k-1}^{(tgt)}, \mathbf{F}_K^{(src)}, \mathbf{t})$。文本指令通过交叉注意力引入。使用2D-RoPE区分源图和目标图token
- **设计动机**: 自回归生成天然支持"保留不变区域+精确修改编辑区域"的分离

### 损失函数 / 训练策略

使用bitwise分类器损失优化目标残差token的索引预测，遵循Infinity的训练方案。VAREdit-2B两阶段训练：256²分辨率8k步 + 512²分辨率7k步。VAREdit-8B直接在512²训练60k步。推理使用CFG强度 $\eta=4$，logits温度 $\tau=0.5$。

## 实验关键数据

### 主实验

EMU-Edit和PIE-Bench上的定量对比：

| 方法 | 参数量 | GPT-Balance(EMU)↑ | GPT-Balance(PIE)↑ | 时间 |
|------|--------|-------------------|-------------------|------|
| InstructPix2Pix | 1.1B | 2.923 | 4.034 | 3.5s |
| UltraEdit | 7.7B | 4.541 | 5.580 | 2.6s |
| ICEdit | 17B | 4.785 | 4.933 | 8.4s |
| **VAREdit-2B** | 2.2B | 5.662 | 6.996 | **0.7s** |
| **VAREdit-8B** | 8.4B | **7.892** | **8.105** | 1.2s |
| Step1X-Edit | 21B | 7.081 | 7.351 | 12.8s |

512×512编辑：VAREdit-8B仅需1.2秒，比同规模UltraEdit快**2.2倍**。

### 消融实验

| 条件化策略 | CLIP-Out.↑ | GPT-Suc.↑ | GPT-Over.↑ | GPT-Bal.↑ |
|-----------|-----------|-----------|-----------|----------|
| 全尺度条件 | 0.275 | 5.781 | 7.087 | 5.346 |
| 最细尺度条件 | 0.264 | 4.926 | 7.077 | 4.584 |
| 最细尺度+SAR(第1层) | **0.271** | **6.210** | **7.055** | **5.662** |
| SAR在所有层 | 0.269 | 5.884 | 7.036 | 5.352 |
| SAR在前3层 | 0.269 | 5.894 | 7.048 | 5.297 |

### 关键发现

- VAREdit-8B的GPT-Balance比最强扩散基线（ICEdit）高**64.9%**（EMU-Edit）和**45.3%**（PIE-Bench）
- SAR仅在第一层应用效果最佳——在所有层应用反而性能下降，验证了注意力分析的洞察
- 在最成功编辑（GPT-Suc.≥9）的子集中，VAREdit的区域保留得分甚至超过OmniGen，证明非保守的真正保留
- 开源模型中，VAREdit-8B的GPT-Balance超越了参数量更大的Step1X-Edit（21B）和FLUX.1 Kontext（12B）

## 亮点与洞察

- **范式创新**: 首次将VAR的多尺度预测成功引入指令引导编辑，突破了扩散模型主导的现状
- **分析驱动设计**: SAR模块的设计完全来自对注意力热力图的系统分析，而非直觉猜测
- **效率与质量兼得**: 2.2B模型在0.7秒内完成512²编辑，质量超越17B的ICEdit
- **自回归范式对编辑的天然优势**: 因果生成机制自然支持区域选择性修改

## 局限与展望

- 依赖离散visual tokenizer，其重建质量限制了编辑的上限
- 目前最大模型为8B，更大规模的scaling效果待验证
- 未支持交互式编辑和多轮对话式编辑
- 可探索与mask引导的结合以进一步提升区域控制精度

## 相关工作与启发

- VAR (Tian et al., 2024) 和 Infinity 为多尺度自回归生成奠定基础，VAREdit将其扩展到编辑
- InstructPix2Pix建立了指令编辑的标准范式，但扩散模型的局限推动了本工作的AR探索
- 启发: 不同于"修补"扩散模型的不足，转换到自回归范式可能是更根本的解决方案

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将VAR范式引入图像编辑，SAR设计有数据驱动的支撑
- 实验充分度: ⭐⭐⭐⭐⭐ 四个benchmark、多种指标、与frontier模型对比
- 写作质量: ⭐⭐⭐⭐ 分析清晰，从问题到解决方案的逻辑链完整
- 价值: ⭐⭐⭐⭐⭐ 性能大幅超越SOTA，开创编辑新范式

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing](editreward_a_human-aligned_reward_model_for_instruction-guided_image_editing.md)
- [\[CVPR 2026\] Depth Adaptive Efficient Visual Autoregressive Modeling](../../CVPR2026/image_generation/depthvar_depth_adaptive_var.md)
- [\[ICCV 2025\] Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing](../../ICCV2025/image_generation/early_timestep_zero-shot_candidate_selection_for_instruction-guided_image_editin.md)
- [\[NeurIPS 2025\] InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation](../../NeurIPS2025/image_generation/infinitystar_unified_spacetime_autoregressive_modeling_for_v.md)
- [\[ICLR 2026\] EditScore: Unlocking Online RL for Image Editing via High-Fidelity Reward Modeling](editscore_unlocking_online_rl_for_image_editing_via_high-fidelity_reward_modelin.md)

<!-- RELATED:END -->
