---
title: >-
  [论文解读] SegAgent: Exploring Pixel Understanding Capabilities in MLLMs by Imitating Human Annotator Trajectories
description: >-
  [CVPR 2025][多模态VLM][交互式分割] SegAgent 将 referring expression segmentation 建模为人类标注员的迭代操作过程——MLLM 观察当前 mask 状态后预测下一个点击位置，交互式分割模型据此更新 mask，经过多轮迭代得到最终分割结果；通过 StaR+ 策略改进和 PRM+树搜索，在复杂场景下大幅提升分割精度。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "交互式分割"
  - "MLLM标注代理"
  - "多步MDP"
  - "过程奖励模型"
  - "策略改进"
---

# SegAgent: Exploring Pixel Understanding Capabilities in MLLMs by Imitating Human Annotator Trajectories

**会议**: CVPR 2025  
**arXiv**: [2503.08625](https://arxiv.org/abs/2503.08625)  
**代码**: 待确认  
**领域**: 多模态VLM  
**关键词**: 交互式分割、MLLM标注代理、多步MDP、过程奖励模型、策略改进

## 一句话总结
SegAgent 将 referring expression segmentation 建模为人类标注员的迭代操作过程——MLLM 观察当前 mask 状态后预测下一个点击位置，交互式分割模型据此更新 mask，经过多轮迭代得到最终分割结果；通过 StaR+ 策略改进和 PRM+树搜索，在复杂场景下大幅提升分割精度。

## 研究背景与动机

**领域现状**：现有的 MLLM 分割方法（如 LISA）通常让 MLLM 输出隐式 token，通过 decoder 一步生成 mask。这类方法本质上是单步预测，缺乏对复杂形状的逐步细化能力。

**现有痛点**：(1) 单步输出方式对细粒度像素理解能力要求极高，MLLM 的视觉编码器经过 Q-former 等抽象后会丢失空间局部性；(2) 现有 RefCOCO 等数据集标注质量有限且复杂度不够（通常 1-2 次点击即可达到目标 IoU），无法充分评估 MLLM 的像素级理解能力；(3) 人类标注员在标注分割时是通过多次交互迭代完成的，而现有方法完全忽略了这种渐进式推理过程。

**核心矛盾**：MLLM 擅长全局语义理解和粗粒度定位，但在像素级精细分割上不擅长；交互式分割模型（SAM 等）擅长精细分割但不理解自然语言。如何结合两者？

**本文目标** 如何让 MLLM 扮演人类标注员的角色，通过多步迭代与交互式分割工具配合完成高质量 referring segmentation？

**切入角度**：将分割任务建模为 MDP——状态是当前 mask（叠加在原图上），动作是预测正/负点击坐标（纯文本输出），转移由交互式分割模型完成，奖励是与 GT 的 IoU。这样 MLLM 只需做"在哪里点击"的决策，精细分割交给专业模型。

**核心 idea**：让 MLLM 模仿人类标注员的点击轨迹，通过多步交互式分割迭代完成 referring segmentation，并用 RL 式策略改进和过程奖励模型提升决策质量。

## 方法详解

### 整体框架
输入为图像 I + 文本提示 P + 空白初始 mask M₀。每步将当前 mask 半透明叠加到原图上作为观测，MLLM 输出下一个点击动作（正/负标记 + 归一化坐标），交互式分割模型（SAM/SimpleClick）据此更新 mask。迭代 T 步（RefCOCO T=7，HRES T=11），取最终 mask 作为结果。

### 关键设计

1. **HLMAT: 人类标注模拟任务（Human-Like Mask Annotation Task）**:

    - 功能：自动生成 MLLM 训练所需的标注轨迹数据
    - 核心思路：给定 GT mask，用规则化的点击模拟器 F_sim 生成轨迹——每步找到当前 mask 与 GT 之间的最大错误区域（假阳/假阴），在其中心放置下一个点击。三个过滤机制保证轨迹质量：限制最大长度 T、IoU 达到 τ_stop 时终止、丢弃 IoU 提升小于 τ_diff 的动作
    - 设计动机：手动收集标注轨迹代价太高，规则模拟器可以从已有分割数据集自动大规模生成训练数据。MLLM 在这些轨迹上做指令微调即可学会"看 mask 状态 → 决定在哪里点击"

2. **StaR+ 策略改进**:

    - 功能：通过自我迭代生成更好的训练数据来提升策略
    - 核心思路：从 SFT 基线模型 S₀ 出发，让模型自己在训练集上生成新轨迹，用奖励函数过滤——每步 IoU 提升低于阈值的动作替换为 F_sim 生成的修正动作。将修正后的新轨迹加入训练集重新微调。与原版 STaR 不同的是按逐步奖励变化（而非整体轨迹正确性）过滤
    - 设计动机：SFT 仅学到模拟器的行为分布，通过模型实际推理的轨迹做迭代改进可以覆盖更多分布、修复错误模式。实验显示 StaR+ 在 ThinObject5K 上提升 +15.12% IoU

3. **过程奖励模型 (PRM) + 贪心树搜索**:

    - 功能：在推理时通过搜索选择每步最优动作
    - 核心思路：训练 MLLM 同时预测每步的奖励值（以文本形式输出"Current mIoU: 0.75"）。推理时每步生成 K 个候选动作（Multinomial Sampling），对每个候选执行交互式分割并用 PRM 预测其奖励，选择 PRM 得分最高的动作。选用简单的贪心搜索而非 MCTS 等复杂方法
    - 设计动机：分割是一个过程奖励天然可以定义（IoU）的任务，PRM 让模型学会自我评估，树搜索在推理时扩大搜索空间。K=3 时 DIS5K 上从 81.17→88.60 IoU（+7.43）

### 损失函数 / 训练策略
标准指令微调 loss（自回归 cross-entropy），冻结图像编码器，微调 LLM + projector。2 epochs，8×80GB GPU，DeepSpeed Zero2。基模型支持 LLaVA-v1.5-7B 和 Qwen-VL-7B，交互式分割支持 SAM 和 SimpleClick。

## 实验关键数据

### 主实验

| 数据集 | 方法 | IoU |
|--------|------|------|
| refCOCO val | LISA+SAM | 74.9 |
| refCOCO val | SAM4MLLM-Qwen | 77.1 |
| refCOCO val | **SegAgent-Qwen+SClick** | **79.69** |
| refCOCO+ val | LISA+SAM | 65.1 |
| refCOCO+ val | **SegAgent-Qwen+SClick** | **72.49** |
| refCOCOg val(U) | LISA+SAM | 67.9 |
| refCOCOg val(U) | **SegAgent-Qwen+SClick** | **75.11** |

### 消融实验（HRES 数据集）

| 配置 | DIS5K IoU | ThinObject5K IoU |
|------|---------|---------|
| Baseline SFT | 71.45 | 71.45 |
| + StaR+ 策略改进 | 78.81 (+7.36) | 86.57 (+15.12) |
| + PRM (K=1) | 81.17 (+2.36) | 75.54 (+4.09) |
| + Tree Search (K=3) | **88.60** (+7.43) | **86.13** (+10.59) |

### 关键发现
- **多步迭代远优于单步**：RefCOCO 等简单数据集只需 1-2 步，但 DIS5K/ThinObject5K 等复杂场景需要 7-11 步，证明了迭代方法的必要性
- **LLaVA vs Qwen 的有趣对比**：LLaVA+SAM 在 mask 精细化（fine-grained）上更强，Qwen+SimpleClick 在粗粒度定位上更强。Q-former 架构在语义抽象过程中丢失了空间局部性
- **StaR+ 的巨大提升**：尤其在 ThinObject5K 上 +15.12% IoU，说明模型自身轨迹的分布与模拟器轨迹有显著差异，自我改进的价值很大
- **PRM + 树搜索的互补效果**：PRM 单独用（K=1）提升有限，但配合 K=3 树搜索提升巨大，说明搜索空间的扩大对质量至关重要

## 亮点与洞察
- **将分割重新建模为 MDP 是一个范式转换**：不再让 MLLM 一步到位输出 mask，而是让它扮演"标注员"做决策，精细操作交给专业工具。这种人机协作的范式可以迁移到其他需要精细标注的任务
- **PRM 的天然适配性**：分割任务的 IoU 是一个完美的过程奖励信号——每步都可以计算，不需要学习奖励模型。这让 RL 式方法在分割任务上的应用变得特别优雅
- **纯文本坐标输出**：不需要修改 MLLM 架构、不需要添加特殊 token，动作完全用文本表示（"Positive point: (175,483)"），保持了 MLLM 的通用性

## 局限与展望
- 只探索了 1 轮 StaR+ 改进，更多轮次可能进一步提升
- 只用贪心搜索而非 MCTS，更复杂的搜索策略可能更好
- 无法回撤（undo）之前的错误点击，复杂场景下可能累积误差
- 对 RefCOCO 这种简单数据集优势不够突出（因为 1-2 步就够了），方法的价值在复杂场景才充分体现
- Qwen 的 Q-former 架构丢失空间局部性的问题值得 VLM 社区关注

## 相关工作与启发
- **vs LISA**: LISA 让 MLLM 输出隐式 token + decoder 一步生成 mask；SegAgent 多步迭代+专业工具，在 refCOCO 上 +4.8% IoU，在 refCOCO+ 上 +7.4% IoU
- **vs SAM4MLLM**: SAM4MLLM 也用交互式分割但只做单步；SegAgent 的多步 MDP + RL 改进显著更强
- **对 MLLM 架构的启示**：Q-former 虽然在语义理解上强，但在需要空间定位的任务上不如直接 projection，这对未来 VLM 设计有参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将分割建模为人类标注员的 MDP 是非常新颖的范式，StaR+ 和 PRM 的引入自然且有效
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多模型组合对比，消融分析清晰，还引入了 HRES 高质量数据集
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，算法伪代码完整
- 价值: ⭐⭐⭐⭐ 范式创新有启发性，但实际应用受限于推理时多步调用的效率开销

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](sparrow_learning_spatial_precision_and_temporal_referential_consistency_in_pixel.md)
- [\[CVPR 2025\] PEACE: Empowering Geologic Map Holistic Understanding with MLLMs](peace_empowering_geologic_map_holistic_understanding_with_mllms.md)
- [\[ACL 2025\] OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preference](../../ACL2025/multimodal_vlm/omnialign-v_towards_enhanced_alignment_of_mllms_with_human_preference.md)
- [\[CVPR 2026\] HumanVBench: Probing Human-Centric Video Understanding in MLLMs with Automatically Synthesized Benchmarks](../../CVPR2026/multimodal_vlm/humanvbench_probing_human_centric_video_understanding_in_mllms_with_automatica.md)
- [\[NeurIPS 2025\] Face-Human-Bench: A Comprehensive Benchmark of Face and Human Understanding for Multi-modal Assistants](../../NeurIPS2025/multimodal_vlm/face-human-bench_a_comprehensive_benchmark_of_face_and_human_understanding_for_m.md)

</div>

<!-- RELATED:END -->
