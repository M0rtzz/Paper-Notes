---
title: >-
  [论文解读] MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization
description: >-
  [CVPR 2026][目标检测][概念定制] 发现并利用跨模态知识迁移现象——修改 LLM 文本编码器中的知识可自然迁移到视觉生成，提出 MoKus 两阶段框架（视觉概念学习 + 文本知识更新）实现知识感知的概念定制。
tags:
  - CVPR 2026
  - 目标检测
  - 概念定制
  - 知识编辑
  - 跨模态知识迁移
  - 扩散模型
  - LLM文本编码器
---

# MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization

**会议**: CVPR 2026  
**arXiv**: [2603.12743](https://arxiv.org/abs/2603.12743)  
**代码**: [项目页](https://chenyangzhu1.github.io/MoKus)  
**领域**: 目标检测  
**关键词**: 概念定制, 知识编辑, 跨模态知识迁移, 扩散模型, LLM文本编码器

## 一句话总结
发现并利用跨模态知识迁移现象——修改 LLM 文本编码器中的知识可自然迁移到视觉生成，提出 MoKus 两阶段框架（视觉概念学习 + 文本知识更新）实现知识感知的概念定制。

## 研究背景与动机
现有概念定制方法使用稀有 token（如 `<sks>`）绑定目标概念，存在两大问题：

**不稳定性能**：稀有 token 缺乏语义含义，在预训练数据中极少出现，与其他文本组合时生成效果不稳定

**知识无感知**：稀有 token 仅绑定视觉外观，无法存储目标概念的内在知识（如"小美人鱼雕塑在丹麦"）

因此提出**知识感知的概念定制**——将多条自然语言描述的知识绑定到目标视觉概念，使模型能识别提示中的知识并生成高保真定制结果。这比传统概念定制更有挑战：模型需要感知提示中的知识、高效绑定多条知识到同一概念。

## 方法详解

### 整体框架
MoKus 采用 LLM 作为文本编码器 + DiT 作为生成骨干，分两阶段：
1. **视觉概念学习**：将目标概念绑定到"锚表示"（稀有 token 的 text embedding）
2. **文本知识更新**：通过知识编辑技术将每条知识的答案更新为锚表示

核心观察：**跨模态知识迁移**——在 LLM 文本编码器中修改知识（如"贝多芬最爱的乐器"的答案从"钢琴"改为"吉他"），生成结果会相应变化。

### 关键设计

1. **视觉概念学习（Stage 1）**：

    - 输入参考图像 $x_i \in \mathcal{X}$，通过 VAE 编码为 $\mathbf{z}_0 = \mathcal{E}(x_i)$
    - 基于 Rectified Flow：$\mathbf{z}_t = t \cdot \mathbf{z}_0 + (1-t) \cdot \mathbf{z}_1$
    - 使用稀有 token $P$（如 `<sks> dog`）生成文本潜变量 $\mathbf{h} = \phi(P)$
    - 在 MMDiT 的 self-attention 层添加 LoRA 参数 $\theta_v$，最小化速度预测 MSE：
    $\mathcal{L}(\theta_v) = \mathbb{E}\left[\|v_{\theta_v}(\mathbf{z}_t, t, h) - (\mathbf{z}_0 - \mathbf{z}_1)\|_2^2\right]$
    - 学到的稀有 token 成为"锚表示"，存储视觉信息

2. **文本知识更新（Stage 2）**：

    - 将知识集 $\mathcal{K} = \{k_i\}_{i=1}^N$ 转化为问题 $q_i$，配对锚表示 $y$ 形成更新样本 $\{(q_i, y)\}$
    - 将 $q_i$ 输入 LLM 编码器获取隐状态 $\mathbf{h}_i$ 和梯度 $\nabla_{\theta_t} y_i$
    - 计算更新方向：$\mathbf{v}_i = -\eta \cdot \|\mathbf{h}_i\|^2 \cdot \nabla y_i$（$\eta = 1e\text{-}6$）
    - 求解正则化最小二乘问题得到闭式解参数偏移：
    $\Delta\theta_t^* = (\mathbf{H}^\top \mathbf{H} + \mathbf{I})^{-1} \mathbf{H}^\top \mathbf{V}$
    - 直接加到原参数：$\hat{\theta}_t = \theta_t + \Delta\theta_t^*$
    - 仅修改 LLM 编码器第 18~26 层 MLP 的 Gate/Up Projection（共 16 个参数矩阵）
    - 每条知识更新仅需几秒

3. **跨模态知识迁移现象**：

    - 在 LLM 文本编码器中用知识编辑技术更新某个事实的答案
    - 生成时使用相关提示，生成结果自然对应更新后的答案
    - 与 GapEval、UniSandbox 等工作不同，MoKus 采用 UltraEdit 而非直接微调，效果显著

### 损失函数 / 训练策略
- Stage 1：Rectified Flow 标准速度匹配损失 + LoRA 微调 MMDiT，lr=2e-4，AdamW
- Stage 2：正则化最小二乘的闭式解，无需迭代优化，批量更新
- 实验环境：Qwen-Image 模型，8×H800 GPU

## 实验关键数据

### 主实验

| 方法 | CLIP-I (Recon.) ↑ | CLIP-I-Seg (Recon.) ↑ | CLIP-T (Gen.) ↑ | Pick Score ↑ | 训练时间 ↓ |
|------|-------------------|----------------------|-----------------|-------------|-----------|
| Naive-DB | 0.874 | 0.758 | 0.291 | 20.80 | ~27min |
| Enc-FT | 0.582 | 0.553 | 0.197 | 18.34 | ~10min |
| **MoKus (Ours)** | 0.867 | **0.764** | **0.305** | **21.30** | **~6min** |

MoKus 在分割后概念保真度（CLIP-I-Seg）、提示保真度和人类偏好上均最优，且效率最高。

### 消融实验

| 知识条数 | CLIP-I-Seg (Recon.) | CLIP-T (Gen.) | 训练时间 |
|---------|---------------------|---------------|---------|
| 1 | 0.761 | 0.304 | 331s |
| 3 | 0.761 | 0.305 | 345s |
| 5 | 0.764 | 0.305 | 360s |

每增加一条知识仅增加约 7 秒训练时间，性能保持稳定。缩放因子 $\eta = 1e\text{-}6$ 最优。

### 关键发现
- Enc-FT（直接微调编码器）严重破坏输出分布，生成质量大幅下降
- MoKus 可扩展到虚拟概念创建（用描述建立新概念）和概念擦除（修改外观描述）
- 在 WISE 世界知识基准上，知识更新后 WiScore 从 0.81 提升到 1.33

## 亮点与洞察
1. **跨模态知识迁移现象**的发现具有独立价值——揭示了 LLM 文本编码器中知识修改如何影响视觉生成
2. **两阶段解耦设计**：视觉学习与知识绑定分离，使每条知识的更新仅需秒级
3. **闭式解参数偏移**：无需迭代训练，效率极高且可控
4. **KnowCusBench 基准**：5975 张评估图像，覆盖 35 个概念 × 6 个知识视角 × 4 个提示视角

## 局限与展望
- 依赖稀有 token 作为中间锚表示，锚表示的质量受 Stage 1 微调效果限制
- 知识编辑仅修改 MLP 层，更复杂的知识关系可能需要更精细的编辑策略
- 仅在 Qwen-Image 上验证，对其他 LLM+DiT 架构的泛化性待测试
- 概念擦除应用可能被滥用，需要安全考量

## 相关工作与启发
- 与 DreamBooth、Textual Inversion 等概念定制工作相比，MoKus 将知识维度引入定制
- 知识编辑技术（ROME、MEMIT 等）从 NLP 迁移到视觉生成的有趣尝试
- 启发：LLM 文本编码器不仅传递语义，还传递可编辑的知识

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 跨模态知识迁移发现和知识感知定制任务定义均有开创性
- 实验充分度: ⭐⭐⭐⭐ 定量定性对比+消融+多应用扩展，配有专门基准
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法流程图详细
- 价值: ⭐⭐⭐⭐ 新任务+新发现+实用框架，对生成模型研究有启发

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection](learning_multi-modal_prototypes_for_cross-domain_few-shot_object_detection.md)
- [\[CVPR 2026\] SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking](slice_semantic_latent_injection_via_compartmentali.md)
- [\[CVPR 2026\] StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars](streamavatar_streaming_diffusion_models_for_real-time_interactive_human_avatars.md)
- [\[CVPR 2026\] UAVGen: Visual Prototype Conditioned Focal Region Generation for UAV-Based Object Detection](uavgen_visual_prototype_conditioned_focal_region_generation_for_uav_based_object_detection.md)
- [\[CVPR 2026\] PHAC: Promptable Human Amodal Completion](phac_promptable_human_amodal_completion.md)

<!-- RELATED:END -->
