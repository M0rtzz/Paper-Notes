---
title: >-
  [论文解读] RnG: A Unified Transformer for Complete 3D Modeling from Partial Observations
description: >-
  [CVPR 2026][3D视觉][3D重建] RnG 提出重构引导因果注意力（Reconstruction-Guided Causal Attention），将 Transformer 的 KV-Cache 重新解释为隐式 3D 表示，用单个前馈 Transformer 统一完成从无位姿稀疏图像到完整 3D 几何与外观的重建与生成，速度比扩散方法快 100 倍以上。
tags:
  - "CVPR 2026"
  - "3D视觉"
  - "3D重建"
  - "novel view synthesis"
  - "Transformer"
  - "KV-Cache"
  - "feed-forward"
---

# RnG: A Unified Transformer for Complete 3D Modeling from Partial Observations

**会议**: CVPR 2026  
**arXiv**: [2603.01194](https://arxiv.org/abs/2603.01194)  
**代码**: [https://npucvr.github.io/RnG](https://npucvr.github.io/RnG)  
**领域**: 3D视觉  
**关键词**: 3D reconstruction, novel view synthesis, transformer, KV-Cache, feed-forward

## 一句话总结

RnG 提出重构引导因果注意力（Reconstruction-Guided Causal Attention），将 Transformer 的 KV-Cache 重新解释为隐式 3D 表示，用单个前馈 Transformer 统一完成从无位姿稀疏图像到完整 3D 几何与外观的重建与生成，速度比扩散方法快 100 倍以上。

## 背景与动机

### 核心问题
当前 3D 重建基础模型（如 VGGT、DUSt3R）能从少量图像恢复可见区域的几何结构，但**无法建模未观测区域**。新视角合成（NVS）方法虽然能渲染未知视角，但通常缺乏一致的 3D 结构，或依赖已知位姿/扩散模型导致推理缓慢。

### 已有方法局限

| 方法 | 无位姿推理 | 相机控制 | 生成未见区域 | 显式3D | 实时推理 |
|------|:---------:|:-------:|:----------:|:-----:|:-------:|
| VGGT | ✓ | N/A | ✗ | ✓ | ✓ |
| DUSt3R | ✓ | N/A | ✗ | ✓ | ✓ |
| LVSM | ✗ | ✓ | ✓ | ✗ | ✓ |
| LGM | ✗ | ✓ | ✗ | ✓ | ✓ |
| Matrix3D | ✓ | ✓ | ✓ | ✓ | ✗ |
| **RnG (Ours)** | **✓** | **✓** | **✓** | **✓** | **✓** |

Matrix3D 虽然实现了统一重建与生成，但基于扩散设计，生成单张新视角需 27 秒，无法满足实时交互需求。

### 核心洞察
3D 重建基础模型的 latent space 中可能已编码了比可见几何更完整的 3D 理解。如果能将视角条件化的神经渲染作为对模型潜空间的查询，就能同时激活重建与生成能力。与以往"用生成先验辅助重建"的方向相反，RnG 证明了**用重建先验驱动生成**同样可行且高效。

## 方法详解

### 整体框架

RnG 要解决的是「从无位姿稀疏图像同时拿到可见区域的精确几何和未见区域的合理生成」，而且要快到能交互。它直接复用 VGGT 的架构与预训练权重，整体是一个前馈 Transformer：源视角图像 $\{\mathbf{I}_s\}$ 先过 DINOv2 抽 token，目标视角则用 Plücker 射线图编码再经线性层映射成 token；两组 token 拼在一起送进 $L=24$ 层交替的全局注意力（Global Attention）和帧级注意力（Frame Attention），最后由三个 head 各取所需——源视角 token 经 Camera Head 估计相机位姿 $\{\hat{\mathbf{g}}_s\}$，目标视角 token 分别经 RGB Head $\mathcal{D}_\text{RGB}$ 与 Point Head $\mathcal{D}_\text{pmap}$ 解出新视角图像 $\hat{\mathbf{I}}_t$ 和点图 $\hat{\mathbf{p}}_t$。

为保住 VGGT 已学的知识，第一个源视角用专属的相机和寄存器 token，其余源视角与目标视角共享同一套 token；训练时把第一个视角的位姿固定为

$$\hat{\mathbf{g}}_{s=1} = \left[I_{3\times3} \mid [0, 0, -1]^\top\right]$$

从而隐式定义了重建的世界坐标系。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["源视角图像 {I_s}<br/>DINOv2 抽 token"] --> C["24 层交替<br/>全局注意力 + 帧级注意力"]
    B["目标视角<br/>Plücker 射线图 → 线性层 token"] --> C
    C --> D["重构引导因果注意力<br/>二值掩码 M 拧成单向：<br/>源 query 只看源，目标 query 看源+目标"]
    D -->|源视角分支| E["KV-Cache 隐式 3D 表示<br/>缓存源 K/V（~0.2s），当可反复查询的 3D 记忆"]
    D -->|源视角分支| G["Camera Head → 相机位姿 ĝ_s"]
    E -.每视角读缓存.-> F["目标 Q 查询 [K_s';K_t]<br/>合成新视角（<0.1s/视角）"]
    D -->|目标视角分支| F
    F --> H["RGB Head → 新视角图像 Î_t"]
    F --> I["Point Head → 点图 p̂_t<br/>多视角累积成完整 3D"]
```

### 关键设计

**1. 重构引导因果注意力：让重建指导生成，但生成不污染重建**

源视角负责「看清楚已有的」、目标视角负责「编出没见过的」，可二者共享同一套参数，如果放任双向注意力，生成的噪声会反过来扰乱重建。RnG 的做法是在全局注意力块里加一个二值掩码 $M$ 把信息流拧成单向：

$$M_{i,j} = \begin{cases} 0 & \text{if } i \in \{s\} \text{ and } j \in \{t\} \\ 1 & \text{elsewhere} \end{cases}$$

其中 $\{s\}$、$\{t\}$ 分别是源、目标视角 token 的索引，注意力变为 $\text{Out} = \text{softmax}\left(\frac{M \odot QK^\top}{\sqrt{d_k}}\right)V$。这样源视角 query 只能 attend 源视角 key（重建完全不受目标视角影响），目标视角 query 则同时看源和目标（生成时能调用重建信息）。妙在不靠拆分模块、只靠一张掩码，就在共享参数的前提下把「感知 + 定位」和「合成外观 + 几何」两个功能在注意力层面解耦，参数高效又能联合训练。

**2. KV-Cache 作为隐式 3D 表示：把缓存的 key/value 当成可反复查询的 3D 记忆**

既然源视角的注意力过程与目标视角完全独立，那源视角那套 key/value 就不再是临时中间量，而是一份与观测方向无关、编码了场景几何与外观的隐式 3D 表示。RnG 据此把推理拆成两段：先用源视角跑一次重建，把每层全局注意力的 key/value 缓存下来（$K_s' = \text{Cache}(K_s),\ V_s' = \text{Cache}(V_s)$，约 0.2 秒）；之后每合成一个新视角都无需再碰源视角，直接读缓存

$$\text{Out}_t = \text{softmax}\left(\frac{Q_t \cdot [K_s'; K_t]^\top}{\sqrt{d_k}}\right)[V_s'; V_t]$$

经 $L$ 层后由两个 DPT Head 解出 $\hat{\mathbf{I}} = \mathcal{D}_\text{RGB}(\text{Out}_t)$ 和 $\hat{\mathbf{P}} = \mathcal{D}_\text{pmap}(\text{Out}_t)$（<0.1 秒/视角）。不断累积多个视角查询得到的点图就能拼出完整 3D，相当于一台「虚拟 3D 扫描仪」——这也是它比扩散方法快两个数量级的根本原因。

### 损失函数 / 训练策略

多任务损失由三部分组成：

$$\mathcal{L} = \mathcal{L}_\text{RGB} + \lambda_\text{pmap}\mathcal{L}_\text{pmap} + \lambda_c\mathcal{L}_\text{cam}$$

新视角图像损失用 MSE + 感知损失 $\mathcal{L}_\text{RGB} = |\mathbf{I}_t - \hat{\mathbf{I}}_t|_2 + \lambda_p \cdot \text{Perceptual}(\mathbf{I}_t, \hat{\mathbf{I}}_t)$；点图损失是不确定性加权的 aleatoric uncertainty loss，Point Head 输出四通道（xyz + 不确定性 $\Sigma_t$）：

$$\mathcal{L}_\text{pmap} = \|\Sigma_t \odot (\mathbf{P}_t - \hat{\mathbf{P}}_t)\| + \|\Sigma_t \odot (\nabla\mathbf{P}_t - \nabla\hat{\mathbf{P}}_t)\| - \alpha \cdot \log\Sigma_t$$

相机位姿损失用 Huber 损失 $\mathcal{L}_\text{cam} = \sum_s |\mathbf{g}_s - \hat{\mathbf{g}}_s|_\epsilon$。超参为 $\lambda_\text{pmap}=0.2$、$\lambda_c=1$、$\lambda_p=0.5$、$\alpha=0.2$。训练用 Objaverse（LVIS 子集 + LGM 筛选列表，共 113.5K 物体），分辨率 $256 \times 256$、patch size = 8，8 × A800、总 batch size = 96、训练 40K steps，bfloat16 + gradient checkpointing。

## 实验

### 主实验结果（GSO 数据集）

| 指标类别 | 指标 | Matrix3D (unposed) | VGGT | LVSM (posed) | **RnG (Ours)** |
|---------|------|:------------------:|:----:|:------------:|:-------------:|
| 位姿 | RA@5↑ | 43.77 | 74.24 | — | **85.15** |
| 位姿 | RT@5↑ | 65.92 | 65.68 | — | **86.02** |
| 位姿 | AUC@30↑ | 66.39 | 77.23 | — | **86.94** |
| 源视角深度 | Rel↓ | 9.43 | 5.96 | — | **0.584** |
| 源视角深度 | a1↑ | 92.26 | 97.72 | — | **99.93** |
| 新视角深度 | Rel↓ | 9.96 | — | — | **0.717** |
| 新视角深度 | a1↑ | 90.28 | — | — | **99.85** |
| 新视角合成 | PSNR↑ | 18.74 | — | 27.52 | **26.28** |
| 新视角合成 | SSIM↑ | 0.786 | — | 0.902 | 0.891 |
| 新视角合成 | LPIPS↓ | 0.193 | — | 0.090 | 0.098 |
| 完整3D | CD↓ | 0.067 | 0.026 | — | **0.0067** |

**关键发现**：
- RnG 在所有重建指标上大幅超越 VGGT 和 Matrix3D，位姿估计 RA@5 从 74.24 提升到 85.15
- 源视角深度 Rel 误差（0.584）比 VGGT（5.96）降低一个数量级
- 作为无位姿方法，RnG 的新视角合成质量（PSNR 26.28）接近需要已知位姿的 LVSM（27.52）
- Chamfer Distance（0.0067）显著优于所有方法，证明多视角融合的 3D 几何高度一致

### 消融实验

| 模型变体 | RA@5↑ | PSNR↑ | LPIPS↓ | 说明 |
|---------|:-----:|:-----:|:------:|------|
| LVSM-100K | — | 27.52 | 0.090 | LVSM 最佳性能（需位姿）|
| LVSM-40K | — | 24.62 | 0.154 | 同等训练步数 |
| **Ours-40K** | **85.15** | **26.28** | **0.098** | 完整模型 |
| Ours-15K | 81.65 | 24.86 | 0.124 | 小数据集 |
| Ours-15K-scratch | 8.25 | 20.78 | 0.204 | 无预训练权重 |
| Ours-15K-w/o cam | — | 24.85 | 0.124 | 去除相机位姿监督 |
| Ours-15K-FullAttn | 82.72 | 24.86 | 0.119 | 全双向注意力 |

**消融发现**：
1. **重建先验至关重要**：从头训练（scratch）性能大幅下降，PSNR 下降 4，证明 VGGT 的预训练权重是关键驱动力
2. **训练效率优势**：Ours-15K 即超越 LVSM-40K，说明重建先验带来的数据效率提升
3. **因果注意力 vs 全注意力**：替换为双向注意力（FullAttn）性能几乎无变化，证明因果设计在不牺牲精度的前提下实现了架构优势
4. **位姿监督兼容**：去除 Camera Head 不影响生成质量，多任务学习中重建与生成不冲突

### 效率对比

KV-Cache 使推理大幅加速：推理时间从 213ms 降至 85ms，FLOPS 从 12.26T 降至 2.29T；对比 Matrix3D 的 27 秒/视角快 **300+ 倍**。

### 泛化能力

虽然仅用 4 张输入图训练，RnG 可直接泛化到任意数量输入。增加源视角时合成质量持续提升；对于具有对称结构的物体，甚至单张图像也能得到合理结果。

## 亮点与创新

- **统一框架**：首个基于前馈 Transformer 同时实现无位姿 3D 重建和新视角几何+外观生成的模型
- **因果注意力**：通过注意力掩码而非独立模块实现任务解耦，参数高效且设计优雅
- **KV-Cache 复用**：将 NLP 中的 KV-Cache 机制赋予全新语义——隐式 3D 表示，首次缓存后可高效多次查询
- **逆向知识迁移**：从重建到生成的知识迁移方向，与主流的"生成先验辅助重建"方向互补

## 局限性

1. **纹理细节不足**：作为确定性前馈模型，无法像扩散模型那样生成极为精细的纹理
2. **世界坐标系定义**：数据准备中相机均朝向世界原点，实际应用中手持设备采集需满足此假设
3. **多视角累积噪声**：完整 3D 需累积多个视角查询的点图，多视角几何融合可能引入噪声和冲突

## 评分

| 维度 | 分数 |
|------|:----:|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 综合推荐 | ⭐⭐⭐⭐⭐ |

> 将 KV-Cache 重新解释为隐式 3D 表示是一个漂亮的设计，重建驱动生成的范式为统一 3D 理解提供了实时可行的新路径。实验全面，在多个任务上均达到 SOTA，且推理速度比扩散方法快两个数量级。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] GGPT: Geometry-Grounded Point Transformer](ggpt_geometry_grounded_point_transformer.md)
- [\[CVPR 2026\] Unified Primitive Proxies for Structured Shape Completion](unified_primitive_proxies_for_structured_shape_completion.md)
- [\[CVPR 2026\] LitePT: Lighter Yet Stronger Point Transformer](litept_lighter_yet_stronger_point_transformer.md)
- [\[CVPR 2026\] MimiCAT: Mimic with Correspondence-Aware Cascade-Transformer for Category-Free 3D Pose Transfer](mimicat_mimic_with_correspondence-aware_cascade-transformer_for_category-free_3d.md)
- [\[CVPR 2026\] PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis](pr-iqa_partial-reference_image_quality_assessment_for_diffusion-based_novel_view.md)

</div>

<!-- RELATED:END -->
