---
title: >-
  [论文解读] GuideFlow3D: Optimization-Guided Rectified Flow For Appearance Transfer
description: >-
  [NeurIPS 2025][图像生成][3D appearance transfer] 提出 GuideFlow3D，一种无需训练的 3D 外观迁移框架，通过在预训练 rectified flow 模型的采样过程中交替注入可微引导损失（部件感知外观损失 + 自相似性损失），实现几何差异显著的物体间鲁棒的纹理与几何细节迁移。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "3D appearance transfer"
  - "rectified flow"
  - "universal guidance"
  - "structured latent"
  - "part-aware loss"
---

# GuideFlow3D: Optimization-Guided Rectified Flow For Appearance Transfer

**会议**: NeurIPS 2025  
**arXiv**: [2510.16136](https://arxiv.org/abs/2510.16136)  
**代码**: [项目页面](https://sayands.github.io/guideflow3d)  
**领域**: 图像生成  
**关键词**: 3D appearance transfer, rectified flow, universal guidance, structured latent, part-aware loss  

## 一句话总结
提出 GuideFlow3D，一种无需训练的 3D 外观迁移框架，通过在预训练 rectified flow 模型的采样过程中交替注入可微引导损失（部件感知外观损失 + 自相似性损失），实现几何差异显著的物体间鲁棒的纹理与几何细节迁移。

## 背景与动机
将外观（纹理 + 精细几何细节）从一个 3D 物体迁移到另一个 3D 物体，在游戏、AR 和数字内容创作中有广泛应用。现有方法在输入物体与外观物体几何差异较大时表现不佳：

- **2D 风格迁移 → 3D 提升**：先在多视图上做 2D 风格迁移，再重建 3D，但视图间几何不一致导致伪影。
- **3D 生成模型直接应用**：如 Trellis 等 rectified flow 模型可以生成高质量 3D，但受限于训练时的条件信号和数据分布，直接用于外观迁移时泛化能力差，尤其当几何差异大时失败。
- **ControlNet 类方法**（如 TEXTure、EasiTex）：依赖特定训练设定和条件模态，泛化性有限。
- **纯优化方法**：直接优化潜空间使其匹配外观目标，会偏离生成网络建模的数据分布，产生不自然结果。

核心动机：能否利用预训练 3D 生成模型的归纳偏置，通过推理时引导实现灵活的外观迁移，而无需重新训练？

## 核心问题
如何在保持输入物体全局几何结构的前提下，将外观物体的纹理和精细几何细节鲁棒地迁移过去——尤其当两者几何差异很大时（如椅子→床、长颈鹿→家具）？

## 方法详解

### 整体框架
GuideFlow3D 基于 Trellis 的 structured latent (SLat) 表示和 rectified flow 生成模型，在推理时通过交替执行"flow 步骤"和"引导优化步骤"来控制生成过程。

### 1. Structured Latent 表示
3D 物体 $\mathcal{O}$ 被编码为 structured latent：

$$\mathbf{z} = \{(z_i, p_i)\}_{i=1}^{L}, \quad z_i \in \mathbb{R}^C, \quad p_i \in \{0, 1, \ldots, N-1\}^3$$

- $p_i$：活跃体素位置（与物体表面相交），勾勒粗略结构
- $z_i$：对应体素的潜向量，捕获精细几何和纹理特征
- 关键设计：**$p_i$ 固定不变**（保持全局几何），只引导 $z_i$ 的生成

### 2. 引导目标函数

**（a）Part-aware 外观损失 $\mathcal{L}_{\text{appearance}}$**（适用于外观物体有 mesh 的情况）：

$$\mathcal{L}_{\text{appearance}} = \frac{1}{L_q} \sum_{i=1}^{L_q} \| \tilde{z}_i^q - z_m^a \|_2^2$$

- 利用 PartField 的几何特征做 co-segmentation 聚类，建立输入物体与外观物体之间的部件级对应关系（如：椅背↔椅背、椅腿↔椅腿）
- $m$ 为通过部件聚类匹配到的外观物体中对应体素的索引
- 确保局部化的纹理和几何对应

**（b）自相似性损失 $\mathcal{L}_{\text{structure}}$**（适用于外观物体仅为图像或文本的情况）：

$$\mathcal{L}_{\text{structure}} = -\frac{1}{L_q} \sum_{i=1}^{L_q} \log \frac{\sum_{j \in \mathcal{C}_q(i), j \neq i} \exp(\text{sim}_{ij})}{\sum_{j \in \mathcal{C}'_q(i)} \exp(\text{sim}_{ij})}$$

- 基于几何聚类的对比损失：同一部件内的体素特征相似（正样本），不同部件间的体素特征不同（负样本）
- 鼓励局部一致性而不全局同质化

### 3. Guided Rectified Flow 采样
标准 rectified flow 的反向过程为：

$$\hat{\mathbf{z}}_t = \mathbf{z}_t + \Delta t \cdot \mathbf{v}_\theta(\mathbf{z}_t, t \mid \mathbf{c})$$

GuideFlow3D 在每步中注入引导梯度：

$$\hat{\mathbf{z}}_t = \mathbf{z}_t + \Delta t \cdot \mathbf{v}_\theta(\mathbf{z}_t, t \mid \mathbf{c}) + \nabla_{\mathbf{z}_t} \mathcal{L}(\mathbf{z} \mid \mathbf{c})$$

- 条件 $\mathbf{c}$ 可以是图像或文本
- 贝叶斯视角：rectified flow 建模先验 $P(\mathcal{O})$ 和似然 $P(\mathbf{c}|\mathcal{O})$，引导项建模额外约束
- 将 universal guidance 从扩散模型推广到任意 rectified flow 模型

### 4. 条件灵活性
- **图像 + Mesh 条件**：使用 $\mathcal{L}_{\text{appearance}}$，迁移纹理和几何细节
- **仅图像条件**：使用 $\mathcal{L}_{\text{structure}}$（可先用 Trellis 从图像生成 mesh）
- **文本条件**：使用 $\mathcal{L}_{\text{structure}}$，仅迁移纹理

## 训练与推理

### 无需训练
GuideFlow3D 是一个完全 **training-free** 的框架，不需要对生成模型进行任何微调或重新训练。所有外观迁移控制均在推理时通过引导注入完成。

### 实现细节
- **基础模型**：使用 Trellis 预训练模型（图像条件用 `trellis-image-large`，文本条件用 `trellis-text-large`），沿用其默认配置
- **部件特征**：通过 PartField 计算每个 mesh 的 part feature field，用体素坐标 $p_i$ 查询得到每个体素的部件特征
- **采样步数**：rectified flow 采样与单实例优化交替执行，共 300 步
- **优化器**：AdamW，学习率 $5 \times 10^{-4}$
- **硬件**：单张 NVIDIA RTX 4090 GPU
- **运行时间**：96 秒（baseline Trellis 为 78 秒，额外开销约 23%）
- 所有条件类型（图像/文本）使用相同的优化设定

### 评估渲染
- 使用 Blender 渲染所有资产，smooth area lighting
- 每个物体从 4 个视角渲染（固定半径 2，pitch 30°，yaw 从 45° 起每 90° 一次）
- 所有 mesh 使用 canonical pose 确保对齐
- 指标按视角和物体分别计算后取平均

## 实验关键数据

### 数据集
- 输入 mesh：程序化生成的简单几何体（simple）
- 外观物体：ABO 数据集（complex），约 8K 个 3D 模型，55 个类别
- 4 种实验设定：simple-complex 类内/类间、complex-complex 类内/类间
- 每种设定 250 对输入-外观组合

### 评估方式
传统编码器度量（PSNR、SSIM、LPIPS、FID 等）需要 ground truth 且无法处理不相似几何，因此采用 **GPT-based 排名系统**，从 6 个维度评判：Style Fidelity、Structure Clarity、Style Integration、Detail Quality、Shape Adaptation、Overall Quality（排名越低越好）。用户研究确认 GPT 排名与人类偏好高度一致。

### 主要结果（simple-complex 类内，图像条件）

| 方法 | Fidelity↓ | Clarity↓ | Overall↓ |
|------|-----------|----------|----------|
| UV Nearest Neighbor | 4.12 | 3.84 | 4.33 |
| MambaST | 4.94 | 3.55 | 4.87 |
| Cross Image Attention | 3.56 | 3.48 | 3.59 |
| EasiTex | 3.18 | 4.30 | 3.81 |
| Trellis | 2.51 | 2.58 | 2.62 |
| **GuideFlow3D (Ours)** | **1.89** | **2.41** | **2.12** |

### 文本条件结果（simple-complex 类内）

| 方法 | Fidelity↓ | Clarity↓ | Overall↓ |
|------|-----------|----------|----------|
| Trellis | 2.01 | 1.89 | 2.39 |
| **GuideFlow3D (Ours)** | **1.54** | **1.63** | **1.95** |

- 在所有设定（类内/类间、simple/complex）和两种条件模态下，GuideFlow3D 均取得最佳排名
- In-the-wild 实验展示跨语义类别的鲁棒迁移（动物→家具、家具→交通工具等）

### 运行时间
- GuideFlow3D：96 秒（NVIDIA 4090 GPU）
- Trellis baseline：78 秒
- 额外开销约 23%，换取显著的质量提升

### 消融实验（Ablation Study）
在 simple-complex 类内图像条件设定下，对不同设计选择进行消融：

| 变体 | Fidelity↓ | Clarity↓ | Overall↓ |
|------|-----------|----------|----------|
| (i) 无 flow + 全局特征 (global feat.) | 4.52 | 4.51 | 4.50 |
| (ii) 无 flow + SLat 空间 NN 匹配 | 3.58 | 3.62 | 3.63 |
| (iii) 有 flow + K-means on SLat (非 PartField) | 2.57 | 2.65 | 2.66 |
| (iv) 有 flow + $\mathcal{L}_{\text{structure}}$ (图像条件) | 2.17 | 2.05 | 2.03 |
| (v) 有 flow + $\mathcal{L}_{\text{appearance}}$ (图像条件) | **1.23** | **1.08** | **1.06** |

关键发现：
1. **全局特征不够**：min/max/avg pooling 的全局潜向量无法捕获语义对应关系
2. **非结构化 NN 匹配不够**：在 SLat 空间直接做最近邻虽改善 fidelity，但缺乏鲁棒的语义对齐
3. **PartField vs K-means on SLat**：语义感知的 PartField 分割显著优于直接在 SLat 特征上做 K-means，说明 part-aware 语义信息是建立准确部件对应的关键
4. **两种损失互补**：$\mathcal{L}_{\text{appearance}}$ 在 fidelity 上更强，$\mathcal{L}_{\text{structure}}$ 在对齐和适应性上更好

### 场景编辑应用
- 在 ScanNet 室内场景上验证了 GuideFlow3D 的场景级编辑能力
- 利用 per-object CAD mesh 标注为场景中每个语义类别选择外观物体做迁移
- 可以在保持空间布局的同时选择性地重新风格化场景中的多个物体
- 展示了交互式场景定制的应用潜力

### 传统指标的局限性
- DINOv2、CLIP Score、DreamSim 等需要 ground truth 或假设几何相似
- 当输入和外观物体几何差异大时，这些度量无法反映真实迁移质量
- 例如 CLIP Score 在文本条件下反而给 Trellis baseline 更高分，因为文本通常描述与输入几何不同的形状

## 亮点
1. **无需训练**：完全在推理时通过引导注入实现外观迁移，不修改生成模型参数
2. **几何鲁棒性**：通过固定体素位置 $p_i$ 保持全局几何，使用部件感知损失处理大几何差异
3. **统一多模态框架**：同一框架下支持 mesh、图像、文本三种外观表示
4. **原理性强**：基于贝叶斯公式将 universal guidance 推广到 rectified flow，理论框架清晰
5. **通用可扩展**：方法可推广到不同的扩散/flow 模型和引导函数
6. **评估创新**：提出基于 GPT 的多维度排名评估体系，并通过用户研究验证其与人类判断的一致性

## 局限与展望
1. **非实时**：基于优化的方法，96 秒推理不适合实时场景；未来可训练自监督前馈模型加速
2. **依赖外部模型**：依赖 Trellis（SLat 编码/解码）和 PartField（部件特征），这些模型的失败会级联影响结果
3. **需要干净 mesh**：假设输入为无噪声网格，限制了对扫描数据等噪声输入的处理
4. **主实验范围有限**：主实验集中在家具类别（ABO 数据集），虽然 in-the-wild 展示了更广泛的泛化，但缺乏系统评估
5. **缺少传统指标对比**：完全依赖 GPT-based 评估，可能遗漏某些客观质量差异

## 与相关工作的对比

| 方法 | 训练需求 | 几何鲁棒性 | 多模态支持 | 部件感知 | 输出表示 |
|------|---------|-----------|-----------|---------|---------|
| StyleGaussian | 需训练 | 弱 | 仅样式 | 否 | 仅渲染 |
| TEXTure | SDS 蒸馏 | 中 | 文本 | 否 | 纹理 |
| EasiTex | ControlNet | 弱（大几何偏差时） | 图像 | 否 | 纹理 |
| Trellis | 无需额外训练 | 弱 | 图像/文本 | 否 | Mesh/3DGS/NeRF |
| Cross Image Attention | 无需训练 | 弱（2D→3D 伪影） | 图像 | 否 | 依赖提升方法 |
| **GuideFlow3D** | **无需训练** | **强** | **Mesh+图像+文本** | **是** | **Mesh/3DGS/NeRF** |

## 启发与关联
- **Universal guidance 的 3D 推广**：将 Bansal et al. 的 2D diffusion universal guidance 思想推广到 3D rectified flow 模型，为 3D 生成的可控性开辟了新方向——任何可微目标函数都可以在推理时注入
- **部件感知与 PartField 的结合**：利用 PartField 的几何 co-segmentation 建立跨物体的部件对应，是一种优雅的解决大几何差异下对应关系的方案
- **结构化潜空间的位置-特征解耦**：$p_i$ 固定 + $z_i$ 可变的设计巧妙地实现了"保几何、改外观"的需求，这一思路可推广到其他 3D 编辑任务
- **GPT-as-evaluator 范式**：在缺乏 ground truth 的生成任务评估中，GPT 排名 + 用户研究验证是一种值得借鉴的方案

## 评分
- 新颖性: ⭐⭐⭐⭐ （将 universal guidance 推广到 3D rectified flow 的思路新颖，部件感知损失设计有创意）
- 实验充分度: ⭐⭐⭐⭐ （多种设定、多种基线对比、in-the-wild 展示、用户研究、消融实验齐全）
- 写作质量: ⭐⭐⭐⭐ （公式推导清晰，图示丰富，方法动机充分）
- 价值: ⭐⭐⭐⭐ （无需训练的 3D 外观迁移框架，实用性强且可扩展性好）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Rectified-CFG++ for Flow Based Models](rectified-cfg_for_flow_based_models.md)
- [\[NeurIPS 2025\] Efficient Rectified Flow for Image Fusion](efficient_rectified_flow_for_image_fusion.md)
- [\[ICCV 2025\] Straighten Viscous Rectified Flow via Noise Optimization](../../ICCV2025/image_generation/straighten_viscous_rectified_flow_via_noise_optimization.md)
- [\[ICML 2026\] Offline Preference Optimization for Rectified Flow with Noise-Tracked Pairs](../../ICML2026/image_generation/offline_preference_optimization_for_rectified_flow_with_noise-tracked_pairs.md)
- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](on_the_relation_between_rectified_flows_and_optimal_transport.md)

</div>

<!-- RELATED:END -->
