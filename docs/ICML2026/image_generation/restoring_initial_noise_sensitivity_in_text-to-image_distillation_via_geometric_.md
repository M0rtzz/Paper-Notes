---
title: >-
  [论文解读] Restoring Initial Noise Sensitivity in Text-to-Image Distillation via Geometric Alignment
description: >-
  [ICML 2026][图像生成][扩散蒸馏] 本文指出现有 T2I 扩散蒸馏只做"逐点输出对齐"导致学生模型对初始噪声的敏感性塌缩，提出 GAD：用一对扰动输入下的 JVP（雅可比向量积）有限差分近似，强制学生匹配教师对噪声扰动的方向性响应，从而在不损失保真度的前提下恢复布局可控性与生成多样性。 领域现状：DM 与 Flo…
tags:
  - "ICML 2026"
  - "图像生成"
  - "扩散蒸馏"
  - "初始噪声敏感性"
  - "Jacobian-Vector Product"
  - "几何对齐"
  - "T2I"
---

# Restoring Initial Noise Sensitivity in Text-to-Image Distillation via Geometric Alignment

**会议**: ICML 2026  
**arXiv**: [2606.01651](https://arxiv.org/abs/2606.01651)  
**代码**: https://github.com/Hannah1102/GAD (有)  
**领域**: 扩散模型 / 文生图蒸馏  
**关键词**: 扩散蒸馏, 初始噪声敏感性, Jacobian-Vector Product, 几何对齐, T2I

## 一句话总结
本文指出现有 T2I 扩散蒸馏只做"逐点输出对齐"导致学生模型对初始噪声的敏感性塌缩，提出 GAD：用一对扰动输入下的 JVP（雅可比向量积）有限差分近似，强制学生匹配教师对噪声扰动的方向性响应，从而在不损失保真度的前提下恢复布局可控性与生成多样性。

## 研究背景与动机

**领域现状**：DM 与 Flow Matching 已成为 T2I 主流，但需要 20-100 次 NFE，因此蒸馏（output matching / distribution matching / score distillation）被广泛用来把多步轨迹压缩成 1-4 步学生模型。

**现有痛点**：现有蒸馏方法只关心 FID/CLIP 这类"平均输出质量"，并把教师当成静态输入-输出映射。结果是：换不同种子 $\mathbf{z}$，学生输出几乎一样——也就是失去了"对初始噪声的敏感性"。这直接破坏一类下游任务：训练-free 的布局控制（attention guidance 通过 $\mathbf{z}$ 注入空间约束）、NoiseQuery 这类靠最优噪声检索做属性控制的方法、以及单纯靠换种子换图带来的生成多样性，全都靠"教师对 $\mathbf{z}$ 的差异化响应"。

**核心矛盾**：标准蒸馏目标 $\mathcal{L}_{\text{base}}=\mathbb{E}_{\mathbf{z}}[\mathcal{D}(\Phi_S(\mathbf{z}),\Phi_T(\mathbf{z}))]$ 是 pointwise alignment——对每个 $\mathbf{z}$ 独立匹配输出。在多模态目标下，MSE / 反向 KL 会让学生收敛到条件期望（平滑的"平均路径"），把教师在 $\mathbf{z}$ 邻域的局部几何（方向梯度、曲率）抹掉。诊断实验给出了直接证据：学生和教师的 pointwise MSE 已经很低，但 JVP MSE 仍然很高（教师 0.000 vs TDM 0.0003，Tab. 1），JVP 余弦相似度只有 0.012——形状对了，但"切向量"全错。

**本文目标**：在不引入新架构、不依赖额外数据、不破坏 base loss 的前提下，让学生模型在 $\mathbf{z}$ 局部的"差分响应"和教师一致，从而恢复噪声敏感性与下游可控性。

**切入角度**：借鉴经典 KD 中"relational knowledge"的思路（Park et al. 2019, Tung & Mori 2019）——别只学绝对输出，要学样本之间的相对关系。在生成场景里，这个"相对关系"就是教师映射 $\Phi_T$ 的雅可比 $\mathbf{J}_{\Phi_T}(\mathbf{z})$ 所刻画的方向性响应。

**核心 idea**：用一对 $(\mathbf{z}, \mathbf{z}+h\mathbf{v})$ 的输出差分作为 JVP 的有限差近似，强制 student response 等于（stop-grad 的）teacher response，把"复制教师对扰动的反应"作为可插拔正则项加到任意 base distillation loss 上。

## 方法详解

### 整体框架
GAD 要解决的是"蒸馏后学生对初始噪声 $\mathbf{z}$ 不再敏感"这个隐性退化。它的做法是不碰原有蒸馏目标，只额外加一个正交的正则项：让学生在 $\mathbf{z}$ 局部对扰动的"差分响应"去对齐教师，从而把被平均掉的局部几何（方向导数、曲率）重新学回来。整个模块 model-agnostic，能直接挂到 output matching（LADD/ADD）、distribution matching（DMD/TDM）、score identity distillation（SiD）三大蒸馏范式上，最终学生 1-4 步即可推理。每个 iteration 在原本一次前向的基础上，对噪声 $\mathbf{z}$ 与扰动点 $\mathbf{z}'=\mathbf{z}+h\mathbf{v}$（$\mathbf{v}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ 为随机方向，$h$ 为扰动幅度）各跑一次教师和学生，再约束两侧的输出差一致。

### 关键设计

**1. JVP 对齐目标：把不可算的"匹配雅可比"压成可算的方向导数**

要让学生恢复对噪声的敏感性，最直接的目标是匹配教师的雅可比，即 $\mathcal{L}_{\text{Jacobian}}=\mathbb{E}_{\mathbf{z}}[\|\mathbf{J}_{\Phi_S}(\mathbf{z})-\mathbf{J}_{\Phi_T}(\mathbf{z})\|_F^2]$。但在 $d\approx 10^5$ 维的 latent 空间显式存雅可比会直接显存爆炸。作者借 Hutchinson trace estimator 绕开：对随机方向 $\mathbf{v}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ 匹配雅可比向量积（JVP），在期望意义下等价于匹配整个 Frobenius 范数，于是目标改写成 $\mathcal{L}_{\text{GAD}}=\mathbb{E}_{\mathbf{z},\mathbf{v}}\|\nabla_{\mathbf{z}}\Phi_S(\mathbf{z})\mathbf{v}-\nabla_{\mathbf{z}}\Phi_T(\mathbf{z})\mathbf{v}\|_2^2$。这相当于对雅可比做"压缩感知"——只算一次方向导数、内存 $O(d)$ 而非 $O(d^2)$，却能隐式覆盖整片 Jacobian 几何。

**2. 有限差分近似 + 配对前向：把 JVP 换成两点输出差，绕开 forward-mode autodiff**

即便有了 JVP 目标，直接算方向导数仍要 forward-mode autodiff，这与 SDXL/PixArt 这类 black-box teacher 不兼容、显存也大。作者用一阶有限差分 $\nabla_{\mathbf{z}}\Phi(\mathbf{z})\cdot\mathbf{v}\approx[\Phi(\mathbf{z}+h\mathbf{v})-\Phi(\mathbf{z})]/h$ 把方向导数替换成"两点输出差"，再把常数 $1/h^2$ 吸进权重 $\lambda$，得到实操目标 $\mathcal{L}_{\text{GAD}}=\mathbb{E}_{\mathbf{z},\mathbf{v}}\|(\Phi_S(\mathbf{z}')-\Phi_S(\mathbf{z}))-\text{sg}(\Phi_T(\mathbf{z}')-\Phi_T(\mathbf{z}))\|_2^2$。教师侧 stop-gradient 把它对扰动的反应锁成"参考切向量"，让学生单方向去对齐。代价只是每个 step 学生/教师各多一次前向（共 4 次），没有反向 JVP、没有二阶计算图，"加几行就能跑"，且 UNet/DiT/Flow-DiT 都通用。

**3. 三种范式下的统一实例化：换什么当 $\Phi$ 而已**

GAD 作为正则项接入不同蒸馏框架时，只需替换被对齐的映射 $\Phi$。对 output matching（LADD/ADD），$\Phi$ 取学生预测的 $\hat{\mathbf{x}}_0=f_\theta(\mathbf{x}_t,t,c)$，配对扰动取 $\mathbf{x}_t'=\mathbf{x}_t+h\mathbf{v}$，得到输出空间的 $\mathcal{L}_{\text{GAD}}^{\text{out}}$。对 distribution / score-based 范式（DMD/TDM/SiD），其 $\mathcal{L}_{\text{base}}$ 本就靠教师 score $\epsilon_{\text{real}}$ 与学生分布辅助 score $\epsilon_{\text{fake}}$ 的差给学生反传梯度；GAD 则在更高阶上匹配两个 score 场对方向扰动的差分 $\Delta\epsilon(\mathbf{x}_t,\mathbf{v})=\epsilon(\mathbf{x}_t+h\mathbf{v},t,c)-\epsilon(\mathbf{x}_t,t,c)$，梯度形式为 $\nabla_\theta\mathcal{L}_{\text{GAD}}^{\text{score}}=\mathbb{E}[\Delta\epsilon_{\text{fake}}-\Delta\epsilon_{\text{real}}]\partial\mathbf{x}_t/\partial\theta$。之所以加上去不和原 loss 打架，是因为 $\mathcal{L}_{\text{base}}$ 管的是一阶矩对齐（学生收敛到高密度区域），而 $\mathcal{L}_{\text{GAD}}$ 管的是局部曲率/散度对齐，两个目标天然正交。

### 损失函数 / 训练策略
总目标 $\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{base}}+\lambda\mathcal{L}_{\text{GAD}}$。每个 iteration 采样配对 $(\mathbf{z},\mathbf{z}+h\mathbf{v})$，各做一次教师 + 学生 forward，教师 forward 走 `torch.no_grad` + stop-grad 当锚点；$h$、$\lambda$ 取值详见附录 D。整套训练沿用 base 框架的 timestep schedule、CFG、优化器，几乎零迁移成本。

## 实验关键数据

### 主实验
在 3 类 backbone（SD2 UNet / PixArt-α DiT / SANA Flow-DiT）× 3 类蒸馏范式（LADD / TDM / SiD）上集成 GAD，对 11 个 distilled baseline 评测。

**Seed 可识别性（Tab. 2，SD2 体系）**：训分类器去预测图像来自哪个种子，越高代表敏感性越强。

| 模型 | Self-Identifiability ↑ | Teacher Alignment ↑ |
|------|------------------------|---------------------|
| SD2 教师（多步） | 93.70% | - |
| SD-Turbo | 77.80% | 63.20% |
| SwiftBrush | 52.90% | 45.80% |
| TCD | 87.30% | 84.50% |
| LADD | 87.60% | 83.70% |
| LADD + GAD（本文） | **92.40%** | **87.40%** |

**一般生成质量（Tab. 3）**：GAD 不仅没掉 CLIP / PickScore，反而普遍小幅提升，尤其 SiD+GAD 把 SANA 上的 CLIP 从 32.75 拉到 34.40。

**Layout 控制（Tab. 5，COCO 800 prompts + bbox）**：

| 模型 | AP ↑ | AP50 ↑ | CLIP ↑ |
|------|------|--------|--------|
| SD2 教师 | 6.6 | 21.3 | 0.3333 |
| SD-Turbo | 3.0 | 8.4 | 0.3237 |
| LADD | 5.0 | 17.4 | 0.3187 |
| LADD + GAD | **5.8** | **20.6** | 0.3184 |

GAD 恢复了教师 87% 的 layout 精度。

### 消融实验
**几何对齐直接度量（Tab. 1，PixArt-α）**：

| 配置 | JVP Cos ↑ | Jac Norm ↑ | Spec KL ↓ | JVP MSE ↓ |
|------|-----------|-----------|-----------|-----------|
| 教师 | 1.000 | 1.000 | 0.000 | 0.000 |
| TDM | 0.012 | 0.98 | 0.008 | 0.0003 |
| TDM + GAD | 0.014 | 0.99 | 0.006 | 0.0002 |

**轨迹累积偏差（Tab. 4，PixArt-α，200 unseen prompts）**：GAD 在四个时间段的累积偏差全面更低，t=0 处终点误差从 0.491 降到 0.427（−13%），说明 GAD 让学生在 unseen prompt 上更贴近教师的去噪轨迹。

### 关键发现
- pointwise MSE 已经被现有方法压得很低（图 2a），增量空间在"几何对齐"上；GAD 几乎不动 pointwise，却把 JVP 行为拉回到接近教师（图 2b），证明"低 MSE ≠ 正确动力学"。
- 恢复噪声敏感性同时提升一般质量：作者把这归因于 GAD 强制学生在 $\mathbf{z}$ 局部邻域一致，相当于 smoothness regularizer，提升了对 unseen prompt 的泛化（Tab. 4 终点误差 −13%）。
- 下游 zero-shot 迁移（Tab. 6，NoiseQuery 把教师挑出的最优 $\mathbf{z}^*$ 直接喂学生）显示，基线蒸馏模型几乎用不了教师挑的噪声，而 GAD 把 noise-to-image 的几何对齐回去之后能直接享受教师端的 test-time enhancement。
- 多样性 / 保真度 trade-off（图 5）：基线在 Vendi vs CLIP 两轴上被压在劣势区，GAD 把三个 backbone 的点都拉到接近教师的右上角。

## 亮点与洞察
- "蒸馏掉了噪声敏感性"这个 framing 本身很值——把一个长期被 FID/CLIP 掩盖的隐性退化显式化，并设计了 seed 分类、JVP cos、Spec KL、轨迹偏差等多角度的直接度量，让这个问题第一次"可看见、可衡量"。
- 用有限差分 + 配对前向去近似 JVP，把"匹配雅可比"这个理论上昂贵的目标降到几乎零额外工程成本（4 次 forward 而已，无二阶导），可以迁移到任何蒸馏框架——这是真正的 plug-and-play。
- Stop-gradient 把教师响应作为锚点，加上 $\lambda$ 解耦"全局保真" vs "局部曲率"两个目标，回答了一个常被忽略的问题：知识蒸馏里"教什么"比"教多准"更重要。
- 这个思路可以迁移：任何"学生用 MSE/KL 蒸馏成轨迹/分布"的场景（语音合成蒸馏、policy distillation、视频生成蒸馏），都可以加一个 JVP-style 配对响应对齐项来保留教师的局部输入敏感性。

## 局限与展望
- 训练成本：每 step 多两次教师 forward + 两次学生 forward，相对原 base loss 大约 ×1.5～×2 的训练耗时；论文未给出 wall-clock 对比。
- 扰动 $h$ 是关键超参：太小被数值噪声淹没，太大破坏一阶近似；论文只在附录里给配置，缺乏对 $h$ 的系统 sensitivity analysis（实战中跨模型可能要重新调）。
- Hutchinson 估计需要 $\mathbf{v}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ 全空间随机方向，对 latent 已经 64×64×4 的 DiT 来说，单方向无偏估计的方差仍然不小，未讨论多 $\mathbf{v}$ 采样的 variance reduction。
- 实验局限：layout 控制只测了 YOLOv4 AP，没在 ControlNet / Grounded-SAM 等更强 detector 上验证；NoiseQuery 也只在 DrawBench 上做。
- 没有评估对 negative prompt adherence、trajectory invertibility 这些"姐妹问题"的影响——这些都和噪声敏感性高度相关，是天然的下一步评测。

## 相关工作与启发
- **vs 标准蒸馏（ADD / LADD / DMD / TDM / SiD）**: 它们都是 pointwise alignment（MSE / 反向 KL / Fisher divergence），GAD 不替换它们而是作为正交正则项叠加，文章实验也是"base + GAD"的形式，体现"互补而非取代"。
- **vs 关系型知识蒸馏（Park 2019 RKD / Tung 2019 SP）**: 经典 KD 也强调匹配样本间关系，但用的是 pairwise 距离 / 角度；GAD 把这个思路推到生成模型的连续输入空间，用 JVP 形式刻画"无限小邻域"的关系，更贴合生成式映射。
- **vs 多样性增强方法（Diverse Distillation / Gandikota & Bau 2025）**: 那些方法往往直接 regularize 输出分布的熵或显式扩散种子；GAD 不显式优化多样性，而是从更上游的"对噪声的局部响应"切入，恢复了多样性作为副产物——更优雅、且能同时解锁布局 / 检索 / 多样性多个下游。
- **vs NoiseQuery / 噪声检索类工作（Wang et al. 2025）**: NoiseQuery 假设教师 $\mathbf{z}\to\mathbf{x}$ 的几何在学生上保留，但实际不成立；GAD 第一次正面解决了这个 assumption mismatch，让 NoiseQuery 这类技术能 zero-shot 迁移到 1-4 步学生上。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把"噪声敏感性塌缩"这个问题首次系统化，并给出一个干净的几何对齐 framework；JVP + 有限差分本身不是新数学，但用在 T2I 蒸馏里是新组合。
- 实验充分度: ⭐⭐⭐⭐⭐ 3 backbone × 3 蒸馏范式 × 11 baseline × 3 下游任务（layout / diversity / NoiseQuery），还有 JVP cos / Spec KL / 轨迹偏差等直接几何度量，覆盖面非常全。
- 写作质量: ⭐⭐⭐⭐ 动机 → 诊断 → 公式 → 实例化 → 实验的叙事线很清晰；图 1/2/3 的"smooth path vs preserved curvature"概念图直观；少量公式排版略密。
- 价值: ⭐⭐⭐⭐⭐ Plug-and-play、几乎零迁移成本、几乎不掉 base 质量、显著恢复下游可控性——这是少见的"加上就能用、对生态友好"的蒸馏侧贡献，值得直接合并进现有 T2I 蒸馏 pipeline。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models.md)
- [\[ICML 2026\] Information-Geometric Adaptive Sampling for Graph Diffusion](information-geometric_adaptive_sampling_for_graph_diffusion.md)
- [\[ICML 2026\] Escaping Mode Collapse in LLM Generation via Geometric Regulation](escaping_mode_collapse_in_llm_generation_via_geometric_regulation.md)
- [\[ICLR 2026\] Diverse Text-to-Image Generation via Contrastive Noise Optimization](../../ICLR2026/image_generation/diverse_text-to-image_generation_via_contrastive_noise_optimization.md)
- [\[NeurIPS 2025\] Ψ-Sampler: Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment in Score Models](../../NeurIPS2025/image_generation/psi-sampler_initial_particle_sampling_for_smc-based_inference-time_reward_alignm.md)

</div>

<!-- RELATED:END -->
