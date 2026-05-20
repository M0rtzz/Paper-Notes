---
title: >-
  [论文解读] Scaling Vision Transformers for Functional MRI with Flat Maps
description: >-
  [ICML 2026][医学图像][fMRI 基础模型] 把 3D fMRI 体积按"皮层展平图"投影成 2D 视频后直接喂给标准 spacetime MAE-ViT，得到一个在 2.1K 小时 HCP 数据上训练的 CortexMAE：在认知状态解码上大幅超 SOTA…
tags:
  - "ICML 2026"
  - "医学图像"
  - "fMRI 基础模型"
  - "Cortical Flat Map"
  - "MAE"
  - "Brainmarks 评测"
  - "Scaling Law"
---

# Scaling Vision Transformers for Functional MRI with Flat Maps

**会议**: ICML 2026  
**arXiv**: [2510.13768](https://arxiv.org/abs/2510.13768)  
**代码**: https://github.com/MedARC-AI/CortexMAE & https://github.com/MedARC-AI/Brainmarks (有)  
**领域**: 医学图像 / 自监督学习 / 神经影像基础模型  
**关键词**: fMRI 基础模型、Cortical Flat Map、MAE、Brainmarks 评测、Scaling Law

## 一句话总结
把 3D fMRI 体积按"皮层展平图"投影成 2D 视频后直接喂给标准 spacetime MAE-ViT，得到一个在 2.1K 小时 HCP 数据上训练的 CortexMAE：在认知状态解码上大幅超 SOTA，验证 flat map 是体素 (volume) 和脑区平均 (parcellation) 之间的"goldilocks zone"；同时发布首个开源 fMRI 基础模型基准 Brainmarks，给出 fMRI 模型的第一份系统 scaling law 与一个"个体特质预测仍打不过简单功能连接 baseline"的诚实 null result。

## 研究背景与动机

**领域现状**：神经科学社区想用 fMRI + 大模型解码大脑活动 (诊断、行为预测、视觉重建)。已有一波 fMRI 自监督基础模型 (BrainLM, Brain-JEPA, NeuroSTORM, SwiFT 等)，多数用 **parcellation** 表示（把 3D 脑体积按 100-400 个脑区平均，得到 1D 时间序列向量）；少数用 **volume** 表示（直接处理 4D 时空 MRI 数据）。

**现有痛点**：(1) Parcellation 计算便宜但**信息损失严重**——一整个 cm 量级的脑区被压成单标量，丢掉 99% 维度；(2) Volume 信息全保留但 sequence length 巨大（一个 fMRI volume 切 patch 后 ~2000+ tokens），训练计算/IO 开销爆炸；(3) 整个 fMRI 基础模型领域**几乎没有可复现 benchmark**——各家用自家 dataset、自家预处理、自家评测设置，谁强谁弱不可比；(4) 之前的 trait prediction 论文经常报告"我们 beat baseline X%"，但用的 baseline 太弱，没和"简单功能连接 (FC) + 逻辑回归"这种 30 年前的方法做严肃对比。

**核心矛盾**：fMRI 数据本质是 4D 时空体积，标准 ViT 假定 2D 输入。要么花大代价直接学 4D（信息全但贵），要么用强归纳偏置 (parcellation) 损失信息——存在一个 **"既保留全皮层信号又给 ViT 友好的 2D 输入"** 的中间表示吗？

**本文目标**：(i) 找到 fMRI 的 "goldilocks" 输入表示；(ii) 用标准 ViT + MAE 训练一组对比清晰的基础模型；(iii) 建立开源、可复现的 fMRI 基础模型评测 (Brainmarks)；(iv) 第一次系统做 fMRI 自监督的 data/model scaling law。

**切入角度**：神经科学早就有 **cortical flat map** ——把大脑皮层这个 2D 流形（其实皮层就是一层 2-4mm 厚的褶皱片）展平到平面网格上。这样既保留了全皮层 BOLD 信号（不像 parcellation 那样平均掉细节），又得到了一张 224×560 的 2D "图像"，可以直接被 spacetime ViT 当 video 处理。

**核心 idea**：用 cortical flat map 把 3D fMRI 投影成 2D 视频，套用现成 MAE-st 训练，**不改 ViT 架构，只换 patch embedding**——一个简单但被错过的好选择，最终带来 SOTA + 第一份 fMRI scaling law + 第一个开源 benchmark。

## 方法详解

### 整体框架
模型 = MAE-st (Feichtenhofer 等 2022) + 三种 patch embedding 互换的输入头。pipeline：(1) HCP-YA 数据预处理 (FreeSurfer / fMRIPrep 表面映射) → 拿到皮层表面网格上的时间序列；(2) 用 pycortex 把表面投到平面网格 → 得到 16 帧 × 224 × 560 的 fMRI flat map 视频；(3) 切 $p_t \times 16 \times 16$ 时空 patch（默认 $p_t = 4$），mask 比例 0.9（tube masking 不跨时间插值）；(4) ViT-B encoder 看到 sparse 观测 patch + [MASK] token，decoder 重建被 mask 的 patch，loss 用 MSE（只在非背景像素上）；(5) 预训练完成后丢掉 decoder，encoder 输出作为特征，配合 linear probe / attentive probe 做下游 trait / state 预测。同步训练 parcellation MAE (用 Schaefer-400 脑区) 和 volume MAE (用 4D patch) 做严格对照。

### 关键设计

1. **Cortical Flat Map Patch Embedding (核心创新)**：

    - 功能：把 3D fMRI 体积转成 2D 图像视频，让标准 spacetime ViT 直接处理，又不丢失皮层全信号。
    - 核心思路：先用神经科学常用的 surface-based 流程 (FreeSurfer + fMRIPrep) 把每帧 fMRI 信号从 3D voxel 映射到皮层表面的 mesh vertex；再用 pycortex 的 flat map 把这张曲面 mesh 展平到 2D 平面 grid（左右半球分别展开拼成一张图）；最后 resample 到固定 224×560 网格。每个时间步就是一帧 2D 图，16 帧拼起来就是 ViT 的 spacetime 输入。patch 大小 $p_t \times 16 \times 16$，背景 (脑外区域) 全 0 patch 直接排除不参与计算，MSE loss 也只在非背景像素上算。
    - 设计动机：parcellation 把信号压成 ~400 维向量（dim 损失 ~100×），而 volume 直接处理需要 ~132K voxel 的稀疏 patch（计算量大且大部分是无效背景）。flat map 在二者之间——保留全皮层 ~77K 维信号但用 2D ViT 高效处理；序列长度 364（vs volume 的 465 和 parcellation 的 400）几乎持平但训练带宽与数据吞吐都更好。论文 Figure 1 把这个 trade-off 形象化为"光谱"。

2. **三种表示的 head-to-head 对比设计**：

    - 功能：用同一架构、同一预训练数据、同一评测协议，把 parcellation/flat/volume 三种表示放在同一起跑线。
    - 核心思路：所有模型都用 ViT-B encoder、同样的 16 帧输入、同样的 0.9 mask ratio；只是 patch embedding 不同——parcel 用 $p_t \times 1$ 仅时间维 patch、flat 用 $p_t \times 16 \times 16$、volume 用 $p_t \times 8 \times 8 \times 8$ 4D patch。每个变体训 8 次取均值，单独跑下游 8 个数据集 (4 个临床诊断 + 性别 + 年龄 + HCP-YA Task21 + NSD COCO24)。
    - 设计动机：之前 fMRI 基础模型论文从不公平对比表示——通常自己只用一种然后宣称 SOTA。本文是第一份 multi-representation fMRI MAE family，结论可信度高得多。

3. **Brainmarks 开源评测套件**：

    - 功能：给 fMRI 基础模型领域提供一个所有方法都能跑、所有数据集都标准化处理的 benchmark。
    - 核心思路：纳入 6 个现有 fMRI 基础模型 (SwiFT, BrainLM, Brain-JEPA, BrainHarmonix-F, NeuroSTORM, Brain-Semantoks) 和 CortexMAE family；包含 7 个公开数据集——4 个临床诊断 (ABIDE/ADHD200/ADNI/PPMI) + HCP-A 年龄/性别 + HCP-YA Task21 + NSD COCO24；对小样本 trait prediction 用 linear probe + 100 次 train-test 随机分割，对大样本 state prediction 用 attentive probe 单 fixed split + 49 个学习率 grid 调优。所有方法用同一 probe 协议，避免每家自己 fine-tune 偷跑。
    - 设计动机：fMRI 模型评测的复现性灾难是社区公认问题；统一 protocol 才能让"谁真的更好"有意义。还专门设计了 NSD COCO24 这种"短 trial 重叠 + 不同 subject 测试集 + 高难度视觉解码"任务来真正区分模型质量。

### 损失函数 / 训练策略
预训练：MAE MSE loss on masked patches; data normalization 是关键——每个 voxel/ROI 时间序列 z-score (coordinate norm) + 每帧空间 z-score (frame norm)，去除 BOLD 信号只有 1-2% 波动的静态噪声；temporal patch $p_t = 4$；training schedule 默认 625K steps、batch 32 (= 512 帧)；repeated sampling 减少 IO 瓶颈。下游：trait prediction 用 average-pooled embedding + logistic regression 5-fold CV；state prediction 用 attentive probe + early stopping。

## 实验关键数据

### 主实验
三种表示在 8 个下游任务上的探针准确率（8 次预训练随机种子均值）：

| Dataset | parcel | flat | volume | FC 基线 |
|---|---|---|---|---|
| ABIDE (ASD 诊断) | 62.0 | 61.4 | 60.4 | 59.8 |
| ADHD200 | 56.8 | 59.2 | 58.8 | 57.0 |
| ADNI (AD) | 61.6 | 62.4 | 64.3 | 58.6 |
| PPMI (PD) | 61.4 | 58.8 | 59.1 | 58.0 |
| HCP-A Age | 44.2 | 47.5 | **53.4** | 45.6 |
| HCP-A Sex | 71.2 | **87.4** | 86.3 | 81.9 |
| HCP-YA Task21 (状态) | 97.5 | **98.9** | 96.2 | 82.4 |
| NSD COCO24 (视觉解码) | 27.5 | **31.0** | 27.7 | 7.4 |

总结：(1) **flat map 在动态状态解码上全面胜出**（Task21、COCO24、性别）；(2) volume 在年龄预测上有优势（可能跟 dense 体素能捕捉皮层厚度等结构线索有关）；(3) parcel 最高效但状态解码弱；(4) 临床诊断 4 个数据集上所有方法几乎打平，且勉强超过 FC baseline——揭示样本太少时 fMRI 基础模型还看不出优势。

跨模型 controlled benchmark（Figure 8）：trait prediction 上**没有任何模型显著超过简单 FC 基线**（包括 BrainLM、Brain-JEPA、NeuroSTORM 等 SOTA）；state decoding 上 **CortexMAE flat map 全面领先**，比 NeuroSTORM 等 volume 模型在 NSD COCO24 上高 3-5 个百分点。

### 消融实验

| 配置 | 现象 |
|---|---|
| Full flat map MAE | baseline |
| 不做 frame normalization | 全局信号漂移污染，下游精度掉 | 
| 不做 coordinate normalization | 静态体素差异主导特征，状态解码崩 |
| tube masking → random masking | 时间内插泄露信息，重建任务变 trivial |
| mask ratio 0.5 → 0.9 | 高 mask ratio 强迫学结构性表示，下游更强 |
| 增大 encoder depth | depth ~9 (37M 参数) 后 saturate |
| 增大 pretrain data | 在 HCP 内部数据严格遵循 power law (指数 -0.01)，OOD NSD 上 saturate |

### 关键发现
- **fMRI 严格遵循 data scaling law，但指数比语言模型弱十倍** (-0.01 vs Kaplan 2020 的 -0.1)，意味着 fMRI 边际收益小，scaling 不会自动救场。
- Model scaling 在 depth 9 (37M 参数) 就饱和——HCP-YA 这种 2K 小时数据集就只能撑住这点容量。
- 模型自发学到大脑的默认模式网络 (DMN)：position embedding 的第一主成分跟 Margulies 2016 的 FC principal gradient 几乎一致，证明 MAE 真的学到神经生物学有意义的结构。
- **诚实 null result**：所有 fMRI 基础模型在 individual trait prediction 上都打不过简单 FC + linear——这对整个领域是个警钟。
- state decoding 上基础模型有 robust 优势，CortexMAE flat 是其中最强。

## 亮点与洞察
- **"换 patch embedding 就完事"是非常优雅的工程选择**：不重新设计架构、不重写 attention，只是把输入从 3D 体积投到 2D 流形——任何 ViT 论文未来想入局 fMRI 都可以这样做。
- **goldilocks zone 的概念可迁移**：在表示学习里"全保留 vs 大压缩"是经典 trade-off，cortical flat map 是一个利用领域几何 (cortex 本质是 2D) 找到完美中间点的例子，类似的可以套到 EEG (1D 时间 + 电极几何)、心电、显微图像等。
- **诚实 null result + 开源 benchmark**：神经影像社区长期有"小数据集 + 各家自评"的复现性灾难，作者直接发布 Brainmarks 并公开承认 trait prediction 打不过 FC baseline，这是社区急需的诚实声音。
- **第一份 fMRI scaling law**：让大家看清"fMRI 不是 NLP"——指数小十倍意味着堆数据收益有限，真正瓶颈可能是数据多样性而非规模。
- **DMN 自然涌现**：自监督表征对应到神经生物学已知结构，是 fMRI MAE 的优秀诠释性证据。

## 局限与展望
- HCP-YA 全是 22-35 岁年轻人 + 健康人群，预训练分布**狭窄**，OOD 泛化弱（论文图 7 已经展示了在 NSD 上 scaling 失效）。
- 临床诊断结果（ABIDE、ADHD200 等）全 60% 上下徘徊，**几乎没用**——基础模型在小样本临床数据上无法 transfer，是社区共性难题，但本文没提供解决方案。
- depth 9 就 saturate，说明现有数据规模不够——但 fMRI 数据采集极贵 (一小时成本几百美元)，要 10× 数据需要全社区合作。
- flat map 投影本身**丢掉了皮层下结构** (subcortical regions like thalamus, basal ganglia)，这些区域对很多临床任务很重要；volume 模型在这块有结构性优势。
- 没探讨 multi-modal fMRI (task + rest + diffusion) 联合预训练，未来工作空间大。
- 评测只在英语母语、北美人群数据上做，存在 demographic bias。

## 相关工作与启发
- **vs BrainLM (Caro et al. 2024) / Brain-JEPA (Dong et al. 2024)**：这些是 parcellation-based fMRI 基础模型，dim 损失大；CortexMAE flat 全皮层信号保留，在状态解码上显著更强。
- **vs SwiFT (Kim et al. 2023) / NeuroSTORM (Wang et al. 2025a)**：volume-based 模型，计算贵但年龄预测有优势；本文 volume MAE 也复现了这一点，说明 dense 表示有 niche，但状态解码上 flat 仍占优。
- **vs functional connectivity baselines**：从 Finn et al. 2015 起，FC + linear 一直是 trait prediction 标准 baseline；本文证明深度 fMRI 模型至今没真正超越它，是对整个深度 fMRI 领域的一次"打脸"。
- **vs vision MAE (He et al. 2022)**：直接搬运 MAE-st，主要贡献在于"找到合适的 2D 投影把 fMRI 装进现成框架"——示范了"领域几何 + 通用架构"是个高效组合。

## 评分
- 新颖性: ⭐⭐⭐⭐ 严格说 cortical flat map 是神经科学几十年的工具，作者首次系统用它做 ViT-friendly representation；技术上小但战略上聪明。
- 实验充分度: ⭐⭐⭐⭐⭐ 三表示严格对比 + 6 个外部模型 + Brainmarks 7 数据集 + scaling law + 解释性分析，几乎是 fMRI MAE 的"实验白皮书"。
- 写作质量: ⭐⭐⭐⭐⭐ 思路清晰、动机和结论都直白，关键 figure (光谱图、DMN 涌现) 极其有说服力。
- 价值: ⭐⭐⭐⭐⭐ Brainmarks + null result + flat map 三件套对 fMRI 基础模型社区是教科书级贡献，会成为后续工作必引论文。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MuViT: Multi-Resolution Vision Transformers for Learning Across Scales in Microscopy](../../CVPR2026/medical_imaging/muvit_multi-resolution_vision_transformers_for_learning_across_scales_in_microsc.md)
- [\[ICML 2026\] Layer-Specific Fine-Tuning for Improved Negation Handling in Medical Vision-Language Models](layer-specific_fine-tuning_for_improved_negation_handling_in_medical_vision-lang.md)
- [\[ICML 2025\] Supercharging Graph Transformers with Advective Diffusion](../../ICML2025/medical_imaging/supercharging_graph_transformers_with_advective_diffusion.md)
- [\[AAAI 2026\] WDT-MD: Wavelet Diffusion Transformers for Microaneurysm Detection in Fundus Images](../../AAAI2026/medical_imaging/wdt-md_wavelet_diffusion_transformers_for_microaneurysm_detection_in_fundus_imag.md)
- [\[ICLR 2026\] Scaling with Collapse: Efficient and Predictable Training of LLM Families](../../ICLR2026/medical_imaging/scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)

</div>

<!-- RELATED:END -->
