---
title: >-
  [论文解读] POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction
description: >-
  [CVPR 2026][物理/科学计算][射电干涉成像] POLISH++在POLISH框架基础上引入分块训练+拼接策略和arcsinh非线性变换，解决了射电干涉成像中宽视场（万级像素）和高动态范围（$10^4$-$10^6$）两大实际部署难题，在T-RECS仿真数据上大幅超越CLEAN方法的源探测精度，且能超分辨恢复PSF尺度附近的强引力透镜系统，有望将DSA巡天的透镜发现数量提升约10倍。
tags:
  - "CVPR 2026"
  - "物理/科学计算"
  - "射电干涉成像"
  - "深度学习去卷积"
  - "超分辨率"
  - "高动态范围"
  - "强引力透镜"
---

# POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction

**会议**: CVPR 2026  
**arXiv**: [2603.09162](https://arxiv.org/abs/2603.09162)  
**代码**: 无（基于 [POLISH](https://github.com/liamconnor/polish-pub) 扩展）  
**领域**: 其他  
**关键词**: 射电干涉成像, 深度学习去卷积, 超分辨率, 高动态范围, 强引力透镜  

## 一句话总结
POLISH++在POLISH框架基础上引入分块训练+拼接策略和arcsinh非线性变换，解决了射电干涉成像中宽视场（万级像素）和高动态范围（$10^4$-$10^6$）两大实际部署难题，在T-RECS仿真数据上大幅超越CLEAN方法的源探测精度，且能超分辨恢复PSF尺度附近的强引力透镜系统，有望将DSA巡天的透镜发现数量提升约10倍。

## 背景与动机
射电干涉成像通过天线阵列合成大口径来实现高角分辨率，本质上是一个图像去卷积问题（从dirty image恢复true sky）。传统方法CLEAN假设点源模型并迭代减去PSF，分辨率受限于PSF且无法处理复杂形态的源。深度学习方法（如原始POLISH）展现了高效推理和超分辨率的潜力，但现有DL方法存在三个未解决的实际部署问题：(1) 测试图像尺寸小（<1000像素），而DSA等下一代望远镜需处理超过$10000 \times 10000$像素的图像；(2) 仅在低动态范围（<$10^3$）上验证，而实际动态范围可达$10^6$；(3) 训练和测试条件匹配的假设在实际中不成立（大气、校准误差等导致PSF变化）。

## 核心问题
如何让深度学习射电干涉成像方法真正可部署到DSA等下一代巡天望远镜？具体需解决：(1) GPU内存无法承载超大图像的训练和推理；(2) 极端动态范围导致网络难以同时学习亮源和暗源的重建；(3) PSF训练-测试不匹配时的鲁棒性。这些问题共同阻碍了DL方法从实验室走向实际天文观测。

## 方法详解

### 整体框架
POLISH++ 要回答一个很实际的问题：怎么让深度学习射电干涉成像方法真正搬得上 DSA 这种下一代巡天望远镜。它的骨架沿用 POLISH——一个基于 WDSR 的端到端 CNN $G_\theta$，输入低分辨率 dirty image、输出 2 倍超分辨的 clean sky。真正让它可部署的是围绕这个骨架的两项改造加一条部署适配，三者也对应下面三个关键设计：① 训练和推理都落在**块（patch）级别**——把上万像素的全视场切成小块逐块处理、推理后再拼回完整视场，绕开 GPU 显存瓶颈；② 在 CNN 前后套一对 **arcsinh 变换**——前向把跨 $10^4$–$10^6$ 量级的动态范围压到同一量级、CNN 在变换空间里重建、再用逆变换还原真实强度；③ 面对部署时 PSF 与训练条件失配，只在理想 PSF 上训练的模型仍能稳定重建，且用极少几个 epoch 的**微调（fine-tuning）**就能适配新指向的 PSF。模型沿 POLISH → POLISH+（加分块）→ POLISH++（再加 arcsinh）逐步演进，论文的 Table 2 正是按"分块"和"arcsinh"这两个开关区分三者。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["全视场 dirty image（12960×12960）"] --> B["分块<br/>切成 J 个 324×324 非重叠块"]
    B --> C["Arcsinh 变换<br/>压缩 10⁴–10⁶ 动态范围"]
    C --> D["CNN Gθ（WDSR 改）<br/>变换空间内 2× 超分辨重建"]
    D --> E["逆 Arcsinh 变换<br/>还原真实强度"]
    E --> F["拼接<br/>各块复原回全视场"]
    F --> G["高分辨 clean sky"]
    G -.->|部署遇 PSF 失配| H["微调适配<br/>11 epoch 适配新 PSF"]
    H -.->|更新权重| D
```

### 关键设计

**1. 分块训练与拼接：让网络学会处理"邻块亮源溢进来"的污染**

DSA 要处理上万像素的图，GPU 装不下整图训练。POLISH++ 把 $12960 \times 12960$ 的全视场切成 $J$ 个 $324 \times 324$ 的非重叠块，构成训练对 $\{(I_{\text{dirty}}^{[j]}, I_{\text{true}}^{[j]})\}$。这里的关键洞察是：从全视场 dirty image 里抠出的 patch 并不等于直接对该 patch 的 ground truth 做正向成像——块边界外的亮源会通过 PSF 旁瓣"污染"进这个 patch，制造跨块伪影。POLISH++ 不去显式物理建模，而是让网络隐式学会处理这种"不属于自己"的信号。18 张全视场图切出 28800 个训练 patch，含约 600 万个可检测星系样本。

**2. Arcsinh 动态范围变换：一个非线性变换同时照顾亮源和暗源**

实际动态范围可达 $10^4$–$10^6$，网络很难同时学好极亮和极暗的源。POLISH++ 定义 $\text{AsinhStretch}(x; a) = \frac{\text{arcsinh}(x/a)}{\text{arcsinh}(1/a)}$，用它的类对数形态把跨多个数量级的像素值压到同一量级；相比摄影里的 gamma 编码，arcsinh 能同时处理正负值（dirty image 含负值），更适配射电图像。训练在变换空间进行，推理后用逆变换 $a \cdot \sinh(x \cdot \text{arcsinh}(1/a))$ 还原原始强度，参数取 $a_{\text{dirty}} = a_{\text{true}} = 0.1$。消融显示它主要靠多恢复暗弱源来改善 precision-recall 平衡。

**3. PSF 不匹配下的鲁棒性与快速适配：训练只见理想 PSF，部署却能扛失配**

真实观测里大气和校准误差会让 PSF 偏离训练条件。POLISH++ 仅在理想 PSF 上训练，但面对随机扭曲 PSF（扰动强度 $\gamma \in [0, 30]$）仍能给出视觉稳定的重建——PSNR 虽随扰动下降，源检测质量却维持得不错。更实用的是，针对新 PSF 分布做 fine-tuning 只需 11 个 epoch 即达最优，而从头训练要 57 个，加速 5 倍以上，这对 DSA 不同指向方向的逐指向部署很关键。

### 损失函数 / 训练策略
- ℓ1损失在arcsinh变换空间中计算：$\theta^* = \arg\min_\theta \frac{1}{NJ}\sum_{i,j}\|G_\theta(\text{AsinhStretch}(I_{\text{dirty}}^{[j]}; a_d)) - \text{AsinhStretch}(I_{\text{true}}^{[j]}; a_t)\|_1$
- Adam优化器，学习率$10^{-4}$，batch size 12
- 18张全视场图训练、5张测试，含28800个patch
- 训练数据由T-RECS仿真生成，含AGN（点源）和SFRG（椭圆Sérsic轮廓），噪声标准差1 μJy

## 实验关键数据

| 方法 | Precision | Recall | F1 Score | Major Axis RMSE (″) | Minor Axis RMSE (″) |
|------|-----------|--------|----------|---------------------|---------------------|
| CLEAN | 0.3612 | 0.2220 | 0.2750 | 1.0046 | 0.7862 |
| POLISH | 0.5560 | 0.4612 | 0.5042 | 0.9642 | 0.3219 |
| POLISH+ | 0.8744 | 0.5751 | 0.6938 | 0.4335 | 0.1889 |
| POLISH++ | 0.8433 | 0.6142 | 0.7107 | 0.4654 | 0.2056 |

**强引力透镜发现**: 在FPR=$10^{-3}$条件下，POLISH/POLISH++的CNN透镜查找器能恢复Einstein半径接近PSF尺度的透镜系统（CLEAN仅能恢复3倍PSF以上的），预计将DSA的透镜发现数量提升约一个数量级。POLISH在透镜上的TPR接近ground truth上限。

**PSF鲁棒性**: 即使在极端PSF扰动（$\gamma=30$）下，重建视觉质量仍保持稳定。Fine-tuning到新PSF仅需11 epoch vs 从头57 epoch（5倍加速）。

### 消融实验要点
- **分块训练（POLISH→POLISH+）**: Precision从0.56提升到0.87，F1从0.50到0.69，说明在patch级别学习（包含跨块伪影）远优于在小图像上直接训练
- **Arcsinh变换（POLISH+→POLISH++）**: Recall从0.58提升到0.61（+4%），F1从0.69到0.71，主要贡献在于恢复更多暗弱源，改善precision-recall平衡
- **CLEAN的flux估计更优**: 这是DL方法的已知局限——非线性重建缺乏显式flux校准机制，CLEAN在基于模型的框架中保持了绝对flux精度

## 亮点
- **"cross-patch contamination"的洞察**: 首次清晰阐明在patch级别训练射电成像网络时，dirty image的patch包含来自邻近patch亮源的PSF旁瓣伪影，这不是简单的小图像训练。让网络学会处理这种"不属于自己"的信号是一个巧妙的隐式建模
- **arcsinh变换的天文适配性**: 比log变换好在能处理负值，比gamma编码好在有明确的逆变换和物理动机，一个简单的非线性变换解决了$10^6$量级的动态范围问题
- **任务相关评估**: 不仅评估PSNR/SSIM等图像质量指标，还评估源探测精度（precision/recall/F1）和形状参数估计误差，更贴近天文实际需求
- **Fine-tuning快速适配**: 对不同PSF条件的适应不需要从头训练，5倍加速的fine-tuning使得逐指向部署成为可能

## 局限与展望
- **Flux估计仍不如CLEAN**: DL方法在绝对flux精度上落后于CLEAN，需要专门的flux校准步骤或后处理策略
- **仅在image-plane评估**: 使用CASA的deconvolve任务（minor cycle），未在visibility空间做端到端验证
- **训练数据量有限**: 仅18张全视场图用于训练，虽然patch-wise扩展到28800个样本，但sky model的多样性可能不足
- **未考虑频率方向信息**: 射电干涉成像是多频段的，当前方法仅处理单频段图像
- **实际系统效应未充分验证**: 虽测试了PSF失配，但真实的电离层扰动、增益变化、RFI污染等效应的组合影响还未验证

## 与相关工作的对比
- **vs CLEAN**: POLISH++在源探测F1上超出159%（0.27→0.71），形状估计RMSE降低54%，但flux估计不如CLEAN。本质差异在于CLEAN受限于PSF分辨率，而POLISH利用学到的先验实现超分辨率
- **vs R2D2（Aghabiglou et al., 2024）**: R2D2是展开迭代优化网络，最大图像512像素，动态范围$5 \times 10^5$。POLISH++扩展到12960像素、$10^6$动态范围，且为单次前向推理，适合高吞吐量场景
- **vs GU-Net/RI-GAN**: 这些方法在合成数据上测试，图像尺寸<360/512，动态范围<600，且未考虑训练-测试PSF不匹配，与实际部署差距较大

## 启发与关联
- arcsinh变换的思路可迁移到其他高动态范围场景的图像恢复（如HDR去噪、医学CT重建）
- 分块训练+拼接的策略在任何超大分辨率图像处理中都适用（遥感、全景图等）
- 任务相关评估（不仅看PSNR，更看下游任务指标）的范式值得在CV领域推广
- 论文将"天文图像恢复"问题成功带入CV社区，展示了物理约束+深度学习的跨学科潜力

## 评分
- 新颖性: ⭐⭐⭐ 分块训练和arcsinh变换本身技术上不算新，但在射电天文成像中的应用和工程规模化有意义
- 实验充分度: ⭐⭐⭐⭐⭐ 从源探测、形状估计、强引力透镜发现、PSF鲁棒性四个角度全面评估，实验设计与天文实际需求高度对齐
- 写作质量: ⭐⭐⭐⭐ 背景介绍清晰完整，问题动机充分，但部分天文术语对CV读者不够友好
- 价值: ⭐⭐⭐⭐ 对DSA等下一代射电望远镜有直接实用价值，强引力透镜发现数量的10倍提升是实质性贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Semi-Supervised High Dynamic Range Image Reconstructing via Bi-Level Uncertain Area Masking](../../AAAI2026/others/semi-supervised_high_dynamic_range_image_reconstructing_via_bi-level_uncertain_a.md)
- [\[CVPR 2026\] Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods](rooftop_wind_field_reconstruction_using_sparse_sen.md)
- [\[ECCV 2024\] Image Demoiréing in RAW and sRGB Domains](../../ECCV2024/others/image_demoiréing_in_raw_and_srgb_domains.md)
- [\[CVPR 2026\] SimRecon: SimReady Compositional Scene Reconstruction from Real Videos](simrecon_simready_compositional_scene_reconstruction_from_real_videos.md)
- [\[CVPR 2026\] Crowdsourcing of Real-world Image Annotation via Visual Properties](crowdsourcing_of_real_world_image_annotation_via_visual_properties.md)

</div>

<!-- RELATED:END -->
