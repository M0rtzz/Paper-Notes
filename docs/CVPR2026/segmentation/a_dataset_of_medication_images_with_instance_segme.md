---
title: >-
  [论文解读] A dataset of medication images with instance segmentation masks for preventing adverse drug events
description: >-
  [CVPR 2026][语义分割][药物识别] 构建了MEDISEG药物图像实例分割数据集（8262张图像，32类药片，含遮挡/重叠的真实场景），YOLOv8/v9验证3类达99.5% mAP@0.5、32类达80.1%，FsDet few-shot证明MEDISEG预训练在遮挡场景比CURE显著提升（1-shot 0.406 vs 0.131）。
tags:
  - "CVPR 2026"
  - "语义分割"
  - "药物识别"
  - "实例分割"
  - "few-shot检测"
  - "数据集"
  - "用药安全"
---

# A dataset of medication images with instance segmentation masks for preventing adverse drug events

**会议**: CVPR 2026  
**arXiv**: [2603.10825](https://arxiv.org/abs/2603.10825)  
**代码**: [github.com/williamcwi/MEDISEG](https://github.com/williamcwi/MEDISEG)  
**领域**: 实例分割 / 医学图像 / 数据集  
**关键词**: 药物识别, 实例分割, few-shot检测, 数据集, 用药安全

## 一句话总结

构建了MEDISEG药物图像实例分割数据集（8262张图像，32类药片，含遮挡/重叠的真实场景），YOLOv8/v9验证3类达99.5% mAP@0.5、32类达80.1%，FsDet few-shot证明MEDISEG预训练在遮挡场景比CURE显著提升（1-shot 0.406 vs 0.131）。

## 研究背景与动机

**领域现状**：用药错误和不良药物事件(ADE)威胁患者安全——1980-2014年ADE致死占AEMT的8.9%且持续上升。75-85岁人群超1/3每天服用5种以上处方药。AI药片识别是有前景的解决方案，但受限于数据集质量。

**现有痛点**：现有药片数据集存在三大不足：(1) NIH Pillbox（最大，133K图像）已于2021年停止维护且无实例分割标注；(2) CURE有部分实例分割标注但不完整且含合成图像；(3) 所有数据集都以单药片、受控环境拍摄为主，不反映dosette药盒、多药片重叠等真实场景。

**核心矛盾**：临床/家庭场景中药片总是多颗堆叠在药盒里，需要实例级分割来区分每颗药，但现有数据集几乎只有单药片图像。

**本文目标** 构建一个包含真实多药片场景（重叠、遮挡、dosette药盒）和完整实例分割标注的药片图像数据集，并验证其对few-shot泛化的价值。

**切入角度**：从临床需求出发设计数据集——使用dosette药盒自然产生多药片重叠/遮挡场景，刻意选择视觉相似的药片类别增加辨别难度。

**核心 idea**：构建覆盖真实多药片场景（遮挡/重叠/dosette盒）的实例分割数据集，证明场景复杂度而非纯数量才是few-shot泛化的关键。

## 方法详解

### 整体框架

MEDISEG 是一个为"防止用药差错"服务的药片实例分割数据集，含两个子集：3-Pills（3 类）和 32-Pills（8262 张、32 类）。整条采集-标注-验证链路是：iPhone 12 Pro Max 拍摄 → 按 dosette 药盒裁切到单格 → 统一到 640×640 → 用 COCO Annotator 手工标注实例分割掩码 → 拿 YOLOv8/v9 做监督训练验证可学性 → 再用 FsDet 做 few-shot 评估，检验数据集对跨场景泛化的价值。

### 关键设计

**1. 真实场景采集：用 dosette 药盒天然制造重叠与遮挡**

现有药片数据集几乎都是单药片、受控环境拍的，而临床和家庭里药片总是多颗堆在药盒里，模型一上真实场景就垮。MEDISEG 直接用老年人常用的标准四排七列 dosette 药盒排列药片拍摄，格子本身就会产生药片重叠、遮挡和背景干扰，再变化光照强度和角度引入真实的阴影、反射与高光，每张图含 1–13 粒药片，从单药一直覆盖到密集场景。这样训练出的模型见过的就是部署时真正会遇到的画面，而不是实验室里干净的单颗药片。

**2. 细粒度类别：故意塞进视觉高度相似的药片，逼模型同时看形状和颜色**

临床里大量是白色、无明显特征的药片，如果数据集类别区分度太高，模型会偷懒只学颜色这类简单线索。MEDISEG 在 32 类里刻意安排"形状像但颜色不同"（Pill A vs Pill B）和"颜色像但形状不同"（Pill B vs Pill C）的组合，并放入多种小白色药片，让仅凭粗粒度特征无法区分，从而迫使模型必须把形状和颜色特征一起学，贴近真实识别难度。

**3. Few-shot 评估协议：验证"场景复杂度"而非"纯数量"才是泛化关键**

数据集到底好在多还是好在难？作者用 FsDet 框架（Faster R-CNN + ResNet + FPN）做 base/novel 类分离的 few-shot 检测：分别拿 MEDISEG 和 CURE 做 base 训练，冻结 backbone+RPN、只 fine-tune ROI heads，在 1/5/10-shot 下评估 novel 类，并专门构造"仅遮挡"测试集去压极端场景。这个设计直接把"真实遮挡/重叠场景的预训练价值"摆上台面对比，证明决定 few-shot 迁移的是场景复杂度而非图像数量。

### 损失函数 / 训练策略

- YOLOv8/v9: 遗传算法70轮超参搜索（lr=0.01009, momentum=0.94, weight_decay=0.00048），最优fitness=0.81253
- 数据分割: 70% train / 20% val / 10% test
- Few-shot: backbone+RPN冻结，仅fine-tune ROI heads并重初始化分类层

## 实验关键数据

### 主实验

| 数据集 | 模型 | mAP@50 | mAP@50-95 | Precision | Recall |
|--------|------|--------|-----------|-----------|--------|
| 3-Pills | YOLOv8 | 99.4% | 95.0% | 99.7% | 99.7% |
| 3-Pills | YOLOv9 | 99.5% | 96.5% | 99.6% | 99.8% |
| 32-Pills | YOLOv8 | 62.2% | 50.9% | 62.8% | 57.4% |
| 32-Pills | YOLOv9 | 80.1% | 68.4% | 81.2% | 73.7% |

### 消融实验

| Few-shot设置 | MEDISEG fg_cls_acc | CURE fg_cls_acc | 提升倍数 |
|-------------|-------------------|-----------------|---------|
| 1-shot (遮挡集) | 0.406 | 0.131 | 3.1× |
| 5-shot (遮挡集) | 0.625 | 0.372 | 1.7× |
| 10-shot (遮挡集) | 0.740 | 0.558 | 1.3× |

| 对比维度 | MEDISEG | NIH Pillbox | CURE |
|----------|---------|-------------|------|
| 图像数 | 8,262 | 133,774 | ~1,000 |
| 类别数 | 32 | 4,392 | 196 |
| 实例分割标注 | ✓完整 | ✗ | 部分 |
| 多药片场景 | ✓(1-13粒/图) | ✗ | ✗ |
| 公开可用 | ✓(CC BY 4.0) | 已停止 | ✓ |

### 关键发现

- YOLOv9在32-Pills上显著优于YOLOv8（mAP@50: 80.1% vs 62.2%），更强的特征融合对细粒度识别至关重要
- 误分类主要来源于视觉相似药片的侧视图——俯视图可区分但侧面极其相似
- Few-shot性能差异在"仅遮挡"测试集上最显著——MEDISEG的多药片场景预训练赋予更好的遮挡鲁棒性
- 随shot数增加MEDISEG优势缩小（3.1×→1.3×），说明MEDISEG优势主要体现在极限低数据场景

## 亮点与洞察

- 数据集设计理念好: 刻意制造视觉相似类别和真实遮挡场景，不是简单数据量堆积
- Few-shot评估不只看标准测试集，专门构造"仅遮挡"子集测极端情况，评估方法比结果更有参考价值
- 验证了有趣发现: 训练数据的场景复杂度（而非纯数量）对few-shot泛化至关重要
- CC BY 4.0开放协议 + COCO格式 + GitHub代码 = 高易用性

## 局限与展望

- 32类药片数量偏少，实际临床药物种类数以千计
- 仅用iPhone 12拍摄，未验证跨设备泛化能力
- 动态光照和不同背景的变化仍然有限
- 仅评估目标检测/分割，未在语义理解任务上评估
- 未做临床环境中的前瞻性验证

## 相关工作与启发

- **vs NIH Pillbox**: 最大药片数据集(133K)但已停维且无实例分割标注，MEDISEG规模小但标注完整且包含多药片场景
- **vs CURE**: 最接近竞品，有部分分割标注但不完整且含合成图像。Few-shot实验直接证明真实遮挡场景的优势
- **设计启发**: "仅遮挡"测试子集的评估思路值得参考——构造针对性难例比在标准集上报告数字更有说服力

## 评分

- ⭐⭐⭐ 新颖性: 数据集贡献为主，方法无显著创新，但数据集设计理念有思考
- ⭐⭐⭐⭐ 实验充分度: 多模型+多子集+few-shot评估+超参搜索，验证扎实
- ⭐⭐⭐⭐ 写作质量: 数据集论文标准写法，结构清晰表格详尽
- ⭐⭐⭐ 价值: 对药物安全AI有实际价值，但领域局限性较强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](heuristic_self-paced_learning_for_domain_adaptive_semantic_segmentation_under_ad.md)
- [\[ECCV 2024\] Dataset Enhancement with Instance-Level Augmentations](../../ECCV2024/segmentation/dataset_enhancement_with_instance-level_augmentations.md)
- [\[CVPR 2026\] CA-LoRA: Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation](ca-lora_concept-aware_lora_for_domain-aligned_segmentation_dataset_generation.md)
- [\[CVPR 2026\] Phrase-Instance Alignment for Generalized Referring Segmentation](phrase-instance_alignment_for_generalized_referring_segmentation.md)
- [\[CVPR 2026\] RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video](robotseg_a_model_and_dataset_for_segmenting_robots_in_image_and_video.md)

</div>

<!-- RELATED:END -->
