---
title: >-
  ECCV2024 遥感方向 5篇论文解读
description: >-
  5篇ECCV2024 遥感方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛰️ 遥感

**🎞️ ECCV2024** · 共 **5** 篇

**[Adapting Fine-Grained Cross-View Localization To Areas Without Fine Ground Truth](adapting_fine-grained_cross-view_localization_to_areas_without_fine_ground_truth.md)**

:   针对细粒度跨视角定位模型在新区域部署时精度下降的问题，提出基于知识自蒸馏的弱监督学习方法——通过模式化伪GT生成、粗粒度监督和离群值过滤三个策略，仅使用目标区域的地面-航拍图像对（无需精确GT），即可在VIGOR和KITTI上将定位误差降低12%~20%。

**[Congeo Robust Cross-View Geo-Localization Across Ground View Variations](congeo_robust_cross-view_geo-localization_across_ground_view_variations.md)**

:   提出 ConGeo，一种模型无关的单视图+跨视图对比学习框架，通过强制同一地点不同地面视角变体之间的特征一致性，使单一模型即可在任意朝向和任意视场角(FoV)下实现鲁棒的跨视图地理定位。

**[Cross-Platform Video Person Reid A New Benchmark Dataset And Adaptation Approach](cross-platform_video_person_reid_a_new_benchmark_dataset_and_adaptation_approach.md)**

:   构建首个地面-无人机跨平台视频行人重识别数据集G2A-VReID，并提出VSLA-CLIP方法，通过视觉-语义对齐和参数高效的Video Set-Level-Adapter将CLIP适配到视频ReID任务。

**[Masked Angle-Aware Autoencoder For Remote Sensing Images](masked_angle-aware_autoencoder_for_remote_sensing_images.md)**

:   提出 MA3E，在 MAE 预训练中显式引入角度变化（通过 scaling center crop 构建旋转裁剪），并用最优传输损失自动分配重建目标，使模型感知遥感目标的多样角度，学习旋转不变表示。

**[Weakly-Supervised Camera Localization By Ground-To-Satellite Image Registration](weakly-supervised_camera_localization_by_ground-to-satellite_image_registration.md)**

:   提出首个弱监督的地面-卫星图像配准定位方法，通过卫星-卫星自监督训练旋转估计器、对比学习训练平移估计器，在无需精确GT姿态标签的条件下实现最佳跨区域泛化能力，超越大多数全监督SOTA方法。
