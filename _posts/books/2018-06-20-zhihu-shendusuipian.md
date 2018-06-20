---
layout: post
title:  知乎 深度碎片 文章 Notes 
date:   2018-06-20 12:16:00 +0800
categories: Notes
tag: ML-Notes
---

* content
{:toc}
Deep Learning 
=========================

up 主的链接 [深度碎片](https://zhuanlan.zhihu.com/c_170572456)

CNN [知乎](https://zhuanlan.zhihu.com/p/34619675)

## 物体分类

### LeNet-5

- 论文的意义在于DNN和backprop的结合
- 实现了文本的自动识别
- 传统的文本识别包括两个步骤，（1）人工设计特征模式，（2）添加分类器
- 论文的写作逻辑，（1）什么是learning from data (2) 模型的forward prop 函数是什么样子 （3）loss function 是什么样子（4）如何找到最优的模型或者参数 （5）如何检验模型好坏 （6）如何通过正则化来帮助改进模型

### AlexNet

- 通过图片翻转，旋转，平移等增加训练样本数量，降低模型的过拟合风险
- PCA方式降低过拟合风险，

### VGG

- VGG 的感受野由11\*11， 7\*7缩小到了3\*3。 模型变得更小，实现了正则化的效果
- VGG 16 还添加了1\*1 的filter



### ResNet





### GoogleNet









## 物体识别

### R-CNN



### Selective Method



### OverFeat

> 第一篇做localization 和 detection 的CNN模型
>
> Overfeat = 模型feature extractor的名称
>
> 特色 = 一个模型完成了classification localization, detection多任务
>
> 工具 = multi-scale and sliding window
>
> localization = bounding box prediction

挑战

> 尺寸大小不统一
>
> 物体出现在不同的区域

### YOLO



### Faster R-CNN









## 人脸识别

### DeepFace



### FaceNet 和 Triplet loss



### visualizing and understanding CNN



### neural style transfer paper





## 引用量较高的论文

1. ([AlexNet视频笔记](https://zhuanlan.zhihu.com/p/30954591)) ImageNet Classification with Deep Convolutional Neural Networks
2. ([VGG视频笔记](https://zhuanlan.zhihu.com/p/30954591)) Very Deep Convolutional Networks for Large-scale Image Recognition
3. ([ResNet视频笔记](https://zhuanlan.zhihu.com/p/30981686)) Deep Residual Learning for Image Recognition
4. ([Inception GoogLeNet视频笔记](https://zhuanlan.zhihu.com/p/31002146)) Going Deeper with Convolutions
5. ([BN视频笔记](https://zhuanlan.zhihu.com/p/31046868)) Batch normalization: Accelerating deep network training by reducing internal covariate shift (2015)
6. ([Dropout视频笔记](https://zhuanlan.zhihu.com/p/31120926)) Dropout: A simple way to prevent neural networks from overfitting (2014)
7. ([OverFeat 视频笔记](https://zhuanlan.zhihu.com/p/31234308)）OverFeat: Integrated recognition, localization and detection using convolutional networks
8. ([YOLO 视频笔记](https://zhuanlan.zhihu.com/p/31251773?group_id=916677747494858752))You only look once: Unified, real-time object detection
9. YOLO 9000 [[pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1612.08242.pdf)]
10. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (2015), S. Ren et al. [[pdf\]](https://link.zhihu.com/?target=http%3A//papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)
11. Fast R-CNN (2015), R. Girshick [[pdf\]](https://link.zhihu.com/?target=http%3A//www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)
12. Rich feature hierarchies for accurate object detection and semantic segmentation (2014), R. Girshick et al. [[pdf\]](https://link.zhihu.com/?target=http%3A//www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)
13. DeepFace: Closing the gap to human-level performance in face verification (2014), Y. Taigman et al. [[pdf\]](https://link.zhihu.com/?target=http%3A//www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf)
14. FaceNet: A Unified Embedding for Face Recognition and Clustering [[pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1503.03832)] [论文笔记（90%）](https://zhuanlan.zhihu.com/p/32049245)[review笔记](https://zhuanlan.zhihu.com/p/32110109)
15. Visualizing and understanding convolutional networks (2014), M. Zeiler and R. Fergus [[pdf\]](https://link.zhihu.com/?target=http%3A//arxiv.org/pdf/1311.2901)
16. A neural algorithm of artistic style (2015), L. Gatys et al. [[pdf\]](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1508.06576)
17. Understanding deep learning requires rethinking generalization (2017), C. Zhang et al. [[pdf\]](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1611.03530)
18. A Closer Look at Memorization in Deep Networks (2017) Bengio et al. [[pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1706.05394)]
19. The loss surfaces of multilayer networks (2015) [[pdf](https://link.zhihu.com/?target=http%3A//proceedings.mlr.press/v38/choromanska15.pdf)]
20. Hinton's capsule [pre-paper1](https://link.zhihu.com/?target=http%3A//www.cs.toronto.edu/%7Efritz/absps/nips99ywt.pdf)[pre-paper2](https://link.zhihu.com/?target=http%3A//www.cs.toronto.edu/%7Efritz/absps/transauto6.pdf)[paper1](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1710.09829)[paper2](https://link.zhihu.com/?target=https%3A//openreview.net/pdf%3Fid%3DHJWLfGWRb) (2000-2018)
21. Hinton's capsule on [medium](https://link.zhihu.com/?target=https%3A//medium.com/ai%25C2%25B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b) 英文([中文](https://zhuanlan.zhihu.com/p/31262148)) 笔记 [part1](https://zhuanlan.zhihu.com/p/31777460)[part2](https://zhuanlan.zhihu.com/p/31789728)[part3](https://zhuanlan.zhihu.com/p/31813017)
22. Hinton's capsule paper 2017 [文档笔记](https://zhuanlan.zhihu.com/p/31834356)[capsnet code](https://link.zhihu.com/?target=https%3A//github.com/soskek/dynamic_routing_between_capsules/blob/master/nets.py%23L103)
23. Hinton's capsule paper by 李宏毅 [原视频](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/av16583439/)[视频笔记](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/av17214881/)
24. [论文分解器：Generalization in Deep Learning](https://zhuanlan.zhihu.com/p/32298476)
25. [论文分解器：on the origin of deep learning （更新到section6）](https://zhuanlan.zhihu.com/p/32338470)
26. [论文分解器:deep learning-a critical appraisal](https://zhuanlan.zhihu.com/p/32679965)
27. [论文分解器：on the information bottleneck theory of deep learning (更新中）](https://zhuanlan.zhihu.com/p/32718190)
28. ([AlignedReID 视频笔记](https://zhuanlan.zhihu.com/p/31401390)) AlignedReID: Surpassing Human-Level Performance in Person Re-Identification
29. Man is to Computer Programmer as Woman is to Homemaker?Debiasing Word Embeddings [原论文](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1607.06520.pdf)[笔记](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/av18994337/%23page%3D23)