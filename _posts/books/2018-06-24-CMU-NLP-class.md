---
layout: post
title:  CMU NLP Class Notes 
date:   2018-06-24 12:16:00 +0800
categories: Notes
tag: ML-Notes
---

* content
{:toc}
NLP
=========================



## 1 intro

***

> NLP需要解决的问题
>
> ![1529824682897](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\1-phenomena.png)

> 简单的句子情感分类模型
>
> - bag of words
> - ![1529824781579](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\1-bag-of-words.png)
> - continuous bag of words, 增加了权重
> - ![1529824849496](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\1-cbow.png)
> - deep CBOW， 增加了隐藏层
> - ![1529824889155](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\1-deep-cbow.png)



## 2 Predicting the Next Word

> 预测语言模型
>
> ![1529825012244](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\2-plm.png)

> Count-based LM, 主要存在的问题是训练数据不够的情况下，概率估计不准，需要加入smoothing
>
> ![1529825194714](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\2-lm-gram.png)
>
> 目标函数
>
> ![1529825250650](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\2-lm-loss.png)
>
> 存在的问题
>
> ![1529825343791](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\2-ml-problems.png)

> 解决方法， Featurized Log-Linear Models![1529825443086](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\2-fllm.png)
>
> 解决了前面的第二个问题

> Unkonwn words 的解决方法
>
> - 使用UNK 表示一些不常用的单词的必要性。不可能把所有单词作为训练数据。考虑所有单词的字典会比较大，占用更大内存和计算时间。
> - 最常用的方法。1. 设定一个阈值，frequency 小于它时，单词用UNK表示。2.对所有单词排序，对frequency最小的一些单词用UNK表示。

> 线性模型不能够很好地表示特征间的combinations。例如：
>
> ![1529825932735](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\2-fcombination.png)
>
> 如果记住所有特征的组合，就会导致特征空间爆炸的问题。神经网络可以解决这个问题。这就解决了前面提到的三个问题中的第一个。
>
> ![1529826060360](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\2-dml.png)

> 训练神经网络的小技巧
>
> - 打乱训练数据，这主要是因为使用梯度下降方法更新参数的原因
> - SGD with Momentum
> - Adagrad
> - Adam 通常比较快地收敛和稳定。 SGD有更好的泛化能力
> - Early stopping 防止over-fitting
> - Learning rate decay
> - Dropout  防止over-fitting
> - Operation Batching 可以提升训练效率。 Mini-batching. TensorFlow 好像有Autobatching 函数

## 3 Models of Words

***

![1529827092611](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\3-word-model.png)

>  手工做单词之间的联系，WordNet
>
> ![1529827171316](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\3-wordNet.png)

>  使用模型学习到单词的低维向量表达， word embedding
>
> - word-context count matrix, 然后使用SVD进行矩阵分解，学习到单词的向量表达
> - 神经网络的方法 word2vec, 包括skip-gram 和cbow. 新的方法Glove

> 如何评价 word embedding 的好或者差？
>
> - embedding的可视化
> - ![1529828050799](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\2-eval-word-em.png)
> - 使用学习到的embedding 训练语言模型，看最终效果好坏

![1529828186048](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\2-word-embeding-userful.png)

![1529828146464](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\2-embedding-use.png)



> Sub-word Embeddings![1529828454990](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\2-sub-word-embedding.png)

> Multi-prototype Embedding, 一个单词有不同的意思，因此一个单词需要用多个embedding 表示
>
> ![1529828626683](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\2-multi.png)

> Multilingual Coordination of Embeddings. 两种不同语言中同义的单词要比较接近
>
> ![1529829148974](C:\Users\jingjing\AppData\Local\Temp\1529829148974.png)

> unsupervised coordination of embeddings. 主要思想是使得单词的概率分布相似
>
> ![1529829239457](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\2-unsupervised-multi.png)

> Retrofitting of embeddings to existing lexicons
>
> ![1529829361624](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\2-retrofitting.png)

> Sparse embeddings
>
> word embedding的每一维无法解释。解决方法是添加限制条件，让单词的向量添加稀疏性限制条件

## 4 Convolutional Networks for Text 

CNN可以解决单词之间的combination 问题，如 not good

> 解决combination的方法- bag of n-grams![1529829718726](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\4-bag-of-n-grams.png)
>
> 存在的问题是，1. 参数数量太大， 2. 相似单词/n-gram之间没有共享

> Time dealy neural networks
>
> ![1529829918601](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\4-time-dealy-net.png)

> CNN 
>
> ![1529829989049](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\4-cnn-for-text.png)

> stacked CNN

> Structured CNN
>
> Graph CNN