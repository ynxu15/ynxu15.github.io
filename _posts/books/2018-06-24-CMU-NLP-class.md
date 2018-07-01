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

## 5 Recurrent Neural Networks 

> RNN 解决了代词指代的问题。可以记住句子中的信息
>
> RNN的梯度下降叫做backpropagation through time (BPTT)

> RNN 可以做什么应用？
>
> - 表示一个句子，来预测
> - 表示句子中的上下文
> - 句子分类
> - conditional generation
> - retrieval
> - tagging，输入单词序列，输出label序列
> - language modeling，输入单词序列，输出下一个单词序列
> - calculating representations for parsing

> Bi-RNNs 可以解决一部分距离太远，信息丢失的问题

> 梯度消失的问题，解决方法LSTM, GRU

> 句子间的依赖
>
> ![1529991550368](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\5-truncated-bptt.png)

> RNN 的优点和缺点
>
> - 很强大，很灵活
> - 需要大量的训练数据
> - 句子中有小的错误时会从句子末尾传导到整个句子

## 6 Efficiency tricks for neural nets

> 神经网络训练慢的原因
>
> - softmax 等复杂的函数运算，解决方法是寻找替代操作，或者使用gpu
> - 使用GPU，但也要减少某些操作，比如使用batching
> - network 很大，数据量很大。解决方法是用并行化

> word2vec 为什么快？
>
> - 负采样或者编码树代替softmax
> - class-based softmax, 每个单词增加了一个类别。模型先预测属于哪个类别，然后给定类别预测单词

> 并行化
>
> - within-operation parallelism
> - operation-wise parallelism
> - example-wise parallelism

> GPU在什么情况下会有较大性能提升
>
> - 神经网络规模很大
> - mini-batching
> - optimize things properly

> 加速的小技巧
>
> - 只需要做一次的操作，不要放在循环中
> - ![1530007139743](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\6-speed-trick1.png)
> - 尽量使用矩阵乘法，而不是使用循环
> - ![1530007229492](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\6-speed-trick2.png)
> - 减少CPU和GPU之间的数据移动
> - ![1530007735097](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\6-speed-trick3.png)
> - 考虑GPU的显存大小，不能超过显存大小

## 7 Using/Evaluating Sentence Representations

> 如何判断学到句子的表达是好的？
>
> - 句子分类
> - paraphrase identification 判断两个句子是不是同一个意思
> - ![1530010014418](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\7-paraphrase-identify.png)
> - semantic similarity 
> - entailment 句子间的包含关系
> - ![1530010715385](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\7-entailment.png)
> - retrieval， 给定一个句子，找到匹配的东西。 text->text, text->image,  anything-> anything

> Efficient retrieval
>
> - 数据库规模太大，不太好检索，使用approximate nearest neighbor search 的方法解决
> - ![1530011112247](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\7-efficient-retrieval.png)

## 8 Conditioned Generation

> 一个例子是编码解码器做翻译任务。
>
> 如何传递hidden state 呢？
>
> - ![1530085161369](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\8-pass-hidden.png)
>
> 产生下一个单词的方法
>
> - 选择概率最大的那个单词。 问题是经常产生简单的单词，偏好common words
> - 根据概率生成单词
> - beam search
>
> Ensembling
>
> - ![1530085379163](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\8-ensemble.png)
> - ![1530085424729](D:\同步盘\github\ynxu15.github.io\figure\books\2018-06-24-CMU-NLP-class\8-ensemble1.png)



> 如何评价模型好坏？
>
> - 产生结果，与reference 比较
> - 人类评价
> - BLEU, 与reference比较n-gram
> - METEOR 
> - Perplexity

