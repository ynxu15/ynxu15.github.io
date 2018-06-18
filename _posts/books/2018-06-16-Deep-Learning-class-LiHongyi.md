---
layout: post
title:  深度学习 李宏奕 Class Notes 
date:   2018-04-12 17:16:00 +0800
categories: Notes
tag: ML-Notes
---

* content
{:toc}

深度学习 -- 台湾 李宏奕课程
======================================
***

注：这是台湾李宏毅教授的2017年春季学期深度学习课程笔记 
+ [课程官网](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML17.html)
+ [B站视频集合](https://www.bilibili.com/video/av9770302/)
+ [知乎专栏别人的笔记](https://zhuanlan.zhihu.com/p/34513073)


## 0 讲 introduction
------------------------------------------

> Machine learning的实质是find a good function

> Structured Learning
+ Regression: output a scalar
+ Classification: output a class -  one-hot vector
+ Stuctured Learning: output a sequence (翻译，语音识别，聊天机器人), a matrix （文本生成图片，图片风格变化，图片彩色化）, a graph, a tree

> challenges:
- output space is very sparse
- Because the output components have dependency,  they should be considered globally
- methods are deep

> 传统机器学习到深度学习的转变
- 遇到问题直接用深度学习，训练一下试试;
- 万事皆可train；神农尝百草

> 为什么需要深度学习？
- 浅层次的神经网络更倾向于记住训练数据
- 我们使用深度学习是因为我们没有足够的训练数据
- 深度学习的每一个隐藏层可以看做不同层次的抽象，可以使用更少的参数，对训练数据进行拟合



## 1讲和2讲  Basic Structures for Deep Learning Models
------------------------------------------

> 深度学习有3个step:
- build neural network->cost function -> optimization
- 对应于机器学习中的: 定义函数 -> 评价函数好坏 -> 选择函数
![Deep learning three steps{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/0-three-step.jpg' | prepend: site.baseurl}})


> 三种基本网络结构
- 全连接网络
- 循环神经网络
- 卷积神经网络

> 循环神经网络 RNN
![RNN{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/1-RNN.png' | prepend: site.baseurl}})

> 深度循环神经网络 Deep RNN
![Deep RNN{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/1-deep-rnn.png' | prepend: site.baseurl}})

> 双向RNN
![Bi RNN{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/1-bi-rnn.png' | prepend: site.baseurl}})

> Pyramidal RNN, 减少time steps 的个数
![Pyramidal RNN{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/1-pyradimidal-rnn.png' | prepend: site.baseurl}})

> LSTM  
![LSTM{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/1-LSTM.png' | prepend: site.baseurl}})

> GRU
![GRU{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/1-gru.png' | prepend: site.baseurl}})

> Stack RNN， information, pop, push, nothing
![Stack RNN{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/1-stack-rnn.png' | prepend: site.baseurl}})



> 相关的任务
语音识别

> Notes:
- RNN 输出有delay的时候，效果会更好，例如输入xt，输出yt-3
- 模型的效果 LSTM>RNN>feedforward
- 模型效果 Bi-direction > uni-direction RNN/LSTM
- LSTM 收敛速度比RNN更快，一个原因是梯度消失的现象更弱
- 遗忘门对LSTM最重要
- 输出门的激活函数很重要

> 卷积神经网络有两个特征
- Sparse connectivity
- Parameter sharing

> 卷积的超参数 filter size (Size of Receptive field ), stride, padding.

> pooling的形式： max, average, l2 pooling

> 不同类型的神经网络层，可以相互结合
![Stack RNN{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/1-multi-net.png' | prepend: site.baseurl}})



## 3讲 Computational Graph & Backpropagation 
------------------------------------------

TensorFlow 计算图可以根据链式法则自动求导


## 4 讲 Deep Learning for Language Modeling
------------------------------------------

> language model: Estimated the probability of word sequence P(w1, w2, w3, …., wn)

> 应用 
- 语音识别
- 机器翻译
- 文本生成

> N-gram 模型
- N-gram language model: P(w1, w2, w3, …., wn) = P(w1|START)P(w2|w1) …... P(wn|wn-1)
- 使用n-gram 模型的原因是训练数据不够
![N-gram{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/4-ngram-eq.png' | prepend: site.baseurl}})


> N-gram 模型的问题
- 因为数据不够多，导致概率估计不准
- 估计概率为0的n-gram不代表不会出现，或者不正确

> n-gram 问题的解决方法 smoothing
- (1) 每个元素加一个很小的值
- (2) 可以使用MF模型来进行矩阵分解，把缺失的地方填补起来，使得很多没有观察到的，概率不是0
- (3) MF模型替换成神经网络模型。不使用朴素贝叶斯模型，而使用神经网络模型的原因也是数据不够，概率估计不准确

### 例子： 潮水  退了  就  知道  誰 …
方案1 前向神经网络  
![Stack RNN{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/4-feedforward-net.png' | prepend: site.baseurl}})


方案2 RNN
![Stack RNN{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/4-rnn.png' | prepend: site.baseurl}})


方案3 Class-based Language Modeling
![Stack RNN{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/4-class-based.png' | prepend: site.baseurl}})


方案4 Soft word class
给每个单词学习一个低维向量表达
![Stack RNN{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/4-soft-class.png' | prepend: site.baseurl}})
![Stack RNN{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/4-soft-class-2.png' | prepend: site.baseurl}})


方案5 Neural Turing Machine for LM (重点看一下)
![Stack RNN{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/4-neural-turing.png' | prepend: site.baseurl}})


## 5讲 Spatial Transformer Layer
------------------------------------------

> 主要是讲，如果把图片中的数字缩放，旋转，摆正了之后，识别的准确率会提高。因此如何设计神经网络，让它自动把这个过程做了。

> CNN的variant 特性
- CNN中的pooling 层使得该网络在图片小幅度变化时，比如移动几个像素，图中的人走动几步，还能够有很好的效果。 
- 但是如果放大、缩小、旋转等，CNN就不会有好的效果了

> Bird recognition
学习如下的参数，将图片中的一部分截取出来，类似于attention的思想
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/5-bird.png' | prepend: site.baseurl}})

![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/5-bird-net.png' | prepend: site.baseurl}})

![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/5-bird-net1.png' | prepend: site.baseurl}})



## 6讲 Highway Network & Grid LSTM
------------------------------------------

> RNN 和 Feedforward Net
- RNN和Feedforward很像，一个是时间尺度上做多次，一个是隐藏层数上做多次。不同是RNN的不同时间上参数是共享的
- highway network是为了把RNN应用成Feedforward Network
- highway network 主要思想就是使用循环神经网络改造前向网络
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/6-feedforward-rnn.png' | prepend: site.baseurl}})



> Highway Network 和残差网络
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/6-high-resi.png' | prepend: site.baseurl}})


> Grid LSTM 
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/6-grid-net.png' | prepend: site.baseurl}})
> Grid LSTM 模块的具体情况  
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/6-grid-net1.png' | prepend: site.baseurl}})


> 3D grid LSTM
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/6-3d-grid-net.png' | prepend: site.baseurl}})



## 7 讲 Recursive Network 
------------------------------------------

> recursive network
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/7-recursive-net.png' | prepend: site.baseurl}})
> recursive net 的连接情况
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/7-recursive-net1.png' | prepend: site.baseurl}})



> recursive neural tensor network
这个网络主要是解决神经网络中线性的乘法（Wx）无法很好地表示两个单词向量间的交互而设计的。
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/7-recursive-tensor.png' | prepend: site.baseurl}})

> Matrix-Vector Recursive Network
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/7-recursive-tensor1.png' | prepend: site.baseurl}})


> Tree LSTM
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/7-tree-lstm.png' | prepend: site.baseurl}})




## 8讲 Conditional Generation by RNN & Attention
------------------------------------------

> Image Caption Generation
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/8-image-caption.png' | prepend: site.baseurl}})


> Chat-bot
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/8-chat-bot.png' | prepend: site.baseurl}})


> Attention Network

> Speech Recognition
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/8-speech.png' | prepend: site.baseurl}})


> Image Caption Generation
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/8-image-caption1.png' | prepend: site.baseurl}})


> memory based network
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/8-memory-net.png' | prepend: site.baseurl}})
memory based network 细节  
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/8-memory-net1.png' | prepend: site.baseurl}})


> Neural Turing Machine
> 图灵机有三个操作，读，写，删除
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/8-turing.png' | prepend: site.baseurl}})

![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/8-turing1.png' | prepend: site.baseurl}})


> 好的attention应该让每个input拥有接近的attention 权重。以视频生成caption为例，视频的每一帧的权重综合应该是差不多的
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/8-attention.png' | prepend: site.baseurl}})


> Mismatch between Train and Test
- Scheduled Sampling. RNN训练的时候，每次的输入应该是从model来，还是reference来？使用reference可以收敛，来自model不可收敛。scheduled sampling 效果会更好，开始先用reference，然后再用model，使用随机数进行选择
- beam search 
- 强化学习，在句子完整生成最后才给一个reward


## 9讲 Point Network
------------------------------------------
> 从一堆已知的点中选择一些作为输出。
> 用途在于，例如chat-bot 或者机器翻译里面，一些专有名词可以直接copy过来，不用修改

## 10讲 Batch Normalization
------------------------------------------

> feature scaling
- 对输入做 feature scaling
- 对每一层隐藏层的输入做feature scaling，可以解决神经网络的Internal Covariate Shift, learning rate 小一些也可以解决这个问题

> batch normalization
- 解决两个问题Internal Covariate Shift 和gradient vanishing
- 可以设置较大的学习速率
- 学习过程受初始化影响较小
- 降低对正则化的需求

## 11讲 SELU 激活函数
------------------------------------------

> ReLU  
- 计算快
- 解决梯度消失问题  
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/11-relu.png' | prepend: site.baseurl}})

> Leaky ReLU  
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/11-leaky-relu.png' | prepend: site.baseurl}})


> Parametric ReLU  
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/11-p-relu.png' | prepend: site.baseurl}})


> Randomized ReLU
- alpha is sampled from a distribution  
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/11-r-relu.png' | prepend: site.baseurl}})


> ELU  -- Exponential Linear Unit  
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/11-e-relu.png' | prepend: site.baseurl}})

> SELU -- Scaled ELU  
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/11-selu.png' | prepend: site.baseurl}})

> SELU 推导的启示
- 初始化的时候让weight的均值是0，方差是1/k + selu

## 12讲 Capsule Network
------------------------------------------

> 和传统网络的区别
- 传统神经元，输出是一个值
- capsule 输出是一个向量，多个capsule的输出通过动态的加权值得到新的capsule, 增加了挤压操作
- capsule 不是针对某一个pattern，是针对某一种pattern。 capsule的长度表示这个pattern存在的可能性大小，而整个向量表示这个pattern是什么内容
- capsule 可以取代CNN中的filter
- 最后输出的是一个向量，如何做分类问题呢？某一个类对应的向量长度，表示判别为该类别的置信度有多大
- CNN只能做invariance, Capsule 能做到invariance(不同)和equivariance

> Capsule network
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/12-capsule.png' | prepend: site.baseurl}})


> Dynamic Routing 为什么会work, 有attention的影子，还有multi-hopping

> c1 和c2是动态获得的，不是固定的，论文里叫动态路由 dynamic routing


> 动态路由
![{100*100}]({{'/figure/books/2018-06-16-Deep-Learning-class-LiHongyi/12-dynamic-routing.png' | prepend: site.baseurl}})




## 13讲 Tuning Hyperparameters
------------------------------------------

> 如何调参数
- Grid search
- Random search -- 这个更好一些
- Model based hyperparameter optimization 方法，根据模型的超参和结果准确度来训练一个拟合模型，然后找到最优点，并且该点的置信度比较高，重新训练模型，根据结果更新拟合模型
- 强化学习， google的learning to learn，使用RNN来学习CNN的filter的超参


## 14讲 Interesting things about deep learning
------------------------------------------

> 深度学习模型loss不再降低的原因
- saddle point 鞍点
- 局部最优

> 给数据加噪声后发现，神经网络在测试集上随着Epoch的次数增加，准确率先上升，再下降。说明神经网络在训练次数较少的时候，获得了有效信息，但是训练次数增多后，强行记忆了很多噪声信息，导致在测试集上的准确率下降。simple pattern first

> 启示： early stop 有可能获得更好的结果

## 15讲 Generative Adversarial Network (GAN)
------------------------------------------

Auto encoder
VAE，只是模仿现有的图片，而不是用于生成图片
GAN 比较适合用于生成图片

GAN模型


GAN模型算法


Evaluating JS divergence

GAN, 判别器很快就可以达到100%的分类准确率，因为是使用采样来代替分布，则判别器有过拟合的风险。判别器过快达到100%准确率之后，会给生成器的优化带来很少的更新信息，导致生成器不收敛

可以使用较弱的判别器，或者加正则化项，dropout等
给判别器的输入加噪声。给判别器的label加噪声
判别器不能够很好地区分真的和生成的数据。使得真实数据分布和生成数据分布有一些重合
Noise decay over time

Conditional GAN
生成器和判别器都加入条件c
应用有 根据文本生成图片


Speech enhancement GAN


## 16讲 Improved Generative Adversarial Network
------------------------------------------

## 17讲 RL and GAN for Sentence Generation and Chat-bot
------------------------------------------


## 18讲 机器学习，美少女化
------------------------------------------

1. DCGAN
2. 训练auto-encoder, fix住generator，训练vectorizer。将vectorizer和generator接到一块
3. 真实图片-> vectorizer ->generator -> classifer -> 笑脸，嘴唇等 attributes。 最后一步只需要训练vectorizer

Generator 一定要fixed

## 19 讲 Imitation Learning
------------------------------------------


## 20讲 Evaluation of Generative Models
------------------------------------------

## 21讲 Ensemble of GAN
------------------------------------------

## 22讲 Energy-based GAN
------------------------------------------

## 23讲 Video Generation by GAN
------------------------------------------

## 24讲 Ensemble of GAN
------------------------------------------

## 25讲 Gated RNN and Sequence Generation
------------------------------------------
