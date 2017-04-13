---
layout: post
title:  Implement LSTM with theano 
date:   2016-08-27 01:08:00 +0800
categories: programming
tag: theano
---

* content
{:toc}


===============

Matters needing attention
---------------
+ copy a minibatch of data everytime. Don't just copy the data needed. Put the training data in a shared variable in GPU. Otherwise it will lead to a large decrease in performance.
 
  [theano学习指南1（翻译）](http://www.cnblogs.com/xueliangliu/archive/2013/04/03/2997437.html)
  [github](https://github.com/lisa-lab/DeepLearningTutorials)
  [tutorial](http://deeplearning.net/tutorial/)
  [Code analysis](http://blog.csdn.net/DeepLearningGroup/article/details/51385136)