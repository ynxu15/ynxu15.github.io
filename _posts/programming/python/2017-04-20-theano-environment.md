---
layout: post
title:  Construct the environment of theano 
date:   2017-4-20 9:43:00 +0800
categories: programming
tag: theano
---

* content
{:toc}

Build environment for theano
=====================
requirement: numpy, scipy

python packages [http://www.lfd.uci.edu/~gohlke/pythonlibs/](http://www.lfd.uci.edu/~gohlke/pythonlibs/)
and [http://www.lfd.uci.edu/~gohlke/pythonlibs/](http://www.lfd.uci.edu/~gohlke/pythonlibs/)

Steps for building
---------------------
+ Install CUDA 
+ Install theano
+ Install MinGW 64 (provides g++)

Problems
----------------------
**problem 1:** Exception: Compilation failed (return status=1): C:\Users\swanheart\AppData\Local\Theano\compiledir_Windows-7-6.1.7601-SP1-Intel64_Family_6_Model_42_Stepping_7_GenuineIntel-3.4.1-64\lazylinker_ext\mod.cpp:1:0: sorry, unimplemented: 64-bit mode not compiled in

**Solution:** This error means you need a 64-bit g++ compiler. You should install MinGW 64, rather than MinGW. You can check the version of g++ with *g++ -v*
Add /"MinGW path"/mingw64/bin to *path* 



**problem 2:** error: '::hypot' has not been declared when compiling with MingGW64

**Solution:** I've fixed it with adding "-D_hypot=hypot" to the cxxflags in ccompile. C:\Users\jingjing
	[blas]
	ldflags =
	# ldflags = -lopenblas # placeholder for openblas support
	[global]
	device = gpu
	floatX = float32
	[gcc]
	cxxflags =-shared -I"D:\program\TDMGCC\include" -D_hypot=hypot
	[nvcc]
	flags=-LG:\program\python2.7\libs
	compiler_bindir=D:\Program Files\vs2013\VC\bin

I also run this *pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git*. I don't know if it helps.

GPU usage
---------------------
run *nvidia-smi*.

The path of this command *C:\Program Files\NVIDIA Corporation\NVSMI*.

**Notes:** I just record some errors which I met