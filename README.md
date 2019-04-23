<center><h1> BranchyNet_chainer</h1></center>
基于提前退出部分样本原理而实现的带分支网络（supported by chainer）

***
## 摘要
<p>
 　　深度神经网络是具有一定复杂程度的神经网络，可以定义为具有输入层、输出层和至少一个隐藏层之间的网络。每个层在一个过程中执行特定类型的分类和排序，这些复杂的神经网络的一个关键用途是处理未标记或未结构化的数据。一般来说，随着网络深度的增大，网络的性能也将会提升。但是，网络模型在数据集训练以及测试的时间也将变得越来越长，并且对机器资源的消耗需求也会增大。因此，我们提出了一种新的深度网络架构，通过在主网络中添加一个或多个分支网络，对退出点的样本置信度进行判断，从而可以提前退出部分样本，减少后继网络层的样本量。<br>
 　　本文我们使用几个比较常见的网络（LeNet, AlexNet, ResNet）和数据集（MNIST， CIFAR10）来研究该分支网络架构。首先我们在主网络上添加一个或多个分支网络，在数据集上进行训练，获取网络模型的总体准确率和损失函数值。然后在已经训练好的网络模型上进行测试，通过比较分析普通网络与添加分支后的网络测试效果，从而显示出该网络架构可以提高网络的准确性并且显著减少网络的推理时间。 
 
</p>

***
## 实现过程
<p>
 　　我们所建立的网络架构主要通过求解关联着退出点的损失函数加权和的联合优化问题进行训练，一旦训练好网络，则可以利用退出点让部分符合要求的样本提前退出，从而降低推理成本。在每个退出点，我们会利用分类结果的信息熵作为对预测的置信度度量，并设定每一个退出点的退出阈值。如果当前测试样本的熵低于退出阈值，即分类器 对预测结果有信心，则该样本在这个退出点待着预测结果退出网络，不会被下一层的网络处理。如果当前测试样本的上不低于退出阈值，则认为分类器对此样本的预测结果不可信，样本继续到网络的下一个退出点进行处理。如果样本已经到达最后一个退出点，也就是网络的最后一层，则将其直接退出。 
</p>

![B-LeNet](https://github.com/librahfacebook/BranchyNet_chainer/blob/master/pic/lenet_mnist/B-LeNet.png?raw=true "B-LeNet")　　　　　　　![B-LeNet](https://github.com/librahfacebook/BranchyNet_chainer/blob/master/pic/alexnet_cifar10/B-AlexNet.png?raw=true "B-AlexNet")

***

## 退出算法设计
当测试样本在训练好的分支网络模型上进行测试时，最终会经过网络层的计算，在退出点产生一个输出结果 z，我们使用 softmax 函数对其输出进行归一化，生成一个 0~1 之间的所有类概率集，其中每个类标签的预测概率定义为y_s，所有可能的类标签集合定 义为 S。  
如果退出点的测试样本 x 输出信息熵越小，则说明该分支的退出点的分类器对于正确标记该测试样本x的预测结果置信度越高，该样本被提前退出网络的可能性也就越大。

*退出算法代码：*
```
	for n=1...N:
	    y=softmax(z)
	    e=entropy(y)
	    if e<Tn:
	       return argmax(y)
	return argmax(y)
```

***

## 实验环境



|硬件环境|Colab 云服务器|
|:----:|:----:|
|CPU|Intel(R) Xeon® CPU @2.30GHZ*2 |
|GPU|Nvidia Tesla K80 |
|内存|12.72GB |

|软件环境|版本名称|
|:----:|:----:|
|操作系统|Ubuntu 16.04 |
|深度学习框架|Chainer 5.3.0|

***

## 实验结果

1. #### 普通网络与分支网络性能比较 

|Network|Acc.(%)|Time(ms)|Gain|Thrshold|Exit(%)|
|:----:|:----:|:----:|:----:|:----:|:----:|
|LeNet|98.85|1.93|-|-|-|
|B-LeNet|98.96|0.25|7.69x|0.5|98.25, 1.75|
|AlexNet|76.25|3.77|-|-|-|
|B-AlexNet|76.25|1.30|2.90x|0.05,1.0|51.65, 46.56, 1.79 |
|ResNet110|80.06|54.97|-|-|-|
|B-ResNet110|74.96|54.05|1.02x|0.0001,0.0001|43.12, 29.83, 27.05|

![LeNet](https://github.com/librahfacebook/BranchyNet_chainer/blob/master/pic/lenet_mnist/lenet_gpu(100).png?raw=true "LeNet")![AlexNet](https://github.com/librahfacebook/BranchyNet_chainer/blob/master/pic/alexnet_cifar10/alexnet_gpu(100).png?raw=true "AlexNet")![ResNet](https://github.com/librahfacebook/BranchyNet_chainer/blob/master/pic/resnet_cifar10/resnet_gpu(100).png?raw=true "ResNet")

2. #### 退出点阈值对分支网络的影响 

下图展示了退出阈值设置对分支网络第一个退出点退出样本数量的影响，在本实验中，所有在第一个分支没有退出的样本全部在最后一个退出点退出，我们可以设置其他退出点的阈值为 0 来对其行为进行影响，保证中间的分支没有样本提前退出。

![B-LeNet](https://github.com/librahfacebook/BranchyNet_chainer/blob/master/pic/lenet_mnist/lenet_branch1(100).png?raw=true "B-LeNet")![B-AlexNet](https://github.com/librahfacebook/BranchyNet_chainer/blob/master/pic/alexnet_cifar10/alexnet_branch1(100).png?raw=true "B-AlexNet")![B-ResNet](https://github.com/librahfacebook/BranchyNet_chainer/blob/master/pic/resnet_cifar10/resnet_branch1(100).png?raw=true "B-ResNet")

3. #### 首分支卷积层数对网络的影响 
下图显示了 B-AlexNet 网络的第一个侧分支中添加额外的卷积层数对最终网络精度 的影响。我们可以看到有一个最优的卷积层数对提高网络的准确性最好，而其他的卷积层数并不能有效提高网络的准确性，甚至更多的卷积层数会损害网络精度。因此，分支中的网络层数和分支结构的大小应该能够保证分支的总体网络深度小于在后面的退出 点退出所需的计算量，即较前的分支网络模型参数量应当小于靠后的分支网络模型参数量。通常，我们发现早期的退出点应当有更多的网络层，而后继的退出点应当有较少的网络层。 

![B-AlexNet](https://github.com/librahfacebook/BranchyNet_chainer/blob/master/pic/alexnet_cifar10/alexnet_layers.png?raw=true "B-AlexNet")

***

## 总结

深度神经网络是一种判别式模型，可以使用反向传播算法来进行训练，常用于图像 分类、语音识别等研究项目中。一般来说，随着网络深度的增加，网络的准确率也会随 着有所提高，但可能存在很多问题，常见的两类问题是过拟合和过长的运算时间。我们提出了一种新的网络结构——通过在主网络上添加侧边分支提前退出部分样本来加快网络的推理进程。通过合理的分支结构和退出原则以及所有退出点的损失函数的联合优 化，整个网络结构可以允许较多的测试样本能被提前分类，而不需要被传递到靠后的网络层，减少了整个网络的推理时间，提高了网络的整体性能。我们对3个网络进行设计 改造，在不同位置上添加分支结构来进行提前退出，从实验结果上可以展示出分支网络的优越性。  

本文针对普通网络在推理过程上的局限性和成本消耗性，利用样本提前退出的方法， 在网络的推理过程中逐个退出点减少测试样本，降低了大量样本对网络的逐层运算量。该分支网络结构在已有的主网络上合理添加分支网络结构，将样本提前退出的方法与深度学习的方法相结合，在相关数据集上得到了良好的实验结果。



