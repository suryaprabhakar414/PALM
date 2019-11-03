# PALM(Pathological Myopia)
This project focuses on the investigation and development of algorithms associated with the diagnosis of Pathological Myopia in fundus photos from PM patients. The goal of the project is to evaluate and compare Deep Learning algorithms for the detection of pathological myopia on a common dataset of retinal fundus images. 

In this Project I have used two Deep Learning Architectures:-

1)Residual Neural Network(ResNet50)

2)DenseNet(DenseNet121)

# Residual Neural Network(ResNet)

A Deeper Comvolutional Neural Network is able to extract complez features, but as we go deeper it becomes difficult to train because of  vanishing gradient problem — as the gradient is back-propagated to earlier layers, repeated multiplication may make the gradient infinitively small. As a result, as the network goes deeper, its performance gets saturated or even starts degrading rapidly.

The core idea of ResNet is introducing a so-called “identity shortcut connection” that skips one or more layers. These shortcut connection or skip connections are used to solve tne vanishing gradient problem.

![alt text](https://miro.medium.com/max/510/1*ByrVJspW-TefwlH7OLxNkg.png)




