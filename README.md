# Distracted-Driver-Detection

## 来源

[Kaggle competition: State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection)

## 项目简介

本项⽬利⽤深度学习训练应⽤于驾驶员⾛神检测的多分类器。分类器的输⼊为⻋内摄像头拍摄的驾驶员画⾯，输出为⼗种⾛神状态的概率值。这样的问题，在数据集充分的情况下，在原理上与常⻅的图像分类任务（例如ImageNet）是⽐较相似的。

其中，⼗种可能的⾛神状态定义为：

C0: 安全驾驶 C1: 右⼿打字 C2: 右⼿打电话 C3: 左⼿打字 C4: 左⼿打电话 C5: 调收⾳机 C6: 喝饮料 C7: 拿后⾯的东⻄ C8: 整理头发和化妆 C9: 和其他乘客说话

## 主要步骤

1. 探索数据，通过随机采样、可视化等⽅式认识数据，检查类别均衡等。
2. 采用同类图像左右拼接的方式进行数据增⼴。 
3. 对图像进⾏裁剪、归⼀化等预处理操作。 
4. 考虑到图像采集时具有时序关系，同一驾驶员的数据相关性较强，对于训练集按驾驶员分类，随机打乱，并划分训练集、验证集。 
5. 对于较⼤的训练集等，抽取一部分作为小训练集，便于验证模型正确运行。 
6. 构造简单卷积神经⽹络（例如三层卷积池化层+三层全连接层），在小训练集上调试超参数。 使⽤batch normalization等⽅式加快⽹络训练。使⽤tensorboard对⽹络结构和训练过程进⾏可视化。通过k-折交叉验证、⽹格搜索等⽅式进⾏超参数选择。 
7. 将简单卷积神经⽹络在全部训练集上训练，并微调超参数，检验性能。
8. 利⽤迁移学习的⽅式，使⽤多种在ImageNet上取得良好效果的⽹络结构，并在此基础上添加新的卷积或全连接层，进行fine-tuning，在小训练集上调试超参数。使⽤batch normalization等⽅式加快⽹络训练。使⽤tensorboard对⽹络结构和训练过程进⾏可视化。通过k-折交叉验证、⽹格搜索等⽅式进⾏超参数选择。将上述训练后的迁移学习模型在全部训练集上进⾏训练，微调超参数。并将训练后的模型与基准模型进⾏性能对⽐。
9. 将多个迁移学习模型（例如基于不同的经典⽹络结构VGG、ResNet等）进⾏集成。



## 目前得分

kaggle leaderboard上的得分，public为0.32046， private为0.29960，private得分排在194名，占全体的13.4%。

## 可能的改进方向

1. 进一步调整超参数，优化训练性能；
2. 进一步通过数据增广、L2正则化、Dropout等方式的综合运用，降低过拟合风险；
3. 将基于同一预训练模型的不同fine-tuning模型，与不同预训练模型的fine-tuning模型集成；
4. 通过Class Activation Mapping (CAM)方法，可视化神经网络对于每个类别的关注区域，指导后续改进。

