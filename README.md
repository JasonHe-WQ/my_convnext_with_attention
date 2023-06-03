# 一个添加了注意力机制的convNext模型

## 1. 项目介绍
使用CBAM注意力机制，对convNext模型进行改进，提高模型的准确率。

具体来说，使用了通道注意力机制和空间注意力机制，在模型的末尾，最终做成了一个7分类模型。

## 2. 项目结构
```
├── README.md                   // 描述文件
├── test.py                     // 用于推理
├── main.py                     // 用于训练
├── cur                         // 存放需要推理的图片
├── jud                         // 存放推理结果
├── model                       // 模型文件夹
     | convnext.py              // 添加了注意力的convNext模型
     | convNext_isotropic.py    // 原有convNext模型
├── pthlib                      // 存放权重
├── test                        // 测试集文件夹
     | right                    // 正确
     | err1                     // 1类
     | err2                     // 2类
     | err3                     // 3类
     | err4                     // 4类
     | err5                     // 5类
     | err6                     // 6类
├── train                       // 训练集文件夹
     | right                    // 正确
     | err1                     // 1类
     | err2                     // 2类
     | err3                     // 3类
     | err4                     // 4类
     | err5                     // 5类
     | err6                     // 6类
```

## 3. 使用方法
### 3.1 训练
直接运行main.py即可，训练完成后，会在pthlib文件夹下生成一个`model_1000.pth`文件，即为训练好的权重。
### 3.2 推理
将需要推理的图片放入cur文件夹下，运行test.py即可，推理结果会在jud文件夹下以json格式生成。

## 4. 致谢
感谢[convNext](https://github.com/facebookresearch/ConvNeXt)的开源代码，本项目在其基础上进行了改进。

感谢[CBAM](https://github.com/luuuyi/CBAM.PyTorch)的开源代码，本项目在其基础上进行了改进。

