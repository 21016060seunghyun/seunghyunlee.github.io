#!/usr/bin/env python
# coding: utf-8

# # Residual Networks (ResNet)
# 
# 이번 실습에서는 매우 깊은 CNN모델인 <b>Residual Networks (ResNet)</b>를 직접 구현해봅니다. 
# 
# 일반적으로 신경망이 깊어질수록(deep) 표현력(representational power)이 증가하여 복잡한 함수도 근사할할 수 있게 되지만, 학습이 어려워진다는 문제가 발생합니다.  
# 
# 논문 [He et al. (2015)](https://arxiv.org/pdf/1512.03385.pdf)에서 소개한 ResNet구조는 아주 깊은 신경망도 효과적으로 학습할 수 있는 방법을 제안하였습니다.

# In[1]:


import torch
from torch import nn


# ## 깊은 신경망의 문제점 (The Problem of Very Deep Neural Networks)
# 
# - 깊은 신경망의 가장 중요한 장점은 아주 복잡한 함수도 근사할 수 있다는 것입니다. 입력에 가까운 앞쪽 레이어에서는 edge, texture와 같은 저수준 특징(low-level features)을 추출하고, 깊은 레이어로 갈수록 좀더 추상적이고 고수준의 특징(high-level features)을 계층적으로 추출함으로써 깊은 신경망은 아주 높은 표현력(representational power)을 가질 수 있습니다.
# - 하지만 무작정 깊게 쌓는다고 좋은 성능이 나오는 것은 아니며, 대표적인 원인은 **기울기 소실(vanishing gradient)** 문제입니다.
# 
# ### 기울기 소실 (vanishing gradient)
# - 역전파(backpropagation) 과정에서, 파라미터에 대한 손실 함수 $\mathcal{L}$의 미분은 연쇄법칙(chain rule)에 의해 마지막 레이어부터 첫번째 레이어로 전파됩니다.
# 
# $$
# \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[1]}} =
# \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \cdot
# \left(
# \prod_{l=L}^{2} 
# \frac{\partial \mathbf{a}^{[l]}}{\partial \mathbf{z}^{[l]}} \cdot
# \frac{\partial \mathbf{z}^{[l]}}{\partial \mathbf{a}^{[l-1]}} 
# \right) \cdot
# \frac{\partial \mathbf{a}^{[1]}}{\partial \mathbf{z}^{[1]}} \cdot
# \frac{\partial \mathbf{z}^{[1]}}{\partial \mathbf{W}^{[1]}}
# $$
# 
# $$
# \mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
# \quad , \quad
# \mathbf{a}^{[l]} = \phi(\mathbf{z}^{[l]})
# $$
# 
# 
# - 이때 각 레이어에서의 편미분 값들이 $< 1$인 경우, 여러 레이어를 거치며 곱해지는 과정에서 gradient가 지수적으로 감소하며며 아주 빠르게 0에 가까워집니다.
# - 이는 곧 앞쪽 레이어(입력에 가까운 레이어)의 가중치가 거의 업데이트되지 않아, 학습 제대로 이루어지지 않음을 의미합니다.
# - (반대로 편미분 값들이 $> 1$인 경우에는 그래디언트가 레이어를 지날수록 지수적으로 증가하여 gradient explode현상이 발생할 수 있습니다. 일반적으로는 gradient vanishing현상이 더 자주 발생합니다.)
# 
# <center><img src="resources/vanishing_grad_kiank.png" style="width:450px;height:220px;"></center>
# 
# - 위 그림을 살펴보면, 입력에 가까운 레이어일수록 gradient의 크기(norm) 급격히 감소하여 학습이 이루어지기 어려움을 알 수 있습니다.
# 
# ## Core idea of ResNet
# 
# ResNet에서는 <b>skip connection</b>(residual connection 또는 shortcut 이라고도 불림)이라고 불리는 구조를 도입하여 이 문제를 해결합니다.
# 
# <center><img src="resources/skip_connection_kiank.png" style="width:650px;height:200px;"></center>
# 
#  - 왼쪽 그림은 일반적인 신경망의 연산 경로(main path)를 나타냅니다.  
#  - 오른쪽 그림은 여기에 skip connection이 추가된 구조로, 입력이 일부 레이어를 건너뛰어 출력에 직접 더해집니다. ResNet에서는 이러한 구조를 <b>residual block</b>이라고 부릅니다
# 
# ### Skip connection의 장점
# - Skip connection은 역전파 시 기울기(gradient)가 직접 전달될 수 있는 경로를 제공하여, 기울기 소실(vanishing gradient) 현상을 크게 줄여줍니다.
# - 또한, Skip connection이 존재하는 <b>residual block</b>은 항등 함수(identity function)을 매우 쉽게 학습할 수 있습니다.
#   - 예를 들어 블록 내 모든 가중치가 0이면, 출력은 입력과 거의 같아져 항등 함수가 됩니다.
#   - 이는 residual block을 아주 깊게 쌓더라도, 더 얕은 네트워크에 비해 성능이 나빠지지 않는다는 특성을 갖게 합니다.
#   - 실제로 일부 연구에서는, ResNet의 성능 향상이 vanishing gradient를 막는 것보다 항등 함수를 빠르게 학습할 수 있는 구조에서 기인한다는 분석도 제시하고 있습니다.
# 
# 따라서 ResNet은 이러한 **residual block**을 반복적으로 쌓는 방식으로, 수십~수백 층에 이르는 매우 깊은 신경망도 안정적으로 학습할 수 있게 해줍니다.
# 
# ## Building a ResNet from scratch
# 
# ResNet에서는 입력과 출력의 차원에 따라 두 가지 종류의 블록을 사용합니다:
# - Identity Block: 입력과 출력의 차원이 같은 경우  
# - Convolutional Block: 차원이 다른 경우
# 
# ### Identity Block
# 
# identity block은 ResNet의 기본 블럭으로, 입력 activation(예 $a^{[l]}$)과 출력 activation (예 $a^{[l+2]}$)의 **차원이 같을 때** 사용됩니다.
# 
# ### Basic Identity block (ResNet-18, ResNet-34)
# <center><img src="resources/idblock2_kiank.png" style="width:650px;height:150px;"></center>
# <caption><center> <b>2개의 레이어를 건너뛰는 Basic Identity block.</b> </center></caption>
# 
# - 위 구조는는 2개의 conv 레이어와 skip connection으로 구성됩니다.
# - 이러한 구조는 ResNet-18, ResNet-34에서 사용됩니다.
# 
# ### Bottleneck Identity block (ResNet-50 이후)
# ResNet-50부터는 더 깊고 네트워크를 효율적으로 학습하기 위해 Bottleneck 구조의 identity block을 사용합니다.
# 
# <center><img src="resources/idblock3_kiank.png" style="width:650px;height:150px;"></center>
#     <caption><center> <b>3개의 레이어를 건너뛰는 Bottleneck Identity block.</b> (Resnet50 이후) </center></caption>
# 
# 이 블록은 다음과 같은 단계로 구성됩니다:
# 1. 1x1 conv: 채널 수를 감소시킴.
# 2. 3x3 conv: 특징 추출
# 3. 1x1 conv: 채널 수를 다시 복원
# 4. Skip connection: 입력을 출력에 더해줌.
# 
# 이처럼 채널 수를 먼저 줄인 뒤 연산을 수행하고, 다시 채널 수를 원래대로 늘려주는 구조를 Bottleneck 구조라고 부릅니다.
# 
# <center><img src="resources/Basic Block_1.png" style="width:264px;">  <img src="resources/Bottleneck Block_2.png" style="width:300px;"></center>
# 
# - bottleneck 구조는 레이어 수는 더 많지만, 채널 수를 줄인 후 3x3 conv를 수행하기 때문에 연산량인 더 적고 효율적입니다.
# - 또한 비선형성(non-linearity)이 더 많아 표현력이 향상되며, 더 적은 연산량으로도 비슷하거나 더 좋은 성능을 달성합니다.
# 
# <mark>실습</mark> Bottleneck Identity block을 구현하세요
# 
# 1. main path의 첫번째 layer
#    - `Conv2d`: `intermediate_channels`개의 1x1 필터와 stride = 1, no zero padding ("valid" padding). BatchNorm을 수행할 것이므로 `bias = False`이다.
#    - `BatchNorm2d`: 'channels'축을 기준으로 정규화.
#    - `ReLU`
# 
# 2. main path의 두번쨰 layer
#    - `Conv2d`: `intermediate_channels`개의 3x3 필터와 stride = 1, "same" padding (출력 feature map의 크기가 입력과 동일), `bias = False`
#    - `BatchNorm2d`
#    - `ReLU`
# 
# 3. main path의 세번째 layer
#    - `Conv2d`: `intermediate_channels` x `expansion`개의 1x1 필터와 stride = 1, no zero padding, `bias = False`
#    - `BatchNorm2d`
#    - <b>NO</b> `ReLU` activation
# 
# 4. Skip connection (Shortcut path)
#    - main path의 결과와 입력값(`x`)을 더해준다 (element-wise addition)
#    - 그 후 `ReLU` activation을 적용한다.
# 
# 아래 documentation을 참고하세요
# - [Conv2D](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
# - [BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
# - [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)
# 

# In[3]:


class IdentityBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, intermediate_channels):
        super().__init__()

        ##### YOUR CODE START #####
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, bias=False)
        self.bn1   = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3   = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
       ##### YOUR CODE END #####

    def forward(self, x):
        ##### YOUR CODE START #####

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        ##### YOUR CODE END #####
        
        return out


# ### Convolutional Block
# 
# "convolutional block"은 ResNet에서 사용하는 두 번째 유형의 블록으로, <b>입력과 출력의 차원이 다를때</b> 사용하는 구조입니다.
# 
# indentity block과의 주요 차이점은 shortcut path에 `Conv2D` 레이어가 포함되어 있다는 점입니다.
# 
# <center><img src="resources/convblock_kiank.png" style="width:650px;height:150px;"></center>
# <caption><center> <b>Convolutional block</b> </center></caption>
# 
# * shortcut path의 `Conv2D`레이어는 입력 $x$의 차원을 main path의 출력 차원과 일치하도록록 변환하는 역할을 합니다다.
#   * 예를들어 입력의 높이(height)와 너비(width)를 1/2로 줄이고 싶다면 1x1 conv를 `stride = 2`로 적용합니다. 
#   * 이 `Conv2D` 레이어의 출력값에는 비선형 활성화 함수(non-linear activation)를 적용하지 않으며, 입력에 대한 단순한 선형 변환을 수행하는 것이 목적입니다.
# 
# <mark>실습</mark> Convolution Block을 구현하세요
# 
# 1. main path의 첫번째 레이어
#    - `Conv2d`: `intermediate_channels`개의 1x1 필터와 stride = 1, no zero padding, `bias = False`
#    - `BatchNorm2d`
#    - `ReLU` activation
# 
# 2. main path의 두번째 레이어
#    - `Conv2d`: `intermediate_channels`개의 3x3필터와 stride = stride, padding = 1, `bias = False`
#    - `BatchNorm2d`
#    - `ReLU` activation
# 
# 3. main path의 세번째 레이어
#    - `Conv2d`: `intermediate_channels` x `expansion`개의 1x1 필터와 stride = 1, no zero padding, `bias = False`
#    - `BatchNorm2d`
#    - **NO** ReLU activation 
# 
# 4. Shortcut path
#    - `Conv2d`: `intermediate_channels` x `expansion`개의 1x1 필터와 stride = stride, no zero padding, `bias = False`
#    - `BatchNorm2d`
# 
# 5. Final step: 
#    - main path의 출력과 shortcut path의 출력을 더합니다.
#    - 그 후 `ReLU` activation을 적용합니다.

# In[5]:


class ConvBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, intermediate_channels, stride):
        super().__init__()

        ##### YOUR CODE START #####

        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, bias=False)
        self.bn1   = nn.BatchNorm2d(intermediate_channels)
        
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(intermediate_channels)
        
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3   = nn.BatchNorm2d(intermediate_channels * self.expansion)
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(intermediate_channels * self.expansion)
        )
        
        self.relu = nn.ReLU(inplace=True)
        ##### YOUR CODE END #####

    def forward(self, x):
        ##### YOUR CODE START #####
        
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        identity = self.shortcut(x)
        
        out += identity
        out = self.relu(out)
        ##### YOUR CODE END #####

        return out


# ## Building the ResNet-50 Architecture
# 
# <center><img src="resources/resnet_kiank.png" style="width:850px;height:150px;"></center>
# <caption><center> <b>ResNet-50 architecture</b> </center></caption>
# - 위 그림은 ResNet-50의 전체 구조를 보여줍니다.
# - 여기서 "ID BLOCK"은 Identity block을, "ID BLOCK x3"은 identity block을 3번번 쌓는것을 의미합니다.
# 
# 
# <mark>실습</mark> 50개의 레이어가 있는 ResNet-50 아키텍쳐를 구현하세요.
# - Stage 1 (Stem):
#     - `Conv2D': 64개의 7x7 필터, stride = 2, padding = 3, bias=False, 입력은 RGB 이미지입니다.
#     - `BatchNorm2d`
#     - `ReLU` activation
#     - `MaxPool2d`: stride = 2, kernel_size = 3, padding = 1
# - Stage 2 (3 blocks):
#     - `ConvBlock`: intermediate_channels = 64, expansion = 4, stride = 1
#     - `IdentityBlock` x 2:  intermediate_channels = 64, expansion = 4
# - Stage 3 (4 blocks):
#     - `ConvBlock`: intermediate_channels = 128, expansion = 4, stride = 2
#     - `IdentityBlock` x 3: intermediate_channels = 128, expansion = 4
# - Stage 4 (6 blocks):
#     - `ConvBlock`: intermediate_channels = 256, expansion = 4, stride = 2
#     -`IdentityBlock` x 5:  intermediate_channels = 256, expansion = 4
# - Stage 5 (3 blocks):
#     - `ConvBlock`: intermediate_channels = 512, expansion = 4, stride = 2
#     - `IdentityBlock` x 2: intermediate_channels = 512, expansion = 4
# - `AdaptiveAvgPool2d`: 2D Average Pooling (output shape = Cx1x1)
# - `Flatten` layer
# - `Linear`: out_features = `num_classes`
# 
# 아래 문서를 참고하세요
# - [MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
# - [flatten](https://pytorch.org/docs/stable/generated/torch.flatten.html)
# - [AdaptiveAvgPool2d](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html)

# In[8]:


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()

                ##### YOUR CODE START #####
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = nn.Sequential(
            ConvBlock(64, 64, stride=1),
            IdentityBlock(256, 64),
            IdentityBlock(256, 64)
        )
        self.layer2 = nn.Sequential(
            ConvBlock(256, 128, stride=2),
            IdentityBlock(512, 128),
            IdentityBlock(512, 128),
            IdentityBlock(512, 128)
        )
        self.layer3 = nn.Sequential(
            ConvBlock(512, 256, stride=2),
            IdentityBlock(1024, 256),
            IdentityBlock(1024, 256),
            IdentityBlock(1024, 256),
            IdentityBlock(1024, 256),
            IdentityBlock(1024, 256)
        )
        self.layer4 = nn.Sequential(
            ConvBlock(1024, 512, stride=2),
            IdentityBlock(2048, 512),
            IdentityBlock(2048, 512)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, num_classes)
             ##### YOUR CODE END #####



    def forward(self, x):
        ##### YOUR CODE START #####
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        logits = self.fc(x)
        ##### YOUR CODE END #####

        return logits


# 직접 구현한 ResNet-50과 PyTorch에 미리 구현되어있는 ResNet-50의 파라미터 수가 같음을 확인해보자.

# <mark>실습</mark> 앞서 정의한 `model = ResNet50(num_classes = 1000)`모델에 입력 이미지 `X = torch.rand(4, 3, 224, 224)`를 통과시켰을때, 각 stage를 지난 후 출력 텐서의 shape을 조사하여 아래 변수들에 채워 넣으세요.
# - 각 shape은 튜플(tuple) 형태로 작성하세요.
# - 튜플의 각 원서(element)는 숫자 계산식으로 입력해도 괜찮으나, 파이썬 변수를 사용하지 마세요

# In[12]:


input_shape = (4, 3, 224, 224)
shape_after_stage1 = (4, 64, 56, 56)  # TODO: output shape after the stage1 (stem)
shape_after_stage2 = (4, 256, 56, 56)   # TODO: output shape after the stage2
shape_after_stage3 = (4, 512, 28, 28) # TODO: output shape after the stage3
shape_after_stage4 = (4, 1024, 14, 14) # TODO: output shape after the stage4
shape_after_stage5 = (4, 2048, 7, 7)  # TODO: output shape after the stage5
shape_after_avgpool = (4, 2048, 1, 1)  # TODO: output shape after the avgpool
shape_after_flatten = (4, 2048)   # TODO: output shape after the flatten
shape_after_fc = (4, 1000)      # TODO: output shape after the fc layer


# # Transfer learning

# In[13]:


import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision import models
from torch.utils.data import Subset

import numpy as np
from sklearn.model_selection import train_test_split

from train_utils import train_model, load_and_evaluate_model


# 전이 학습(Transfer Learning)은 대규모 데이터셋을 이용해 <b>사전 학습(pretraining)</b>된 모델을 기반으로,
# 보다 작은 규모의 타겟 데이터셋(target dataset)에 대해 파인튜닝(fine-tuning)을 수행함으로써, 적은 양의 데이터로도 높은 성능을 낼 수 있는 학습 방법입니다.
# 
# 이번 실습 시간에는 ImageNet 데이터셋(약 120만장의 이미지, 1,000 classes, ~140 GB)으로 사전학습된 ResNet-50모델을 이용하여 CIFAR-10 데이터셋을 학습해봅니다.
# 
# 이를 위해서는 먼저 이미지의 전처리(`transform`) 방식을 사전 학습된 모델과 일치하도록 맞추어 주어야 합니다.
#  - <b>정규화(standardization)</b>: ImageNet 데이터 학습시 사용한 평균/표준편차 값을 그대로 사용해야 합니다 : `mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]`
#  - <b>입력 이미지 크기 조정</b>: 사전학습된 모델은 `(224, 224)`크기의 이미지를 입력으로 받도록 설계되어 있으므로, CIFAR-10 데이터셋도 이 크기로 `Resize`합니다.

# In[14]:


def load_cifar10_datasets(data_root_dir):
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])  # pretraining에서 사용한 normalize로 수정

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # pretraining에서 사용된 이미지 사이즈로 수정
        transforms.ToTensor(),
        normalize,
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # pretraining에서 사용된 이미지 사이즈로 수정
        transforms.ToTensor(),
        normalize,
    ])
    
    
    train_dataset_full = datasets.CIFAR10(root=data_root_dir, train=True, download=False, transform=train_transforms)
    val_dataset_full = datasets.CIFAR10(root=data_root_dir, train=True, download=False, transform=eval_transforms)
    test_dataset = datasets.CIFAR10(root=data_root_dir, train=False, download=False, transform=eval_transforms)

    train_indices, val_indices = train_test_split(np.arange(len(train_dataset_full)), 
                                                  test_size = 0.2, random_state = 42)
    
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    train_dataset.classes = train_dataset_full.classes  # monkey‐patch class informations
    train_dataset.class_to_idx = train_dataset_full.class_to_idx

    return train_dataset, val_dataset, test_dataset


# ## 사전학습된 ResNet-50 모델 불러오기
# 
# PyTorch에서는 [링크](https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights)와 같이 다양한 사전학습 모델(pre-trained model)들을 제공합니다.
# 
# 아래 코드와 같이 `torchvision.models` 모듈을 통해 ImageNet 데이터셋으로 사전학습된 ResNet-50 모델을 불러올 수 있습니다.

# 먼저 ResNet-50 모델의 전체 구조를 확인해보겠습니다.

# ResNet-50은 여러 개의 서브 모듈(sub-module) 로 구성되어 있으며, `.` 연산자를 사용해 각 구성 요소에 접근할 수 있습니다.
# 
# 예를 들어, `model.layer1`과 같이 입력하면 서브 모듈 `layer1`에 접근할 수 있습니다.
#  - `layer1` ~ `layer4`는 우리가 앞서 앞서 직접 구현했던 ResNet-50 아키텍처의 Stage 2 ~ Stage 5에 해당합니다.

# ## Fine-turning을 위해 모델 수정하기
# 사전 학습된 ResNet-50 모델을 이용하여 CIFAR-10 데이터을 학습하기 위해서는, 다음과 같이 모델 구조를 일부 수정해야 합니다:
# 1. 마지막 `Linear` 레이어의 출력 차원(`out_features`)을 CIFAR-10 데이터셋의 클래스 수(num_classes = 10)와 일치하도록 수정
# 2. 학습 대상 파라미터 설정: 필요한 레이어만 학습하도록, 특정 레이어는 동결(freeze) 하고, 나머지는 학습 가능(trainable) 하도록 설정합니다.
# 
# ### 마지막 fc 레이어 수정하기
# 
# 사전 학습된 ResNet-50 모델의 마지막 fully connected 레이어(`fc`)를 다음과 같이 새로 정의할 수 있습니다:

# ## 학습 대상 파라미터 설정
# 
# 모델 파라미터의 `require_grad`값을 조정함으로써, 원하는 레이어의 파라미터들을 동결(freeze) 하거나 학습 가능(trainable) 하도록 설정할 수 있습니다.
# 
# 아래 예시는 마지막 `fc` 레이어만 학습가능하도록(`requires_grad = True`) 설정하는 방법을 보여줍니다.

# <mark>실습</mark> `get_model` 함수를 완성하세요.
# - `layer4`와 `fc`만 학습 가능하도록 설정하고, 나머지 파라미터들은 모두 동결(freeze)하세요.

# In[26]:


def get_model(config, num_classes):
    if config["model_architecture"] == "resnet50":
        print(f'Using pretrained model {config["pretrained"]}')
        model = models.resnet50(weights = config["pretrained"])

        ##### YOUR CODE START #####
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

        ##### YOUR CODE END #####
    else:
        raise ValueError(f"Unknown model architecture: {config['model_architecture']}")
    
    return model


# <mark>실습</mark> 아래 코드를 통해 사전 학습된 ResNet-50 모델을 사용해 CIFAR-10 데이터셋을 학습하고, 성능을 평가해보세요.

# In[28]:


config = {
    "mode": "train",  # Options: "train", "eval"
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    ## data and preprocessing settings
    "data_root_dir": '/datasets',
    "num_workers": 4,

    ## model settings
    "model_architecture": "resnet50",
    'pretrained' : 'IMAGENET1K_V2',

    ## Training Hyperparams
    "batch_size": 64,       # use 1.7GB of GPU memory
    "learning_rate": 1e-2,
    "num_epochs": 10,

    ## checkpoints
    "checkpoint_path": "checkpoints/checkpoint.pth",    # Path to save the most recent checkpoint
    "best_model_path": "checkpoints/best_model.pth",    # Path to save the best model checkpoint
    "checkpoint_save_interval": 10,                     # Save a checkpoint every N epochs
    "resume_training": None,    # Options: "latest", "best", or None

    ## WandB logging
    "wandb_project_name": "CIFAR10-experiments",
    "wandb_experiment_name" : "ResNet-50_pretrained",
    "dataset_name": "CIFAR10"
}

def main_train_resnet50(config):
    train_dataset, val_dataset, test_dataset = load_cifar10_datasets(config["data_root_dir"])
    num_classes = len(train_dataset.classes)

    model = get_model(config, num_classes = num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    print(f"Using {config['device']} device")
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())} ({sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable)")
    
    if config["mode"] == "train":
        val_accuracy = train_model(model, train_dataset, val_dataset, criterion, optimizer, scheduler, config)
        return val_accuracy
    elif config["mode"] == "eval":
        test_accuracy = load_and_evaluate_model(model, test_dataset, criterion, optimizer, scheduler, config)
        return test_accuracy
    else:
        raise ValueError(f"Unknown mode: {config['mode']}")


# #### Lab을 마무리 짓기 전 저장된 checkpoint를 모두 지워 저장공간을 확보합니다.

# In[ ]:




