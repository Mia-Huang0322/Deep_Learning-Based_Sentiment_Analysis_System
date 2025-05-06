import torch.nn as nn
import torchvision.models as models

class ResNetModel(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNetModel, self).__init__()
        # 加载预训练的 ResNet50
        self.resnet = models.resnet18(pretrained=False)
        # 修改全连接层输出为7类情感
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


from django.db import models
from django.contrib.auth.models import User


# 功能使用记录
class FeatureUsage(models.Model):
    FEATURE_CHOICES = [
        ('text', 'Text Sentiment'),
        ('visual', 'Visual Sentiment'),
        ('combined', 'Combined Sentiment'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    feature_name = models.CharField(max_length=20, choices=FEATURE_CHOICES)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.feature_name} - {self.timestamp}"


# 页面访问记录
class PageView(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    page_name = models.CharField(max_length=100)
    timestamp = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)

    def __str__(self):
        return f"{self.page_name} - {self.timestamp}"
