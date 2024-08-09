import torch
import torch.nn as nn
import torchvision.models as models

class VGG16FineTune(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG16FineTune, self).__init__()
        # Load the pretrained VGG16 model
        self.vgg16 = models.vgg16(pretrained=True)
        
        # Freeze the early layers (Optional: depending on your task)
        for param in self.vgg16.features.parameters():
            param.requires_grad = False
        
        # Replace the classifier part to match the number of classes
        self.vgg16.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.vgg16(x)
        return x

# import torch.nn as nn
# import torch.nn.functional as F

# class CNNModel(nn.Module):
#     def __init__(self):
#         super(CNNModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.fc1 = nn.Linear(128 * 16 * 16, 512)
#         self.fc2 = nn.Linear(512, 2)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(-1, 128 * 16 * 16)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x