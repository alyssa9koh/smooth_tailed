# Neural network implementation for ResNet 18
# for a visual walkthrough of this neural network, see: https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8
# code from: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
class BasicBlock(nn.Module):
  # the basic block implements a basic residual block
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample = None,
            groups: int = 1,
            base_width = 64,
            dilation = 1,
            norm_layer = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
      # the actual residual part!
        out += identity
        out = self.relu(out)

        return out