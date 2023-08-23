# actual ResNet class implementation
class ResNet18(nn.Module):
    def __init__(
            self,
            block, # what kind of block to use. Small ResNets only use basic blocks
            layers, # number of BasicBlocks to make in each layer of ResNet
            num_classes: int = 63,
            groups: int = 1,
            supervised = False):
        super().__init__()

        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
       # magic numbers to match the matrix sizes for matrix multiplication
        self.inplanes = 64
        self.dilation = 1
        # for ResNet18 and BasicBlock, the group and base_width must always be these vals
        self.groups = 1
        self.base_width = 64
        # whether or not to use the final linear layer for classification or just embed the image
        self.supervised = supervised
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # at some point, you may want to do the math so that your feature
        # extraction layer has > 512 features in it. In the past, I've found
        # 2048 to be a better final layer size.
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

      # initialize the model weights to be small real values drawn from a Z normal distribution
      # turns out starting all the parameter values at 0 can lead to either untrainable or unstable NNs
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                nn.init.constant_(m.bn2.weight, 0)

    # sets up all the gross math so the weight matrices line up correctly
    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.supervised:
          x = self.fc(x)

        return x

    def prep_finetuning(self):
      # gets the model ready for finetuning
      # of the last layer
      self.supervised = True
      # turn off other model parameters
      # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#set-model-parameters-requires-grad-attribute
      for param in model.parameters():
          param.requires_grad = False
      # turn back on last linear layer parameters
      for param in model.fc.parameters():
        param.requires_grad = True

      params_to_update = []
      for name,param in self.named_parameters():
          if param.requires_grad == True:
              params_to_update.append(param)
      return params_to_update