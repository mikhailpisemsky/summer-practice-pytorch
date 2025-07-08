import torch
import torch.nn as nn

class CustomConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, custom_param=1.0):
        super().__init__()
        self.custom_param = custom_param
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)

    def forward(self, x):
        conv_output = self.conv(x)
        output_with_logic = conv_output * self.custom_param
        return output_with_logic

class CNNWithCustomConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.custom_conv = CustomConv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1, custom_param=1.5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.custom_conv(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CNNWithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.se1 = Attention(channels=16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.se2 = Attention(channels=32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.se1(self.relu1(self.conv1(x))))
        x = self.pool2(self.se2(self.relu2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return x

class CustomActivation(nn.Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return x * torch.sigmoid(x) + self.alpha * F.relu(x)
    
class CNNWithCustomActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.custom_activation1 = CustomActivation(alpha=0.02)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.custom_activation2 = CustomActivation(alpha=0.02)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.custom_activation1(self.conv1(x)))
        x = self.pool2(self.custom_activation2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.custom_activation2(self.fc1(x))
        x = self.fc2(x)
        return x

class Pooling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, p):
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.p = p

        output = F.unfold(input, kernel_size=kernel_size, stride=stride)
        output = output.view(input.size(0), input.size(1), kernel_size*kernel_size, -1)

        abs_output_pow_p = torch.pow(torch.abs(output) + 1e-8, p)
        sum_abs_output_pow_p = torch.sum(abs_output_pow_p, dim=2)

        lp_norm_output = torch.where(sum_abs_output_pow_p > 1e-12,
                                     torch.pow(sum_abs_output_pow_p, 1.0 / p),
                                     torch.zeros_like(sum_abs_output_pow_p))

        out_height = (input.size(2) - kernel_size) // stride + 1
        out_width = (input.size(3) - kernel_size) // stride + 1
        output_final = lp_norm_output.view(input.size(0), input.size(1), out_height, out_width)

        ctx.save_for_backward(input, output_final)

        return output_final

    @staticmethod
    def backward(ctx, grad_output):
        input, output_final = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        p = ctx.p

        grad_input = torch.zeros_like(input)

        grad_output_unfolded = grad_output.view(grad_output.size(0), grad_output.size(1), -1)
        input_unfolded = F.unfold(input, kernel_size=kernel_size, stride=stride)
        input_unfolded_view = input_unfolded.view(input.size(0), input.size(1), kernel_size * kernel_size, -1)
        lp_norm_output_expanded = output_final.view(input.size(0), input.size(1), 1, -1).expand_as(input_unfolded_view)

        if p == 1:
            grad_local_unfolded = torch.sign(input_unfolded_view)
        elif p == 2:
            grad_local_unfolded = input_unfolded_view / (lp_norm_output_expanded + 1e-8)
        else:
            abs_input_unfolded_pow_p_minus_2 = torch.where(input_unfolded_view != 0,
                                                           torch.pow(torch.abs(input_unfolded_view), p - 2),
                                                           torch.zeros_like(input_unfolded_view))
            numerator = input_unfolded_view * abs_input_unfolded_pow_p_minus_2
            denominator = torch.pow(lp_norm_output_expanded + 1e-8, p - 1)
            grad_local_unfolded = numerator / denominator

        grad_output_expanded = grad_output_unfolded.unsqueeze(2).expand_as(input_unfolded_view)
        grad_unfolded = grad_local_unfolded * grad_output_expanded

        grad_input = F.fold(grad_unfolded.view(grad_unfolded.size(0), -1, grad_unfolded.size(3)),
                             output_size=(input.size(2), input.size(3)),
                             kernel_size=kernel_size,
                             stride=stride)

        return grad_input, None, None, None

class CustomLpPooling(nn.Module):
    def __init__(self, kernel_size, stride=None, p=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.p = p

    def forward(self, x):
        return Pooling.apply(x, self.kernel_size, self.stride, self.p)

class CNNWithCustomPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.lp_pool1 = CustomLpPooling(kernel_size=2, stride=2, p=2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.lp_pool2 = CustomLpPooling(kernel_size=2, stride=2, p=2)

        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.lp_pool1(self.relu1(self.conv1(x)))
        x = self.lp_pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x

class BottleneckResidualBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.shortcut = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.net(x)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        return self.relu(out)

class CNNWithBottleneckResidual(nn.Module):
    def __init__(self, num_blocks=[3, 4, 6, 3], num_classes=10, base_channels=64):
        super().__init__()
        self.in_channels = base_channels
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BottleneckResidualBlock, base_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BottleneckResidualBlock, base_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BottleneckResidualBlock, base_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BottleneckResidualBlock, base_channels * 8, num_blocks[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8 * BottleneckResidualBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class WideResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.net(x)
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out += identity
        return self.relu(out)

class CNNWithWideResidual(nn.Module):
    def __init__(self, depth, width_factor=2, dropout_rate=0.0, num_classes=10):
        super().__init__()
        assert (depth - 4) % 6 == 0,
        n = (depth - 4) // 6

        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(WideResidualBlock, 16 * width_factor, n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(WideResidualBlock, 32 * width_factor, n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(WideResidualBlock, 64 * width_factor, n, stride=2, dropout_rate=dropout_rate)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * width_factor * WideResidualBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, dropout_rate):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout_rate))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=1, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
