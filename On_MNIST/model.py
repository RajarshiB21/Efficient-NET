from math import ceil
import torch
import torch.nn as nn




############# Efficient Net - B0 Baseline Network ##################
#  Stage      Operator              Resolution    Channels   Layers
#   i           F(i)                H(i)xW(i)      C(i)       L(i)
#   1          Conv3x3              224x224         32          1
#   2       MBConv 1, k3,3          112x112         16          1
#   3       MBConv 6, k3,3          112x112         24          2
#   4       MBConv 6, k5,5          56x56           40          2
#   5       MBConv 6, k3,3          28x28           80          3
#   6       MBConv 6, k5,5          14x14           112         3
#   7       MBConv 6, k5,5          14x14           192         4
#   8       MBConv 6, k3,3          7x7             320         1
#   9   Conv1x1 & Pooling & FC      7x7             1280        1

####################################################################

#list of lists, values are taken from the above structure
#Expand ratio is found next to the MBConv under Operator

base_model = [
    # expand_ration, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

#Phi Values for every efficient net model
#alpha,beta,gamma, depth=alpha**phi
#Dictionary of tuples
#Tuple of: (phi_value, resolution, drop_rate)

phi_values ={
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


class CNNBlock(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups, #For depth wise convolution
            #if we set group = 1 as done by default then it is a normal conv,
            #if we set it to groups = in_channels, then it is a depthwise conv,
            #in depth wise conv, the kernel convolves over each channel independently
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU() #Used in the paper instead of the ReLU

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))


#To compute attention scores for each channels
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # if CxHxW -> Cx1x1 since we want only 1 value as specified
            nn.Conv2d(in_channels, reduced_dim, 1),#Reduces number of channels
            nn.SiLU(),#Passes through activation function
            nn.Conv2d(reduced_dim, in_channels, 1),#Brings it back
            nn.Sigmoid()#Passes through sigmoid and gives us a value of 0 and 1
        )

    def forward(self, x):
        return x * self.se(x)#elementwise multiplication of the attention scores


#Takes an input and expands it to a higher number of channels
#Uses a depth wise conv
#Then brings it back to org number of channels
#Reduction is for the reduced dimensionality in the squeeze and excitation
#4 as in reducing it to 1/4th of the dimension that comes in
#survival_prob is stochastic depth
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio, reduction=4, survival_prob=0.8):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = 0.8
        #If we downsample then we cant use a skip connection since the height and width wont match
        #Similarly if the in_channels and out_channels do not match then we cant use a skip_connection
        self.use_residual = in_channels == out_channels and stride ==1
        hidden_dim = in_channels * expand_ratio
        #We expand in between the stages
        #check structure on top
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels/reduction)

        #If we are going to expand then we call the cnnblock method and pass in the inchannels and
        #hidden_dim which expands the in_channels to some hidden_dimensions
        if self.expand:

            self.expand_conv = CNNBlock(
                in_channels, hidden_dim, kernel_size = 3, stride=1, padding=1
            )

            #Then we use the inverted residual block
            #Depth wise conv
        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim,
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    #Randomly drops the layers during training
    #Not testing
    def stochastic_depth(self, x):
        if not self.training:
            return x

        device = "cuda"
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device= x.device) < self.survival_prob
        # Computes a value of 0 or 1 for each example
        return torch.div(x, self.survival_prob) * binary_tensor
        #The above line is from the stochastic depth paper
        #To preserve the mean and standard deviation during testing

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        #We are sending it through a conv layer and then use the stochastic depth
        #Where we remove the output from some of them
        #To make sure we dont lose all of the information we add the residual connection
        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs

        #if we dont use residual connection that is if we downsampled or if in_channels out_channels are not equal
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    def __init__(self, version, num_classes, in_channels):
        super(EfficientNet, self).__init__()

        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280*width_factor)
        self.in_channels = in_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),#model dependent
            nn.Linear(last_channels, num_classes),
        )

    #alpha for depth scaling or how many layers to increase
    #beta for width scaling or how many channels to increase
    #We already have resolution so we wont use gamma
    #alpha and beta values are from the paper
    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta **phi
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        #########***Change channels near CNNBlock for different datasets***#############
        features = [CNNBlock(self.in_channels, channels, 3, stride=2, padding=1)]
        in_channels = channels

        #Now we iterate through all stages in the base model
        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4*ceil(int(channels * width_factor)/4)
            #for some cases in the squeeze excitation we have a reduced dimension
            #so it has to modulas of 4 or reduction_dim
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride = stride if layer == 0 else 1,
                        #Since we downsample at one of the layers for every 2 steps
                        #Check structure above
                        kernel_size= kernel_size,
                        padding = kernel_size//2, #if k=1:pad=0, k=3:pad=1
                    )
                )

                in_channels = out_channels

        features.append(
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )

        return nn.Sequential(*features)


    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))
        #Since we need to flatten before we run it through the linear layers


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    version = "b0"
    phi, res, drop_rate = phi_values[version]
    num_examples, num_classes = 4,10
    x = torch.randn((num_examples, 3, res, res)).to(device)
    model = EfficientNet(
        version=version,
        num_classes=num_classes,
    ).to(device)

    #print(model(x).shape)

#if __name__ == "__main__":
    #test()

#######################################################################################################################

