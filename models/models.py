import torch
from torch import nn
import math
from torch.autograd import Function
from torch.nn.utils import weight_norm
import torch.nn.functional as F

# from utils import weights_init

def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]


##################################################
##########  BACKBONE NETWORKS  ###################
##################################################

###################### LITEMV ######################
class Hybrid_block(nn.Module):
    def __init__(self, input_channels, keep_track, kernel_sizes=[2, 4, 8, 16, 32, 64]):
        super(Hybrid_block, self).__init__()
        self.keep_track = keep_track

        self.hybrid_block = nn.ModuleList()

        for kernel_size in kernel_sizes:
            # Depthwise convolution
            depthwise = nn.Conv1d(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=kernel_size,
                padding='same',
                groups=input_channels,  # ← depthwise
                bias=False
            )

            # Initialize alternating sign filters manually (optional)
            with torch.no_grad():
                filter_ = torch.ones((input_channels, 1, kernel_size))
                indices_ = torch.arange(kernel_size)
                filter_[:, :, indices_ % 2 == 0] *= -1
                depthwise.weight = nn.Parameter(filter_, requires_grad=False)

            # Pointwise convolution to mix features
            pointwise = nn.Conv1d(
                in_channels=input_channels,
                out_channels=1,
                kernel_size=1,
                bias=False
            )

            # Stack the two in a Sequential block
            separable_conv = nn.Sequential(depthwise, pointwise)
            self.hybrid_block.append(separable_conv)

            self.keep_track += 1

        for kernel_size in kernel_sizes:
            # Create alternating pattern filter (same as your logic)
            filter_ = torch.ones((input_channels, 1, kernel_size))
            indices_ = torch.arange(kernel_size)
            filter_[:, :, indices_ % 2 > 0] *= -1

            # Depthwise convolution (1 conv per channel)
            depthwise = nn.Conv1d(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=kernel_size,
                padding='same',
                groups=input_channels,
                bias=False
            )
            with torch.no_grad():
                depthwise.weight = nn.Parameter(filter_, requires_grad=False)

            # Pointwise convolution (to mix the features)
            pointwise = nn.Conv1d(
                in_channels=input_channels,
                out_channels=1,  # collapse to 1 channel per kernel size
                kernel_size=1,
                bias=False
            )
            # Freeze pointwise weights too (optional)
            with torch.no_grad():
                pointwise.weight = nn.Parameter(
                    torch.ones_like(pointwise.weight), requires_grad=False
                )

            # Combine into sequential block
            separable_conv = nn.Sequential(depthwise, pointwise)
            self.hybrid_block.append(separable_conv)

            self.keep_track += 1


        for kernel_size in kernel_sizes[1:]:
            total_kernel_size = kernel_size + kernel_size // 2

            # Shape: (total_kernel_size, input_channels, 1) → will permute later to (input_channels, 1, total_kernel_size)
            filter_ = torch.zeros((total_kernel_size, input_channels, 1))

            xmash = torch.linspace(start=0, end=1, steps=kernel_size // 4 + 1)[1:].reshape((-1, 1, 1))

            filter_left = xmash**2
            filter_right = filter_left.flip(0)

            filter_[0 : kernel_size // 4] = -filter_left
            filter_[kernel_size // 4 : kernel_size // 2] = -filter_right
            filter_[kernel_size // 2 : 3 * kernel_size // 4] = 2 * filter_left
            filter_[3 * kernel_size // 4 : kernel_size] = 2 * filter_right
            filter_[kernel_size : 5 * kernel_size // 4] = -filter_left
            filter_[5 * kernel_size // 4 :] = -filter_right

            # → filter_ shape becomes (input_channels, 1, total_kernel_size)
            filter_ = filter_.permute(1, 2, 0)  # (input_channels, 1, kernel_size)

            # Depthwise convolution: one filter per input channel
            depthwise = nn.Conv1d(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=total_kernel_size,
                padding='same',
                groups=input_channels,
                bias=False
            )
            with torch.no_grad():
                depthwise.weight = nn.Parameter(filter_, requires_grad=False)

            # Pointwise convolution to mix channels to 1 output
            pointwise = nn.Conv1d(
                in_channels=input_channels,
                out_channels=1,
                kernel_size=1,
                bias=False
            )
            with torch.no_grad():
                pointwise.weight = nn.Parameter(
                    torch.ones_like(pointwise.weight), requires_grad=False
                )

            # Combine depthwise + pointwise into sequential
            separable_conv = nn.Sequential(depthwise, pointwise)
            self.hybrid_block.append(separable_conv)

            self.keep_track += 1
            
        self.relu = nn.ReLU()

    def forward(self, x):
        
        conv_outputs = [conv(x) for conv in self.hybrid_block]
        
        x = torch.cat(conv_outputs, dim=1)
        x = self.relu(x)

        return x

class Inception_block(nn.Module):
    
    def __init__(
              self, 
              in_channels,
              n_filters,
              kernel_size,
              dilation_rate=1, 
              stride=1,
              keep_track=0,
              use_hybrid_layer=True,
              use_multiplexing=True):
        super(Inception_block, self).__init__()

        self.use_hybrid_layer = use_hybrid_layer

        n_convs = 3

        kernel_size_s = [kernel_size // (2**i) for i in range(n_convs)]

        self.inception_layers  = nn.ModuleList()

        for i in range(len(kernel_size_s)):
            if n_filters[i] != 0:
                self.inception_layers.append(nn.Conv1d(
                                                in_channels=in_channels,
                                                out_channels=n_filters[i],
                                                kernel_size=kernel_size_s[i],
                                                stride=stride,
                                                padding='same',
                                                dilation=dilation_rate,                                            
                                                bias=False
                                                ))

        self.hybrid = Hybrid_block(input_channels=in_channels, keep_track=keep_track)
        
            
        self.bn = nn.BatchNorm1d(sum(n_filters) +17)
        self.relu = nn.ReLU()

    def forward(self, x):
        input = x
        inception_outputs = []
        for conv_layer in self.inception_layers:
            inception_outputs.append(conv_layer(x))
        
        x = torch.cat(inception_outputs, 1)
        if self.use_hybrid_layer:
            h = self.hybrid(input)
            # print('x dimension: ', x.shape)            
            # print('h dimension: ', h.shape)
            x = torch.cat([x, h], 1)
        
        x = self.bn(x)
        x = self.relu(x)

        return x

class FCN_block(nn.Module):
    
    def __init__(
            self,
            in_channels,
            kernel_size,
            n_filters,
            dilation_rate,
            stride=1,
    ):
        super(FCN_block, self).__init__()
        if in_channels != 0:
            self.depthwise_conv = nn.Conv1d(
                                    in_channels=in_channels, 
                                    out_channels=in_channels, 
                                    kernel_size=kernel_size, 
                                    stride=stride,
                                    padding='same', 
                                    dilation=dilation_rate, 
                                    groups=in_channels, 
                                    bias=False)

        if n_filters != 0:
            self.pointwise_conv = nn.Conv1d(
                                    in_channels=in_channels, 
                                    out_channels=n_filters, 
                                    kernel_size=1, 
                                    bias=False
                                    )

        self.bn = nn.BatchNorm1d(n_filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply depthwise convolution
        depth_out = self.depthwise_conv(x)
        # Apply pointwise convolution
        x = self.pointwise_conv(depth_out)

        x = self.bn(x)
        x = self.relu(x)

        return x, depth_out

class LITEMV(nn.Module):

    def __init__(
            self,
            configs,
            # config,
            batch_size=64,
            n_filters=[[32, 32, 32], 32, 32],
            kernel_size=41,
            n_epochs=1500,
            verbose=True,
            use_custom_filters=True,
            use_dialtion=True,
            use_multiplexing=True):
        
        super(LITEMV, self).__init__()
        # channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]

        # self.length_TS = seq_len
        self.n_classes = configs.num_classes
        # self.channel_size = channel_size
        self.verbose = verbose
        self.n_filters = n_filters
        self.use_custom_filters = use_custom_filters
        self.use_dilation = use_dialtion
        self.use_multiplexing = use_multiplexing
        self.kernel_size = kernel_size - 1
        self.batch_size = batch_size 
        self.n_epochs = n_epochs

        self.keep_track = 0
        # self.input_shape = (self.length_TS)
        self.inception = Inception_block(in_channels=configs.input_channels, n_filters=self.n_filters[0], kernel_size=self.kernel_size, 
                                         dilation_rate=1, keep_track=self.keep_track, use_hybrid_layer=self.use_custom_filters)
        
        self.kernel_size //= 2

        self.fcn_module1 = FCN_block(in_channels=sum(self.n_filters[0]) + 17, kernel_size=self.kernel_size // (2**0), n_filters=self.n_filters[1], dilation_rate=2)
        self.fcn_module2 = FCN_block(in_channels=self.n_filters[1], kernel_size=self.kernel_size // (2**1), n_filters=self.n_filters[2], dilation_rate=4)        

        self.avgpool1 = nn.AdaptiveAvgPool1d(configs.features_len)

    
    def forward(self, x):
        
        x = self.inception(x)
        
        x, _ = self.fcn_module1(x)
        
        x, _ = self.fcn_module2(x)

        x = self.avgpool1(x)
        
        x = torch.flatten(x, start_dim=1)

        return x
    
########## CNN #############################
class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)

        x_flat = x.reshape(x.shape[0], -1)
        return x_flat

class classifier(nn.Module):
    def __init__(self, configs):
        super(classifier, self).__init__()
        self.logits = nn.Linear(configs.features_len * configs.final_out_channels, configs.num_classes)
        self.configs = configs

    def forward(self, x):

        predictions = self.logits(x)

        return predictions


##################################################
##########  OTHER NETWORKS  ######################
##################################################

class codats_classifier(nn.Module):
    def __init__(self, configs):
        super(codats_classifier, self).__init__()
        model_output_dim = configs.features_len
        self.hidden_dim = configs.hidden_dim
        self.logits = nn.Sequential(
            nn.Linear(model_output_dim * configs.final_out_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, configs.num_classes))

    def forward(self, x_in):
        predictions = self.logits(x_in)
        return predictions

class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, configs):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, 2)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


#### Codes required by DANN ##############
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

#### Codes required by CDAN ##############
class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


class Discriminator_CDAN(nn.Module):
    """Discriminator model for CDAN ."""

    def __init__(self, configs):
        """Init discriminator."""
        super(Discriminator_CDAN, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels * configs.num_classes, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, 2)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out

