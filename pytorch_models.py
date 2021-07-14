import torch 
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class FilterResponseNorm_layer(nn.Module):
    def __init__(self, num_filters, eps=1e-6):
        super(FilterResponseNorm_layer, self).__init__()
        self.eps = eps
        par_shape = (1, num_filters, 1, 1)  # [1,C,1,1]
        self.tau = torch.nn.Parameter(torch.zeros(par_shape))  #self.register_parameter("tau", torch.zeros(par_shape)) 
        self.beta = torch.nn.Parameter(torch.zeros(par_shape))  #self.register_parameter("beta", torch.zeros(par_shape)) 
        self.gamma = torch.nn.Parameter(torch.ones(par_shape)) #self.register_parameter("gamma", torch.ones(par_shape))

    def forward(self, x):

        nu2 = torch.mean(torch.square(x), dim=[2, 3], keepdim=True)
        x = x * 1 / torch.sqrt(nu2 + self.eps)
        y = self.gamma * x + self.beta
        z = torch.max(y, self.tau)
        return z

class resnet_block(nn.Module):
    def __init__(self, normalization_layer, input_size, num_filters, kernel_size=3, strides=1,
      activation=torch.nn.Identity, use_bias=True):
        super(resnet_block, self).__init__()
        # input size = C, H, W
        p0 = int(strides * (input_size[2] - 1) + kernel_size - input_size[2])
        if p0%2 == 0:
            p0 /= 2
            p2 = p0
        else:
            p2 = (p0+1)/2
            p0 = (p0-1)/2
        # height padding
        p1 = strides * (input_size[1] - 1) + kernel_size - input_size[1]
        if p1%2 == 0:
            p1 /= 2
            p3 = p1
        else:
            p3 = (p1+1)/2
            p1 = (p1-1)/2
        self.pad1 = torch.nn.ZeroPad2d((int(p0), int(p1), int(p2), int(p3)))
        self.conv1 = torch.nn.Conv2d(input_size[0], num_filters,
                                     kernel_size=kernel_size, stride=strides, padding=0,
                                     bias=use_bias)
        self.norm1 = normalization_layer(num_filters)
        self.activation1 = activation()


    def forward(self, x):

        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.activation1(out)

        return out
        

class stacked_resnet_block(nn.Module):
      def __init__(self, normalization_layer, num_filters, input_num_filters,
                 stack, res_block, activation, use_bias):
        super(stacked_resnet_block, self).__init__()
        self.stack = stack
        self.res_block = res_block
        strides = 1
        if stack > 0 and res_block == 0:  # first layer but not first stack
          strides = 2  # downsample
        self.res1 = resnet_block(normalization_layer=normalization_layer, num_filters=num_filters,
                            input_size=(input_num_filters,32,32), strides=strides,
                            activation=activation, use_bias=use_bias)
        self.res2 = resnet_block(normalization_layer=normalization_layer, num_filters=num_filters,
                            input_size=(num_filters,32,32), use_bias=use_bias)
        if stack > 0 and res_block == 0:  # first layer but not first stack
          # linear projection residual shortcut connection to match changed dims
          self.res3 = resnet_block(normalization_layer=normalization_layer, num_filters=num_filters,
                            input_size=(input_num_filters,32,32), strides=strides, kernel_size = 1,
                            use_bias=use_bias)
          
        self.activation1 = activation()

      def forward(self, x):
        y = self.res1(x)
        y = self.res2(y)
        if self.stack > 0 and self.res_block == 0:
          x = self.res3(x)
        

        out = self.activation1(x+y)
        return out

class make_resnet_fn(nn.Module):
    def __init__(self, num_classes, depth, normalization_layer,
                 width=16, use_bias=True, activation=torch.nn.ReLU(inplace=True)):
        super(make_resnet_fn,self).__init__()
        self.num_res_blocks = (depth - 2) // 6
        self.normalization_layer = normalization_layer
        self.activation = activation
        self.use_bias = use_bias
        self.width = width
        if (depth - 2) % 6 != 0:
            raise ValueError('depth must be 6n+2 (e.g. 20, 32, 44).')

        # first res_layer
        self.layer1 = resnet_block(normalization_layer=normalization_layer, num_filters=width,
                                   input_size=(3,32,32), kernel_size=3, strides=1,
                                   activation=torch.nn.Identity, use_bias=True)
        # stacks
        self.stacks = self._make_res_block()
        # avg pooling
        self.avgpool1 = torch.nn.AvgPool2d(kernel_size=(8, 8), stride=8, padding=0)
        # linear layer 
        self.linear1 = nn.Linear(768, num_classes)
    
    def forward(self, x):
      # first res_layer
      out = self.layer1(x) # shape out torch.Size([5, 16, 32, 32])
      out = self.stacks(out)
      out = self.avgpool1(out)
      out = torch.flatten(out,start_dim=1)
      logits = self.linear1(out)
      return logits

    def _make_res_block(self):
      layers = list()
      num_filters = self.width
      input_num_filters = num_filters
      for stack in range(3):
        for res_block in range(self.num_res_blocks):
          layers.append(stacked_resnet_block(self.normalization_layer, num_filters, input_num_filters,
                 stack, res_block, self.activation, self.use_bias))
          input_num_filters = num_filters
        num_filters *= 2
      return nn.Sequential(*layers)

def make_resnet20_frn_fn(data_info, activation=torch.nn.ReLU):
    num_classes = data_info["num_classes"]
    return make_resnet_fn(
      num_classes, depth=20, normalization_layer=FilterResponseNorm_layer,
      activation=activation)

# pytorch version
def get_model(model_name, data_info, **kwargs):
  _MODEL_FNS = {
    "resnet20_frn": make_resnet20_frn_fn,
    "resnet20_frn_swish": partial(
      make_resnet20_frn_fn, activation=torch.nn.SiLU),
  }
  net_fn = _MODEL_FNS[model_name](data_info, **kwargs)
  return net_fn