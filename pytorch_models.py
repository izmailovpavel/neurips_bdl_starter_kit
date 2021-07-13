import torch 
import torch.nn as nn
import torch.nn.functional as F

# import haiku as hk
# import jax
# import jax.numpy as jnp
# import functools

# # jax version
# def _resnet_layer( inputs, num_filters, normalization_layer, kernel_size=3, strides=1,
#     activation=lambda x: x, use_bias=True, is_training=False):
#     x = inputs
#     x = hk.Conv2D(
#       num_filters, kernel_size, stride=strides, padding='same',
#       w_init=he_normal, with_bias=use_bias)(x)
#     x = normalization_layer()(x, is_training=is_training)
#     x = activation(x)
#     return x
    
# pytorch version
def _resnet_layer( 
    inputs, num_filters, normalization_layer, kernel_size=3, strides=1,
    activation=lambda x: x, use_bias=True, is_training=False):
     
    x = inputs
    # same padding => input size = output size
    input_size, output_size = inputs.shape, inputs.shape
    # padding_left, padding_right, padding_top, \padding_bottom)
    # input shape should be Input: (N, C, H_{in}, W_{in}) 
    # width padding
    p0 = int(strides * (output_size[3] - 1) + kernel_size - input_size[3])
    if p0%2 == 0:
        p0 /= 2
        p2 = p0
    else:
        p2 = (p0+1)/2
        p0 = (p0-1)/2
    # height padding
    p1 = strides * (output_size[2] - 1) + kernel_size - output_size[2]
    if p1%2 == 0:
        p1 /= 2
        p3 = p1
    else:
        p3 = (p1+1)/2
        p1 = (p1-1)/2
    # padding layer
    x = torch.nn.ZeroPad2d((int(p0), int(p1), int(p2), int(p3)))(x)
    
    # conv layer
    x = torch.nn.Conv2d(input_size[1], num_filters,
                    kernel_size=kernel_size, stride=strides, padding=0,
                   bias=use_bias)(x)
    x = normalization_layer(inputs.shape[1])(x)
    x = activation(x)
    return x

# # jax version
# def make_resnet_fn(
#     num_classes: int,
#     depth: int,
#     normalization_layer,
#     width: int = 16,
#     use_bias: bool = True,
#     activation=jax.nn.relu,
# ):
#     num_res_blocks = (depth - 2) // 6
#     if (depth - 2) % 6 != 0:
#         raise ValueError('depth must be 6n+2 (e.g. 20, 32, 44).')
  
#     def forward(x, is_training):
#         num_filters = width
#         x = _resnet_layer(
#             x, num_filters=num_filters, activation=activation, use_bias=use_bias,
#             normalization_layer=normalization_layer
#         )
#         for stack in range(3):
#             for res_block in range(num_res_blocks):
#                 strides = 1
#                 if stack > 0 and res_block == 0:  # first layer but not first stack
#                     strides = 2  # downsample
#                 y = _resnet_layer(
#                     x, num_filters=num_filters, strides=strides, activation=activation,
#                     use_bias=use_bias, is_training=is_training,
#                     normalization_layer=normalization_layer)
#                 y = _resnet_layer(
#                     y, num_filters=num_filters, use_bias=use_bias,
#                     is_training=is_training, normalization_layer=normalization_layer)
#                 if stack > 0 and res_block == 0:  # first layer but not first stack
#                   # linear projection residual shortcut connection to match changed dims
#                   x = _resnet_layer(
#                       x, num_filters=num_filters, kernel_size=1, strides=strides,
#                       use_bias=use_bias, is_training=is_training,
#                       normalization_layer=normalization_layer)
#                 x = activation(x + y)
#             num_filters *= 2
#         x = hk.AvgPool((8, 8, 1), 8, 'VALID')(x)
#         x = hk.Flatten()(x)
#         logits = hk.Linear(num_classes, w_init=he_normal)(x)
#         return logits
#     return forward

# pytorch version
def make_resnet_fn(
    num_classes: int,
    depth: int,
    normalization_layer,
    width: int = 16,
    use_bias: bool = True,
    activation=torch.nn.ReLU()):
    
    num_res_blocks = (depth - 2) // 6
    if (depth - 2) % 6 != 0:
        raise ValueError('depth must be 6n+2 (e.g. 20, 32, 44).')
    
    def forward(x):
        num_filters = width
        x = _resnet_layer(
            x, num_filters=num_filters, activation=activation, use_bias=use_bias,
            normalization_layer=normalization_layer
        )
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = _resnet_layer(
                    x, num_filters=num_filters, strides=strides, activation=activation,
                    use_bias=use_bias,
                    normalization_layer=normalization_layer)
                y = _resnet_layer(
                    y, num_filters=num_filters, use_bias=use_bias,
                    normalization_layer=normalization_layer)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                  # linear projection residual shortcut connection to match changed dims
                  x = _resnet_layer(
                      x, num_filters=num_filters, kernel_size=1, strides=strides,
                      use_bias=use_bias,
                      normalization_layer=normalization_layer)
                x = activation(x + y)
            num_filters *= 2

        # valid padding = no padding
        x = torch.nn.AvgPool2d(kernel_size=(8, 8), stride=8, padding=0)(x)
        x = torch.flatten(x)
        
        # need to know the input size 
        # need to enforce a he_normal initialization 
        logits = nn.Linear(len(x), num_classes)(x)
        
        return logits
    return forward

# # jax version
# class FilterResponseNorm(hk.Module):
#     def __init__(self, eps=1e-6, name='frn'):
#         super().__init__(name=name)
#         self.eps = eps
  
#     def __call__(self, x, **unused_kwargs):
#         del unused_kwargs
#         par_shape = (1, 1, 1, x.shape[-1])  # [1,1,1,C]
#         tau = hk.get_parameter('tau', par_shape, x.dtype, init=jnp.zeros)
#         beta = hk.get_parameter('beta', par_shape, x.dtype, init=jnp.zeros)
#         gamma = hk.get_parameter('gamma', par_shape, x.dtype, init=jnp.ones)
#         nu2 = jnp.mean(jnp.square(x), axis=[1, 2], keepdims=True)
#         x = x * jax.lax.rsqrt(nu2 + self.eps)
#         y = gamma * x + beta
#         z = jnp.maximum(y, tau)
#         return z

# pytorch version
class FilterResponseNorm(torch.nn.Module):
    def __init__(self, eps=1e-6, name='frn'):
        super().__init__()
        self.eps = eps
  
    def __call__(self, x, **unused_kwargs):
        del unused_kwargs
        par_shape = (1, 1, 1, x.shape[-1])  # [1,1,1,C]
        tau = torch.nn.Parameter(torch.zeros(par_shape)) 
        # nn.Module.get_parameter(name='tau', shape=par_shape, initializer=torch.zeros)
        beta = torch.nn.Parameter(torch.zeros(par_shape)) 
        # nn.Module.get_parameter(name='beta', shape=par_shape, initializer=torch.zeros)
        gamma = torch.nn.Parameter(torch.ones(par_shape)) 
        # nn.Module.get_parameter(name='gamma', shape=par_shape, initializer=torch.ones)

        nu2 = torch.mean(torch.square(x), dim=[1, 2], keepdim=True)

        x = x * 1 / torch.sqrt(nu2 + self.eps)
        y = gamma * x + beta
        z = torch.max(y, tau)
        return z
    
    
# def make_resnet20_frn_fn(data_info, activation=jax.nn.relu):
#     num_classes = data_info["num_classes"]
#     return make_resnet_fn(
#       num_classes, depth=20, normalization_layer=FilterResponseNorm,
#       activation=activation)

def make_resnet20_frn_fn(data_info, activation=torch.nn.ReLU()):
    num_classes = data_info["num_classes"]
    return make_resnet_fn(
      num_classes, depth=20, normalization_layer=FilterResponseNorm,
      activation=activation)
