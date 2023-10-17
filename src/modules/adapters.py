import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def init_weights(module):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # std defaults to 0.02, this might need to be changed
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class ParallelAdapter(nn.Module):
    def __init__(self,
                 d_model=None,
                 bottleneck_dim=None,
                 dropout=0.,
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in",
                 activation_function="relu",
                 ):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck_dim

        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == 'out':
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones([1]))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)

        # self.non_linear_func = nn.ReLU()
        self.non_linear_func = None
        if activation_function == "gelu":
            self.non_linear_func = nn.GELU()
        elif activation_function == "tanh":
            self.non_linear_func = nn.Tanh()
        elif activation_function == "relu":
            self.non_linear_func = nn.ReLU()
        else:
            raise ValueError

        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout

        self.apply(init_weights)

    def forward(self, x, add_residual=True, residual=None, mask=None):
        residual = x if residual is None else residual
        # if self.adapter_layernorm_option == "in":
        #     x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up
        return output

