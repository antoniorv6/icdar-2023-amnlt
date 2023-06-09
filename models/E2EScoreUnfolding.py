import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from torchinfo import summary
from itertools import groupby

class DepthSepConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=None, padding=True, stride=(1,1), dilation=(1,1)):
        super(DepthSepConv2D, self).__init__()

        self.padding = None
        
        if padding:
            if padding is True:
                padding = [int((k-1)/2) for k in kernel_size]
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    padding_h = kernel_size[1] - 1
                    padding_w = kernel_size[0] - 1
                    self.padding = [padding_h//2, padding_h-padding_h//2, padding_w//2, padding_w-padding_w//2]
                    padding = (0, 0)

        else:
            padding = (0, 0)

        self.depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, dilation=dilation, kernel_size=(1,1))
        self.activation = activation

    def forward(self, inputs):
        x = self.depth_conv(inputs)
        if self.padding:
            x = F.pad(x, self.padding)
        if self.activation:
            x = self.activation(x)
        
        x = self.point_conv(x)

        return x

class MixDropout(nn.Module):
    def __init__(self, dropout_prob=0.4, dropout_2d_prob=0.2):
        super(MixDropout, self).__init__()

        self.dropout = nn.Dropout(dropout_prob)
        self.dropout2D = nn.Dropout2d(dropout_2d_prob)
    
    def forward(self, inputs):
        if random.random() < 0.5:
            return self.dropout(inputs)
        return self.dropout2D(inputs)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=(1,1), kernel=3, activation=nn.ReLU, dropout=0.5):
        super(ConvBlock, self).__init__()

        self.activation = activation()
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel, padding=kernel//2)
        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=kernel, padding=kernel//2)
        self.conv3 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3,3), padding=(1,1), stride=stride)
        self.normLayer = nn.InstanceNorm2d(num_features=out_c, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_prob=dropout, dropout_2d_prob=dropout/2)

    def forward(self, inputs):
        pos = random.randint(1,3)

        x = self.conv1(inputs)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)
        
        x = self.normLayer(x)
        x = self.conv3(x)
        x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        
        return x

class DSCBlock(nn.Module):

    def __init__(self, in_c, out_c, stride=(2, 1), activation=nn.ReLU, dropout=0.5):
        super(DSCBlock, self).__init__()

        self.activation = activation()
        self.conv1 = DepthSepConv2D(in_c, out_c, kernel_size=(3, 3))
        self.conv2 = DepthSepConv2D(out_c, out_c, kernel_size=(3, 3))
        self.conv3 = DepthSepConv2D(out_c, out_c, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm_layer = nn.InstanceNorm2d(out_c, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_prob=dropout, dropout_2d_prob=dropout/2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)
        #x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        return x

class Encoder(nn.Module):

    def __init__(self, in_channels, dropout=0.5):
        super(Encoder, self).__init__()

        self.conv_blocks = nn.ModuleList([
            ConvBlock(in_c=in_channels, out_c=32, stride=(1,1), dropout=dropout),
            ConvBlock(in_c=32, out_c=64, stride=(2,2), dropout=dropout),
            ConvBlock(in_c=64, out_c=128, stride=(2,2), dropout=dropout),
            ConvBlock(in_c=128, out_c=256, stride=(2,1), dropout=dropout),
            ConvBlock(in_c=256, out_c=512, stride=(2,1), dropout=dropout),
            ConvBlock(in_c=512, out_c=512, stride=(2,1), dropout=dropout)
        ])

        self.dscblocks = nn.ModuleList([
            DSCBlock(in_c=512, out_c=512, stride=(1,1), dropout = dropout),
            DSCBlock(in_c=512, out_c=512, stride=(1,1), dropout = dropout),
            DSCBlock(in_c=512, out_c=512, stride=(1,1), dropout = dropout),
            DSCBlock(in_c=512, out_c=512, stride=(1,1), dropout = dropout)
        ])
    
    def forward(self, x):
        for layer in self.conv_blocks:
            x = layer(x)
        
        for layer in self.dscblocks:
            xt = layer(x)
            x = x + xt if x.size() == xt.size() else xt

        return x

class PageDecoder(nn.Module):

    def __init__(self, out_cats):
        super(PageDecoder, self).__init__()
        self.dec_conv = nn.Conv2d(in_channels= 512, out_channels= out_cats, kernel_size=(5,5), padding=(2,2))
        self.lsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, inputs):
        x = self.dec_conv(inputs)
        x = self.lsoftmax(x)
        b, c, h, w = x.size()
        x = x.reshape(b, c, h*w)
        x = x.permute(2,0,1)
        return x

class RecurrentDecoder(nn.Module):

    def __init__(self, out_cats):
        super(RecurrentDecoder, self).__init__()
        self.dec_lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, dropout=0.2, bidirectional=True, batch_first=True)
        self.out_dense = nn.Linear(in_features=512, out_features=out_cats)
    
    def forward(self, inputs):
        x = inputs
        b, c, h, w = x.size()
        x = x.reshape(b, c, h*w)
        x = x.permute(0,2,1)
        x, _ = self.dec_lstm(x)
        x = self.out_dense(x)
        x = x.permute(1,0,2)
        return F.log_softmax(x, dim=2)

class PositionalEncoding2D(nn.Module):

    def __init__(self, dim, h_max, w_max, device):
        super(PositionalEncoding2D, self).__init__()
        self.h_max = h_max
        self.max_w = w_max
        self.dim = dim
        self.pe = torch.zeros((1, dim, h_max, w_max), device=device, requires_grad=False)

        div = torch.exp(-torch.arange(0., dim // 2, 2) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(1)
        w_pos = torch.arange(0., w_max)
        h_pos = torch.arange(0., h_max)
        self.pe[:, :dim // 2:2, :, :] = torch.sin(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        self.pe[:, 1:dim // 2:2, :, :] = torch.cos(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        self.pe[:, dim // 2::2, :, :] = torch.sin(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)
        self.pe[:, dim // 2 + 1::2, :, :] = torch.cos(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)

    def forward(self, x):
        """
        Add 2D positional encoding to x
        x: (B, C, H, W)
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)].to(x.device)

    def get_pe_by_size(self, h, w, device):
        return self.pe[:, :, :h, :w].to(device)

class PositionalEncoding1D(nn.Module):

    def __init__(self, dim, len_max, device):
        super(PositionalEncoding1D, self).__init__()
        self.len_max = len_max
        self.dim = dim
        self.pe = torch.zeros((1, dim, len_max), device=device, requires_grad=False)

        div = torch.exp(-torch.arange(0., dim, 2) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(1)
        l_pos = torch.arange(0., len_max)
        self.pe[:, ::2, :] = torch.sin(l_pos * div).unsqueeze(0)
        self.pe[:, 1::2, :] = torch.cos(l_pos * div).unsqueeze(0)

    def forward(self, x, start=0):
        """
        Add 1D positional encoding to x
        x: (B, C, L)
        start: index for x[:,:, 0]
        """
        if isinstance(start, int):
            return x + self.pe[:, :, start:start+x.size(2)].to(x.device)
        else:
            for i in range(x.size(0)):
                x[i] = x[i] + self.pe[0, :, start[i]:start[i]+x.size(2)]
            return 

class TransformerDecoder2D(nn.Module):

    def __init__(self, out_cats, max_h, max_w):
        super(TransformerDecoder2D, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.pos_encoding = PositionalEncoding2D(dim=512, h_max=max_h, w_max=max_w, device=self.dummy_param.device)
        transf_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, batch_first=True)
        self.dec_transf = nn.TransformerEncoder(transf_layer, num_layers=1)
        self.out_dense = nn.Linear(in_features=512, out_features=out_cats)

    def forward(self, inputs):
        x = inputs
        b, c, h, w = x.size()
        x = self.pos_encoding(x)
        x = x.reshape(b, c, h*w)
        x = x.permute(0,2,1).contiguous()
        x = self.dec_transf(x)
        x = self.out_dense(x)
        x = x.permute(1,0,2).contiguous()
        return F.log_softmax(x, dim=2)

class TransformerDecoder1D(nn.Module):
    def __init__(self, out_cats, max_len):
        super(TransformerDecoder1D, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.pos_encoding = PositionalEncoding1D(dim=512, len_max=max_len, device=self.dummy_param.device)
        transf_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, batch_first=True)
        self.dec_transf = nn.TransformerEncoder(transf_layer, num_layers=1)
        self.out_dense = nn.Linear(in_features=512, out_features=out_cats)

    def forward(self, inputs):
        x = inputs
        b, c, h, w = x.size()
        x = x.reshape(b, c, h*w)
        x = self.pos_encoding(x)
        x = x.permute(0,2,1).contiguous()
        x = self.dec_transf(x)
        x = self.out_dense(x)
        x = x.permute(1,0,2).contiguous()
        return F.log_softmax(x, dim=2)


class E2EScore_FCN(nn.Module):

    def __init__(self, in_channels, out_cats):
        super(E2EScore_FCN, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = PageDecoder(out_cats=out_cats)
    
    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x


class E2EScore_CRNN(nn.Module):

    def __init__(self, in_channels, out_cats, pretrain_path=None):
        super(E2EScore_CRNN, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = RecurrentDecoder(out_cats=out_cats)
    
    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

class E2EScore_CNNT2D(nn.Module):

    def __init__(self, in_channels, out_cats, mh, mw, pretrain_path=None):
        super(E2EScore_CNNT2D, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = TransformerDecoder2D(out_cats=out_cats, max_w=mw, max_h=mh)
    
    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

class E2EScore_CNNT1D(nn.Module):

    def __init__(self, in_channels, out_cats, max_len, pretrain_path=None):
        super(E2EScore_CNNT1D, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = TransformerDecoder1D(out_cats=out_cats, max_len=max_len)
    
    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x


def get_FCN_model(in_channels, out_size, mh, mw, maxlen=None):
    model = E2EScore_FCN(in_channels=in_channels, out_cats=out_size)
    return model

def get_CRNN_model(in_channels, out_size, mh, mw, maxlen=None):
    model = E2EScore_CRNN(in_channels=in_channels, out_cats=out_size)
    return model

def get_CNNT2D_model(in_channels, out_size, mh, mw, maxlen=None):
    model = E2EScore_CNNT2D(in_channels=in_channels, out_cats=out_size, mh=mh, mw=mw)
    return model

def get_CNNT1D_model(in_channels, out_size, mh, mw, maxlen=5000):
    model = E2EScore_CNNT1D(in_channels=in_channels, out_cats=out_size, max_len=maxlen)
    return model

