from abc import ABC

import torch
import math
from functools import partial
from models import pvt
from models import pvt_v2
from timm.models import create_model
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

checkpoint_path = {
    'pvt_tiny': '../pre-train/PVT/pvt_tiny.pth',
    'pvt_small': '../pre-train/PVT/pvt_small.pth',
    'pvt_v2_b0': '../pre-train/PVT/pvt_v2_b0.pth',
    'pvt_v2_b2_li': '../pre-train/PVT/pvt_v2_b2_li.pth',
}

out_channels = {
    'pvt_tiny': 1024,
    'pvt_small': 1024,
    'pvt_medium': 1024,
    'pvt_v2_b0': 512,
    'pvt_v2_b1': 1024,
    'pvt_v2_b2': 1024,
    'pvt_v2_b2_li': 1024,
}

embed_dims = {
    'pvt_tiny': [64, 128, 320, 512],
    'pvt_small': [64, 128, 320, 512],
    'pvt_medium': [64, 128, 320, 512],
    'pvt_v2_b0': [32, 64, 160, 256],
    'pvt_v2_b1': [64, 128, 320, 512],
    'pvt_v2_b2': [64, 128, 320, 512],
    'pvt_v2_b2_li': [64, 128, 320, 512]
}


def conv_block(in_channel, out_channel, kernel_size=3, stride=1, padding=1, norm=True,
               norm_type=torch.nn.BatchNorm2d, relu_type=torch.nn.LeakyReLU, trans=False):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=not norm) if not trans else
        torch.nn.ConvTranspose2d(in_channel, out_channel, 4, 2, padding, bias=not norm),
        torch.nn.Identity() if not norm else norm_type(out_channel),
        relu_type(inplace=True)
    )


def projection_block(in_channel, out_channel):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=True),
    )


def skip_connection(in_channel, out_channel=64, kernel_size=3, stride=1, padding=1, norm=True,
                    norm_type=torch.nn.BatchNorm2d, relu_type=torch.nn.LeakyReLU, trans=False):
    return torch.nn.Sequential(
        conv_block(in_channel, out_channel, kernel_size, stride, padding, norm, norm_type, relu_type, trans),
        conv_block(out_channel, out_channel, kernel_size, stride, padding, norm, norm_type, relu_type)
    )


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor,coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input* 1.0
        return output
    
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg ( ) * Ctx.coeff, None


class task_attention(torch.nn.Module, ABC):
    def __init__(self, embed_dim=256):
        super(task_attention, self).__init__()

        self.query_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.key_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.value_proj = torch.nn.Linear(embed_dim, embed_dim)

        self.embed_dim = embed_dim

    def forward(self, task_token, image_token):
        query = self.query_proj(image_token)
        key = self.key_proj(task_token)
        value = self.value_proj(task_token)

        attention_map = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(self.embed_dim)
        output_token = torch.matmul(torch.nn.functional.sigmoid(attention_map), value)

        return output_token + image_token


class res_block(torch.nn.Module, ABC):
    def __init__(self, in_channel=64, out_channel=64, norm=True,
                 norm_type=torch.nn.BatchNorm2d, relu_type=torch.nn.LeakyReLU, up=False):
        super(res_block, self).__init__()

        if up:
            self.up = torch.nn.UpsamplingNearest2d(scale_factor=2)
        else:
            self.up = torch.nn.Identity()

        self.up_branch = skip_connection(in_channel, out_channel, 3, 1, 1, norm, norm_type, relu_type, trans=up)
        self.down_branch = conv_block(in_channel, out_channel, 1, 1, 0, norm, norm_type, relu_type)

    def forward(self, x):
        up = self.up_branch(x)
        down = self.down_branch(self.up(x))
        output = up + down
        return output


class transform_layers(torch.nn.Module, ABC):
    def __init__(self, in_channels, size, layer_num=2, heads=2, mlp_ratio=4, sr_ratio=4):
        super(transform_layers, self).__init__()
        self.norm = torch.nn.LayerNorm(in_channels)
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, size * size, in_channels))
        self.transformer_block = torch.nn.ModuleList([pvt.Block(
            dim=in_channels, num_heads=heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., drop_path=0., norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), sr_ratio=sr_ratio
        ) for _ in range(layer_num)])

        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        x = x + self.pos_embed
        for block in self.transformer_block:
            x = block(x, h, w)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return x


def decode_block(in_channel=64, out_channel=64, norm=True,
                 norm_type=torch.nn.BatchNorm2d, relu_type=torch.nn.LeakyReLU):
    return torch.nn.Sequential(
        res_block(in_channel, out_channel, norm, norm_type, relu_type, up=True),
        res_block(out_channel, out_channel, norm, norm_type, relu_type, up=False)
    )


class simple_decoder(torch.nn.Module, ABC):
    def __init__(self, in_channels, **kwargs):
        super(simple_decoder, self).__init__()

        self.up2 = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.up4 = torch.nn.UpsamplingNearest2d(scale_factor=4)
        self.up8 = torch.nn.UpsamplingNearest2d(scale_factor=8)

        self.conv_1 = conv_block(in_channels, in_channels // 2, 3, 1, 1, norm=False)
        self.conv_2 = conv_block(in_channels // 2, in_channels // 4, 3, 1, 1, norm=False)
        self.conv_final = torch.nn.Conv2d(in_channels // 4, 1, 1)

    def forward(self, x1, x2, x3, x4):
        x2, x3, x4 = self.up2(x2), self.up4(x3), self.up8(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.up2(self.conv_1(x))
        x = self.up2(self.conv_2(x))
        x = self.conv_final(x)

        output = (torch.tanh(x) + 1.0) / 2.0

        return output


class unet_decoder(torch.nn.Module, ABC):
    def __init__(self, dims, **kwargs):
        super(unet_decoder, self).__init__()

        self.skip_connections = torch.nn.ModuleList()
        self.decode_blocks = torch.nn.ModuleList()
        for dim in dims[2::-1]:
            self.skip_connections.append(skip_connection(dim, norm=False))
        for i, dim in enumerate(dims[:0:-1]):
            self.decode = self.decode_blocks.append(decode_block(dim if i == 0 else 64, norm=False))

        self.conv = res_block(64, 128, norm=False, up=False)
        # self.conv1 = res_block(64, 64, norm=False, up=True)
        # self.conv2 = res_block(64, 64, norm=False, up=True)
        self.final_conv = torch.nn.Conv2d(8, 1, 1, 1)

    def forward(self, x1, x2, x3, x4):
        embedding = [x4, x3, x2, x1]
        for i in range(len(embedding)):
            if i != 0:
                embedding[i] = self.skip_connections[i-1](embedding[i])
                embedding[i] = embedding[i] + embedding[i-1]
            if i != len(embedding) - 1:
                embedding[i] = self.decode_blocks[i](embedding[i])

        x = embedding[-1]
        # x = self.conv2(self.conv1(x))
        x = self.conv(x)

        b, c, h, w = x.shape
        x = x.reshape(b, 4, 4, c // 16, h, w).permute(0, 3, 4, 1, 5, 2).reshape(b, c // 16, h * 4, w * 4)

        x = self.final_conv(x)
        output = (torch.tanh(x) + 1.0) / 2.0

        return output


class patch_decoder(torch.nn.Module, ABC):
    def __init__(self, in_channels, task_wise=False, image_size=320, **kwargs):
        super(patch_decoder, self).__init__()

        self.up2 = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.up4 = torch.nn.UpsamplingNearest2d(scale_factor=4)
        self.up8 = torch.nn.UpsamplingNearest2d(scale_factor=8)

        self.projection = torch.nn.Conv2d(256, 256, 1, 1, 0, bias=True)
        self.norm = torch.nn.LayerNorm(256)
        for i in range(len(in_channels)):
            setattr(self, f'proj_{i+1}', projection_block(in_channels[i], 256))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, image_size // 4 * image_size // 4, 256))

        self.prediction_patch = torch.nn.Conv2d(256, 3, 1, 1, 0, bias=True)
        self.prediction_pixel = torch.nn.Conv2d(16, 3, 1, 1, 0, bias=True)
        # self.sigmoid = torch.nn.Sigmoid()         `

        self.transformer_block = torch.nn.ModuleList([pvt.Block(
            dim=256, num_heads=2, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), sr_ratio=4
        ) for _ in range(2)])

        self.task_wise = task_wise
        if task_wise:
            self.patch_token = torch.nn.Parameter(torch.randn(1, 1, 256), requires_grad=True)
            self.pixel_token = torch.nn.Parameter(torch.randn(1, 1, 256), requires_grad=True)

            self.patch_attention = task_attention()
            self.pixel_attention = task_attention()

        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x1, x2, x3, x4):
        x1, x2, x3, x4 = self.proj_1(x1), self.proj_2(x2), self.proj_3(x3), self.proj_4(x4)
        x2, x3, x4 = self.up2(x2), self.up4(x3), self.up8(x4)
        x = x1 + x2 + x3 + x4
        b, c, h, w = x.shape

        x = self.norm(self.projection(x).flatten(2).transpose(1, 2))
        x = x + self.pos_embed

        if self.task_wise:
            patch_token = self.patch_token.repeat(b, 1, 1)
            pixel_token = self.pixel_token.repeat(b, 1, 1)
            x = torch.cat([patch_token, pixel_token, x], dim=1)

        for block in self.transformer_block:
            x = block(x, h, w)

        if self.task_wise:
            patch_token, pixel_token, image_token = torch.split(x, [1, 1, h*w], dim=1)
            x_patch = self.patch_attention(patch_token, image_token)
            x_pixel = self.pixel_attention(pixel_token, image_token)
            x_patch = x_patch.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            x_pixel = x_pixel.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        else:
            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            x_patch, x_pixel = x, x

        x_patch_predict = self.prediction_patch(x_patch)
        b, c, h, w = x_pixel.shape
        x_pixel = x_pixel.reshape(b, 4, 4, c // 16, h, w).permute(0, 3, 4, 1, 5, 2).reshape(b, c // 16, h * 4, w * 4)
        x_pixel_predict = self.prediction_pixel(x_pixel)

        return x_patch_predict, x_pixel_predict, x_patch, x_pixel


class regression_decoder(torch.nn.Module, ABC):
    def __init__(self, in_channels, image_size=320, **kwargs):
        super(regression_decoder, self).__init__()
        self.image_size = image_size

        for i, channel in enumerate(in_channels):
            setattr(self, f'skip_projection_{i}', torch.nn.Linear(in_channels[i], 256, bias=True))
            setattr(self, f'skip_norm_{i}', torch.nn.LayerNorm(256))

            setattr(self, f'decode_transform_block_{i}', torch.nn.ModuleList([pvt.Block(
                dim=256, num_heads=2, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0., norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), sr_ratio=4
            ) for _ in range(2)]))
            size = image_size // pow(2, i + 2)
            setattr(self, f'decode_pos_embed_{i}', torch.nn.Parameter(torch.zeros(1, size * size, 256)))
            setattr(self, f'decode_projection_{i}', torch.nn.Linear(256+3, 256, bias=True))
            setattr(self, f'decode_norm_{i}', torch.nn.LayerNorm(256))

        self.act = torch.nn.GELU()

    def forward(self, x, cls):
        patch_prediction, _, patch_features, _ = cls
        patch_prediction = torch.nn.functional.softmax(patch_prediction, 1)

        for i, feature in enumerate(x):
            feature = feature.flatten(2).transpose(1, 2)
            skip_projection = getattr(self, f'skip_projection_{i}')
            skip_norm = getattr(self, f'skip_norm_{i}')
            x[i] = self.act(skip_norm(skip_projection(feature)))
        for i in range(len(x)):
            patch_p = torch.nn.functional.interpolate(patch_prediction, scale_factor=1. / pow(2, 3 - i))
            patch_f = torch.nn.functional.interpolate(patch_features, scale_factor=1. / pow(2, 3 - i))
            patch_p = patch_p.flatten(2).transpose(1, 2)
            patch_f = patch_f.flatten(2).transpose(1, 2)
            if i != 0:
                x[3-i] = x[3-i] + x[3-i+1]
            x[3-i] = torch.cat([x[3-i]+patch_f, patch_p], 2)
            decode_projection = getattr(self, f'decode_projection_{3-i}')
            decode_norm = getattr(self, f'decode_norm_{3-i}')
            decode_pos_embed = getattr(self, f'decode_pos_embed_{3-i}')
            decode_transform_block = getattr(self, f'decode_transform_block_{3-i}')
            x[3-i] = decode_norm(decode_projection(x[3-i])) + decode_pos_embed
            b, n, c = x[3-i].shape
            size = int(n ** 0.5)
            for block in decode_transform_block:
                x[3-i] = block(x[3-i], size, size)

            if i != len(x) - 1:
                x[3-i] = x[3-i].reshape(b, size, size, -1).permute(0, 3, 1, 2).contiguous()
                x[3-i] = torch.nn.functional.interpolate(x[3-i], scale_factor=2)
                x[3-i] = x[3-i].flatten(2).transpose(1, 2)

        for i in range(len(x)):
            b, n, _ = x[i].shape
            size = int(n ** 0.5)
            x[i] = x[i].reshape(b, size, size, -1).permute(0, 3, 1, 2).contiguous()
        return x


class local_domain_align(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()
        
        self.mlp = torch.nn.Linear(256, 256, bias=True)
        self.discriminator = torch.nn.Linear(256, 2, bias=True)
        
    def forward(self, x, domain_label, masks):
        features = self.mlp(x)
        features = GradientReverseFunction.apply(*features)
        _, _, masks = torch.split(masks, 1, 1)
        features = features * domain_label * masks
        features = self.discriminator(features)
        
        return features


class regression_head(torch.nn.Module, ABC):
    def __init__(self):
        super(regression_head, self).__init__()

        self.prediction = torch.nn.Conv2d(16, 1, 1, 1, 0)

    def forward(self, x, cls):
        b, c, h, w = x.shape
        x = x.reshape(b, 4, 4, c // 16, h, w).permute(0, 3, 4, 1, 5, 2).reshape(b, c // 16, h * 4, w * 4)
        x = self.prediction(x)

        _, cls_prediction, _, _ = cls
        # cls_prediction = torch.nn.functional.interpolate(cls_prediction, scale_factor=4)
        cls_prediction = torch.nn.functional.softmax(cls_prediction, 1)
        bg, fg, un = torch.split(cls_prediction, 1, 1)

        # output = fg + un * x
        output = x

        return output


def _init_weights(m):
    if isinstance(m, torch.nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, torch.nn.Linear) and m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, torch.nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


class network(torch.nn.Module, ABC):
    def __init__(self, backbone_model, decoder_type='baseline', pretrain=True, image_size=320, **kwargs):
        super(network, self).__init__()

        self.backbone = create_model(backbone_model, pretrained=False)

        if decoder_type == 'baseline':
            channels = embed_dims[backbone_model]
            self.cls_decoder = patch_decoder(channels, image_size=image_size, **kwargs)
            self.reg_decoder = regression_decoder(channels, image_size=image_size, **kwargs)
            self.reg_head = regression_head()
            
            self.local_domain = local_domain_align()
            self.global_domain = pvt.Block(
                dim=256, num_heads=2, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), sr_ratio=4, align=True
                )

        self.apply(_init_weights)

        # if pretrain:
        #     checkpoint = torch.load(checkpoint_path[backbone_model], map_location='cpu')
        #     self.backbone.load_state_dict(checkpoint, strict=False)

    def forward(self, x, domain_label=None):
        output = {}
        x1, x2, x3, x4 = self.backbone(x)
        
        x4, global_align = self.global_domain(x4)

        cls_output, reg_output = None, None
        if hasattr(self, 'cls_decoder'):
            cls_output = self.cls_decoder(x1, x2, x3, x4)
            output['cls'] = cls_output

        if hasattr(self, 'reg_decoder'):
            reg_feature = self.reg_decoder([x1, x2, x3, x4], cls_output)
            reg_feature = reg_feature[0]
            reg_output = self.reg_head(reg_feature, cls_output)
            output['reg'] = reg_output
        
        local_features = torch.cat([x1, torch.nn.functional.interpolate(x2, scale_factor=2)])
        local_features = local_features.flatten(2).transpose(1, 2)
        local_align = self.local_domain(local_features, domain_label, cls_output[0])
        
        output['align'] = (local_align, global_align)

        return output


@register_model
def trans_matting(backbone_model, decode_type, image_size=320, **kwargs):
    model = network(backbone_model, decoder_type=decode_type, image_size=image_size, **kwargs)
    return model


if __name__ == '__main__':
    # cp = torch.load('../../pre-train/PVT/pvt_small.pth', map_location='cpu')

    # img = torch.randn(1, 3, 320, 320).cuda()
    # net = network('pvt_small', 'baseline', pretrain=False, task_wise=True).cuda()
    model = create_model('trans_matting', backbone_model='pvt_v2_b2_li', decode_type='baseline', image_size=512)
    # model = model.cuda()
    model.eval()
    # net.backbone.load_state_dict(cp, strict=False)

    # reg = o['reg']
    # print(reg.shape)
    # for r in reg:
    #     print(r.shape)
