# =============================================================================
# Install necessary packages
# =============================================================================
# pip install inplace-abn
# pip install timm


# =============================================================================
# Import required libraries
# =============================================================================
import torch
from torch import nn
import torch.nn.functional as F
import timm

from transformer import Transformer


# =============================================================================
# Backbone (CNN)
# =============================================================================
class Backbone(nn.Module):
    '''
        images dim: (batch-size, 3, image-size, image-size)
        
        features dim: (batch-size, num-channels, encoded-image-size, encoded-image-size)
    '''

    def __init__(self, args, pretrained):
        super(Backbone, self).__init__()
        if args.data == 'VG-500':
            tresnet = timm.create_model('tresnet_v2_l', num_classes=11221)
            if pretrained:
                load_weight = torch.load(
                    './pretrained_model/tresnet_l_v2_miil_21k.pth')
                tresnet.load_state_dict(load_weight['state_dict'])
        else:
            tresnet = timm.create_model(
                'tresnet_m_miil_in21k', pretrained=pretrained)
        #
        self.features = nn.Sequential(tresnet.body)

    def forward(self, images):
        return self.features(images)


# =============================================================================
# Head (Transformer & fully connected)
# =============================================================================
class Head(nn.Module):
    '''
        features dim: (batch-size, num-channels, encoded-image-size, encoded-image-size)
        
        y-hats dim: (batch-size, num-classes)
        
        norm_first:
            if True, layer norm is done prior to self attention, cross attention 
            and feedforward operations, respectively. Otherwise it’s done after. 
        remove_self_attn:
            if True, self attention layer will be removed, and only cross attention 
            and feedforward operations will remain. 
        keep_query_position:
            if False, the 'query,' 'key', and 'value' will be the same as the label embbedings.
    '''

    def __init__(self,
                 args,
                 num_classes,
                 hidden_size,
                 num_heads,
                 num_decoder_layers,
                 feedforward_size,
                 norm_first,
                 remove_self_attn,
                 keep_query_position):
        super(Head, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.channel_size = 2048

        self.transformer = Transformer(hidden_size=hidden_size,
                                       num_heads=num_heads,
                                       num_decoder_layers=num_decoder_layers,
                                       feedforward_size=feedforward_size,
                                       dropout=0.1,
                                       activation="relu",
                                       norm_first=norm_first,
                                       remove_self_attn=remove_self_attn,
                                       keep_query_position=keep_query_position)

        self.query_emb = nn.Embedding(num_classes, hidden_size)
        if remove_self_attn:
            self.query_emb.requires_grad_(False)

        self.linear_projection = nn.Conv2d(
            self.channel_size, hidden_size, kernel_size=1)

        self.w = nn.Parameter(torch.Tensor(1, num_classes, hidden_size))
        self.b = nn.Parameter(torch.Tensor(1, num_classes))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.w)
        nn.init.constant_(self.b, 0)

    def forward_fc(self, hidden_state):
        output = (self.w * hidden_state).sum(-1)
        output = output + self.b
        return output

    def forward(self, features):
        # (batch-size, hidden-size, encoded-image-size, encoded-image-size)
        features = self.linear_projection(features)
        features = F.relu(features, inplace=True)
        # (num-classes, hidden-size)
        queries = self.query_emb.weight
        # (batch-size, num-classes, hidden-size)
        hidden_state, attn_weights = self.transformer(features,
                                                      queries)
        y_hats = self.forward_fc(hidden_state)
        return y_hats, attn_weights


# =============================================================================
# Annotator
# =============================================================================
class Annotator(nn.Module):
    def __init__(self,
                 args,
                 num_classes,
                 hidden_size,
                 num_heads,
                 num_decoder_layers,
                 feedforward_size,
                 norm_first,
                 remove_self_attn,
                 keep_query_position):
        super(Annotator, self).__init__()
        self.path = args.save_dir + 'Annotator_' + args.data + '.pth'
        # load backbone
        self.backbone = Backbone(args, pretrained=True)
        # load head
        self.head = Head(args,
                         num_classes,
                         hidden_size,
                         num_heads,
                         num_decoder_layers,
                         feedforward_size,
                         norm_first,
                         remove_self_attn,
                         keep_query_position)

    def forward(self, images):
        features = self.backbone(images)
        y_hats, attn_weights = self.head(features)
        return y_hats, attn_weights
