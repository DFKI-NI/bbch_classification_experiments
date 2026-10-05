"""Task-specific decoder heads for the MTL model.
 
All decoders get the feature map of the encoder with 2048 channels (output of the
dilated ResNet). The pixel-wise tasks use a DeepLabv3 head (ASPP). The plant-level
tasks use small heads that pool the feature map. For most tasks there are several
decoder variants, which are chosen with the "decoder" name in task_definition of
the config.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


# DeepLabHead + ASPP are taken from https://github.com/median-research-group/LibMTL/blob/main/examples/nyu/aspp.py
class DeepLabHead(nn.Sequential):
    """DeepLabv3 head: ASPP, a 3x3 conv and a 1x1 conv to num_classes channels.
 
    The output has the same spatial size as the input feature map.
    """

    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


class ASPPConv(nn.Sequential):
    """One ASPP branch: dilated 3x3 conv, BatchNorm and ReLU."""
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    """Image-level ASPP branch: global pooling, 1x1 conv, BatchNorm and ReLU."""
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):  # type: ignore
        """Apply the branch and upsample the result to the input size.
 
        Returns the feature map.
        """
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling with five parallel branches.
 
    The branches are a 1x1 conv, three dilated 3x3 convs (atrous_rates) and the
    image-level pooling. Their outputs are concatenated and projected to 256
    channels.
    """

    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        """Run all branches and project the concatenated output.
 
        Returns a feature map with 256 channels.
        """
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def get_decoder(task_name, decoder_name, num_classes=None):
    """Create the decoder for one task.
 
    num_classes is only needed for the classification and segmentation tasks.
    Raises ValueError for an unknown task.
 
    Returns the decoder module.
    """
    if task_name == "bbch_classification":
        return get_bbch_classification_decoder(num_classes, decoder_name)
    elif task_name == "bbch_regression":
        return get_bbch_regression_decoder(decoder_name)
    elif task_name == "damaged_classification":
        return get_damaged_classification_decoder(num_classes, decoder_name)
    elif task_name == "pprcm_regression":
        return get_ppr_cm_regression_decoder(decoder_name)
    elif task_name == "distance_regression":
        return get_distance_regression_decoder(decoder_name)
    elif task_name == "direction_regression":
        return get_directions_regression_decoder(decoder_name)
    elif task_name == "semantic_segmentation":
        return get_maize_segmentation_decoder(num_classes, decoder_name)
    else:
        raise ValueError(f"No support for {task_name}")


def get_maize_segmentation_decoder(num_classes, decoder_name):
    """Create the decoder for the semantic segmentation.
 
    Options: "deeplab" (DeepLabv3 head, outputs logits). Raises ValueError for an
    unknown decoder_name.
 
    Returns the decoder module.
    """
    feature_dim = 2048
    if decoder_name == "deeplab":
        return DeepLabHead(feature_dim, num_classes)
    else:
        raise ValueError(f"Decoder {decoder_name} for semantic_segmentation not supported.")


def get_directions_regression_decoder(decoder_name):
    """Create the decoder for the direction regression.
 
    Options: "deeplab" (DeepLabv3 head with 2 output channels for cos and sin, and
    Tanh). Raises ValueError for an unknown decoder_name.
 
    Returns the decoder module.
    """
    feature_dim = 2048
    
    if decoder_name == "deeplab":
        head = DeepLabHead(feature_dim, num_classes=2)
        return nn.Sequential(
            head,
            nn.Tanh()
        )    
    else:
        raise ValueError(f"Decoder {decoder_name} for directions_regression not supported.")


def get_distance_regression_decoder(decoder_name):
    """Create the decoder for the distance regression.
 
    Options: "deeplab" (DeepLabv3 head without activation) and "deeplab_tanh" (with
    Tanh, output range [-1, 1]). Raises ValueError for an unknown decoder_name.
 
    Returns the decoder module.
    """
    feature_dim = 2048
    
    if decoder_name == "deeplab":
        return DeepLabHead(feature_dim, num_classes=1)
    
    elif decoder_name == "deeplab_tanh":
        head = DeepLabHead(feature_dim, num_classes=1)
        return nn.Sequential(
            head,
            nn.Tanh()
        )
    else:
        raise ValueError(f"Decoder {decoder_name} for distance_regression not supported.")


def get_ppr_cm_regression_decoder(decoder_name):
    """Create the decoder for the ppr_cm regression.
 
    All options end with a Sigmoid, so the output is in [0, 1] like the normalized
    target:
    - "minimal": global average pooling and one linear layer.
    Raises ValueError for an unknown decoder_name.
 
    Returns the decoder module.
    """
    feature_dim = 2048 
    out_dim = 1

    if decoder_name == "minimal":
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, out_dim),
            nn.Sigmoid()
        )
    else:
        raise ValueError(f"Decoder {decoder_name} for pprcm_regression not supported.")


def get_damaged_classification_decoder(num_classes, decoder_name):
    """Create the decoder for the damage classification.
 
    All options output logits for num_classes classes:
    - "minimal": global average pooling and one linear layer.
    Raises ValueError for an unknown decoder_name.
 
    Returns the decoder module.
    """
    feature_dim = 2048

    if decoder_name == "minimal":
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, num_classes)
        )
    else:
        raise ValueError(f"Decoder {decoder_name} for damaged_classification not supported.")


def get_bbch_regression_decoder(decoder_name):
    """Create the decoder for the BBCH regression.
     
    All options end with a Sigmoid, so the output is in [0, 1] like the normalized
    target:
    - "minimal": global average pooling and one linear layer.
    Raises ValueError for an unknown decoder_name.
    
    Returns the decoder module.
    """
    feature_dim = 2048
    out_dim = 1

    if decoder_name == "minimal":
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, out_dim),
            nn.Sigmoid()
        )
    else:
        raise ValueError(f"Decoder {decoder_name} for bbch_regression not supported.")


def get_bbch_classification_decoder(num_classes, decoder_name):
    """Create the decoder for the BBCH classification.
 
    All options output logits for num_classes classes:
    - "minimal": global average pooling and one linear layer.
    Raises ValueError for an unknown decoder_name.
 
    Returns the decoder module.
    """
    feature_dim = 2048 

    if decoder_name == "minimal":
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, num_classes)
        )
    else:
        raise ValueError(f"Decoder {decoder_name} for bbch_classification not supported.")

