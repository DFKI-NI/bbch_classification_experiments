"""Shared encoder (backbone) of the MTL model."""


from LibMTL.model import resnet_dilated, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d


def get_encoder(params):
    """Create the encoder chosen with params.encoder_variant.
 
    All variants are ResNet or ResNeXt backbones from LibMTL with 2048 output
    channels, optionally with ImageNet weights (params.pretrained):
    - dil_resnet50, dil_resnet101, dil_resnext50_32x4d, dil_resnext101_32x8d:
      dilated versions. The last stages use dilation instead of stride, so the
      feature map is only 8x or 16x smaller than the image (params.dilate_scale).
      This keeps more detail for the pixel-wise tasks.
    - resnet50, resnet101, resnext50_32x4d, resnext101_32x8d: standard versions.
    Raises ValueError for an unknown variant.
 
    Returns the encoder module.
    """
    print(f"Encoder Variant:\n        {params.encoder_variant}")
    if params.encoder_variant == 'dil_resnet50':
        return resnet_dilated("resnet50", pretrained=params.pretrained, dilate_scale=params.dilate_scale)
    elif params.encoder_variant == 'dil_resnet101':
        return resnet_dilated("resnet101", pretrained=params.pretrained, dilate_scale=params.dilate_scale)
    elif params.encoder_variant == 'dil_resnext50_32x4d':
        return resnet_dilated("resnext50_32x4d", pretrained=params.pretrained, dilate_scale=params.dilate_scale)
    elif params.encoder_variant == 'dil_resnext101_32x8d':
        return resnet_dilated("resnext101_32x8d", pretrained=params.pretrained, dilate_scale=params.dilate_scale)
    elif params.encoder_variant == 'resnet50':
        return resnet50(pretrained=params.pretrained)
    elif params.encoder_variant == 'resnet101':
        return resnet101(pretrained=params.pretrained)
    elif params.encoder_variant == 'resnext50_32x4d':
        return resnext50_32x4d(pretrained=params.pretrained)
    elif params.encoder_variant == 'resnext101_32x8d':
        return resnext101_32x8d(pretrained=params.pretrained)
    else:
        raise ValueError(f"No support for {params.encoder_variant}")
    
    
    
