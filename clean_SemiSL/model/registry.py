from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, convnext_small, ConvNeXt_Small_Weights, convnext_base, ConvNeXt_Base_Weights, convnext_large, ConvNeXt_Large_Weights

MODELS = {
    "resnet18": lambda pretrained, kwargs: resnet18(ResNet18_Weights.DEFAULT if pretrained else None, **kwargs),
    "resnet50": lambda pretrained, kwargs: resnet50(ResNet50_Weights.DEFAULT if pretrained else None, **kwargs),
    "resnet101": lambda pretrained, kwargs: resnet101(ResNet18_Weights.DEFAULT if pretrained else None, **kwargs),
    "convnext_tiny": lambda pretrained, kwargs: convnext_tiny(ConvNeXt_Tiny.DEFAULT if pretrained else None, **kwargs),
    "convnext_small": lambda pretrained, kwargs: convnext_small(ConvNeXt_Small.DEFAULT if pretrained else None, **kwargs),
    "convnext_base": lambda pretrained, kwargs: convnext_base(ConvNeXt_Base.DEFAULT if pretrained else None, **kwargs),
    "convnext_large": lambda pretrained, kwargs: convnext_large(ConvNeXt_Large.DEFAULT if pretrained else None, **kwargs),
}

def load_model(name, pretrained: bool, **kwargs):
    return MODELS[name](pretrained, kwargs) 
