from torchvision.models import resnet50, ResNet50_Weights

resn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

print(resn)