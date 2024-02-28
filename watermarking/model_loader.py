import timm
import torch

def load_model(model_path, architecture, num_channels, num_classes, pretrained=False):
    
    try:
        model = timm.create_model(model_name=architecture, pretrained=pretrained, in_chans=num_channels, num_classes=num_classes)
    except:
        print(f'There is no such model in the timm module: {architecture}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model