import torch
import torch.nn as nn
# from model import My_CNN
# import sys
 
# # adding Folder_2 to the system path
# sys.path.insert(0, '..')

'''
For now, I just stole this architecture from
"Fast processing of environmental DNA metabarcoding 
sequence data using convolutional neural networks"
- it seems somewhat silly, and will likely need to be replaced'''
class My_CNN(nn.Module):
    def __init__(self, in_channels=4, 
                    conv1_out_channels=16, conv1_kernel_size=13,
                    conv2_out_channels=32, conv2_kernel_size=13,
                    conv3_out_channels=64, conv3_kernel_size=13,
                    in_len=150, n_classes=1):
        super().__init__()
        self.in_channels = in_channels
        self.in_len = in_len

        self.conv1_out_channels = conv1_out_channels
        self.conv2_out_channels = conv2_out_channels
        self.conv3_out_channels = conv3_out_channels

        self.conv1_kernel_size = conv1_kernel_size
        self.conv2_kernel_size = conv2_kernel_size
        self.conv3_kernel_size = conv3_kernel_size

        print("conv1_out_channels: ", conv1_out_channels)
        print("conv2_out_channels: ", conv2_out_channels)
        print("conv3_out_channels: ", conv3_out_channels)
        print("NOTE: This model uses 6 convolutional layers total")
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, 
                    out_channels=conv1_out_channels, 
                    kernel_size=conv1_kernel_size,
                    padding=conv1_kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv1_out_channels, 
                    out_channels=conv2_out_channels, 
                    kernel_size=conv2_kernel_size,
                    padding=conv2_kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv2_out_channels, 
                    out_channels=conv3_out_channels, 
                    kernel_size=conv3_kernel_size,
                    padding=conv3_kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.Conv1d(in_channels=conv3_out_channels, 
                    out_channels=conv3_out_channels*2, 
                    kernel_size=conv1_kernel_size,
                    padding=conv1_kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv3_out_channels*2, 
                    out_channels=conv3_out_channels*4, 
                    kernel_size=conv2_kernel_size,
                    padding=conv2_kernel_size // 2),    
            nn.ReLU(),
            nn.Conv1d(in_channels=conv3_out_channels*4, 
                    out_channels=conv3_out_channels*8, 
                    kernel_size=conv3_kernel_size,
                    padding=conv3_kernel_size // 2),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv_layers(x)
        # Reshape out to feed into linear layer
        # out = out.view(out.shape[0], -1)
        return out

    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    

def base_model_features(pretrained=False, conv1_out_channels=16, **kwargs):
    """Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    conv1_out_channels=conv1_out_channels
    model = My_CNN(conv1_out_channels=conv1_out_channels, conv1_kernel_size=13,
                            conv2_out_channels=conv1_out_channels*2, conv2_kernel_size=13,
                            conv3_out_channels=conv1_out_channels*4, conv3_kernel_size=13).cuda()
    if pretrained:
        my_dict = torch.load('../2_12_threshold_4_base_model_out_ch_{}.pth'.format(conv1_out_channels))
        # model.load_state_dict(torch.load('../base_model_weights.pth'))
        # model.eval()
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('last_layer'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)
    return model

if __name__ == '__main__':
    base = base_model_features(pretrained=True)
    print(base)