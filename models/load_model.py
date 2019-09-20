
import pickle
import torch

from .inceptionresnetv2 import InceptionResNetV2
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .resnet_vggface2 import resnet50_vggface2
from .model_irse import IR_50
from .embeddingNet import EmbeddingNet

def load_model(model_arch,
               device,
               checkpoint_path=None,
               embedding_size=128,
               imgnet_pretrained=False):

    is_standard = True
    if model_arch == "resnet18":
        model = resnet18(num_classes=embedding_size, pretrained=imgnet_pretrained)
    elif model_arch == "resnet34":
        model = resnet34(num_classes=embedding_size, pretrained=imgnet_pretrained)
    elif model_arch == "resnet50":
        model = resnet50(num_classes=embedding_size, pretrained=imgnet_pretrained)
    elif model_arch == "resnet101":
        model = resnet101(num_classes=embedding_size, pretrained=imgnet_pretrained)
    elif model_arch == "resnet152":
        model = resnet152(num_classes=embedding_size, pretrained=imgnet_pretrained)
    elif model_arch == "inceptionresnetv2":
        model = InceptionResNetV2(bottleneck_layer_size=embedding_size)
    elif model_arch == "lenet":
        model = EmbeddingNet(n_outputs=embedding_size)
    elif model_arch == "ir50":
        model = IR_50([112, 112])
        if not (checkpoint_path is None):
            print('Loading from checkpoint {}'.format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint)
        is_standard = False
    elif model_arch == "vggface2_resnet50":
        model = resnet50_vggface2(num_classes=8631, include_top=False)
        if not (checkpoint_path is None):
            print('Loading from checkpoint {}'.format(checkpoint_path))
            with open(checkpoint_path, 'rb') as f:
                obj = f.read()
            weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
            model.load_state_dict(weights)
        is_standard = False
    else:
        raise Exception("Model architecture {} is not supported.".format(model_arch))

    if not (checkpoint_path is None) and is_standard:
        print('Loading from checkpoint {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)

    return model