from coatt_models import CoResNet, WordEmbedResNet, WordEmbedCoResNet, WordEmbedProtoResNet
from improved_models import IterativeWordEmbedCoResNet
from base_models import ResNet, FilMGen, Bottleneck


def Res_Deeplab(data_dir='./datasets/', num_classes=2, model_type='nwe_coatt', filmed=False):
    if filmed and 'nwe' in model_type:
        bottleneck_module = Bottleneck#FilMedBottleneck
        film_gen = FilMGen()
    else:
        bottleneck_module = Bottleneck
        film_gen = None

    if model_type == 'iter_nwe_coatt':
        model = IterativeWordEmbedCoResNet(bottleneck_module, [3, 4, 6, 3],
                                           num_classes, data_dir=data_dir)
    elif model_type == 'nwe_coatt':
        model = WordEmbedCoResNet(bottleneck_module, [3, 4, 6, 3],
                                  num_classes, film_gen, data_dir=data_dir)
    elif model_type == 'coatt':
        model = CoResNet(bottleneck_module, [3, 4, 6, 3],
                         num_classes)
    elif model_type == 'nwe':
        model = WordEmbedResNet(bottleneck_module, [3, 4, 6, 3],
                                num_classes, data_dir=data_dir)
    elif model_type == 'nwe_proto':
        model = WordEmbedProtoResNet(bottleneck_module, [3, 4, 6, 3],
                                num_classes, data_dir=data_dir)
    else:
        model = ResNet(bottleneck_module,[3, 4, 6, 3],
                       num_classes)
    return model
