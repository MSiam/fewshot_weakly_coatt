from coatt_models import CoResNet, WordEmbedResNet, WordEmbedCoResNet, WordEmbedProtoResNet, SimplerNet, SimpleWordEmbedCoResNet
from improved_models import IterativeWordEmbedCoResNet
from base_models import ResNet, FilMGen, Bottleneck


def Res_Deeplab(data_dir='./datasets/', num_classes=2, model_type='nwe_coatt',
                filmed=False, embed='word2vec', dataset_name='pascal',
                backbone='resnet50', multires_flag=True):

    if filmed and 'nwe' in model_type:
        bottleneck_module = Bottleneck#FilMedBottleneck
        film_gen = FilMGen()
    else:
        bottleneck_module = Bottleneck
        film_gen = None

    if backbone == 'resnet50':
        block_list = [3, 4, 6, 3]
    elif backbone == 'resnet101':
        block_list = [3, 4, 23, 3]

    if model_type == 'iter_nwe_coatt':
        model = IterativeWordEmbedCoResNet(bottleneck_module, block_list,
                                           num_classes, data_dir=data_dir,
                                           embed=embed, dataset_name=dataset_name,
                                           multires_flag=multires_flag)
    elif model_type == 'nwe_coatt':
        model = WordEmbedCoResNet(bottleneck_module, block_list,
                                  num_classes, film_gen, data_dir=data_dir,
                                  embed=embed, dataset_name=dataset_name)
    elif model_type == 'coatt':
        model = CoResNet(bottleneck_module, block_list,
                         num_classes)
    elif model_type == 'simple':
        model = SimplerNet(bottleneck_module, block_list, num_classes)
    elif model_type == 'simple_nwe_coatt':
        model = SimpleWordEmbedCoResNet(bottleneck_module, block_list, num_classes,
                                        data_dir=data_dir, embed=embed,
                                        dataset_name=dataset_name)
    elif model_type == 'nwe':
        model = WordEmbedResNet(bottleneck_module, block_list,
                                num_classes, data_dir=data_dir,
                                embed=embed, dataset_name=dataset_name)
    elif model_type == 'nwe_proto':
        model = WordEmbedProtoResNet(bottleneck_module, block_list,
                                     num_classes, data_dir=data_dir,
                                     embed=embed, dataset_name=dataset_name)
    else:
        model = ResNet(bottleneck_module,block_list,
                       num_classes)
    return model
