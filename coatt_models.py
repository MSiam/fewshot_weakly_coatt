import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from base_models import ResNet, FilMedBottleneck, FiLM
import os


class CoResNet(ResNet):
    def __init__(self, block, layers, num_classes):
        super(CoResNet, self).__init__(block, layers, num_classes)
        self.linear_e = nn.Linear(256, 256, bias = False)
        self.gate = nn.Conv2d(256, 1, kernel_size  = 1, bias = False)
        self.gate_s = nn.Sigmoid()

    def coattend(self, va, vb, sprt_l):
        """
        Performs coattention between support set and query set
        va: query features
        vb: support features
        sprt_l: support image-level label
        """
        channel = va.shape[1]
        fea_size = va.shape[2:]

        exemplar_flat = vb.view(vb.shape[0], vb.shape[1], -1) #N,C,H*W
        query_flat = va.view(va.shape[0], va.shape[1], -1)

        exemplar_t = torch.transpose(exemplar_flat,1,2).contiguous()
        exemplar_corr = self.linear_e(exemplar_t)
        S = torch.bmm(exemplar_corr, query_flat)
        Sc = F.softmax(S, dim = 1) #

        za = torch.bmm(exemplar_flat, Sc).contiguous()
        za = za.view(-1, channel, fea_size[0], fea_size[1])

        input2_mask = self.gate(za)
        input2_mask = self.gate_s(input2_mask)

        za = za * input2_mask
        return za

    def forward(self, query_rgb, support_rgb, support_lbl, history_mask):
        # important: do not optimize the RESNET backbone
        query_rgb = self.conv1(query_rgb)
        query_rgb = self.bn1(query_rgb)
        query_rgb = self.relu(query_rgb)
        query_rgb = self.maxpool(query_rgb)
        query_rgb = self.layer1(query_rgb)
        query_rgb = self.layer2(query_rgb)
        query_feat_layer2=query_rgb
        query_rgb = self.layer3(query_rgb)
        query_rgb=torch.cat([query_feat_layer2,query_rgb],dim=1)
        query_rgb = self.layer5(query_rgb)

        feature_size = query_rgb.shape[-2:]

        #side branch,get latent embedding z
        support_rgb = self.conv1(support_rgb)
        support_rgb = self.bn1(support_rgb)
        support_rgb = self.relu(support_rgb)
        support_rgb = self.maxpool(support_rgb)
        support_rgb = self.layer1(support_rgb)
        support_rgb = self.layer2(support_rgb)
        support_feat_layer2 = support_rgb
        support_rgb = self.layer3(support_rgb)
        support_rgb = torch.cat([support_feat_layer2, support_rgb], dim=1)
        support_rgb = self.layer5(support_rgb)


        h,w=support_rgb.shape[-2:][0],support_rgb.shape[-2:][1]
        z = self.coattend(query_rgb, support_rgb, support_lbl)

        history_mask=F.interpolate(history_mask,feature_size,mode='bilinear',align_corners=True)

        out=torch.cat([query_rgb,z],dim=1)
        out = self.layer55(out)
        out_plus_history=torch.cat([out,history_mask],dim=1)
        out = out + self.residule1(out_plus_history)
        out = out + self.residule2(out)
        out = out + self.residule3(out)

        global_feature=F.avg_pool2d(out,kernel_size=feature_size)
        global_feature=self.layer6_0(global_feature)
        global_feature=global_feature.expand(-1,-1,feature_size[0],feature_size[1])
        out=torch.cat([global_feature,self.layer6_1(out),self.layer6_2(out),self.layer6_3(out),self.layer6_4(out)],dim=1)
        out=self.layer7(out)

        out=self.layer9(out)
        return out

class WordEmbedCoResNet(CoResNet):
    def __init__(self, block, layers, num_classes, film_gen=None, data_dir='./datasets/',
                 embed='word2vec', dataset_name='pascal'):
        super(WordEmbedCoResNet, self).__init__(block, layers, num_classes)
        self.film_gen = film_gen
        if embed == 'word2vec':
            self.linear_word_embedding = nn.Linear(300, 256, bias=False)
        elif embed == 'fasttext':
            self.linear_word_embedding = nn.Linear(300, 256, bias=False)
        elif embed == 'concat':
            self.linear_word_embedding = nn.Linear(600, 256, bias=False)

        self.word2vec = np.load(os.path.join(data_dir, 'embeddings_%s_%s.npy'%(embed, dataset_name)),\
                                allow_pickle=True).item()
        if dataset_name == 'pascal':
            self.classes = ['plane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat', 'chair',
                            'cow', 'table', 'dog', 'horse',
                            'motorbike', 'person', 'plant',
                            'sheep', 'sofa', 'train', 'monitor']
        elif dataset_name == 'coco':
            self.classes = []
            classes_f = open(os.path.join(data_dir, 'coco_classes.txt'), 'r')
            for line in classes_f:
                self.classes.append(line.strip().replace(' ', '_'))
            classes_f.close()

        if self.film_gen is None:
            self.linear_e = nn.Linear(512, 512, bias = False)
            self.reduction = nn.Conv2d(512, 256, 1, bias=False)
            self.gate = nn.Conv2d(512, 1, kernel_size  = 1, bias = False)
        else:
            self.inplanes = 512
            self.layer3 = self._make_layer(FilMedBottleneck, 256, layers[2], stride=1, dilation=2)
            self.linear_e = nn.Linear(256, 256, bias = False)
            self.gate = nn.Conv2d(256, 1, kernel_size  = 1, bias = False)
            self.film = FiLM(linear=True)

    def filmed_coattend(self, va, vb, gammas, betas):
        """
        Performs coattention between support set and query set
        va: query features
        vb: support features
        sprt_l: support image-level label
        """
        channel = va.shape[1]
        fea_size = va.shape[2:]

        exemplar_flat = vb.view(vb.shape[0], vb.shape[1], -1) #N,C,H*W
        query_flat = va.view(va.shape[0], va.shape[1], -1)

        exemplar_t = torch.transpose(exemplar_flat,1,2).contiguous()
        exemplar_corr = self.linear_e(exemplar_t)
        exemplar_corr = self.film(exemplar_corr, gammas, betas)
        S = torch.bmm(exemplar_corr, query_flat)
        Sc = F.softmax(S, dim = 1) #

        za = torch.bmm(exemplar_flat, Sc).contiguous()
        za = za.view(-1, channel, fea_size[0], fea_size[1])

        input2_mask = self.gate(za)
        input2_mask = self.gate_s(input2_mask)

        za = za * input2_mask
        return za

    def extract_nwe(self, lbl):
        word_embedding = []
        for cls in lbl:
            cls = self.classes[cls-1]
            word_embedding.append(torch.tensor(self.word2vec[cls]))

        word_embedding = torch.stack(word_embedding).cuda().float()
        return word_embedding

    def coattend(self, va, vb, word_embedding):
        """
        Performs coattention between support set and query set
        va: query features
        vb: support features
        sprt_l: support image-level label
        """
        channel = va.shape[1]*2
        fea_size = va.shape[2:]

        word_embedding = self.linear_word_embedding(word_embedding)

        word_embedding = word_embedding.unsqueeze(2).unsqueeze(2)
        word_embedding_tiled = word_embedding.repeat(1, 1, va.shape[2], va.shape[3])

        va = torch.cat((va, word_embedding_tiled), 1)
        vb = torch.cat((vb, word_embedding_tiled), 1)

        exemplar_flat = vb.view(vb.shape[0], vb.shape[1], -1) #N,C,H*W
        query_flat = va.view(va.shape[0], va.shape[1], -1)

        exemplar_t = torch.transpose(exemplar_flat,1,2).contiguous()
        exemplar_corr = self.linear_e(exemplar_t)
        S = torch.bmm(exemplar_corr, query_flat)
        Sc = F.softmax(S, dim = 1) #

        za = torch.bmm(exemplar_flat, Sc).contiguous()
        za = za.view(-1, channel, fea_size[0], fea_size[1])

        input2_mask = self.gate(za)
        input2_mask = self.gate_s(input2_mask)

        za = za * input2_mask
        za = self.reduction(za)
        return za

    def forward(self, query_rgb, support_rgb, support_lbl, history_mask):
        nwe = self.extract_nwe(support_lbl)

        if self.film_gen is not None:
            gammas_256 = self.film_gen.gen_gammas_256(nwe)
            betas_256 = self.film_gen.gen_betas_256(nwe)
            gammas_1024 = self.film_gen.gen_gammas_1024(nwe)
            betas_1024 = self.film_gen.gen_betas_1024(nwe)

        # important: do not optimize the RESNET backbone
        query_rgb = self.conv1(query_rgb)
        query_rgb = self.bn1(query_rgb)
        query_rgb = self.relu(query_rgb)
        query_rgb = self.maxpool(query_rgb)
        query_rgb = self.layer1(query_rgb)
        query_rgb = self.layer2(query_rgb)
        query_feat_layer2=query_rgb

        if self.film_gen is not None:
            inputs ={'x': query_rgb, 'gammas': gammas_1024, 'betas': betas_1024}
            query_rgb = self.layer3(inputs)['x']
        else:
            query_rgb = self.layer3(query_rgb)

        query_rgb=torch.cat([query_feat_layer2,query_rgb],dim=1)
        query_rgb = self.layer5(query_rgb)

        feature_size = query_rgb.shape[-2:]

        #side branch,get latent embedding z
        support_rgb = self.conv1(support_rgb)
        support_rgb = self.bn1(support_rgb)
        support_rgb = self.relu(support_rgb)
        support_rgb = self.maxpool(support_rgb)

        support_rgb = self.layer1(support_rgb)
        support_rgb = self.layer2(support_rgb)
        support_feat_layer2 = support_rgb

        if self.film_gen is not None:
            inputs ={'x': support_rgb, 'gammas': gammas_1024, 'betas': betas_1024}
            support_rgb = self.layer3(inputs)['x']
        else:
            support_rgb = self.layer3(support_rgb)

        support_rgb = torch.cat([support_feat_layer2, support_rgb], dim=1)
        support_rgb = self.layer5(support_rgb)

        h,w=support_rgb.shape[-2:][0],support_rgb.shape[-2:][1]

        if self.film_gen is not None:
            z = self.filmed_coattend(query_rgb, support_rgb, gammas_256,
                                     betas_256)
        else:
            z = self.coattend(query_rgb, support_rgb, nwe)

        history_mask=F.interpolate(history_mask,feature_size,mode='bilinear',align_corners=True)

        out=torch.cat([query_rgb,z],dim=1)
        out = self.layer55(out)
        out_plus_history=torch.cat([out,history_mask],dim=1)
        out = out + self.residule1(out_plus_history)
        out = out + self.residule2(out)
        out = out + self.residule3(out)

        global_feature=F.avg_pool2d(out,kernel_size=feature_size)
        global_feature=self.layer6_0(global_feature)
        global_feature=global_feature.expand(-1,-1,feature_size[0],feature_size[1])
        out=torch.cat([global_feature,self.layer6_1(out),self.layer6_2(out),self.layer6_3(out),self.layer6_4(out)],dim=1)
        out=self.layer7(out)

        out=self.layer9(out)
        return out

class WordEmbedResNet(CoResNet):
    def __init__(self, block, layers, num_classes, data_dir='./datasets/', embed='word2vec', dataset_name='pascal'):
        super(WordEmbedResNet, self).__init__(block, layers, num_classes)
        if embed == 'word2vec':
            self.linear_word_embedding = nn.Linear(300, 256, bias=False)
        elif embed == 'fasttext':
            self.linear_word_embedding = nn.Linear(300, 256, bias=False)
        elif embed == 'concat':
            self.linear_word_embedding = nn.Linear(600, 256, bias=False)

        if dataset_name == 'pascal':
            self.classes = ['plane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat', 'chair',
                            'cow', 'table', 'dog', 'horse',
                            'motorbike', 'person', 'plant',
                            'sheep', 'sofa', 'train', 'monitor']
        elif dataset_name == 'coco':
            self.classes = []
            classes_f = open(os.path.join(data_dir, 'coco_classes.txt'), 'r')
            for line in classes_f:
                self.classes.append(line.strip().replace(' ', '_'))
            classes_f.close()

        self.reduction = nn.Conv2d(512, 256, 1, bias=False)

    def coattend(self, va, vb, sprt_l):
        """
        Performs coattention between support set and query set
        va: query features
        vb: support features
        sprt_l: support image-level label
        """
        channel = va.shape[1]*2
        fea_size = va.shape[2:]

        word_embedding = []
        for cls in sprt_l:
            cls = self.classes[cls-1]
            word_embedding.append(torch.tensor(self.word2vec[cls]))

        word_embedding = torch.stack(word_embedding).cuda().float()
        word_embedding = self.linear_word_embedding(word_embedding)

        word_embedding = word_embedding.unsqueeze(2).unsqueeze(2)
        word_embedding_tiled = word_embedding.repeat(1, 1, va.shape[2], va.shape[3])
        va = torch.cat((va, word_embedding_tiled), 1)
        za = self.reduction(va)
        return za

