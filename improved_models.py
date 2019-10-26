import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from coatt_models import WordEmbedCoResNet

#code of dilated convolution part is referenced from https://github.com/speedinghzl/Pytorch-Deeplab

class IterativeWordEmbedCoResNet(WordEmbedCoResNet):
    def __init__(self, block, layers, num_classes, data_dir='./datasets/'):
        super(IterativeWordEmbedCoResNet, self).__init__(block, layers, num_classes, data_dir=data_dir)
        self.reduction_cat = nn.Conv2d(512, 256, 1, bias=False)

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

        word_embedding = torch.stack(word_embedding).cuda()
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
        Sr = F.softmax(torch.transpose(S, 1, 2), dim = 1) #

        uq = torch.bmm(exemplar_flat, Sc).contiguous()
        uq = uq.view(-1, channel, fea_size[0], fea_size[1])

        us = torch.bmm(query_flat, Sr).contiguous()
        us = us.view(-1, channel, fea_size[0], fea_size[1])

        input2_mask = self.gate(uq)
        input2_mask = self.gate_s(input2_mask)

        input1_mask = self.gate(us)
        input1_mask = self.gate_s(input1_mask)

        uq = uq * input2_mask
        us = us * input1_mask

        uq = self.reduction(uq)
        us = self.reduction(us)

        return uq, us

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
        va1, vb1 = self.coattend(query_rgb, support_rgb, support_lbl)

        va1 = self.relu(self.reduction_cat(torch.cat([query_rgb, va1], dim=1)))
        vb1 = self.relu(self.reduction_cat(torch.cat([support_rgb, vb1], dim=1)))

        va2, vb2 = self.coattend(va1, vb1, support_lbl)

        va2 = self.relu(self.reduction_cat(torch.cat([query_rgb, va2], dim=1)))
        vb2 = self.relu(self.reduction_cat(torch.cat([support_rgb, vb2], dim=1)))

        va2 = va1 + va2
        vb2 = vb1 + vb2

        va3, _ = self.coattend(va2, vb2, support_lbl)

        z = va2 + va3

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
