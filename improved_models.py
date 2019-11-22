import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from coatt_models import WordEmbedCoResNet

#code of dilated convolution part is referenced from https://github.com/speedinghzl/Pytorch-Deeplab

class IterativeWordEmbedCoResNet(WordEmbedCoResNet):
    def __init__(self, block, layers, num_classes, data_dir='./datasets/', embed='word2vec', dataset_name='pascal'):
        super(IterativeWordEmbedCoResNet, self).__init__(block, layers, num_classes,
                                                         data_dir=data_dir, embed=embed,
                                                         dataset_name=dataset_name)
        self.reduction_cat = nn.Conv2d(512, 256, 1, bias=False)

    def coattend(self, va, vb, sprt_l, srgb_size):
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
        word_embedding_rep = word_embedding.unsqueeze(1).repeat(1, srgb_size[1], 1)
        word_embedding_rep = word_embedding_rep.view(-1, word_embedding.shape[1])

        word_embedding_rep = self.linear_word_embedding(word_embedding_rep)

        word_embedding_rep = word_embedding_rep.unsqueeze(2).unsqueeze(2)
        word_embedding_tiled = word_embedding_rep.repeat(1, 1, va.shape[2], va.shape[3])
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

        return uq, us, input2_mask, input1_mask

    def forward(self, query_rgb, support_rgb, support_lbl, history_mask,
                history_masks_sprt, side_output=False):
        srgb_size = support_rgb.shape
        support_rgb = support_rgb.view(-1, srgb_size[2], srgb_size[3], srgb_size[4])

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
        query_rgb_rep = query_rgb.unsqueeze(1).repeat(1, srgb_size[1], 1, 1, 1)
        sqry_size = query_rgb_rep.shape
        query_rgb_rep = query_rgb_rep.view(-1, sqry_size[2], sqry_size[3], sqry_size[4])

        va1, vb1, _, _ = self.coattend(query_rgb_rep, support_rgb, support_lbl, srgb_size)

        va1 = self.relu(self.reduction_cat(torch.cat([query_rgb_rep, va1], dim=1)))
        vb1 = self.relu(self.reduction_cat(torch.cat([support_rgb, vb1], dim=1)))

        va2, vb2, _, _ = self.coattend(va1, vb1, support_lbl, srgb_size)

        zq = va1 + va2
        zs = vb1 + vb2

        # Upsample history masks for support set
        history_masks_sprt = history_masks_sprt.view(-1, history_masks_sprt.shape[2],
                                                     history_masks_sprt.shape[3],
                                                     history_masks_sprt.shape[4])
        history_masks_sprt=F.interpolate(history_masks_sprt, feature_size,mode='bilinear',align_corners=True)

        # Decode predictions for support set and consutrct prototypes
        outs, visual_feats = self.decode(zs, support_rgb, srgb_size, history_masks_sprt,
                                         feature_size, sprt_flag=True, prototypes=None)
        pred_sprt_mask = outs.argmax(dim=1, keepdim=True)
        binary_sprt_masks = [pred_sprt_mask == i for i in range(2)]
        pred_sprt_mask = torch.stack(binary_sprt_masks, dim=1).float()
        masked_feats = visual_feats.unsqueeze(1) * pred_sprt_mask
        masked_feats = masked_feats.view(srgb_size[0], srgb_size[1],
                                         masked_feats.shape[1],
                                         masked_feats.shape[2],
                                         masked_feats.shape[3],
                                         masked_feats.shape[4])
        pred_sprt_mask = pred_sprt_mask.view(srgb_size[0], srgb_size[1],
                                             pred_sprt_mask.shape[1],
                                             pred_sprt_mask.shape[2],
                                             pred_sprt_mask.shape[3],
                                             pred_sprt_mask.shape[4])
        protos = torch.sum(masked_feats, dim=(1, 4, 5))
        protos = protos / (pred_sprt_mask.sum((1, 4, 5)) + 1e-5)

        # upsample history_mask for decoding of query
        history_mask=F.interpolate(history_mask,feature_size,mode='bilinear',align_corners=True)

        # Decode predictions for query based on prototypes
        outq, _ = self.decode(zq, query_rgb, srgb_size, history_mask,
                           feature_size, sprt_flag=False, prototypes=protos)
#        pred_mask = outq.argmax(dim=1, keepdim=True)
#        binary_masks = [pred_mask == i for i in range(2)]
#        pred_mask = torch.stack(binary_masks, dim=1).float()
#        protos_qry = torch.sum(query_rgb.unsqueeze(1) * pred_mask, dim=(3, 4))
#        protos_qry = protos_qry / (pred_mask.sum((3, 4)) + 1e-5)  # (1 + Wa) x C

        if side_output:
            gate_s = torch.cat((1-gate_s, gate_s), dim=1)
            gate_q = torch.cat((1-gate_q, gate_q), dim=1)
            extras = [gate_s, gate_q]
        else:
            extras = None
        return outq, outs, extras#, protos, protos_qry

