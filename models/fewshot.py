"""
Fewshot Semantic Segmentation
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as bkpt
from .vgg import Encoder
import numpy as np

class FewShotSeg(nn.Module):
    """
    Fewshot Segmentation model

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}

        # Encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', Encoder(in_channels, self.pretrained_path)),]))

        self.layer55 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2,
                      bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5))        
    
        self.layer9=nn.Conv2d(256, 2, kernel_size=1, stride=1, bias=True)
        self.linear_word_embedding = nn.Linear(300, 256, bias=False)
        self.word2vec = np.load('embeddings_word2vec_pascal.npy',\
                        allow_pickle=True).item()
        self.classes = ['plane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'table', 'dog', 'horse',
                       'motorbike', 'person', 'plant',
                       'sheep', 'sofa', 'train', 'monitor']        
        self.linear_e = nn.Linear(512, 512, bias = False)
        self.reduction = nn.Conv2d(512, 256, 1, bias=False)
        self.gate = nn.Conv2d(512, 1, kernel_size = 1, bias = False)        
        self.gate_s = nn.Sigmoid()

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, support_lbl):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        nwe = self.extract_nwe(support_lbl)
#        bkpt()
       # srgb_size = supp_imgs[0][0].shape
        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)
        img_fts = self.encoder(imgs_concat)
        img_fts = self.layer55(img_fts)
        fts_size = img_fts.shape[-2:]

        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size) # N x B x C x H' x W'
        
        supp_fts = supp_fts.squeeze()
        query_rgb = qry_fts.squeeze(0)

        query_rgb_rep = query_rgb.unsqueeze(1).repeat(1, 5, 1, 1, 1)
        sqry_size = query_rgb_rep.shape
        query_rgb_rep = query_rgb_rep.view(-1, sqry_size[2], sqry_size[3], sqry_size[4])

        nwe_rep = nwe.unsqueeze(1).repeat(1, 5, 1)
        nwe_rep = nwe_rep.view(-1, nwe.shape[1])

        z = self.coattend(query_rgb_rep, supp_fts, nwe_rep)

        z = z.view(1, 5, z.shape[1], z.shape[2], z.shape[3]) #batch-size, n_shot, ,,,
        z = torch.mean(z, dim=1)        
        out=self.layer9(z)
        return out


#        fore_mask = torch.stack([torch.stack(way, dim=0)
#                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
#        back_mask = torch.stack([torch.stack(way, dim=0)
#                                 for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'

        ###### Compute loss ######
#        align_loss = 0
#        outputs = []
#        for epi in range(batch_size):
            ###### Extract prototype ######
#            supp_fg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
#                                             fore_mask[way, shot, [epi]])
#                            for shot in range(n_shots)] for way in range(n_ways)]
#            supp_bg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
#                                             back_mask[way, shot, [epi]])
#                            for shot in range(n_shots)] for way in range(n_ways)]

            ###### Obtain the prototypes######
#            fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)

            ###### Compute the distance ######
#            prototypes = [bg_prototype,] + fg_prototypes
#            dist = [self.calDist(qry_fts[:, epi], prototype) for prototype in prototypes]
#            pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H' x W'
#            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

            ###### Prototype alignment loss ######
#            if self.config['align'] and self.training:
#                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi],
#                                                fore_mask[:, :, epi], back_mask[:, :, epi])
#                align_loss += align_loss_epi

#        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
#        output = output.view(-1, *output.shape[2:])
#        return output, align_loss / batch_size


    def calDist(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
        return dist


    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
            / (mask[None, ...].sum(dim=(2, 3)) + 1e-5) # 1 x C
        return masked_fts


    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype


    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding fatures for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'
        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3, 4))
        qry_prototypes = qry_prototypes / (pred_mask.sum((0, 3, 4)) + 1e-5)  # (1 + Wa) x C

        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            prototypes = [qry_prototypes[[0]], qry_prototypes[[way + 1]]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, [shot]]
                supp_dist = [self.calDist(img_fts, prototype) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:],
                                          mode='bilinear')
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss = loss + F.cross_entropy(
                    supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways
        return loss


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

    def extract_nwe(self, lbl):
        word_embedding = []
        for cls in lbl:
            cls = self.classes[cls-1]
            word_embedding.append(torch.tensor(self.word2vec[cls]))

        word_embedding = torch.stack(word_embedding).cuda().float()
        return word_embedding    


