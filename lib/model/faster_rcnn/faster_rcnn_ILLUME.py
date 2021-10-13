import random
import torch
import string
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, \
    _affine_theta,grad_reverse, \
    prob2entropy2,  self_entropy, global_attention, prob2entropy

#from model.faster_rcnn.vgg16_ILLUME import vgg16

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic, gc1, gc2, gc3):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        #self.labelst = labelst
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.gc1 = gc1
        self.gc2 = gc2
        self.gc3 = gc3
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        #---
        self.fc2 = nn.Linear(128, 2)
        self.fc3 = nn.Linear(128, 2)
        #---

    def forward(self, im_data, im_info, gt_boxes, num_boxes,target=False,eta=1.0):
        
        #torch.cuda.empty_cache()
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        labelst = 'target'

        if target == False :
            labelst = 'source'

        #-------------------------------------------------------------
        # feed image data to base model to obtain base feature map
        base_feat1 = self.RCNN_base1(im_data)
        #print('\nbase_feat1: ', base_feat1.shape)
        #print("\nBAse_feat1",base_feat1.requires_grad)
        domain_p1 = self.netD1(grad_reverse(base_feat1, lambd=eta)) # get att map * base_feat1, use 1 atten only

        #print(domain_p1.shape)
        #print("\ndomain_p1",domain_p1.requires_grad)
        #domain_p1_en = prob2entropy2(domain_p1)
        #self.in_channels = 256
        #self.st = 'source'
        #vgg_ob = vgg16()
        #vgg_ob.self_attn2(in_channels=256,st=labelst)
        #ab = self.self_attn
        #ab1 = type(self)()
        #print(self.__class__())
        #print("\n******",ab1)
        #self1 = ab(in_channels=256,st=labelst)

        #base_feat1 = base_feat1 * domain_p1_en

        #print(base_feat1.shape[1])
        #ch_attn1 = base_feat1.shape[1]
        #self.self_attn2.__init__(ch_attn1,labelst)

        self.self_attn2.__init__(256,labelst)
        domain_p1_en = self.self_attn2(base_feat1)

        #f = open("/home2/vkhindkar/ILLUME-PyTorch-main/attnvk.pt", "wb")
        #print("\n________________WRITING_____________\n")
        #f.write(str(domain_p1_en.cpu().detach().numpy()))
  
        #domain_p1_en = domain_p1_en_obj.forward(base_feat1)
        #domain_p1_en = self1(base_feat1)
        domain_p1_en.requires_grad_(True)
        #logger2 = open('/home2/vkhindkar/ILLUME-PyTorch-main/attnvk.txt', 'a')

        #logger2.write(str(domain_p1_en))

        #print(domain_p1_en,"\n\n___________________________________________________")
        #print("\ndomain_p1_en",domain_p1_en.requires_grad)
        #domain_p1_en = self.self_attn1(base_feat1)
        #print("\nentrppy:",domain_p1_en.shape)
        #domain_p1_en = -torch.mul(domain_p1, domain_p1_en_attn)
        #print("\nFinal ebtropyn* domain",domain_p1_en.shape)

        base_feat1 = domain_p1 * domain_p1_en

        #print("\nbase_feat1",base_feat1.requires_grad)
        #print('\nbase_feat1 af: ', base_feat1.shape)
        # print('\natt_map: ', att_map.shape)
        #print('\ndomain_p1: ', domain_p1.shape) 
        # base_feat1 = base_feat1 * att_map_256
        # print('\n att base_feat1 map: ', base_feat1.shape)

        feat1 = self.netD_forward1(base_feat1.detach())

        # base_feat1.detach(): the gradients of self.netD_forward1() will update parameters of ifself,
        # don't update the previous ones! Example:
        # def forward(self, x):
        #   x = self.net1(x)
        #   return self.net2(x.detach()) # training will only update parameters on net2, not net1

        feat1_p = F.softmax(feat1, 1)
        ##feat1_en = prob2entropy(feat1_p)
        ##feat1 = feat1 * feat1_en
        # feat1 = self.netD_forward1(base_feat1) 
        
        # feat1 = feat1 * att_map # atten 2

        # print('\nfeat1: ', feat1.shape) 

        #----------------------------------------------------------------
        base_feat2 = self.RCNN_base2(base_feat1)
        #print(base_feat2.shape[1])
        #ch_attn2 = base_feat2.shape[1]
        #self.self_attn2.__init__(ch_attn2,labelst)
        self.self_attn2.__init__(512,labelst)
        base_feat2 = self.self_attn2(base_feat2)
        #print("base_feat2:",base_feat2.shape)

        domain_p2 = self.netD2(grad_reverse(base_feat2, lambd=eta))
        # print('\ndomain_p2: ', domain_p2.shape)
        # base_feat2 = base_feat2 * att_map_512
        feat2 = self.netD_forward2(base_feat2.detach())
       

        feat2_p = self.fc2(feat2.view(-1, 128)) # nn.Linear(128,2)
        ##feat2 = global_attention(feat2, feat2_p)

        #self.self_attn2.__init__(128,labelst)
        #feat2 = self.self_attn2(feat2)
        #print("feat2",feat2.shape)


        #print("\n______________________________\n")
        #print("base_feat2:",base_feat2.shape,"domain_p2:",domain_p2.shape,"feat2:",feat2.shape,"feat2_p:",feat2_p.shape,"feat2after:",feat2.shape)

        # feat2 = self.netD_forward2(base_feat2)
        # feat2 = feat2 * att_map_128

        # print('\nbase_feat2: ', base_feat2.shape) 
        # print('\ncam_logit_p2: ', cam_logit_p2.shape)

        # print('\nfeat2: ', feat2.shape)  

        # domain_p2_sig, _ = self.netD12(grad_reverse(base_feat2, lambd=eta))
        # base_feat2 = base_feat2 * att_map_512
        #----------------------------------------------------------------

        base_feat = self.RCNN_base3(base_feat2)
        #print(base_feat.shape[1])
        #ch_attn3 = base_feat.shape[1]
        #self.self_attn2.__init__(ch_attn3,labelst)
        self.self_attn2.__init__(512,labelst)
        base_feat = self.self_attn2(base_feat)
        #print(base_feat.shape)

        domain_p3 = self.netD3(grad_reverse(base_feat, lambd=eta))
        # print('\ndomain_p3: ', domain_p3.shape) 
        # print('\nbase_feat: ', base_feat.shape) 

        # print('\ncam_logit_p3: ', cam_logit_p3.shape) 


        # base_feat = base_feat * att_map_1024
        feat3 = self.netD_forward3(base_feat.detach())
        feat3_p = self.fc3(feat3.view(-1, 128))
        ##feat3 = global_attention(feat3, feat3_p)

        # feat3_en = prob2entropy(F.sigmoid(feat3))
        # feat3 = feat3 * feat3_en

        # feat3 = self.netD_forward3(base_feat)
        # print('\nfeat3: ', feat3.shape) 


        # feat3 = feat3 * att_map_128
        # domain_p3_sig, _ = self.netD13(grad_reverse(base_feat, lambd=eta))
        # base_feat = base_feat * att_map_1024

        #----------------------------------------------------------------


        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground truth bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)

        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))

        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        feat1 = feat1.view(1, -1).repeat(pooled_feat.size(0), 1)
        pooled_feat = torch.cat((feat1, pooled_feat), 1)

        feat2 = feat2.view(1, -1).repeat(pooled_feat.size(0), 1)
        pooled_feat = torch.cat((feat2, pooled_feat), 1)

        
        feat3 = feat3.view(1, -1).repeat(pooled_feat.size(0), 1)
        pooled_feat = torch.cat((feat3, pooled_feat), 1)

        #---------------------------------------------------------------
        d_inst = self.netD_inst(grad_reverse(pooled_feat, lambd=eta)) ## Add entropy!!?
        #---------------------------------------------------------------
        # print('\nd_inst: ', d_inst.shape) #torch.Size([128, 2])
        #---
        # add entropy loss here




        #---
        if target:
            return d_inst, domain_p1, domain_p2, domain_p3, \
                feat1_p, feat2_p, feat3_p
                # cam_logit_p1, cam_logit_p2, cam_logit_p3
                # domain_p12, domain_p2_sig, domain_p3_sig

        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
                RCNN_loss_cls, RCNN_loss_bbox, rois_label, \
                d_inst, domain_p1, domain_p2, domain_p3, \
                feat1_p, feat2_p, feat3_p 
                # cam_logit_p1, cam_logit_p2, cam_logit_p3
                # domain_p12, domain_p2_sig, domain_p3_sig

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
