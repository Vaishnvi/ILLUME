# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from lib.model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, \
#     _affine_theta,grad_reverse, \
#     prob2entropy, self_entropy, global_attention, prob2entropy2

import os
#os.environ = '0,1,2,3'
#os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
import sys
import numpy as np
import pprint
import time
import _init_paths

import torch
import cv2
from torch.autograd import Variable
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.parser_func_multi import parse_args,set_dataset_args

import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3



lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)
  print(args.vis, "   ", args.dataset)
  # exit() 
  args = set_dataset_args(args,test=True)
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
  np.random.seed(cfg.RNG_SEED)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  # print('Using config:')
  # pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  # initilize the network here.
  from model.faster_rcnn.vgg16_ILLUME import vgg16
  from model.faster_rcnn.resnet_ILLUME import resnet

  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic,gc1 = args.gc1, gc2=args.gc2, gc3 = args.gc3)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic,
                gc1 = args.gc1, gc2=args.gc2, gc3 = args.gc3)
  #elif args.net == 'res50':
  #  fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic,context=args.context)

  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (args.load_name))
  checkpoint = torch.load(args.load_name)
  #checkpoint.keys()
  #print("______________________\n\n",checkpoint)
  fasterRCNN.load_state_dict(checkpoint['model'], strict=False)
  #print(fasterRCNN)
  #model.load_state_dict(torch.load(PATH), strict=False)
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    fasterRCNN.cuda()

  start = time.time()
  max_per_image = 100

  thresh = 0.0
  ##################################################
  vis = args.vis
  if vis:
    thresh = 0.05
  else:
    thresh = 0.0
  ##################################################  

  save_name = args.load_name.split('/')[-1]
  num_images = len(imdb.image_index)
  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb, save_name)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        imdb.num_classes, training=False, normalize = False, path_return=True)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  fasterRCNN.eval()

  #activation = {}
  #def get_activation(name):
      #print("\nHERE")
      #def hook(model, input, output):
          #activation[name] = output.detach()
          #print("\nact",activation[name])
      #return hook


  #model = fasterRCNN
  #model.self_attn2.register_forward_hook(get_activation('self_attn2'))
  #x = torch.randn(1, 25)
  #output = model(num_images[0])
  #print(activation['self_attn2'])  

  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  for i in range(num_images):

      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      #print(data[0].size())
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])

      # print('\ndata path: ', data[-1]) # image path
      # img_name = data[-1][0].split('/')[-1].split('.')[0]
      # print('\nimg_name: ', img_name)
      det_tic = time.time()
      
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label, d_pred, domain_p1, domain_p2, domain_p3,\
      out_d11, out_d12, out_d13 = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)


      #model = fasterRCNN
      #model = list(fasterRCNN.children())[:-8]
      #print(model)
      #outputs=model(im_data)
      #print(outputs)
      #break
      #model.self_attn2.register_forward_hook(get_activation('softmax'))
      #print(model)
      #x = torch.randn(1, 25)
      #output = model(num_images[0])
      #wt=model(im_data, im_info, gt_boxes, num_boxes)
      #print(wt)
      #print(activation.keys())
      #print(activation['softmax'])
      #print(wt)
      #print(activation['self_attn2'])
      #break

      # VIZ
      # def prob2entropy2(prob):
      #   # convert prob prediction maps to weighted self-information maps
      #   n, c, h, w = prob.size()
      #   return -torch.mul(prob, torch.log2(prob + 1e-30))
      # # print('\nim_info: ', im_info[:2])
      # tmp_f = fasterRCNN.RCNN_base1(im_data) #worked
      # # print('\ntmp_f: ', tmp_f.shape) #[1, 256, 150, 300]

      # # # entropy
      # domain_p1_en = prob2entropy2(domain_p1)
      # tmp_f = tmp_f *  domain_p1_en

      # # # resize
      # _, _, h, w = im_data.shape
      # tmp_f = torch.nn.functional.interpolate(tmp_f, size=(h, w), mode='bilinear')
      # tmp_f = torch.mean(tmp_f.squeeze(0), 0)



      # # print('\ntmp_f mean: ', tmp_f.shape)
      # sns.heatmap(tmp_f.detach(), cbar=False, cbar_ax=False, cmap=cm.jet)

      # plt.savefig('/home/basic/mm20-may10/output/%s_feat_map_%s_entropy.png' %(img_name, str(i)))#, dpi=400)


      # # print('\n cls_prob: ', cls_prob.shape)      
      # # print('\n rois: ', rois.shape)      
      # # print('\n d_pred: ', d_pred.shape)
      # print('\n domain_p1: ', domain_p1.shape) # (1, 1, h, w), [1, 1, 150, 300]
      # # print('\n out_d11: ', out_d11.shape)

      # domain_p1 = prob2entropy2(domain_p1)
      # x = domain_p1.squeeze(0).squeeze(0)
      # sns.heatmap(x.detach(), cbar=False, cbar_ax=False, cmap=cm.jet)
      # plt.savefig('/home/basic/mm20-may10/output/predict_map_%s_entropy.png' %(str(i)), dpi=400)


      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]
      path = data[4]

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= data[1][0][2].item()

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      # print(vis)
      if vis:
        im = cv2.imread(imdb.image_path_at(i))
        im2show = np.copy(im)

      for j in xrange(1, imdb.num_classes):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            if vis:
              im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
            all_boxes[j][i] = cls_dets.cpu().numpy()
          else:
            all_boxes[j][i] = empty_array

      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
          image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in xrange(1, imdb.num_classes)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(1, imdb.num_classes):
                  keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                  all_boxes[j][i] = all_boxes[j][i][keep, :]

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
          .format(i + 1, num_images, detect_time, nms_time))
      sys.stdout.flush()
      os.makedirs('./test_images/'+str(args.dataset), exist_ok=True)
      # print(vis)
      if vis:
        path = './test_images/'+str(args.dataset)+'/'+str(i)+'.jpg'
        cv2.imwrite(path, im2show)
        print("saved image ", path)

  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

  end = time.time()
  print("test time: %0.4fs" % (end - start))
