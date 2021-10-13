# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn_ILLUME import _fasterRCNN
#from model.faster_rcnn.faster_rcnn_imgandpixellevel_gradcam  import _fasterRCNN
from model.utils.config import cfg

import pdb
def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
           padding=0, bias=False)


from torch.nn.utils import spectral_norm

from torch.nn.init import xavier_uniform_


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)
        #print(m,"\n")

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))


#def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    #return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   #stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

class self_attn2(nn.Module):
    #Self attention Layer"""

    def __init__(self,in_channels,st):
        super(self_attn2, self).__init__()
        self.in_channels = in_channels
        #print("\nchannels:",self.in_channels)
        #with torch.no_grad():
        cuda3 = torch.device('cuda:3')
        cuda2 = torch.device('cuda:2')
        self.st = st
        #self.dom = domain
        if st=='source':
            cud = cuda2
        else :
            cud = cuda3

        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0).cuda(cud)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0).cuda(cud)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0).cuda(cud)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0).cuda(cud)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0).cuda(cud)
        self.softmax  = nn.Softmax(dim=-1).cuda(cud)
        self.sigma = nn.Parameter(torch.zeros(1)).cuda(cud)
        self._init_weights()

        #print(self_attn2.snconv1x1_attn.weight_orig)

    def _init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            xavier_uniform_(m.weight)
            m.bias.data.fill_(0.)
            #print(m,"\n")


    def forward(self, x):
        #self.__init__(in_channels,st)
        #super(self_attn2, self).__init__()
        cuda3 = torch.device('cuda:3')
        cuda2 = torch.device('cuda:2')
        cuda0 = torch.device('cuda:0')

        if self.st=='source':
            cud = cuda2
        else :
            cud = cuda3

        x = x.cuda(cud)
        #with torch.no_grad():

        self.apply(init_weights)

        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        #print(theta.dtype)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        #print("\nx:",x.shape)
        phi = self.snconv1x1_phi(x)
        #print("phiconv:",phi.shape)
        phi = self.maxpool(phi)
        #print("phimax:",phi.shape)
        #phi = phi.view(-1, ch//8, h*w//4)
        phi = phi.view(-1, ch//8, phi.shape[2]*phi.shape[3])
        #print("phiview:",phi.shape)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        #g = g.view(-1, ch//2, h*w//4)
        g = g.view(-1, ch//2, g.shape[2]*g.shape[3])
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        #attn_g = attn_g.cuda()
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)

        #print("\n________________WRITING_____________\n")
        #f.write(str(domain_p1_en.cpu().detach().numpy()))
        #torch.save(attn_g.cpu().detach().numpy(), "/home2/vkhindkar/ILLUME-PyTorch-main/attn_g1.pt")
        #print("\n____Done____\n")


        #Out
        out = x + self.sigma*attn_g
        out = out.cuda(cuda0)
        #out1 = Variable(out, requires_grad = True).cuda(cuda0)
        #out1 = out.cuda(cuda0)

        del self.snconv1x1_theta,
        self.snconv1x1_phi,
        self.snconv1x1_g,
        self.snconv1x1_attn,
        self.maxpool,   self.softmax ,
        self.sigma

        del x, theta, phi, attn, g
        del self
        del cud
        torch.cuda.empty_cache()
        #return -torch.mul(domain, attn_g)
        
        return out



'''
class self_attn1(nn.Module):
    #Self attention Layer
    def __init__(self, in_dim):
        super().__init__()
        #with torch.no_grad():
        # Construct the conv layers
        #cuda0 = torch.device('cuda:0')
        cuda1 = torch.device('cuda:1') 
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1).cuda(cuda1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1).cuda(cuda1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1).cuda(cuda1)
        print("\n_____________________________HERE:VALUE_CONV_________________\n")
        # Initialize gamma as 0
        self.gamma = nn.Parameter(torch.zeros(1)).cuda(cuda1)
        self.softmax  = nn.Softmax(dim=-1).cuda(cuda1)
        print("\n______________________HERE : SOFTMAX ______________\n")
        self._init_weights()
        print("\n___________________HERE : INIT WEIGHTS ______________\n")
    
    def _init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            xavier_uniform_(m.weight)
            m.bias.data.fill_(0.)
            print(m,"\n")

    
    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            
            #weight initalizer: truncated normal and random normal.
            
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                #m.bias.data.zero_()
        normal_init(self.query_conv, 0, 0.01)
        normal_init(self.key_conv, 0, 0.01)
        normal_init(self.value_conv, 0, 0.01)
        #normal_init(self.softmax,0, 0.01)
        
    def forward(self,x):
       
            #inputs :
                #x : input feature maps( B * C * W * H)
            #returns :
                #out : self attention value + input feature 
                #attention: B * N * N (N is Width*Height)
        
        #cuda0 = torch.device('cuda:0')
        cuda1 = torch.device('cuda:1') 
        m_batchsize,C,width ,height = x.size()
        
        proj_query  = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1) # B * N * C
        proj_key =  self.key_conv(x).view(m_batchsize, -1, width*height) # B * C * N
        energy =  torch.bmm(proj_query, proj_key).cuda(cuda1) # batch matrix-matrix product
        
        attention = self.softmax(energy).cuda(cuda1) # B * N * N
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height).cuda(cuda1) # B * C * N
        out = torch.bmm(proj_value, attention.permute(0,2,1)) # batch matrix-matrix product
        out = out.view(m_batchsize,C,width,height) # B * C * W * H
        
        # Add attention weights onto input
        #out = self.gamma*out + x
        return out



class self_attn(nn.Module):
    # Self attention Layer

    def __init__(self, in_channels):
        super(self_attn, self).__init__()
        self.in_channels = in_channels
        #print("\nchannels:",self.in_channels)
        #with torch.no_grad():
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))


    def _init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            xavier_uniform_(m.weight)
            m.bias.data.fill_(0.)
            print(m,"\n")


    def forward(self, x):
        
        #x = x.cpu()
        #with torch.no_grad():

        #self.apply(init_weights)

        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        #attn_g = attn_g.cuda()
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        return attn_g
'''

class netD_forward1(nn.Module):
    def __init__(self):
        super(netD_forward1, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                  padding=1, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
        return feat

class netD_forward2(nn.Module):
    def __init__(self):
        super(netD_forward2, self).__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1,
                  padding=1, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
        return feat

class netD_forward3(nn.Module):
    def __init__(self):
        super(netD_forward3, self).__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1,
                  padding=1, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
        return feat


class netD_inst(nn.Module):
  def __init__(self, fc_size=2048):
    super(netD_inst, self).__init__()
    self.fc_1_inst = nn.Linear(fc_size, 1024)
    self.fc_2_inst = nn.Linear(1024, 256)
    self.fc_3_inst = nn.Linear(256, 2)
    self.relu = nn.ReLU(inplace=True)
    #self.softmax = nn.Softmax()
    #self.logsoftmax = nn.LogSoftmax()
    # self.bn = nn.BatchNorm1d(128)
    self.bn2 = nn.BatchNorm1d(2)

  def forward(self, x):
    x = self.relu(self.fc_1_inst(x))
    x = self.relu((self.fc_2_inst(x)))
    x = self.relu(self.bn2(self.fc_3_inst(x)))
    return x

class netD1(nn.Module):
    def __init__(self,context=False):
        super(netD1, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1,
                  padding=0, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.context = context
        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.context:
          feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
          x = self.conv3(x)
          return F.sigmoid(x),feat
        else:
          x = self.conv3(x)
          return F.sigmoid(x)

class netD2(nn.Module):
    def __init__(self,context=False):
        super(netD2, self).__init__()
        self.conv1 = conv3x3(512, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)
        self.context = context
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)
        if self.context:
          feat = x
        x = self.fc(x)
        if self.context:
          return x,feat
        else:
          return x


class netD3(nn.Module):
    def __init__(self,context=False):
        super(netD3, self).__init__()
        self.conv1 = conv3x3(512, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)
        self.context = context
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)
        if self.context:
          feat = x
        x = self.fc(x)
        if self.context:
          return x,feat
        else:
          return x

class netD_dc(nn.Module):
    def __init__(self):
        super(netD_dc, self).__init__()
        self.fc1 = nn.Linear(2048,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100,2)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))),training=self.training)
        x = self.fc3(x)
        return x

class vgg16(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False,gc1=False, gc2=False, gc3=False):
    self.model_path = cfg.VGG_PATH
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.gc1 = gc1
    self.gc2 = gc2
    self.gc3 = gc3

    _fasterRCNN.__init__(self, classes, class_agnostic,self.gc1,self.gc2, self.gc3)

  def _init_modules(self):
    vgg = models.vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # not using the last maxpool layer
    #print(vgg.features)
    self.RCNN_base1 = nn.Sequential(*list(vgg.features._modules.values())[:14])
    self.RCNN_base2 = nn.Sequential(*list(vgg.features._modules.values())[14:21])
    self.RCNN_base3 = nn.Sequential(*list(vgg.features._modules.values())[21:-1])
    #print(self.RCNN_base1)
    #print(self.RCNN_base2)
    self.netD1 = netD1()
    self.netD_forward1 = netD_forward1()
    #cuda0 = torch.device('cuda:0')
    #cuda1 = torch.device('cuda:1')

    #self1 = self_attn2()
    #self1.in_channels = 256
    #self1.st = 'source'
    self.self_attn2 = self_attn2(512,'source') 
    #self.self_attn1 = self_attn1(256).cuda(cuda1)
    #print("\n*******attn******\n",self.self_attn1)
    self.netD2 = netD2()
    self.netD_forward2 = netD_forward2()
    self.netD3 = netD3()
    self.netD_forward3 = netD_forward3()
    feat_d = 4096
    feat_d += 128
    feat_d += 128
    feat_d += 128

    # Fix the layers before conv3:
    self.netD_inst = netD_inst(fc_size = feat_d)

    for layer in range(10):
      for p in self.RCNN_base1[layer].parameters(): p.requires_grad = False
      
    #for p in self.self_attn1.parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier

    self.RCNN_cls_score = nn.Linear(feat_d, self.n_classes)
    if self.class_agnostic:
        self.RCNN_bbox_pred = nn.Linear(feat_d, 4)
    else:
        self.RCNN_bbox_pred = nn.Linear(feat_d, 4 * self.n_classes)


  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

