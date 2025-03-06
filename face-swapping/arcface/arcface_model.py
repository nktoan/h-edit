
from PIL import Image

import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from .facial_recognition.model_irse import Backbone
import lpips

class IDLoss(nn.Module):
    def __init__(self, ref_path=None):
        super(IDLoss, self).__init__()

        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load("./arcface/weights/model_ir_se50.pth"))
        self.facenet.eval()
        
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))

        self.to_tensor = torchvision.transforms.ToTensor()

        """
        Prepare the ref image
        """

        self.ref_path = "../assets/demo/0.jpg" if not ref_path else ref_path

        img = Image.open(self.ref_path)
        image = img.resize((256, 256), Image.BILINEAR)
        img = self.to_tensor(image)
        img = img * 2 - 1
        img = torch.unsqueeze(img, 0)
        img = img.cuda()

        self.ref = img

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def get_cosine_sim(self, image):
        # Extract features of images
        img_feat = self.extract_feats(image)
        ref_feat = self.extract_feats(self.ref)

        # Normalize features
        img_feat = F.normalize(img_feat, p=2, dim=-1)
        ref_feat = F.normalize(ref_feat, p=2, dim=-1)
    
        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(ref_feat, img_feat, dim=-1)
        
        return cosine_sim
    
    def get_cosine_loss(self, image):
        # Compute cosine loss from cosine similarity
        cosine_sim = self.get_cosine_sim(image)
        cosine_loss = 1 - cosine_sim
        
        return cosine_loss.mean()
    
class LPIPS_Loss(nn.Module):
    def __init__(self, src_path=None):
        super(LPIPS_Loss, self).__init__()

        self.lpips_loss = lpips.LPIPS(net='vgg')
        self.to_tensor = torchvision.transforms.ToTensor()

        """
        Prepare the source image
        """

        self.src_path = "../assets/demo/0.jpg" if not src_path else src_path
        
        img = Image.open(self.src_path)
        image = img.resize((256, 256), Image.BILINEAR)
        img = self.to_tensor(image)
        img = img * 2 - 1
        img = torch.unsqueeze(img, 0)
        img = img.cuda()

        self.src = img
    
    def get_lpips_loss(self, x):
        # Compute the LPIPS loss
        lpips_loss = self.lpips_loss(x, self.src)
        
        return lpips_loss.mean()