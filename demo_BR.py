import time
import numpy as np
import io
import os
from PIL import Image
import cv2
import saverloader
import imageio.v2 as imageio
from nets.pips import Pips
import utils.improc
import random
import glob
from utils.basic import print_, print_stats
import torch
# from tensorboardX import SummaryWriter
import torch.nn.functional as F
random.seed(125)
np.random.seed(125)

import visualization
from visualization import image_show

import pickle

#%%

def model_def():
    exp_name = '00' # (exp_name is used for logging notes that correspond to different runs)

    init_dir = 'reference_model'

    ## choose hyps
    B = 1
    S = 8
    N = 16**2 # number of points to track
    filenames = glob.glob('./demo_images/*.jpg')
    filenames = sorted(filenames)
    print('filenames', filenames)
    max_iters = len(filenames)//S # run each unique subsequence

    log_freq = 2 # when to produce visualizations 
    
    ## autogen a name
    model_name = "%02d_%d_%d" % (B, S, N)
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    # log_dir = 'logs_demo'
    # writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    global_step = 0

    model = Pips(stride=4).cuda()
    parameters = list(model.parameters())
    if init_dir:
        _ = saverloader.load(init_dir, model)
    global_step = 0
    model.eval()
    return model
#%%

model = model_def()

#%%
import pdb

def inferred_value(model , rgbs , p = 4, xy = None):
    with torch.no_grad():
        rgbs = rgbs.cuda().float() # B, S, C, H, W
    
        B, S, C, H, W = rgbs.shape
        rgbs_ = rgbs.reshape(B*S, C, H, W)
        H_, W_ = 360, 640
        rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
        H, W = H_, W_
        rgbs = rgbs_.reshape(B, S, C, H, W)
    
        # pick N points to track; we'll use a uniform grid
        if xy == None:
            N_ = np.sqrt(p**2).round().astype(np.int32)
            grid_y, grid_x = utils.basic.meshgrid2d(B, N_, N_, stack=False, norm=False, device='cuda')
            grid_y = 8 + grid_y.reshape(B, -1)/float(N_-1) * (H-16)
            grid_x = 8 + grid_x.reshape(B, -1)/float(N_-1) * (W-16)
            xy = torch.stack([grid_x, grid_y], dim=-1) # B, N_*N_, 2
            _, S, C, H, W = rgbs.shape
        
        
        print_stats('rgbs', rgbs)
        preds, preds_anim, vis_e, stats = model(xy, rgbs, iters=6)
        trajs_e = preds[-1]
        return rgbs, trajs_e




#%%

data_br = pickle.load(open("../data_1.pkl", "rb"))
# data_br = data_br[:,:,:,np.newaxis]



def main():
    fr_gp = 5
    vid_st = 40
    data_rs = data_br[::fr_gp]
    xy = None
    
    traj_l , rgbs_l = [], []
    
    for i in range(90):
        
        data_rgbs = data_rs[vid_st:vid_st+8]
        data_rgbs = torch.from_numpy(data_rgbs).permute(0,3,1, 2)
        data_rgbs = data_rgbs.unsqueeze(0)
        # data_rgbs =  torch.cat([data_rgbs, data_rgbs, data_rgbs], dim = 2)
        rgbs, trajs_e = inferred_value(model, data_rgbs, p = 6, xy = xy)

        
        if i%10 == 0:
            xy = None
        else:
            xy = trajs_e[:,-1,:,:]
            
        traj_l.append(trajs_e.cpu())
        rgbs_l.append(rgbs.cpu())
        vid_st = vid_st +8
    
    
    rgbs = torch.cat(rgbs_l, dim= 1)
    trajs_e = torch.cat(traj_l, dim= 1)
    
    # image_show(rgbs, trajs_e)
    torch.cuda.empty_cache()
    return trajs_e

#%%

if __name__ == '__main__':
    trajs_e = main()
