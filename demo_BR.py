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
from scipy import signal 
np.random.seed(125)

import visualization
from visualization import image_show

import pickle

import matplotlib.pyplot as plt

#%%

def model_def():
    exp_name = '00' # (exp_name is used for logging notes that correspond to different runs)

    init_dir = 'reference_model'

    ## choose hyps
    B = 1
    S = 8
    N = 16**2 # number of points to track

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

data_br = pickle.load(open("../data.pkl", "rb"))
# data_br = data_br[:,:,:,np.newaxis]



def main(fr_gp = 40, vid_st = 40, window_ = 40, p = 5):
    data_rs = data_br[::fr_gp]
    xy = None
    
    traj_l , rgbs_l = [], []
    
    for i in range(window_):
        
        data_rgbs = data_rs[vid_st:vid_st+8]
        data_rgbs = torch.from_numpy(data_rgbs).permute(0,3,1, 2)
        data_rgbs = data_rgbs.unsqueeze(0)
        # data_rgbs =  torch.cat([data_rgbs, data_rgbs, data_rgbs], dim = 2)
        rgbs, trajs_e = inferred_value(model, data_rgbs, p = p, xy = xy)

        
        if i%5 == 0:
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
    frame_gap = 10
    video_start = 11
    window_ = 20
    points = 5
    trajs_e = main(fr_gp=frame_gap, vid_st=video_start, window_= window_, p = points)
    
#%%
t_s = []

for pos_ in range(points**2):
    v_1,t = [], []
    for i in range(len(trajs_e[0])):
        v_1.append(sum((trajs_e[0,i,pos_] - trajs_e[0,0,pos_])**2).cpu())
        t.append(i*frame_gap/30)
                            
    t_s.append(np.array(v_1))


plt.plot(np.array(t), np.array(v_1))

#%% Filter Design

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#%% Filtering Process

y = butter_bandpass_filter(np.array(t_s[10]), 0.06, 0.8, 30/frame_gap)
plt.plot(t,y)

print('zero crossing: ', (np.diff(np.sign(y)) != 0).sum())


#%% Frequency response. 

import scipy.fftpack

yf = scipy.fftpack.fft(y)

xf = np.linspace(0.0, 30/(2.0*frame_gap), len(y)//2)

plt.figure()

plt.plot(xf, 2.0/len(y) * np.abs(yf[:len(y)//2]))


#%%

t_s_filt = []
for i in range(len(t_s)):
    t_s_filt.append(butter_bandpass_filter(np.array(t_s[i]), 0.06, 0.8, 30/frame_gap))

#%% Clustering Algorithms

from tslearn.clustering import KernelKMeans
gak_km = KernelKMeans(n_clusters=2, kernel="gak")
X =  np.array(t_s_filt)[:,:, np.newaxis]
labels_gak = gak_km.fit_predict(X)


from tslearn.clustering import TimeSeriesKMeans, silhouette_score
km = TimeSeriesKMeans(n_clusters=2, metric="dtw")
labels = km.fit_predict(X)
silhouette_score(X, labels, metric="dtw")

#%% 

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2).fit(X[:,:,0])