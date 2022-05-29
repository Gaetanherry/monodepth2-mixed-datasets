from __future__ import absolute_import, division, print_function

import os
import glob
import cv2
import numpy as np
import re
import sys

import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from PIL import Image
from numpy import savetxt

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4
input_dir = '/home/fusion4268/3dvision/Project/dataset/TUM_test/'
#gt_dir = '/home/fusion4268/3dvision/Project/dataset/TUM/rgbd_dataset_freiburg3_long_office_household/depth/*.png'
#input_dir = '/home/fusion4268/3dvision/Project/dataset/ICL_NUIM/living_room_traj0_frei_png/rgb/*.png'
#gt_dir = '/home/fusion4268/3dvision/Project/dataset/ICL_NUIM/living_room_traj0_frei_png/depth/*.png'
color_new_height = int(1704 / 2)


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    dataset = []
    seq_num = []
    gt_num = []
    compare_num = []

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        input_seqs_file = open(input_dir + 'test.txt')
        input_seqs = input_seqs_file.readlines()
        #print(len(input_seqs))
        for i in range(len(input_seqs)):
            seq_data = glob.glob(input_dir+input_seqs[i].rstrip()+'rgb/*.png')
            seq_data.sort(key=natural_keys)
            seq_num.append(len(seq_data))
            dataset.append(seq_data)

        #for i in range(len(input_seqs)):
        #    print(len(dataset[i]))
        
        #print(dataset[0][0])
        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
        
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))
        
        with torch.no_grad():
            for sequence in dataset:
                for filename in sequence:
                    image = cv2.imread(filename)
                    height, width, channel = image.shape
                    #cv2.imshow('image',image)
                    #image = image[ int((2272 - color_new_height)/2):int((2272 + color_new_height)/2),:,:]
                    input_color = cv2.resize(image/255.0, (width, height), interpolation=cv2.INTER_NEAREST)
                    input_color = torch.tensor(input_color, dtype = torch.float).cuda().permute(2,0,1)[None,:,:,:]
                

                    #save_image(input_color, 'ICL_depth.png')
                    #sys.exit()
                    output = depth_decoder(encoder(input_color))
    
                    pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                    pred_disp = pred_disp.cpu()[:, 0].numpy()

                    pred_disps.append(pred_disp)
                    #print(pred_disp.shape)
                    #sys.exit()

        pred_disps = np.concatenate(pred_disps)
        sys.exit()
        
    for i in range(len(input_seqs)):
        gt_path = glob.glob(input_dir+input_seqs[i].rstrip()+'depth/*png')
        gt_path.sort(key=natural_keys)
        gt_depth.append(gt_path)

    for i in range(len(input_seqs)):
        print(len(gt_depth[i]))

    sys.exit()
    #print(gt_depth.shape)
    #savetxt('depth_ICL.txt', gt_depth, fmt='%1.1f')
    #sys.exit()

    print("-> Evaluating")
    print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    print(pred_disps.shape[0])
    for i in range(pred_disps.shape[0]):
    #for i in range(10):
        #print(gt_path[i])
        img = Image.open(gt_path[i])
        gt_depth = np.array(img)
        gt_height, gt_width = gt_depth.shape[:2]
        gt_depth = gt_depth / 5000
        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        mask = gt_depth > 0
        #print(mask)
        #sys.exit()
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        
        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        
        #print(gt_depth)
        #print(pred_depth)
        #sys.exit()
        savetxt('gt_depth_ICL.txt', gt_depth, fmt='%1.1f')
        savetxt('pred_depth_ICL.txt', pred_depth, fmt='%1.1f')
        sys.exit()
        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")
    


if __name__ == "__main__":
    options = MonodepthOptions()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(options.parse())
