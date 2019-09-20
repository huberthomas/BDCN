import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import time
import re
import os
import sys
import cv2
import bdcn
from datasets.dataset import Data
import argparse
import cfg
from matplotlib import pyplot as plt
from os.path import splitext, join
import logging
import fnmatch
import multiprocessing as mp

def createDataList(inputDir = None, outputFileName='data.lst', supportedExtensions = ['png', 'jpg', 'jpeg']):
    '''
    Get files e.g. (png, jpg, jpeg) from an input directory. It is case insensitive to the extensions.

    inputDir Input directory that contains images.

    supportedExtensions Only files with supported extensions are included in the final list. Case insensitive.

    Returns a list of images file names.
    '''
    if inputDir is None:
        raise ValueError('Input directory must be set.')

    if supportedExtensions is None or len(supportedExtensions) == 0:
        raise ValueError('Supported extensions must be set.')

    res = []

    dirList = os.listdir(inputDir)

    for extension in supportedExtensions:
        pattern = ''
        for char in extension:
            pattern += ('[%s%s]' % (char.lower(), char.upper()))

        res.extend(fnmatch.filter(dirList, '*.%s' % (pattern)))

    out = open(join(inputDir, outputFileName), "w")

    for f in res:
        out.write('%s %s\n' % (f, f))

    out.close()
    return res

def sigmoid(x):
    return 1./(1+np.exp(np.array(-1.*x)))


def forwardAll(model, args):
    test_root = cfg.config_test[args.dataset]['data_root']

    if(args.inputDir is not None):
      test_root = args.inputDir

    logging.info('Processing: %s' % test_root)
    test_lst = cfg.config_test[args.dataset]['data_lst']

    imageFileNames = createDataList(test_root, test_lst)
    
    mean_bgr = np.array(cfg.config_test[args.dataset]['mean_bgr'])
    test_img = Data(test_root, test_lst, mean_bgr=mean_bgr, shuffle=False, crop_padding=0, crop_size=None)
    testloader = torch.utils.data.DataLoader(test_img, batch_size=1, shuffle=False, num_workers=mp.cpu_count())
    # nm = np.loadtxt(test_name_lst, dtype=str)
    # print(len(testloader), len(nm))
    # assert len(testloader) == len(nm)
    # save_res = True
    save_dir = join(test_root, args.res_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if args.cuda:
        model.cuda()

    model.eval()
    # data_iter = iter(testloader)
    # iter_per_epoch = len(testloader)
    start_time = time.time()
    all_t = 0
    timeRecords = open(join(save_dir, 'timeRecords.txt'), "w")
    timeRecords.write('# filename time[ms]\n')

    for i, (data, _) in enumerate(testloader):
        if args.cuda:
            data = data.cuda()

            with torch.no_grad():
                data = Variable(data)#, volatile=True)
                tm = time.time()
        
                out = model(data)
                fuse = torch.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]

                elapsedTime = time.time() - tm
                timeRecords.write('%s %f\n'%(imageFileNames[i], elapsedTime * 1000))

                cv2.imwrite(os.path.join(save_dir, '%s' % imageFileNames[i]), fuse*255)

                all_t += time.time() - tm

    timeRecords.close()
    print(all_t)
    print('Overall Time use: ', time.time() - start_time)


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logging.info('Loading model...')
    model = bdcn.BDCN()
    logging.info('Loading state...')
    model.load_state_dict(torch.load('%s' % (args.model)))
    logging.info('Start image processing...')

    inputDirs = [
        '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/bsr_bsds500/BSR/BSDS500/data/images/test_png/'
        #'/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/test_dataset/hdr_fusion/flicker_synthetic/flicker_1'
        ]
    #baseDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/'
    #inputDirs = [
      #args.inputDir
      #os.path.join(baseDir, 'rgbd_dataset_freiburg1_desk/rgb/'),
      #os.path.join(baseDir, 'rgbd_dataset_freiburg1_desk2/rgb/'),
      #os.path.join(baseDir, 'rgbd_dataset_freiburg1_plant/rgb/'),
      #os.path.join(baseDir, 'rgbd_dataset_freiburg1_room/rgb/'),
      #os.path.join(baseDir, 'rgbd_dataset_freiburg1_rpy/rgb/'),
      #os.path.join(baseDir, 'rgbd_dataset_freiburg1_xyz/rgb/'),
      #os.path.join(baseDir, 'rgbd_dataset_freiburg2_desk/rgb/'),
      #os.path.join(baseDir, 'rgbd_dataset_freiburg2_xyz/rgb/'),
      #os.path.join(baseDir, 'rgbd_dataset_freiburg3_long_office_household/rgb/'),
    #]   
    # baseDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/test_dataset'
    # inputDirs = [
    #     os.path.join(baseDir, 'hdr_fusion', 'flicker_synthetic', 'flicker_1'),
    #     os.path.join(baseDir, 'hdr_fusion', 'smooth_synthetic', 'flicker_2'),
    #     os.path.join(baseDir, 'nyu_depth_v2', 'basements', 'basement_001c'),
    #     os.path.join(baseDir, 'nyu_depth_v2', 'cafe', 'cafe_0001c'),
    #     os.path.join(baseDir, 'nyu_depth_v2', 'classrooms', 'classroom_0014'),
    #     os.path.join(baseDir, 'tum', 'rgbd_dataset_freiburg1_desk', 'rgb'),
    #     os.path.join(baseDir, 'tum', 'rgbd_dataset_freiburg1_xyz', 'rgb'),
    #     os.path.join(baseDir, 'tum', 'rgbd_dataset_freiburg2_xyz', 'rgb'),
    # ]


    for inputDir in inputDirs:
      args.inputDir = inputDir
      args.cuda = True      
      forwardAll(model, args)


def parse_args():
    parser = argparse.ArgumentParser('test BDCN')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_test.keys(), default='bsds500', help='The dataset to train')
    parser.add_argument('-i', '--inputDir', type=str, default=None, help='Input image directory for testing.')
    parser.add_argument('-c', '--cuda', action='store_true', help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default='models/bdcn_pretrained_on_bsds500.pth', help='the model to test')
    parser.add_argument('--res-dir', type=str, default='bdcn', help='the dir to store result')
    parser.add_argument('-k', type=int, default=1, help='the k-th split set of multicue')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:\t%(message)s', level=logging.INFO)
    main()
