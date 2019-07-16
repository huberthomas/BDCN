import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
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

def createDataList(inputDir, outputFileName='data.lst', supportedExtensions=['.png', '.jpg', '.jpeg']):
  '''
  @brief Get image files (png, jpg, jpeg) from an input directory and save it as pair to data_lst.txt.
  @param inputDir Input directory that contains images.
  @param supportedExtensions Only files with supported extensions are included in the final list.
  @return Found images as list.
  '''
  out = open(join(inputDir, outputFileName), "w")
  res = []
  for root, directories, files in os.walk(inputDir):
      for f in files:
          for extension in supportedExtensions:
            fn, ext = splitext(f.lower())

            if extension == ext:
              out.write('%s %s\n' % (f, f))
              res.append(f)

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
    test_img = Data(test_root, test_lst, mean_bgr=mean_bgr)
    testloader = torch.utils.data.DataLoader(test_img, batch_size=1, shuffle=False, num_workers=8)
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
        data = Variable(data, volatile=True)
        tm = time.time()
        
        out = model(data)
        fuse = F.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]

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

    # baseDir = '/home/user/results/'
    # inputDirs = [
    #   os.path.join(baseDir, 'dir_1', 'rgb'),
    #   os.path.join(baseDir, 'dir_2', 'rgb'),      
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
