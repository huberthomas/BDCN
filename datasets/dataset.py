import os
import cv2
import numpy as np
from PIL import Image
import scipy.io
import torch
from torch.utils import data
import random
from io import StringIO
import cv2


def load_image_with_cache(path, cache=None, lock=None):
	if cache is not None:
		if path not in cache:
			cache[path] = Image.open(path)
			# with open(path, 'rb') as f:
			# 	cache[path] = f.read()
		return cache[path]  # Image.open(StringIO(cache[path]))

	#im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
	#im = cv2.pyrDown(im)
	#im = cv2.pyrDown(im)
	#return Image.fromarray(im)
	return Image.open(path)


class Data(data.Dataset):
	def __init__(self,
              root,
              lst,
              yita=0.5,
              mean_bgr=np.array([104.00699, 116.66877, 122.67892]),
              crop_size=None,
              rgb=True,
              scale=None,
			  crop_padding=0,
			  shuffle=False):

		self.mean_bgr = mean_bgr
		self.root = root
		self.lst = lst
		self.yita = yita
		self.crop_size = crop_size
		self.crop_padding = crop_padding
		self.rgb = rgb
		self.scale = scale
		self.cache = {}

		lst_dir = os.path.join(self.root, self.lst)
		# self.files = np.loadtxt(lst_dir, dtype=str)
		with open(lst_dir, 'r') as f:
			self.files = f.readlines()
			self.files = [line.strip().split(' ') for line in self.files]

		if shuffle:
			random.shuffle(self.files)


	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		data_file = self.files[index]
		# load Image
		img_file = os.path.join(self.root, data_file[0])
		# print(img_file)
		if not os.path.exists(img_file):
			img_file = img_file.replace('jpg', 'png')
		# img = Image.open(img_file)
		img = load_image_with_cache(img_file, self.cache)
		# load gt image
		gt_file = os.path.join(self.root, data_file[1])
		# gt = Image.open(gt_file)
		gt = load_image_with_cache(gt_file, self.cache)
		if gt.mode == '1':
			gt = gt.convert('L')  # convert to grayscale

		return self.transform(img, gt)

	def transform(self, img, gt):
		gt = np.array(gt, dtype=np.float32)
		if len(gt.shape) == 3:
			gt = gt[:, :, 0]
		gt /= 255.
		gt[gt >= self.yita] = 1  # threshold ground truth
		gt = torch.from_numpy(np.array([gt])).float()
		img = np.array(img, dtype=np.float32)
		if self.rgb:
			img = img[:, :, ::-1]  # RGB->BGR
		img -= self.mean_bgr
		data = []
		if self.scale is not None:
			for scl in self.scale:
				img_scale = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
				data.append(torch.from_numpy(img_scale.transpose((2, 0, 1))).float())
			return data, gt
		img = img.transpose((2, 0, 1))  # change channel order
		img = torch.from_numpy(img.copy()).float()
		if self.crop_size:
			_, h, w = gt.size()
			assert(self.crop_size < (h - 2 * self.crop_padding) and self.crop_size < (w - 2 * self.crop_padding))
			i = random.randint(self.crop_padding, h - self.crop_padding - self.crop_size)
			j = random.randint(self.crop_padding, w - self.crop_padding - self.crop_size)
			img = img[:, i:i+self.crop_size, j:j+self.crop_size]
			gt = gt[:, i:i+self.crop_size, j:j+self.crop_size]
		return img, gt
