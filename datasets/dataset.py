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
import copy
import math
import matplotlib.pyplot as plt

def load_image_with_cache(path, cache=None, lock=None):
	if cache is not None:
		if path not in cache:
			cache[path] = cv2.imread(path, cv2.IMREAD_UNCHANGED) #Image.open(path)  # 
			#Image.open(path)
			# with open(path, 'rb') as f:
			# 	cache[path] = f.read()
		return cache[path]  # Image.open(StringIO(cache[path]))

	#im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
	#im = cv2.pyrDown(im)
	#im = cv2.pyrDown(im)
	# return Image.fromarray(cv2.imread(path, cv2.IMREAD_UNCHANGED))
	#return Image.fromarray(im)
	return cv2.imread(path, cv2.IMREAD_UNCHANGED) #Image.open(path)


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
              shuffle=False,
              flip=False,
              blur=False,
              brightness=False,
              rotate=False):

		self.mean_bgr = mean_bgr
		self.root = root
		self.lst = lst
		self.yita = yita
		self.crop_size = crop_size
		self.crop_padding = crop_padding
		self.rgb = rgb
		self.scale = scale
		self.cache = {}
		self.flip = flip
		self.blur = blur
		self.brightness = brightness
		self.rotate = rotate

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
		
		if len(gt.shape) == 3:
			# convert to grayscale
			_, _, c = gt.shape
			if c == 3:
				gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
			elif c == 4:
				gt = cv2.cvtColor(gt, cv2.COLOR_BGRA2GRAY)

		# if gt.mode == '1':
		# 	gt = gt.convert('L')  # convert to grayscale

		return self.transform(img, gt)

	def transform(self, image, groundTruth):
		'''
		Transform image
		'''
		img = image.copy()
		gt = groundTruth.copy()
		## Image manipulations

		# if self.scale is not None:
		#	data = []
		# 	for scl in self.scale:
		# 		img_scale = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
		# 		data.append(torch.from_numpy(img_scale.transpose((2, 0, 1))).float())
		# 	return data, gt
		if self.rotate:
			if random.randint(0, 1):
				angles = self.__calculateRotationAngles(16)#[0, 45, 90, 135, 180, 225, 270, 315]
				angle = angles[random.randint(0, len(angles) - 1)]
				r, c = img.shape[:2]
				# extend image canvas to avoid black border effects
				img = self.__extendImageCanvas(img, int(c*2), int(r*2))
				gt = self.__extendImageCanvas(gt, c*2, r*2)
				img = self.__rotateImage(img, angle, True)
				gt = self.__rotateImage(gt, angle, True)

		if self.crop_size:
			#_, h, w = gt.size()
			h, w = groundTruth.shape
			assert(self.crop_size < (h - 2 * self.crop_padding) and self.crop_size < (w - 2 * self.crop_padding))
			i = random.randint(self.crop_padding, h - self.crop_padding - self.crop_size)
			j = random.randint(self.crop_padding, w - self.crop_padding - self.crop_size)
			gtTmp = gt[i:(i + self.crop_size), j:(j + self.crop_size)]
			
			if len(np.nonzero(gtTmp)[0]) > 0:
				# be sure that there exists a gt in the cropped area
				gt = gtTmp
				img = img[i:(i + self.crop_size), j:(j + self.crop_size), :]
			else:
				# otherwise take the crop from the original image
				gt = groundTruth[i:(i + self.crop_size), j:(j + self.crop_size)]
				img = image[i:(i + self.crop_size), j:(j + self.crop_size), :]

		if self.brightness:
			if random.randint(0, 1):
				if random.randint(0, 1):
					# increase
					img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
					increase = random.randint(0, 128) #50
					v = img[:, :, 2]
					v = np.where(v <= 255 - increase, v + increase, 255)
					img[:, :, 2] = v
					img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
				else:
					# decrease
					img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
					decrease = random.uniform(0.1, 1.0) #0.75, 1
					img[..., 2] = img[..., 2] * decrease
					img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

		if self.flip:
			if random.randint(0, 1):
				flip = random.randint(-1, 1)
				cv2.flip(img, flip)
				cv2.flip(gt, flip)

		if self.blur:
			if random.randint(0, 1):
				blurKernelSize = [3, 5, 7]
				blurKernelIndex = random.randint(0, len(blurKernelSize) - 1)
				img = cv2.blur(img, (blurKernelSize[blurKernelIndex], blurKernelSize[blurKernelIndex]))


		gt = np.array(gt, dtype=np.float32)
		gt /= 255.0
		gt[gt >= self.yita] = 1  # threshold ground truth

		img = np.array(img, dtype=np.float32)
		img -= self.mean_bgr
		img = img.transpose((2, 0, 1))  # change channel order

		img = torch.from_numpy(img).float()
		gt = torch.from_numpy(np.array([gt])).float()

		return img, gt

	def __getCropCoordinates(self, angle, width, height):
		'''
		Rotation sometimes results in black borders around the image. This function calculates the
		cropping area to avoid black borders. For more information see
		https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders

		angle Angle in degrees.

		width Width of the image.

		height Height of the image.

		Returns the cropping tuple (x, y, width, height). 
		'''
		ang = math.radians(angle)
		quadrant = int(math.floor(ang / (math.pi / 2.))) & 3
		sign_alpha = ang if (quadrant & 1) == 0 else math.pi - ang
		alpha = (sign_alpha % math.pi + math.pi) % math.pi

		bb = {
			'w': width * math.cos(alpha) + height * math.sin(alpha),
			'h': width * math.sin(alpha) + height * math.cos(alpha)
		}

		gamma = math.atan2(bb['w'], bb['h']) if width < height else math.atan2(bb['h'], bb['w'])
		delta = math.pi - alpha - gamma

		length = height if width < height else width
		d = length * math.cos(alpha)
		a = d * math.sin(alpha) / math.sin(delta)

		y = a * math.cos(gamma)
		x = y * math.tan(gamma)

		return x, y, bb['w'] - 2 * x, bb['h'] - 2 * y

	def __rotateImage(self, img, angle, removeCropBorders):
		"""
		Rotates an image (angle in degrees) and expands image to avoid cropping

		img OpenCV image.

		angle Angle in degrees.

		removeCropBorders Removes black borders that can occure during rotation.

		Returns rotated image. 
		"""
		height, width = img.shape[:2]  # image shape has 3 dimensions
		# getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
		imageCenter = (width/2, height/2)

		M = cv2.getRotationMatrix2D(imageCenter, angle, 1.0)

		# rotation calculates the cos and sin, taking absolutes of those.
		absCos = abs(M[0, 0])
		absSin = abs(M[0, 1])

		# find the new width and height bounds
		boundW = int(height * absSin + width * absCos)
		boundH = int(height * absCos + width * absSin)

		# subtract old image center (bringing image back to origo) and adding the new image center coordinates
		M[0, 2] += boundW/2 - imageCenter[0]
		M[1, 2] += boundH/2 - imageCenter[1]

		# rotate image with the new bounds and translated rotation matrix
		rotatedImg = cv2.warpAffine(img, M, (boundW, boundH))

		# remove black borders
		if removeCropBorders:
			x, y, bbW, bbH = self.__getCropCoordinates(angle, width, height)
			c = list(map(int, [x, y, bbW, bbH]))
			rotatedImg = rotatedImg[c[1]:c[1]+c[3], c[0]:c[0]+c[2]]

		return rotatedImg

	def __calculateRotationAngles(self, numOfAngles):
		'''
        Calculate angles given by a number of angles.

        numOfAngles Number angles, e.g. 4 will give [0 90 180 270]

        Returns calculated angles as list of floats.
        '''
		assert(numOfAngles > 0, 'Parameter must be an integer greater than 0.')

		if numOfAngles == 1:
			return [0]

		factor = 360.0 / float(numOfAngles)

		angles = [0]

		while(angles[len(angles) - 1] < (360 - factor)):
			angles.append(angles[len(angles) - 1] + factor)

		return angles

	def __extendImageCanvas(self, img, cols, rows):
		'''
		Extend image canvas by mirroring the image in all directions.

		img Input image.

		cols New width.

		rows New height.

		Returns extended image with new dimensions.
		'''
		r, c = img.shape[:2]

		assert(cols > c or rows > r, 'New image size must be greater than the old one.')

		dr = rows - r
		dc = cols - c

		assert(dr >= 0 or dc >= 0, 'New image size is too big. Must be smaller or equal than twice the image size.')

		# horizontal
		flipudImg = np.flipud(img)
		# vertical
		fliplrImg = np.fliplr(img)
		flipudlrImg = np.fliplr(flipudImg)
		
		if img.ndim < 3:
			extendedImg = np.ones((rows, cols), img.dtype)
			extendedImg[:r, :c] = img
			extendedImg[r:, c:] = flipudlrImg[:dr, :dc]
			extendedImg[r:, :c] = flipudImg[:dr, :c]
			extendedImg[:r, c:] = fliplrImg[:r, :dc]
		else:
			extendedImg = np.ones((rows, cols, img.ndim), img.dtype)
			extendedImg[:r, :c, :] = img
			extendedImg[r:, c:, :] = flipudlrImg[:dr, :dc, :]
			extendedImg[r:, :c, :] = flipudImg[:dr, :c, :]
			extendedImg[:r, c:, :] = fliplrImg[:r, :dc, :]

		return extendedImg
