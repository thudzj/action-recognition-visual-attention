import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy
import scipy

import cv2
import skimage
import skimage.transform
import skimage.io
from PIL import Image

import sys
#sys.path.append('../')

from util.data_handler import DataHandler
from util.data_handler import TrainProto
from util.data_handler import TestTrainProto
from util.data_handler import TestValidProto
from util.data_handler import TestTestProto

import src.actrec

def overlay(bg,fg):
    """
    Overlay attention over the video frame
    """
    src_rgb = fg[..., :3].astype(numpy.float32) / 255.
    src_alpha = fg[..., 3].astype(numpy.float32) / 255.
    dst_rgb = bg[..., :3].astype(numpy.float32) / 255.
    dst_alpha = bg[..., 3].astype(numpy.float32) / 255.

    out_alpha = src_alpha + dst_alpha * (1. - src_alpha)
    out_rgb = (src_rgb * src_alpha[..., None] + dst_rgb * dst_alpha[..., None] * (1. - src_alpha[..., None])) / out_alpha[..., None]

    out = numpy.zeros_like(bg)
    out[..., :3] = out_rgb * 255
    out[..., 3] = out_alpha * 255

    return out
		
model ='model_obj101.npz'
dataset ='obj101'

with open('%s.pkl'%model, 'rb') as f:
    options = pkl.load(f)
batch_size = 10

print '-----'
if dataset == "obj101":
	from obj101 import obj101
	test = obj101("obj101/googlenet_test_features_tmp.hkl", "obj101/googlenet_test_labels.hkl")
num_batches = test.num_examples / batch_size
print '-----'

params  = src.actrec.init_params(options)
params  = src.actrec.load_params(model, params)
tparams = src.actrec.init_tparams(params)

trng, use_noise, inps, alphas, cost, opt_outs, preds = src.actrec.build_model(tparams, options)
f_alpha = theano.function(inps,alphas,name='f_alpha',on_unused_input='ignore')
f_preds = theano.function(inps,preds,profile=False,on_unused_input='ignore')
	
x, y, fname = test.random_batch(batch_size)
alpha = f_alpha(x,y)

alpha = numpy.transpose(alpha, (1,0,2))

space = 255.0*numpy.ones((224*2,20,4))
space[:,:,0:3] = 255.0*numpy.ones((224*2,20,3))

for i in xrange(alpha.shape[0]):
	imgf = numpy.array([]).reshape(2*224,0,4)
	for ii in xrange(alpha.shape[1]):
    # read frame
		image = skimage.io.imread(fname[i])
		# add an Alpha layer to the RGB image
		image = skimage.transform.resize(image, (224, 224))
		img = numpy.array(image)
		alphalayer = numpy.ones((224,224,1))*255
		img = numpy.dstack((img,alphalayer)) #rgba

		# create the attention map and add an Alpha layer to the RGB image
		alpha_img = skimage.transform.pyramid_expand(alpha[i,ii,:].reshape(7,7), upscale=32, sigma=20)
		alpha_img = alpha_img*255.0/numpy.max(alpha_img)
		alpha_img = skimage.color.gray2rgb(alpha_img)
		alpha_img = numpy.dstack((alpha_img,0.8*alphalayer)) #rgba

		old_img = img
		img = overlay(img,alpha_img)

		img  = numpy.concatenate((old_img,img),axis=0)
		imgf = numpy.concatenate((imgf,img),axis=1)
		imgf = numpy.concatenate((imgf,space),axis=1)
	skimage.io.imsave("draw/" + fname[i].split('/')[-1], imgf)