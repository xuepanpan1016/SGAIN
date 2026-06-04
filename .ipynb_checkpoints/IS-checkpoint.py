# Code adapted from
# https://github.com/openai/improved-gan/blob/master/inception_score/model.py
# which was in turn derived from
# tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
# import imageio.v2 as imageio  # 注释掉imageio导入
# from PIL import Image  # 注释掉PIL导入
import cv2
import math
from tqdm import tqdm, trange

parser = argparse.ArgumentParser()
parser.add_argument('--input_npy_file', default=None)
parser.add_argument('--input_image_dir', default='')
parser.add_argument('--input_image_dir_list', default=None)
parser.add_argument('--input_image_superdir', default=None)
parser.add_argument('--image_size', default=128, type=int)
parser.add_argument('--num_splits', default=1, type=int)
parser.add_argument('--tensor_layout', default='NHWC', choices=['NHWC', 'NCHW'])

IMAGE_EXTS = ['.png', '.jpg', '.jpeg']


def main(args):
    got_npy_file = args.input_npy_file is not None
    got_image_dir = args.input_image_dir is not None
    got_image_dir_list = args.input_image_dir_list is not None
    got_image_superdir = args.input_image_superdir is not None
    inputs = [got_npy_file, got_image_dir, got_image_dir_list, got_image_superdir]
    if sum(inputs) != 1:
        raise ValueError('Must give exactly one input type')

    if args.input_npy_file is not None:
        images = np.load(args.input_npy_file)
        images = np.split(images, images.shape[0], axis=0)
        images = [img[0] for img in images]
        mean, std = get_inception_score(args, images)
        print('Inception mean: ', mean)
        print('Inception std: ', std)
    elif args.input_image_dir is not None:
        images = load_images(args, args.input_image_dir)
        mean, std = get_inception_score(args, images)
        print('Inception mean: ', mean)
        print('Inception std: ', std)
    elif got_image_dir_list:
        with open(args.input_image_dir_list, 'r') as f:
            dir_list = [line.strip() for line in f]
        for image_dir in dir_list:
            images = load_images(args, image_dir)
            mean, std = get_inception_score(args, images)
            print('Inception mean: ', mean)
            print('Inception std: ', std)
            print()
    elif got_image_superdir:
        for fn in sorted(os.listdir(args.input_image_superdir)):
            if not fn.startswith('result'): continue
            image_dir = os.path.join(args.input_image_superdir, fn, 'images')
            if not os.path.isdir(image_dir): continue
            images = load_images(args, image_dir)
            mean, std = get_inception_score(args, images)
            print('Inception mean: ', mean)
            print('Inception std: ', std)
            print()

def load_images(args, image_dir):
    print('Loading images from ', image_dir)
    images = []
    
    for fn in os.listdir(image_dir):
        ext = os.path.splitext(fn)[1].lower()
        if ext not in IMAGE_EXTS:
            continue
        
        img_path = os.path.join(image_dir, fn)
        
        try:
            # 使用OpenCV读取图像
            img = cv2.imread(img_path)
            if img is None:
                print(f'Error: Could not load image {img_path}')
                continue
            # OpenCV读取的图像是BGR格式，需要转换为RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f'Error loading image {img_path}: {e}')
            continue

        # 检查图像通道
        if len(img.shape) != 3 or img.shape[2] != 3:
            print('Skipping one channel image:', fn)
            continue
        
        if args.image_size is not None:
            # 使用OpenCV进行图像缩放
            img = cv2.resize(img, (args.image_size, args.image_size))
        
        images.append(img)
    
    print('Found %d images' % len(images))
    return images


MODEL_DIR = './tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None


def get_inception_score(args, images):
    splits = args.num_splits
    layout = args.tensor_layout

    assert (type(images) == list)
    assert (type(images[0]) == np.ndarray)
    assert (len(images[0].shape) == 3)
    inps = []
    for img in images:
        img = img.astype(np.float32)
        inps.append(np.expand_dims(img, 0))
    bs = 1
    with tf.compat.v1.Session() as sess:
        preds = []
        n_batches = int(math.ceil(float(len(inps)) / float(bs)))
        for i in trange(n_batches, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            sys.stdout.write(".")
            sys.stdout.flush()
            inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
            inp = np.concatenate(inp, 0)
            if layout == 'NCHW':
                inp = inp.transpose(0, 2, 3, 1)
            pred = sess.run(softmax, {'ExpandDims:0': inp})
            preds.append(pred)
        preds = np.concatenate(preds, 0)
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return np.mean(scores), np.std(scores)


def _init_inception():
    global softmax
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    
    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
    
    with tf.io.gfile.GFile(os.path.join(MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    
    with tf.compat.v1.Session() as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        ops = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape().as_list()
                new_shape = []
                for j, s in enumerate(shape):
                    if s is None:
                        new_shape.append(None)
                    elif s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.set_shape(tf.TensorShape(new_shape))
        
        w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
        softmax = tf.nn.softmax(logits)

if softmax is None:
    _init_inception()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    print("Test on {} Done".format(args.input_image_dir))