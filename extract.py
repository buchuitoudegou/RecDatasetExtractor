from PIL import Image, ImageFile
import mxnet as mx
from tqdm import tqdm
import os
import numpy as np
import cv2

def load_mx_rec(rec_path):
    save_path = rec_path + '/imgs'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path + '/train.idx'), str(rec_path + '/train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        img = np.array(img)
        # cv2.imwrite(save_path + '/' + str(label) + '/{}.jpg'.format(idx), img)
        # exit(0)
        # img = Image.fromarray(img)
        label_path = save_path + '/' + str(label)
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        # img.save(label_path + '/{}.jpg'.format(idx), quality=95)
        cv2.imwrite(label_path + '/{}.jpg'.format(idx), img)

load_mx_rec('.')