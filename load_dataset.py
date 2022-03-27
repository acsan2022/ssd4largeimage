#! -*- coding:utf-8 -*-

import numpy as np
import os
import chainer
from chainercv.utils import read_image
from xml.etree.ElementTree import *


class OriginalDetectionDataset(chainer.dataset.DatasetMixin):
    def __init__(self, data_dir, label_names):
        self.data_dir = data_dir
        self.label_names = label_names
        # print("*************************************")
        # print(self.label_names)

        self.img_filenames = []
        self.anno_filenames = []

        for root, dirs, files in os.walk(data_dir):
            for name in sorted(files):
                if os.path.splitext(name)[1] != '.jpg':
                    continue
                img_filename = os.path.join(root, name)
                anno_filename = os.path.splitext(img_filename)[0] + '.xml'
                if not os.path.exists(anno_filename):
                    continue
                self.img_filenames.append(img_filename)
                self.anno_filenames.append(anno_filename)


    def __len__(self):
        # print("len>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        return len(self.img_filenames)


    def get_example(self, i):
        img_filename = self.img_filenames[i]
        anno_filename = self.anno_filenames[i]
        img = read_image(img_filename)
        # print("*************************************")
        # print("filename")
        # print(img_filename)
        width = img.shape[2]
        height = img.shape[1]

        with open(anno_filename, 'r') as f:
            tree = parse(f)
            elem = tree.getroot()

        bbox = []
        label = []

        for e in elem.getiterator("object"):
            x_min = float(e[4][0].text)
            y_min = float(e[4][1].text)
            x_max = float(e[4][2].text)
            y_max = float(e[4][3].text)
 
            
            # print("label")
            # print(e[0].text)
            # print("*************************************")

            l = self.label_names.index(e[0].text)
            # print("l")
            # print(l)
            # print("*************************************")
            bbox.append([y_min, x_min, y_max, x_max])
            label.append(l)

        bbox = np.array(bbox, dtype=np.float32)
        # print("bbox")
        # print(bbox)
        # print("*************************************")
        label = np.array(label, dtype=np.int32)
        # print("img read end")
        return img, bbox, label

