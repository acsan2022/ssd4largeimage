import argparse
import yaml
import chainer
import cv2
import glob
import os
import numpy as np
import time
from chainercv.links import SSD512
from dlsurf_util import *

def main():
    chainer.config.train = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--window-size", type=str, required=True)
    parser.add_argument("--margin", type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument('--model')
    parser.add_argument("--image-format", type=str)
    parser.add_argument('--label-names', default="label.yml", help='The path to the yaml file with label names')
    parser.add_argument('--thresh', type=float, default=0.7)
    parser.add_argument('--target-height', type=int, default=0)
    parser.add_argument('--target-width', type=int, default=0)
    parser.add_argument('--display-gt', action='store_true')

    args = parser.parse_args()
    data_dir = os.path.join(args.data_dir, "*." + args.image_format)

    with open(args.label_names, 'r') as f:
        label_names = tuple(yaml.load(f))

    model = SSD512(
        n_fg_class=len(label_names),
        pretrained_model=args.model)
    model.score_thresh = args.thresh

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    if ',' in args.window_size:
        window_size_list = [int(ws) for ws in args.window_size.split(',')]
    else:
        window_size_list = [int(args.window_size)]
    if ',' in args.margin:
        margin_list = [int(margin) for margin in args.margin.split(',')]
    else:
        margin_list = [int(args.margin)]
    # if ',' in args.batch_size:
    #     batch_size_list = [int(bs) for bs in args.batch_size.split(',')]
    # else:
    #     batch_size_list = [int(args.batch_size)]


    target_height = args.target_height
    target_width = args.target_width
    batch_size = args.batch_size

    for path in glob.iglob(data_dir):
        start1 = time.time()
        img = cv2.imread(path)
        display_img = img.copy()
        # sample_img = np.ones(img.shape, np.uint8)*192
        dir, filename = os.path.split(path)
        print('processing : ' + filename)
        basefilename = os.path.splitext(filename)[0]

        for window_size, margin in zip(window_size_list, margin_list):
            print('predict with window size : ' + str(window_size) + ', margin : ' + str(margin))
            target_img = img.copy()
            num_horizon, num_vert, add_height, add_width = calculateImg(target_img, window_size, margin, target_height,
                                                                        target_width)
            print("total size of inputs are " + str(num_vert * num_horizon))
            processed_img = cv2.vconcat([target_img, add_height])
            processed_img = cv2.hconcat([processed_img, add_width])
            bboxes, labels, scores = predict(model, processed_img, num_vert, num_horizon, window_size, margin, batch_size)
            writePredictedBB(display_img, label_names, num_vert, num_horizon, bboxes, labels, scores, window_size, margin, True)
            # writePredictedBB(sample_img, label_names, num_vert, num_horizon, bboxes, labels, scores, window_size,
            #                  margin, False)

        if args.display_gt:
            cv2.imwrite(os.path.join(args.output_dir, 'out-' + basefilename + ".jpg"), writeGT(display_img, os.path.join(args.data_dir, basefilename + '.xml')))
        else:
            cv2.imwrite(os.path.join(args.output_dir, 'out-' + basefilename + ".jpg"), display_img)

        # cv2.imwrite(os.path.join(args.results_dir, 'out-' + basefilename + "-r2.jpg"), writeGT(sample_img, os.path.join(args.data_dir, basefilename + '.xml')))

        elapsed_time1 = time.time() - start1
        print("processing time : {0}".format(elapsed_time1) + "[sec]")
        print('\n')


if __name__ == '__main__':
    main()
