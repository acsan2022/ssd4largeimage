import cv2
import numpy as np
import xml.etree.ElementTree as ET
from chainercv.datasets import voc_semantic_segmentation_label_colors

def calculateImg(img, window_size, margin, target_height, target_width):
    height, width, channels = img.shape
    remain_height = height % (window_size - margin)
    remain_width = width % (window_size - margin)
    add_height = np.ones((window_size - remain_height, width, 3), np.uint8)
    add_width = np.ones((height + add_height.shape[0], window_size - remain_width, 3), np.uint8)

    new_width = width + window_size - remain_width
    new_height = height + window_size - remain_height
    if target_height == 0:
        target_height = new_height
    if target_width == 0:
        target_width = new_width

    q, mod = divmod(target_height, (window_size - margin))
    if (mod > margin):
        num_vert = q + 1
    else:
        num_vert = q

    q, mod = divmod(target_width, (window_size - margin))
    if (mod > margin):
        num_horizon = q + 1
    else:
        num_horizon = q

    return num_horizon, num_vert, add_height, add_width


def predict(model, processed_img, num_vert, num_horizon, window_size, margin, batch_size):
    image_list = []
    bboxes = []
    labels = []
    scores = []
    for i in range(num_vert):
        for j in range(num_horizon):
            height_start = i * (window_size - margin)
            height_end = height_start + window_size
            width_start = j * (window_size - margin)
            width_end = width_start + window_size
            clopped = processed_img[height_start:height_end, width_start:width_end]
            im_size = (512, 512)
            resized = cv2.resize(clopped, im_size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            image = rgb.transpose((2, 0, 1))
            image_list.append(image)
            if len(image_list) == batch_size:
                pred_bboxes, pred_labels, pred_scores = model.predict(image_list)
                bboxes.extend(pred_bboxes)
                labels.extend(pred_labels)
                scores.extend(pred_scores)
                image_list = []

    if not len(image_list) == 0:
        pred_bboxes, pred_labels, pred_scores = model.predict(image_list)
        bboxes.extend(pred_bboxes)
        labels.extend(pred_labels)
        scores.extend(pred_scores)


    return bboxes , labels, scores

def writePredictedBB(display_img, label_names, num_vert, num_horizon, bboxes, labels, scores, window_size, margin, needPrint):
    numoftile = 0
    for i in range(num_vert):
        for j in range(num_horizon):
            bbox, label, score = bboxes[numoftile], labels[numoftile], scores[numoftile]
            numoftile += 1
            if len(bbox) != 0:
                for k, bb in enumerate(bbox):
                    lb = label[k]
                    conf = score[k].tolist()
                    ymin = int(bb[0] * window_size / 512 + i * (window_size - margin))
                    xmin = int(bb[1] * window_size / 512 + j * (window_size - margin))
                    ymax = int(bb[2] * window_size / 512 + i * (window_size - margin))
                    xmax = int(bb[3] * window_size / 512 + j * (window_size - margin))


                    class_num = int(lb)
                    if(label_names[class_num]=='nazca_element'):
                        cv2.rectangle(display_img, (xmin, ymin), (xmax, ymax),
                                      voc_semantic_segmentation_label_colors[class_num], 2)
                    text = 'nasca_element' + " " + ('%.2f' % conf)
                    coor = " (" + str(xmin) + "," + str(ymin) + ") (" + str(xmax) + "," + str(ymax) + ")"
                    if needPrint:
                        print('  ' + text + ' ' + coor)
                    text_top = (xmin, ymin - 15)
                    text_bot = (xmin + 180, ymin + 5)
                    text_pos = (xmin + 5, ymin)
                    if(label_names[class_num]=='nazca_element'):
                        cv2.rectangle(display_img, text_top, text_bot,
                                      voc_semantic_segmentation_label_colors[class_num], -1)
                        cv2.putText(display_img, text, text_pos,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def writeGT(img, xml_file):
    tree = ET.parse(xml_file)
    objects = tree.findall('object')
    for object in objects:
        name = object.findtext('name')
        xmin = int(object.findtext('bndbox/xmin'))
        xmax = int(object.findtext('bndbox/xmax'))
        ymin = int(object.findtext('bndbox/ymin'))
        ymax = int(object.findtext('bndbox/ymax'))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (255,51,255), 2)
        # text_top = (xmin, ymin - 15)
        # text_bot = (xmin + 220, ymin + 5)
        # text_pos = (xmin + 5, ymin)
        # cv2.rectangle(img, text_top, text_bot,
        #               (255, 51, 255), -1)
        # cv2.putText(img, name, text_pos,
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        text_top = (xmax, ymax - 5)
        text_bot = (xmax + 180, ymax + 15)
        text_pos = (xmax + 5, ymax + 10)
        cv2.rectangle(img, text_top, text_bot,
                      (255, 51, 255), -1)
        cv2.putText(img, name, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img

def writeGTwithRatio(img, xml_file, ratio):
    tree = ET.parse(xml_file)
    objects = tree.findall('object')
    for object in objects:
        name = object.findtext('name')

        if(name=='nazca_element'):
            xmin = int(int(object.findtext('bndbox/xmin')) * ratio)
            xmax = int(int(object.findtext('bndbox/xmax')) * ratio)
            ymin = int(int(object.findtext('bndbox/ymin')) * ratio)
            ymax = int(int(object.findtext('bndbox/ymax')) * ratio)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                          (255,51,255), 2)
            text_top = (xmin, ymin - 15)
            text_bot = (xmin + 220, ymin + 5)
            text_pos = (xmin + 5, ymin)
            cv2.rectangle(img, text_top, text_bot,
                          (255, 51, 255), -1)
            cv2.putText(img, name, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img
