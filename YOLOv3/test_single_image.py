# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import cv2
import glob
import os

from YOLOv3.utils.simple_object import SimpleObject
from YOLOv3.utils.misc_utils import parse_anchors, read_class_names
from YOLOv3.utils.nms_utils import gpu_nms
from YOLOv3.utils.plot_utils import get_color_table, plot_one_box
from YOLOv3.utils.data_aug import letterbox_resize

from YOLOv3.model import yolov3

base_path = 'D:/Python/PycharmProjects/images/'

args = SimpleObject()
# The path of the anchor txt file.
args.anchor_path = base_path + "marker_anchors.txt"
# Resize the input image with `new_size`, size format: [width, height]
args.new_size = [416, 416]
# Whether to use the letterbox resize.
args.letterbox_resize = True
# The path of the class names.
args.class_name_path = base_path + "data.names"
# The path of the weights to restore.
args.restore_path = "./data/darknet_weights/yolov3.ckpt"
# The probability threshold that the proposed bounding box needs to meet
args.score_thresh = 0.3
# The IoU threshold for non-maximum suppression (NMS) of similar bounding boxes; lower value cuts more
args.nms_thresh = 0.45

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

if not os.path.exists(base_path + "test_eval/"):
    os.mkdir(base_path + "test_eval/")

# img_ori = cv2.imread(args.input_image)
# if args.letterbox_resize:
#     img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
# else:
#     height_ori, width_ori = img_ori.shape[:2]
#     img = cv2.resize(img_ori, tuple(args.new_size))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = np.asarray(img, np.float32)
# img = img[np.newaxis, :] / 255.

with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    yolo_model = yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)

    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class,
                                    max_boxes=200, score_thresh=args.score_thresh, nms_thresh=args.nms_thresh)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)

    test_images = glob.glob(base_path + "test/*.jpg")
    results = ""

    for test_image in test_images:
        img_ori = cv2.imread(test_image)
        name = os.path.splitext(os.path.basename(test_image))[0]  # get the filename without extension
        if args.letterbox_resize:
            img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
        else:
            height_ori, width_ori = img_ori.shape[:2]
            img = cv2.resize(img_ori, tuple(args.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.

        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

        # rescale the coordinates to the original image
        if args.letterbox_resize:
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        else:
            boxes_[:, [0, 2]] *= (width_ori / float(args.new_size[0]))
            boxes_[:, [1, 3]] *= (height_ori / float(args.new_size[1]))

        # print("box coords:")
        # print(boxes_)
        # print('*' * 30)
        # print("scores:")
        print(name, " ", scores_)
        # print('*' * 30)
        # print("labels:")
        # print(labels_)

        results += name
        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            score = f"{scores_[i] * 100:.2f}"
            results += f" {x0:.3f} {y0:.3f} {x1:.3f} {y1:.3f} {score}"

            plot_one_box(
                img_orig, [x0, y0, x1, y1],
                label=args.classes[labels_[i]] + f", {score}%",
                color=[200, 200, 200]
                # color=color_table[labels_[i]]
            )
        results += "\n"
        # cv2.imshow('Detection result', img_ori)
        # cv2.waitKey(0)
        cv2.imwrite(base_path + 'test_eval/' + name + '_eval.jpg', img_ori)

    with open(base_path + "predicted.txt", "w") as file:
        file.write(results)
