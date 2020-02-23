# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import glob

from YOLOv3.utils.misc_utils import parse_anchors, read_class_names
from YOLOv3.utils.nms_utils import gpu_nms
from YOLOv3.utils.plot_utils import get_color_table, plot_one_box
from YOLOv3.utils.data_aug import letterbox_resize

from YOLOv3.model import yolov3

#################
# ArgumentParser
#################
parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")

images_folder = "images5"
parser.add_argument("--input_image", type=str, default="todo",
                    help="The path of the input image.")
parser.add_argument("--anchor_path", type=str, default="./data/my_data/" + images_folder + "/marker_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="./data/my_data/" + images_folder + "/data.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

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

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)

    test_images = glob.glob("D:/Python/PycharmProjects/" + images_folder + "/test/*.jpg")
    results = ""

    for test_image in test_images:
        img_ori = cv2.imread(test_image)
        name = test_image[39:-4]
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
        for j in range(0, len(boxes_)):
            results += (" " + str(boxes_[j][0]) + " " + str(boxes_[j][1]) + " " + str(boxes_[j][2]) + " " + str(boxes_[j][3]))
        results += "\n"

        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            plot_one_box(img_ori, [x0, y0, x1, y1],
                         label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100),
                         color=color_table[labels_[i]])
        # cv2.imshow('Detection result', img_ori)
        # cv2.imwrite('./test_eval/' + name + '_eval.jpg', img_ori)
        # cv2.waitKey(0)

    with open("./predicted.txt", "w") as file:
        file.write(results)
