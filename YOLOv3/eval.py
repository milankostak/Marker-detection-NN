from __future__ import division, print_function

import tensorflow as tf
from tqdm import trange

from YOLOv3.model import yolov3
from YOLOv3.utils.data_utils import get_batch_data
from YOLOv3.utils.eval_utils import get_preds_gpu, voc_eval, parse_gt_rec
from YOLOv3.utils.misc_utils import parse_anchors, read_class_names, AverageMeter
from YOLOv3.utils.nms_utils import gpu_nms
from YOLOv3.utils.simple_object import SimpleObject

base_path = "D:/Python/PycharmProjects/marker_testing/T_cross_real2/"

args = SimpleObject()
# The path of the anchor txt file.
args.anchor_path = base_path + "marker_anchors.txt"
# Resize the input image with `new_size`, size format: [width, height]
args.img_size = [416, 416]
# Whether to use the letterbox resize.
args.letterbox_resize = True
# The path of the class names.
args.class_name_path = base_path + "data.names"
# The path of the weights to restore.
# args.restore_path = "./data/darknet_weights/yolov3.ckpt"
args.restore_path = base_path + "checkpoint/50/best_model_Epoch_36_step_2404_mAP_0.8841_loss_0.7208_lr_3e-05"
# The path of the validation or test txt file.
args.val_file = base_path + "val.txt"
# Number of threads for image processing used in tf.data pipeline.
args.num_threads = 10
# Prefetech_buffer used in tf.data pipeline.
args.prefetech_buffer = 5
# IOU threshold in nms operation.
args.nms_threshold = 0.45
# Threshold of the probability of the classes in nms operation.
args.score_threshold = 0.01
# Keep at most nms_topk outputs after nms.
args.nms_topk = 400
# Whether to use the voc 2007 mAP metrics.
args.use_voc_07_metric = False

# args params
args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.class_num = len(args.classes)
args.val_img_cnt = len(open(args.val_file, 'r').readlines())

# setting placeholders
is_training = tf.placeholder(dtype=tf.bool, name="phase_train")
handle_flag = tf.placeholder(tf.string, [], name="iterator_handle_flag")
pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
gpu_nms_op = gpu_nms(pred_boxes_flag, pred_scores_flag, args.class_num, args.nms_topk, args.score_threshold,
                     args.nms_threshold)

##################
# tf.data pipeline
##################
val_dataset = tf.data.TextLineDataset(args.val_file)
val_dataset = val_dataset.batch(1)
val_dataset = val_dataset.map(
    lambda x: tf.py_func(get_batch_data,
                         [x, args.class_num, args.img_size, args.anchors, "val", False, False, args.letterbox_resize],
                         [tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=args.num_threads
)
val_dataset.prefetch(args.prefetech_buffer)
iterator = val_dataset.make_one_shot_iterator()

image_ids, image, y_true_13, y_true_26, y_true_52 = iterator.get_next()
image_ids.set_shape([None])
y_true = [y_true_13, y_true_26, y_true_52]
image.set_shape([None, args.img_size[1], args.img_size[0], 3])
for y in y_true:
    y.set_shape([None, None, None, None, None])

##################
# Model definition
##################
yolo_model = yolov3(args.class_num, args.anchors)
with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(image, is_training=is_training)
loss = yolo_model.compute_loss(pred_feature_maps, y_true)
y_pred = yolo_model.predict(pred_feature_maps)

saver_to_restore = tf.train.Saver()

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer()])
    saver_to_restore.restore(sess, args.restore_path)

    print('\n----------- start to eval -----------\n')

    val_loss_total, val_loss_xy, val_loss_wh, val_loss_conf, val_loss_class = \
        AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    val_preds = []

    for j in trange(args.val_img_cnt):
        __image_ids, __y_pred, __loss = sess.run([image_ids, y_pred, loss], feed_dict={is_training: False})
        pred_content = get_preds_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, __image_ids, __y_pred)

        val_preds.extend(pred_content)
        val_loss_total.update(__loss[0])
        val_loss_xy.update(__loss[1])
        val_loss_wh.update(__loss[2])
        val_loss_conf.update(__loss[3])
        val_loss_class.update(__loss[4])

    # calc mAP
    rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
    gt_dict = parse_gt_rec(args.val_file, args.img_size, args.letterbox_resize)
    print('mAP eval:')
    for ii in range(args.class_num):
        npos, nd, rec, prec, ap = voc_eval(gt_dict, val_preds, ii, iou_thres=0.5, use_07_metric=args.use_voc_07_metric)
        rec_total.update(rec, npos)
        prec_total.update(prec, nd)
        ap_total.update(ap, 1)
        print("Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}".format(ii, rec, prec, ap))

    mAP = ap_total.average
    print("final mAP: {:.4f}".format(mAP))
    print("recall: {:.4f}, precision: {:.4f}".format(rec_total.average, prec_total.average))
    print("total_loss: {:.3f}, loss_xy: {:.3f}, loss_wh: {:.3f}, loss_conf: {:.3f}, loss_class: {:.3f}".format(
        val_loss_total.average, val_loss_xy.average, val_loss_wh.average, val_loss_conf.average, val_loss_class.average
    ))
