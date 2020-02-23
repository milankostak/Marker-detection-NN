import numpy as np


# https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
# box_a - ground truth; box_b - prediction
def bb_intersection_over_union(box_a, box_b):
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    # compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    # return the intersection over union value
    return iou


images_folder = "images5"

with open('D:\\Python\\PycharmProjects\\Marker-detection-NN\\YOLOv3\\data\\my_data\\' + images_folder + '\\test.txt') as file:
    gt_lines = [line.rstrip() for line in file]

with open('D:\\Python\\PycharmProjects\\' + images_folder + '\\predicted.txt') as file:
    pred_lines = [line.rstrip() for line in file]

all_ious = list()
count = len(gt_lines)
for i in range(count):
    gt_line = gt_lines[i].split(" ")
    gt_box = [float(gt_line[5]), float(gt_line[6]), float(gt_line[7]), float(gt_line[8])]

    pred_line = pred_lines[i].split(" ")
    possible_ious = list()
    pred_count = len(pred_line)
    if pred_count > 1:
        for j in range(1, pred_count, 4):
            pred_box = [float(pred_line[j]), float(pred_line[j + 1]), float(pred_line[j + 2]), float(pred_line[j + 3])]
            possible_iou = bb_intersection_over_union(gt_box, pred_box)
            possible_ious.append(possible_iou)

    if len(possible_ious) != 0:
        max_iou = max(possible_ious)
    else:
        max_iou = 0

    print(pred_line[0], max_iou)
    all_ious.append(max_iou)

result = np.array(all_ious).mean()
print(result)
