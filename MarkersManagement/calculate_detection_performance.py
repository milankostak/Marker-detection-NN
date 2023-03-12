import math

import numpy as np
from shapely.geometry import Polygon

from MarkersManagement.get_cropped import get_cropped
from MarkersManagement.get_data import get_data


def length(pp1, pp2):
    return math.sqrt(math.pow(pp2[0] - pp1[0], 2) + math.pow(pp2[1] - pp1[1], 2))


def get_content(pp1, pp2, pp3, pp4):
    l1 = length(pp1, pp2)
    l2 = length(pp2, pp3)
    l3 = length(pp3, pp4)
    l4 = length(pp4, pp1)

    length_average = (l1 + l2 + l3 + l4) / 4
    # print(l1, l2, l3, l4, int(length_average))
    return length_average * length_average


def get_content2(pp1, pp2, pp3, pp4):
    # https://en.wikipedia.org/wiki/Quadrilateral#Non-trigonometric_formulas
    a = length(pp1, pp2)
    b = length(pp2, pp3)
    c = length(pp3, pp4)
    d = length(pp4, pp1)
    s = (a + b + c + d) / 2.0
    p = length(pp1, pp3)
    q = length(pp2, pp4)

    value = (s - a) * (s - b) * (s - c) * (s - d) - 0.25 * (a * c + b * d + p * q) * (a * c + b * d - p * q)
    return math.sqrt(value) if value > 0 else 0


base_path = "D:/Python/PycharmProjects/images/"

with open(base_path + "test.txt") as file:
    markers = [line.rstrip() for line in file]

show_outputs = True

data = []
times = []
count = len(markers)
for i in range(count):
    marker = markers[i].split(" ")
    # image_id = int(marker[0])
    image_path = marker[1]
    image_id = image_path.split("\\")[-1].split(".")[0]
    if show_outputs:
        print(image_id)
    true_center_x = int(marker[9])
    true_center_y = int(marker[10])
    true_angle = float(marker[11])

    true_p1 = (int(marker[12]), int(marker[13]))
    true_p2 = (int(marker[14]), int(marker[15]))
    true_p3 = (int(marker[16]), int(marker[17]))
    true_p4 = (int(marker[18]), int(marker[19]))

    true_content = get_content2(true_p1, true_p2, true_p3, true_p4)

    true_box = [true_p1, true_p2, true_p3, true_p4]

    # image_id_zeros = f"{image_id:04d}"
    img, base_x, base_y = get_cropped(image_id=image_id)
    detected_values = get_data(img=img, show_outputs=False)

    if len(detected_values) == 0:
        if show_outputs:
            print("False negative case")
    else:
        detected_center = detected_values[0]
        detected_angle = detected_values[1]
        angle_diff = abs(true_angle - detected_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        center_diff = length(detected_center, (true_center_x - base_x, true_center_y - base_y))
        if show_outputs:
            print(center_diff)
            print(true_angle, detected_angle, angle_diff)

        detected_content = get_content2(detected_values[2], detected_values[3], detected_values[4], detected_values[5])
        content_diff = abs(true_content - detected_content)
        if show_outputs:
            print(content_diff)

        detected_box = [
            [detected_values[2][0] + base_x, detected_values[2][1] + base_y],
            [detected_values[3][0] + base_x, detected_values[3][1] + base_y],
            [detected_values[4][0] + base_x, detected_values[4][1] + base_y],
            [detected_values[5][0] + base_x, detected_values[5][1] + base_y]
        ]

        detected_polygon = Polygon(detected_box)
        true_polygon = Polygon(true_box)

        polygon_intersection = detected_polygon.intersection(true_polygon).area
        polygon_union = detected_polygon.union(true_polygon).area
        iou = polygon_intersection / polygon_union if polygon_union > 0 else 0
        if show_outputs:
            print(iou)

        data.append([image_id, center_diff, angle_diff, content_diff, iou])
        times.append(detected_values[6])

    if show_outputs:
        print()

# print("TIMES")
# print(np.mean(np.array(times), axis=0))

data = np.array(data)
print("Total count:", data.shape[0])
print()

centers = data[:, 1].astype(np.float)
angles = data[:, 2].astype(np.float)
contents = data[:, 3].astype(np.float)
ious = data[:, 4].astype(np.float)


def get_metrics(values: np.ndarray, name: str, limit: float, reverse: bool = False):
    print(f"{name} median: {np.median(values):.2f}")
    print(f"{name} mean: {np.mean(values):.2f}")
    values_good = values[values < limit] if not reverse else values[values > limit]
    print(f"{name} good count:", values_good.shape[0])
    print(f"{name} bad count:", data.shape[0] - values_good.shape[0])
    print(f"{name} good percent: {(values_good.shape[0] / values.shape[0]) * 100:.2f}%")
    print(f"{name} good median: {np.median(values_good):.2f}")
    print(f"{name} good mean: {np.mean(values_good):.2f}")
    print()


get_metrics(centers, "Centers", 60)
get_metrics(angles, "Angles", 30)
get_metrics(contents, "Contents", 10000)
get_metrics(ious, "IoUs", 0.3, reverse=True)

if show_outputs:
    print(data)
