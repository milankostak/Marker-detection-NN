import math

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


base_path = "D:/Python/PycharmProjects/images/"

with open(base_path + "test/marker_data.txt") as file:
    markers = [line.rstrip() for line in file]

min_c = 100_000
count = len(markers)
for i in range(count):
    marker = markers[i].split(" ")
    image_id = int(marker[0])
    image_path = marker[1]
    center_x = int(marker[2])
    center_y = int(marker[3])
    angle = float(marker[4])

    p1 = (int(marker[5]), int(marker[6]))
    p2 = (int(marker[7]), int(marker[8]))
    p3 = (int(marker[9]), int(marker[10]))
    p4 = (int(marker[11]), int(marker[12]))

    content = get_content(p1, p2, p3, p4)

    image_id_zeros = f"{image_id:04d}"
    print(image_id_zeros)
    img, base_x, base_y = get_cropped(image_id=image_id_zeros)
    detected_values = get_data(img=img, show_outputs=False)

    if len(detected_values) == 0:
        print("False negative case")
    else:
        detected_center = detected_values[0]
        detected_angle = detected_values[1]
        angle_diff = abs(angle - detected_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        detected_content = get_content(detected_values[2], detected_values[3], detected_values[4], detected_values[5])

        print(length(detected_center, (center_x - base_x, center_y - base_y)))
        print(angle, detected_angle, angle_diff)
        c = abs(content - detected_content)
        if c < min_c:
            min_c = c
            print("MIN CONTENT")
        print(abs(content - detected_content))
    print("")
