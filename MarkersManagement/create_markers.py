import cv2
import os
import glob
import random
import shutil
import numpy as np

files2015 = glob.glob("..../*.jpg")
files2016 = glob.glob("..../*.jpg")
files2017 = glob.glob("..../*.jpg")
files2018 = glob.glob("..../*.jpg")
files2019 = glob.glob("..../*.jpg")

# mode = "rectangle_filled"
# mode = "rectangle_empty"
mode = "triangle_filled"
# mode = "triangle_empty"
# mode = "star"
# mode = "star_in_filled_rect"
# mode = "star_th1"
# mode = "star_th1_in_rectangle"
# mode = "star_th1_in_filled_rect"
# mode = "cross_in_filled_rect"
# mode = "cross_in_filled_rect_color"
# mode = "cross_th1_in_filled_rect"
# mode = "cross_th1_in_filled_rect_color"
# mode = "at_sign"

files = files2015 + files2016 + files2017 + files2018 + files2019
print("Total images count:", files.__len__())

folder = "D:/Python/PycharmProjects/" + mode + "/"
trainFolder = folder + "train/"
valFolder = folder + "val/"
testFolder = folder + "test/"

if not os.path.exists(folder):
    os.mkdir(folder)
if not os.path.exists(trainFolder):
    os.mkdir(trainFolder)
if not os.path.exists(valFolder):
    os.mkdir(valFolder)
if not os.path.exists(testFolder):
    os.mkdir(testFolder)

test_txt = ""
train_txt = ""
val_txt = ""

counter = 0
for file in files:
    print(counter)
    name = f'{counter:04d}'

    newFile = ""
    if counter % 10 < 7:  # <0;6>
        newFile = trainFolder + name + ".jpg"
    elif counter % 10 < 9:
        newFile = valFolder + name + ".jpg"
    else:
        newFile = testFolder + name + ".jpg"

    shutil.copy(file, newFile)
    source = cv2.imread(newFile)

    imgH, imgW, channels = source.shape

    if mode == "at_sign":
        width = height = random.randint(15, 45)
    else:
        width = random.randint(5, 60)
        height = random.randint(6, 50)
    x = random.randint(1, imgW - width - 1)
    y = random.randint(1, imgH - height - 1)
    color = (0, 200, 0)
    thickness = 2
    data = [counter, os.path.abspath(newFile), imgW, imgH, 0, x, y, x + width, y + height]

    if mode == "rectangle_filled":
        cv2.rectangle(source, (x, y), (x + width, y + height), color, -1)

    elif mode == "rectangle_empty":
        cv2.line(source, (x, y), (x + width, y), color, thickness)  # top line
        cv2.line(source, (x, y + height), (x + width, y + height), color, thickness)  # bottom line
        cv2.line(source, (x, y), (x, y + height), color, thickness)  # left line
        cv2.line(source, (x + width, y), (x + width, y + height), color, thickness)  # right line

    elif mode == "triangle_filled" or mode == "triangle_empty":
        pt0 = (int(x + width / 2), y)  # top
        pt1 = (x, y + height)  # left bottom
        pt2 = (x + width, y + height)  # right bottom

        if mode == "triangle_filled":
            triangle_points = np.array([pt0, pt1, pt2])
            cv2.drawContours(source, [triangle_points], 0, color, -1)
        else:
            cv2.line(source, pt0, pt1, color, thickness)  # left line
            cv2.line(source, pt0, pt2, color, thickness)  # right line
            cv2.line(source, pt1, pt2, color, thickness)  # bottom line

    elif "star" in mode:
        if "th1" in mode:
            thickness = 1
        if "filled_rect" in mode:
            cv2.rectangle(source, (x, y), (x + width, y + height), color, -1)
            color = (200, 0, 0)
        # x += 1
        # y += 1
        # width -= 2
        # height -= 2

        # left top to right bottom
        cv2.line(source, (x, y), (x + width, y + height), color, thickness)
        # middle - top to bottom
        x2 = int(x + width / 2)
        cv2.line(source, (x2, y), (x2, y + height), color, thickness)
        # right top to left bottom
        cv2.line(source, (x + width, y), (x, y + height), color, thickness)
        # middle - left to right
        y2 = int(y + height / 2)
        cv2.line(source, (x, y2), (x + width, y2), color, thickness)

        if mode == "star_th1_in_rectangle":
            cv2.line(source, (x, y), (x + width, y), color, thickness)  # top line
            cv2.line(source, (x, y + height), (x + width, y + height), color, thickness)  # bottom line
            cv2.line(source, (x, y), (x, y + height), color, thickness)  # left line
            cv2.line(source, (x + width, y), (x + width, y + height), color, thickness)  # right line

    elif "cross" in mode:
        color2 = (200, 0, 0)
        if "color" in mode:
            a = random.randint(1, 3)
            b = random.randint(200, 255)
            g = random.randint(200, 255)
            r = random.randint(200, 255)
            if a == 1:
                color = (0, g, r)
            elif a == 2:
                color = (b, 0, r)
            else:
                color = (b, g, 0)

            a = random.randint(1, 3)
            if a == 1:
                color2 = (b, 0, 0)
            elif a == 2:
                color2 = (0, g, 0)
            else:
                color2 = (0, 0, r)
        if "th1" in mode:
            thickness = 1
        cv2.rectangle(source, (x, y), (x + width, y + height), color, -1)
        # left top to right bottom
        cv2.line(source, (x, y), (x + width, y + height), color2, thickness)
        # right top to left bottom
        cv2.line(source, (x + width, y), (x, y + height), color2, thickness)

    elif mode == "at_sign":
        font_scale = width / 23
        cv2.putText(source, "@", (x - 2, y - 1 + height), cv2.FONT_HERSHEY_COMPLEX, font_scale, color)
        # thickness = 1
        # cv2.line(source, (x, y), (x + width, y), color, thickness)  # top line
        # cv2.line(source, (x, y + height), (x + width, y + height), color, thickness)  # bottom line
        # cv2.line(source, (x, y), (x, y + height), color, thickness)  # left line
        # cv2.line(source, (x + width, y), (x + width, y + height), color, thickness)  # right line

    row = ' '.join(str(e) for e in data)
    if counter % 10 < 7:  # <0;6>
        train_txt += row + "\n"
    elif counter % 10 < 9:
        val_txt += row + "\n"
    else:
        test_txt += row + "\n"

    cv2.imwrite(newFile, source)
    counter += 1

with open(folder + "train.txt", "w") as file:
    file.write(train_txt)

with open(folder + "val.txt", "w") as file:
    file.write(val_txt)

with open(folder + "test.txt", "w") as file:
    file.write(test_txt)

with open(folder + "data.names", "w") as file:
    file.write("marker")

with open(folder + "marker_anchors.txt", "w") as file:
    file.write("")
