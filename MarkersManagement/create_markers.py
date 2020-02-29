import cv2
import os
import glob
import random
import shutil

files2015 = glob.glob("..../*.jpg")
files2016 = glob.glob("..../*.jpg")
files2017 = glob.glob("..../*.jpg")
files2018 = glob.glob("..../*.jpg")
files2019 = glob.glob("..../*.jpg")

# mode = "rectangle"
mode = "triangle_empty"

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
    data = []

    if mode == "rectangle":
        width = random.randint(5, 60)
        height = random.randint(6, 50)
        x = random.randint(1, imgW - width - 1)
        y = random.randint(1, imgH - height - 1)
        color = (0, 200, 0)

        cv2.rectangle(source, (x, y), (x + width, y + height), color, -1)
        print(counter, os.path.abspath(newFile), imgW, imgH, 0, x, y, x + width, y + height)
        data = [counter, os.path.abspath(newFile), imgW, imgH, 0, x, y, x + width, y + height]

    elif mode == "triangle_empty":
        wh = random.randint(15, 60)
        x = random.randint(1, imgW - wh - 1)
        y = random.randint(1, imgH - wh - 1)
        color = (0, 200, 0)
        thickness = 2

        cv2.line(source, (x, y + wh), (x + wh, y + wh), color, thickness)  # bottom line
        cv2.line(source, (x, y + wh), (int(x + wh / 2), y), color, thickness)  # left line
        cv2.line(source, (x + wh, y + wh), (int(x + wh / 2), y), color, thickness)  # right line
        data = [counter, os.path.abspath(newFile), imgW, imgH, 0, x, y, x + wh, y + wh]

    row = ' '.join(str(e) for e in data)
    if counter % 10 < 7:  # <0;6>
        train_txt += row + "\n"
    elif counter % 10 < 9:
        val_txt += row + "\n"
    else:
        test_txt += row + "\n"

    cv2.imwrite(newFile, source)
    counter += 1

file = open(folder + "train.txt", "w")
file.write(train_txt)
file.close()

file = open(folder + "val.txt", "w")
file.write(val_txt)
file.close()

file = open(folder + "test.txt", "w")
file.write(test_txt)
file.close()
