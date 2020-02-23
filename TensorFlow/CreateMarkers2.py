import cv2
import os
import glob
import random
import shutil

files2015 = glob.glob("C:/Programy/EasyPHP-Devserver-17/--/E-fotky/2015/thumbs/*/*.jpg")
# files2016 = glob.glob("C:/Programy/EasyPHP-Devserver-17/eds-www/2016/thumbs/*/*.jpg")
# files2017 = glob.glob("C:/Programy/EasyPHP-Devserver-17/eds-www/2017/thumbs/*/*.jpg")
# files2018 = glob.glob("C:/Programy/EasyPHP-Devserver-17/eds-www/2018/thumbs/*/*.jpg")
# files2019 = glob.glob("C:/Programy/EasyPHP-Devserver-17/eds-www/2019/thumbs/*/*.jpg")

# files = files2015 + files2016 + files2017 + files2018 + files2019
files = files2015
print("Total images count: ", files.__len__())

folder = "D:/Python/PycharmProjects/images5/"
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

counter = 0
for file in files:
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

    wh = random.randint(15, 60)
    x = random.randint(1, imgW - wh - 1)
    y = random.randint(1, imgH - wh - 1)
    color = (0, 200, 0)
    thickness = 2

    # draw triangle
    cv2.line(source, (x, y + wh), (x + wh, y + wh), color, thickness)  # bottom line
    cv2.line(source, (x, y + wh), (int(x + wh / 2), y), color, thickness)  # left line
    cv2.line(source, (x + wh, y + wh), (int(x + wh / 2), y), color, thickness)  # right line
    print(counter, os.path.abspath(newFile), imgW, imgH, 0, x, y, x + wh, y + wh)

    cv2.imwrite(newFile, source)
    counter += 1
