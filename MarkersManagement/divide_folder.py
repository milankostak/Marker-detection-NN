import glob
import os
import shutil

mode = "T_cross_real"
sourceFolder = "D:/images/"
targetFolder = "D:/Python/PycharmProjects/" + mode + "/"

files = glob.glob(sourceFolder + "*.jpg")
print("Total images count:", len(files))

with open(sourceFolder + "results.txt") as file:
    sourceBB = [line.rstrip() for line in file]

trainFolder = targetFolder + "train/"
valFolder = targetFolder + "val/"
testFolder = targetFolder + "test/"

if not os.path.exists(targetFolder):
    os.mkdir(targetFolder)
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

    name = os.path.basename(file)
    bb = sourceBB[counter]

    if counter % 10 < 7:  # <0;6>
        newFile = trainFolder + name
        train_txt += bb + "\n"
    elif counter % 10 < 9:
        newFile = testFolder + name
        test_txt += bb + "\n"
    else:
        newFile = valFolder + name
        val_txt += bb + "\n"

    shutil.copy(file, newFile)

    counter += 1

with open(targetFolder + "train.txt", "w") as file:
    file.write(train_txt)

with open(targetFolder + "val.txt", "w") as file:
    file.write(val_txt)

with open(targetFolder + "test.txt", "w") as file:
    file.write(test_txt)

with open(targetFolder + "data.names", "w") as file:
    file.write("marker")

with open(targetFolder + "marker_anchors.txt", "w") as file:
    file.write("")
