import glob
import os
import shutil

mode = "T_cross_real"
source_folder = "./images/"
target_folder = "D:/Python/PycharmProjects/" + mode + "/"

source_files = glob.glob(source_folder + "*.jpg")
print("Total images count:", len(source_files))

with open(source_folder + "results.txt") as file:
    sourceBB = [line.rstrip() for line in file]

train_folder = target_folder + "train/"
val_folder = target_folder + "val/"
test_folder = target_folder + "test/"

if not os.path.exists(target_folder):
    os.mkdir(target_folder)
if not os.path.exists(train_folder):
    os.mkdir(train_folder)
if not os.path.exists(val_folder):
    os.mkdir(val_folder)
if not os.path.exists(test_folder):
    os.mkdir(test_folder)

test_txt = ""
train_txt = ""
val_txt = ""

counter = 0
for file in source_files:
    print(counter)

    name = os.path.basename(file)
    bb = sourceBB[counter]

    if counter % 10 < 7:  # <0;6>
        newFile = train_folder + name
        train_txt += bb + "\n"
    elif counter % 10 < 9:
        newFile = test_folder + name
        test_txt += bb + "\n"
    else:
        newFile = val_folder + name
        val_txt += bb + "\n"

    shutil.copy(file, newFile)

    counter += 1

with open(target_folder + "train.txt", "w") as file:
    file.write(train_txt)

with open(target_folder + "val.txt", "w") as file:
    file.write(val_txt)

with open(target_folder + "test.txt", "w") as file:
    file.write(test_txt)

with open(target_folder + "data.names", "w") as file:
    file.write("marker")

with open(target_folder + "marker_anchors.txt", "w") as file:
    file.write("")
