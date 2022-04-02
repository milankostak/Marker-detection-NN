import cv2


def get_cropped():
    image_id = "0000"

    with open("D:/Python/PycharmProjects/images/predicted.txt", "r") as file:
        lines = [line.rstrip().split(" ") for line in file]

    matches = (x for x in lines if x[0] == image_id)
    bb = next(matches)
    x1 = float(bb[1]) - 15
    y1 = float(bb[2]) - 15
    x2 = float(bb[3]) + 15
    y2 = float(bb[4]) + 15
    width = round(x2 - x1)
    height = round(y2 - y1)
    print("width", width)
    print("height", height)

    image_path = f"D:/Python/PycharmProjects/images/test/{image_id}.jpg"
    img = cv2.imread(image_path)
    cropped_image = img[round(y1):round(y2), round(x1):round(x2)]
    return cropped_image
