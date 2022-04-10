import cv2

base_path = "D:/Python/PycharmProjects/images/"

with open(f"{base_path}predicted.txt", "r") as file:
    lines = [line.rstrip().split(" ") for line in file]


def get_cropped(image_id: str, padding: int = 15):
    matches = (x for x in lines if x[0] == image_id)
    bb = next(matches)
    if len(bb) <= 1:
        return [None, -1, -1]

    x1 = float(bb[1]) - padding
    if x1 < 0:
        x1 = 0
    y1 = float(bb[2]) - padding
    if y1 < 0:
        y1 = 0
    x2 = float(bb[3]) + padding
    y2 = float(bb[4]) + padding

    # width = round(x2 - x1)
    # height = round(y2 - y1)
    # print("width", width)
    # print("height", height)

    # image_eval_path = f"{base_path}test_eval/{image_id}_eval.jpg"
    # img_eval = cv2.imread(image_eval_path)
    # cv2.imshow("original", img_eval)

    image_path = f"{base_path}/test/{image_id}.jpg"
    img = cv2.imread(image_path)
    cropped_image = img[round(y1):round(y2), round(x1):round(x2)]
    return [cropped_image, round(x1), round(y1)]
