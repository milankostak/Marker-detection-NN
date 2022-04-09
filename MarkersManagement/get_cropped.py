import cv2

base_path = "D:/Python/PycharmProjects/images/"

with open(f"{base_path}predicted.txt", "r") as file:
    lines = [line.rstrip().split(" ") for line in file]


def get_cropped(image_id: str = None, padding: int = 15):
    if image_id is None:
        # image_id = "0549"  # ! 15
        # image_id = "0243"  # !
        # image_id = "0039"  # fialová na žluté   # ! 5
        # image_id = "0313"  # modrá na zelené
        # image_id = "0037"  # ! 5
        # image_id = "0048"
        # image_id = "0049"  # ! 15
        # image_id = "0058"  # ! 15
        # image_id = "0107"
        # image_id = "0127"  # ?
        # image_id = "0128"  # svislá
        # image_id = "0149"
        # image_id = "0181"
        # image_id = "0211"  # svislá
        image_id = "0385"  # !
        # image_id = "0416"

    matches = (x for x in lines if x[0] == image_id)
    bb = next(matches)
    if len(bb) == 1:
        return None

    x1 = float(bb[1]) - padding
    y1 = float(bb[2]) - padding
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
    return cropped_image
