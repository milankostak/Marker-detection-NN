import math
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

from MarkersManagement.get_cropped import get_cropped


def get_xy(k1, q1, k2, q2):
    """
    Get the intersection of two lines base on their "k" and "q" values.
    :param k1: slope 1
    :param q1: y intersection 1
    :param k2: slope 2
    :param q2: y intersection 2
    :return: the intersection as a list of two values [x,y]
    """
    x = (q2 - q1) / (k1 - k2)
    y = k2 * x + q2
    return [int(round(x)), int(round(y))]


def get_data(img: np.ndarray, show_outputs: bool = True):
    if img is None:
        return []

    start = time.time()
    if show_outputs:
        cv2.imshow("cropped", img)

    # div = 16
    # img = img // div * div + div // 2
    # quantization of the image did not bring any good results

    # RGB histogram
    colors = ("b", "g", "r")
    for k, color in enumerate(colors):
        histogram = cv2.calcHist(images=[img], channels=[k], mask=None, histSize=[256], ranges=[0, 256])
        plt.plot(histogram, color=color)
        plt.xlim([0, 256])
    plt.title("RGB histogram")
    # plt.show()

    # convert image to HSV model; get hue histogram; then get the most common hue value
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # TODO pixels towards the center of the image should have more weight
    # TODO maybe do it only on a circle (ellipsis) around the center?
    hist = cv2.calcHist(images=[hsv], channels=[0], mask=None, histSize=[180], ranges=[0, 180])
    target_hue = np.argmax(hist)
    if show_outputs:
        print("Final hue:", target_hue * 2)
    plt.title("Hue histogram")
    # plt.plot(hist)
    # plt.show()

    h, s, v = cv2.split(hsv)
    # take only the hue component and filter all other pixels that have different hue
    # filter by hue_threshold
    # the resulting image becomes a base for other operations
    hue_threshold = 4
    # handle circular hue
    if hue_threshold <= target_hue <= 180 - hue_threshold:
        h[h < target_hue - hue_threshold] = 0
        h[h > target_hue + hue_threshold] = 0
    elif target_hue < hue_threshold:
        # if target_hue is too low
        h[np.logical_and(target_hue - hue_threshold + 180 > h, h > target_hue + hue_threshold)] = 0
    else:
        # if target_hue is too high
        h[np.logical_and(target_hue - hue_threshold > h, h > target_hue + hue_threshold - 180)] = 0

    if show_outputs:
        cv2.imshow("Hue HSV component", h)
    # s[s < 100] = 0
    # v[v < 128] = 0
    # cv2.imshow("Saturation HSV component", s)
    # cv2.imshow("Value HSV component", v)

    # do erosion operation to fill noise spaces (there many of these, and the operation is absolutely necessary)
    erosion_size = 5
    element = cv2.getStructuringElement(
        shape=cv2.MORPH_RECT,
        ksize=(2 * erosion_size + 1, 2 * erosion_size + 1),
        anchor=(erosion_size, erosion_size)
    )
    erosion_dst = cv2.erode(src=h, kernel=element)
    if show_outputs:
        cv2.imshow("Erosion", erosion_dst)

    # after erosion, apply dilatation to get to the original size of the marker
    # some noise space might again appear, but there should be only a few, so they do not break the further steps
    dilatation_size = 5
    element = cv2.getStructuringElement(
        shape=cv2.MORPH_RECT,
        ksize=(2 * dilatation_size + 1, 2 * dilatation_size + 1),
        anchor=(dilatation_size, dilatation_size)
    )
    dilatation_dst = cv2.dilate(src=erosion_dst, kernel=element)
    if show_outputs:
        cv2.imshow("Dilatation", dilatation_dst)

    # apply Gaussian blur before thresholding
    blur_dst = cv2.GaussianBlur(src=dilatation_dst, ksize=(11, 11), sigmaX=0)
    if show_outputs:
        cv2.imshow("Blur", blur_dst)

    # threshold the result with Otsu's adaptive thresholding
    threshold_value, threshold_dst = cv2.threshold(
        src=blur_dst, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if show_outputs:
        print("Otsu's threshold value:", threshold_value)
        cv2.imshow("After Otsu's thresholding", threshold_dst)

    # apply canny edges detector
    edges_dst = cv2.Canny(image=threshold_dst, threshold1=50, threshold2=200, edges=None, apertureSize=3)

    # copy the edge-detection results for later use of displaying hough lines
    lines_dst = cv2.cvtColor(src=edges_dst, code=cv2.COLOR_GRAY2BGR)
    lines_p_dst = np.copy(lines_dst)

    # apply classic Hough lines algorithm
    # automatically change the threshold accordingly to get the optimal number of lines (not too much, but not too few)
    threshold_hough = 30
    while_iteration_limit = 0
    change = 5
    while True:
        lines_hough = cv2.HoughLines(image=edges_dst, rho=1, theta=np.pi / 180, threshold=threshold_hough, srn=0, stn=0)
        if threshold_hough < 0:
            # if no lines cannot be detected in the image
            break
        if while_iteration_limit > 7:
            change = 1
        if lines_hough is None or len(lines_hough) <= 5:
            threshold_hough = threshold_hough - change
        elif len(lines_hough) >= 50:
            threshold_hough = threshold_hough + change
        else:
            break

        while_iteration_limit = while_iteration_limit + 1
        if while_iteration_limit > 20:
            break
    if lines_hough is None:
        lines_hough = []

    if show_outputs:
        print("Final Hough threshold:", threshold_hough)
        print("Hough lines count:", len(lines_hough))

    # process the result of the classic Hough lines algorithm to get the actual lines out of it and also draw them
    lines_hough_points = []
    for i in range(len(lines_hough)):
        lines_hough_points.append([])
        rho = lines_hough[i][0][0]
        theta = lines_hough[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        lines_hough_points[i].append([pt1[0], pt1[1], pt2[0], pt2[1]])
        cv2.line(lines_dst, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

    # apply probabilistic Hough lines algorithm
    # automatically change the threshold accordingly to get the optimal number of lines (not too much, but not too few)
    threshold_hough_p = 20
    while_iteration_limit = 0
    change = 5
    while True:
        lines_hough_p = cv2.HoughLinesP(
            image=edges_dst,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold_hough_p,
            minLineLength=15,
            maxLineGap=50
        )
        if threshold_hough_p < 0:
            # if no lines cannot be detected in the image
            break
        if while_iteration_limit > 7:
            change = 1
        if lines_hough_p is None or len(lines_hough_p) <= 10:
            threshold_hough_p = threshold_hough_p - change
        elif len(lines_hough_p) >= 40:
            threshold_hough_p = threshold_hough_p + change
        else:
            break

        while_iteration_limit = while_iteration_limit + 1
        if while_iteration_limit > 20:
            break
    if lines_hough_p is None:
        lines_hough_p = []

    if show_outputs:
        print("Final Hough-P threshold:", threshold_hough_p)
        print("Hough-P lines count:", len(lines_hough_p))

    # draw the resulting lines of probabilistic Hough Lines algorithm
    for i in range(len(lines_hough_p)):
        line = lines_hough_p[i][0]
        cv2.line(
            img=lines_p_dst,
            pt1=(line[0], line[1]),
            pt2=(line[2], line[3]),
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA
        )
        # print(line)

    final_img = np.copy(img)
    # continue only if there are enough lines to process
    if len(lines_hough_p) < 5:
        return []
    else:
        lines = []
        for i in range(len(lines_hough_p)):
            x1 = lines_hough_p[i][0][0]
            y1 = lines_hough_p[i][0][1]
            x2 = lines_hough_p[i][0][2]
            y2 = lines_hough_p[i][0][3]
            length = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
            if x2 == x1:  # TODO solve in a better way including better handling of |k| > 1 situations
                continue
            k = (y2 - y1) / (x2 - x1)
            q = y1 - k * x1
            lines.append([k, q, length, x1, y1, x2, y2])
        lines.sort(key=lambda val: val[0])
        # kqs.sort()

        # calculate differences between the slopes of the lines
        # calculated as atan because the differences are not equally distributed in the plain slope "k" values
        diffs = []
        for i in range(1, len(lines)):
            diffs.append(math.atan(lines[i][0]) - math.atan(lines[i - 1][0]))
        if len(diffs) == 0:
            return []
        # find 3 largest spaces between the values - there are usually very distinctive from other values
        # the 3 values are the dividers between 4 clusters of lines with different slopes
        # given the marker characteristics, the slopes of the lines can be clearly divided into 4 clusters
        max_index_1 = np.argsort(diffs)[-1] + 1
        max_index_2 = np.argsort(diffs)[-2] + 1
        max_index_3 = np.argsort(diffs)[-3] + 1
        max_indices = sorted([max_index_1, max_index_2, max_index_3])

        # after getting the indices, divide the data into 4 clusters
        kqs_1 = lines[:max_indices[0]]
        kqs_2 = lines[max_indices[0]:max_indices[1]]
        kqs_3 = lines[max_indices[1]:max_indices[2]]
        kqs_4 = lines[max_indices[2]:]

        # now, it is necessary to find out which 2 groups are the rectangular area and which are inner "T-cross" shape
        # the trick is that the lines of the outer rectangular area can be further divided into 2 groups
        # the lines in such a group will have the same slope but very different "q" values
        # by the calculated standard deviation of the "q" values in all four groups, it is possible to divide the
        #   groups, because the lines of the outer rectangular area will have much higher std
        # also, given the characteristics of the slope,
        #   the outer rectangle will be either groups 1 and 3 or groups 2 and 4
        # the same applies to the inner shape - either groups 1 and 3 or groups 2 and 4
        q_std_1 = np.std(list(map(lambda val: val[1], kqs_1)))
        q_std_2 = np.std(list(map(lambda val: val[1], kqs_2)))
        q_std_3 = np.std(list(map(lambda val: val[1], kqs_3)))
        q_std_4 = np.std(list(map(lambda val: val[1], kqs_4)))

        # TODO this will need further refinement
        # firstly, the sum of 1+3 or 2+4 STDs were used to determine the inner shape,
        #   which should have much lower values
        # however, this was not working as fine as expected
        # a better approach lies in using only the minimum value and then taking the other corresponding group
        # if q_std_1 + q_std_3 > q_std_2 + q_std_4:
        if not ((q_std_1 < q_std_2 and q_std_1 < q_std_4) or (q_std_3 < q_std_2 and q_std_3 < q_std_4)):
            main_1 = kqs_1
            main_2 = kqs_3
            inner_1 = kqs_2
            inner_2 = kqs_4
        else:
            main_1 = kqs_2
            main_2 = kqs_4
            inner_1 = kqs_1
            inner_2 = kqs_3
        # now, the division of the lines between the outer and inner shape is done

        ######
        # 1. get the position of the center
        ######
        # position of the center of the marker is calculated by using the mean "k" and "q" values of the two inner lines
        # and a follow-up calculation of their intersection
        # later, some kind of weighted mean might lead to better results
        k_inner_1 = np.mean(list(map(lambda val: val[0], inner_1)))
        k_inner_2 = np.mean(list(map(lambda val: val[0], inner_2)))
        q_inner_1 = np.mean(list(map(lambda val: val[1], inner_1)))
        q_inner_2 = np.mean(list(map(lambda val: val[1], inner_2)))

        x_center, y_center = get_xy(k_inner_1, q_inner_1, k_inner_2, q_inner_2)
        if show_outputs:
            cv2.circle(final_img, center=(x_center, y_center), radius=5, color=(0, 0, 255), thickness=4)

        ######
        # 2. get the orientation
        ######
        # to get the orientation, it is necessary to find out which of the inner lines is the shorter one
        # - that one is used to determine the orientation
        # a mean length of the lines was not working fine, so the line with the maximum length is used
        #   to get the longer line - and then the other line is the shorter one
        # a possible future better approach might consist of calculating a maximal distance between all the points
        #   in the given group; the shorter line is much less likely to have a longer distance between any
        #   of its two points and maybe also use the information about the mean length to filter possible outliers
        length_mean_1 = np.max(list(map(lambda val: val[2], inner_1)))
        length_mean_2 = np.max(list(map(lambda val: val[2], inner_2)))
        if length_mean_1 < length_mean_2:
            lines_or = inner_1
        else:
            lines_or = inner_2

        # now we have the lines for getting the orientation
        # the goal is to get an orientated line segment

        # get the mean X1,Y1,X2,Y2 coordinates of all their points (in the list, they are already sorted by X1 < X2)
        ox_1 = np.mean(list(map(lambda val: val[3], lines_or)))
        oy_1 = np.mean(list(map(lambda val: val[4], lines_or)))
        ox_2 = np.mean(list(map(lambda val: val[5], lines_or)))
        oy_2 = np.mean(list(map(lambda val: val[6], lines_or)))

        # only the simple points do not have all the information to get the orientation
        # but, the calculated center is supposed to be very close to the beginning of the orientated
        #   line segment we are looking for
        # so, the point that is closer to the center is going to be considered the beginning
        distance_1 = math.sqrt(math.pow(x_center - ox_1, 2) + math.pow(y_center - oy_1, 2))
        distance_2 = math.sqrt(math.pow(x_center - ox_2, 2) + math.pow(y_center - oy_2, 2))
        # if necessary then switch the points
        if distance_1 > distance_2:
            temp = ox_1
            ox_1 = ox_2
            ox_2 = temp
            temp = oy_1
            oy_1 = oy_2
            oy_2 = temp

        orientation_vector = (ox_2 - ox_1, oy_2 - oy_1)

        # use orientation_vector and vector (1,0) to get the angle of the line relative to X-axis (to the vector (1,0))
        # since the other vector is (1,0), the formula is used in its simplified form
        cos_angle = orientation_vector[0] / math.sqrt(
            orientation_vector[0] * orientation_vector[0] + orientation_vector[1] * orientation_vector[1]
        )
        # get the final angle of the orientation
        angle = math.degrees(math.acos(cos_angle))
        # the resolution is only <0;180>, so fix those lines which Y > 0
        # then we get resolution of <0;360>
        if orientation_vector[1] > 0:
            angle = 360 - angle

        if show_outputs:
            print("orientation:", str(round(angle)) + "°")
            cv2.putText(
                img=final_img,
                text=str(round(angle)),
                org=(2, 12),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 0, 0)
            )
            cv2.line(final_img, (int(ox_1), int(oy_1)), (int(ox_2), int(oy_2)), (255, 0, 0), 1, cv2.LINE_AA)
            cv2.line(final_img, (int(ox_1), int(oy_1)), (int(ox_2), int(oy_2)), (255, 0, 0), 1, cv2.LINE_AA)
            cv2.circle(img=final_img, center=(int(ox_2), int(oy_2)), radius=3, color=(255, 0, 0), thickness=5)

        # if length_mean_1 < length_mean_2:
        #     k_or = k_inner_1
        #     q_or = q_inner_1
        # else:
        #     k_or = k_inner_2
        #     q_or = q_inner_2
        # angle = math.degrees(math.atan(k_or))
        # print("orientation:", str(round(angle)) + "°")
        # ox_1 = 0
        # oy_1 = round(int(q_or))
        # ox_2 = img_rect2.shape[1]
        # oy_2 = round(int(k_or * ox_2 + q_or))
        # cv2.line(img_rect2, (ox_1, oy_1), (ox_2, oy_2), (255, 0, 0), 1, cv2.LINE_AA)

        ######
        # 3. get the size
        ######
        # from the previous steps, we already have the 2 clusters of lines that represent the outer rectangle
        #   (quadrilateral)
        # these two clusters each need to be further divided into 2 other clusters
        #   - two opposite sides of the rectangle
        k_main_1 = np.mean(list(map(lambda val: val[0], main_1)))
        k_main_2 = np.mean(list(map(lambda val: val[0], main_2)))

        # the described division is done by "q" values which should have two very distinct clusters
        # get the values and then sort them
        qs_main_1 = list(map(lambda val: val[1], main_1))
        qs_main_2 = list(map(lambda val: val[1], main_2))

        qs_main_1.sort()
        qs_main_2.sort()

        # after sorting, calculated the difference between the values
        # one of the differences should be very high - this will be the division point
        diffs_qs_1 = []
        for i in range(1, len(qs_main_1)):
            diffs_qs_1.append(qs_main_1[i] - qs_main_1[i - 1])
        max_index_qs_1 = np.argmax(diffs_qs_1) + 1 if len(qs_main_1) > 1 else 1

        qs_main_1_1 = qs_main_1[:max_index_qs_1] if len(diffs_qs_1) > 0 else qs_main_1
        qs_main_1_2 = qs_main_1[max_index_qs_1:] if len(diffs_qs_1) > 0 else qs_main_1

        # after getting the two clusters of "q" values, get their mean values
        q_main_1_1 = np.mean(qs_main_1_1)
        q_main_1_2 = np.mean(qs_main_1_2)

        # repeat for the second set of values (the two other sides of the rectangle)
        diffs_qs_2 = []
        for i in range(1, len(qs_main_2)):
            diffs_qs_2.append(qs_main_2[i] - qs_main_2[i - 1])
        max_index_qs_2 = np.argmax(diffs_qs_2) + 1 if len(diffs_qs_2) > 1 else 1

        qs_main_2_1 = qs_main_2[:max_index_qs_2] if len(diffs_qs_2) > 0 else qs_main_2
        qs_main_2_2 = qs_main_2[max_index_qs_2:] if len(diffs_qs_2) > 0 else qs_main_2

        q_main_2_1 = np.mean(qs_main_2_1)
        q_main_2_2 = np.mean(qs_main_2_2)

        # now we have two "k" values and four "q" values
        # based on this, we have the information about all four lines
        # it is now possible to calculate the intersection of the four liens to get the four points of the rectangle
        x_main_1, y_main_1 = get_xy(k_main_1, q_main_1_1, k_main_2, q_main_2_1)
        x_main_2, y_main_2 = get_xy(k_main_1, q_main_1_1, k_main_2, q_main_2_2)
        x_main_3, y_main_3 = get_xy(k_main_1, q_main_1_2, k_main_2, q_main_2_1)
        x_main_4, y_main_4 = get_xy(k_main_1, q_main_1_2, k_main_2, q_main_2_2)

        if show_outputs:
            cv2.line(final_img, (x_main_1, y_main_1), (x_main_2, y_main_2), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(final_img, (x_main_1, y_main_1), (x_main_3, y_main_3), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(final_img, (x_main_2, y_main_2), (x_main_4, y_main_4), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(final_img, (x_main_3, y_main_3), (x_main_4, y_main_4), (0, 255, 0), 2, cv2.LINE_AA)
            end = time.time()
            print(end - start)

            cv2.imshow("Canny", edges_dst)
            cv2.imshow("Standard Hough Line Transform", lines_dst)
            cv2.imshow("Probabilistic Hough Line Transform", lines_p_dst)
            cv2.imshow("Rectangle", final_img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return [
            (x_center, y_center),
            angle,
            (x_main_1, y_main_1), (x_main_2, y_main_2),
            (x_main_1, y_main_1), (x_main_3, y_main_3),
            (x_main_2, y_main_2), (x_main_4, y_main_4),
            (x_main_3, y_main_3), (x_main_4, y_main_4),
        ]


if __name__ == '__main__':
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
    # image_id = "0385"  # !
    image_id = "0416"
    img_, x, y, = get_cropped(image_id)
    print(get_data(img_))
