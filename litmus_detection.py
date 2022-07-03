# %%
# !/usr/bin/env python
import math
import cv2
import numpy as np
import random as rng
import qr_extractor as qe

# Gaussian Smoothing
BLUR_VALUE = 3

# Litmus Ratio
LITMUS_RATIO = [0.989, 0.925, 0.82, 0.757]

# Helper Function For Location Calculation
def get_center(c):
    m = cv2.moments(c)
    return [int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])]

def get_midpoint(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

def rotate_img(img, center, degree):
    M = cv2.getRotationMatrix2D(center, degree, 1)
    rotated = cv2.warpAffine(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated

def extend_scale(a, b, scale):
    return [(int)(b[0] + (b[0] - a[0]) * scale), (int)(b[1] + (b[1] - a[1]) * scale)]

# Helper Function For canny edge detection
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def pre_processing_img_for_edge_detection(image, blur_value, canny_min_val, canny_max_val):
    '''
    pre-processing the img and return the canny edge of it
    :param image: input img
    :param blur_value: kernel_size for the cv2.GaussianBlur
    :param canny_min_val: minVal for Canny Edge Detection
    :param canny_max_val: maxVal for Canny Edge Detection
    :return: img with canny edge
    '''
    gray = grayscale(image)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = gaussian_blur(gray, blur_value)
    edged = canny(gray, canny_min_val, canny_max_val)
    return edged

# Helper Function For White Balance
def white_balance(img):
    """ 
    used to improve the performance of the QR-Code-Extractor and cv2.edge_detection
    """
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def simple_white_balance(img, strip_squares):
    """
    white_balanced based on the white pixel of the strip
    For each RBG channel, find the pixel's peak value, and then extend the value range of the strip from [0, peak value] to [0, 255]. 
    The reason behind it is that we already know that the RGB value of the strip white part should be (255, 255, 255)
    """
    assert img.shape[2] == 3

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)
        normalized = cv2.normalize(channel, channel.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

def get_background_mask_from_qr_pos(rectangles):
    """
    get the mask of background approximate position, to narrow down the search range
    :param rectangles: the qr code positions
    :return: the approximate position of black background
    """
    x1, x2, x3, x4 = rectangles
    y4 = extend_scale(x3, x4, 1)
    y3 = x4
    y2 = extend_scale(x4, x1, 3)
    y1 = extend_scale(y3, get_midpoint(y2, y4), 1)
    y4 = extend_scale(y1, y4, 0.05)
    y3 = extend_scale(y2, y3, 0.05)
    return np.array([[y1, y2, y3, y4]], dtype=np.int32)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def get_background_contours(image, area_threshold):
    '''
    get the black background contour based on the following criteria
    Methodology:
        1. use the canny edge detection to find all the contours
        2. traverse each contour, calculate the bounding area of it’s minAreaRect
        3. Set the  qr_area * ratio as the lower bound
        4. The contour that contains the background should be the contour with minimum bounding area above the lower bound
    :param image: input img
    :param area_threshold: the area of the qr code
    :return: the contours that containing the black background
    '''
    edged = pre_processing_img_for_edge_detection(image, BLUR_VALUE, 1, 200)

    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_bounding_area = area_threshold * 5
    bg_contours = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        bounding_area = cv2.contourArea(box)
        # Approximate the contour
        if bounding_area > area_threshold * 1.16 and bounding_area < min_bounding_area:
            min_bounding_area = bounding_area
            bg_contours = [box]

    return bg_contours


def get_bg_rectangle(image, area_threshold):
    # get the background contour
    rectangle = get_background_contours(image, area_threshold)
    # masked the trivial part
    masked_image = region_of_interest(image.copy(), rectangle)
    return masked_image, rectangle

def get_bounding_contour(contour):
    return np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))


def get_strip_rectangle(bg_img, bg_area, bg_rectangle):
    '''
    Get the strip contour based on the following criteria
    Methodology:
        Find the major part of the strip:
        1. use the canny edge detection to find all the contours
        2. Set the bg_area * ratio as the upper bound
        3. traverse each contour, calculate the bounding area of it’s minAreaRect
        4. the contour of major part should be the contour with maximum bounding area under the upper bound
        Find the minor parts of the strip:
        5. build a virtual box by extending the edges of the rectangle containing major part of strip
        6. traverse each contour, find the contour meets the following criteria:
            a. the contour the moment of the contour is inside the virtual box
            b. (edge point to the virtual rectangle is inside the virtual box
                or the distance of edge point to the virtual rectangle within tolerance)
        7. we assume it's also part of the strip, concatenate this contour to the contour of majority_strip
    The rationale behind this Methodology: canny edge detection may detect the strip as separated parts
        and the morphological transformations is not enough to group the separated contours of the strip
    :param bg_img: input img, containing the background contour
    :param bg_area: the area of the background contour
    :param bg_square: the vertices of the background rectangle
    :return masked_image: the output img containing the strip only
    :return final_strip: the vertices of the strip
    :return rotated: straightened img rotated based on the strip angle
    '''
    img = bg_img.copy()
    edged = pre_processing_img_for_edge_detection(img, 1, 1, 200)
    thresh_gray = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # contours, hierarchy = cv2.findContours(thresh_gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
    # Draw contours + hull results
    drawing = np.zeros((edged.shape[0], edged.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv2.drawContours(drawing, contours, i, color)
        cv2.drawContours(drawing, hull_list, i, color)

    squares = []
    max_bounding_area = 0
    max_rect = []
    strip_contours = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        bounding_area = cv2.contourArea(box)
        # filter the contours box that are not in the background
        box_inside_bg = True
        for point in box:
            box_inside_bg = box_inside_bg and is_point_inside_box(point, bg_rectangle[0])
        if not box_inside_bg:
            # print("bounding box is outside the background")
            continue

        if bounding_area < bg_area * 0.7:
            squares.append(box)
            if bounding_area > max_bounding_area:
                max_bounding_area = bounding_area
                max_rect = rect
                strip_contours = [box]
        # else:
        #     print("bounding_area > bg_area * 0.7 ")

    if (len(squares) == 0):
        print("strip contours not founded")
    squares = sorted(squares, key=lambda x: cv2.contourArea(x))

    # find the major part of the strip contour
    majority_strip = strip_contours[-1]

    # build the virtual box
    virtual_box = build_virtual_box(majority_strip, 0.35)

    # find the minor parts of the strip
    similar_contours = [majority_strip]
    tolerant_distance = abs(virtual_box[0][0] - virtual_box[1][0])
    # we only consider the top 10 largest contour
    for contour in squares[-10:]:
        if is_contour_part_of_strip(contour, virtual_box, tolerant_distance):
            similar_contours.append(contour)

    similar_concat = np.concatenate(similar_contours)

    bounding_box = cv2.minAreaRect(np.array(similar_concat, dtype=np.int32))
    bounding_pts = cv2.boxPoints(bounding_box).astype(int)

    strip_vertices = find_strip_exact_pos(virtual_box, bounding_pts)
    masked_image = region_of_interest(bg_img.copy(), [strip_vertices])

    # straighten the img based on the angle of the strip
    rotated_img = bg_img.copy()
    if max_rect[2] > 20:
        angel = 90 - max_rect[2]
    elif max_rect[2] < -20:
        angel = max_rect[2] + 90
    else:
        angel = max_rect[2]
    M = cv2.getRotationMatrix2D(max_rect[0], angel, 1)
    rotated = cv2.warpAffine(rotated_img, M, rotated_img.shape[1::-1])

    return masked_image, [strip_vertices], rotated

def find_strip_exact_pos(virtual_box, approximate_rect):
    '''
    find the vertices of the strip precisely
    input:
        virtual_box: virtual_box extended by the majoriry contour of strip
        approximate_rect: the approximate rect containg the strip
    output:
        edge conner of the strip
    '''
    approximate_rect = build_virtual_box(approximate_rect, 0)
    upper_line = [approximate_rect[0], approximate_rect[1]]
    right_line = [virtual_box[1], virtual_box[2]]
    bottom_lie = [approximate_rect[2], approximate_rect[3]]
    left_line = [virtual_box[3], virtual_box[0]]

    upper_left_edge = line_intersection(upper_line, left_line)
    upper_right_edge = line_intersection(upper_line, right_line)
    bottom_left_edge = line_intersection(bottom_lie, left_line)
    bottom_right_edge = line_intersection(bottom_lie, right_line)
    return np.array([upper_left_edge, upper_right_edge, bottom_right_edge, bottom_left_edge], dtype=np.int32)


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def is_contour_part_of_strip(contour, virtual_box, tolerant_distance):
    '''
    check wheather the contour meets the following criteria:
        a. the contour the moment of the contour is inside the virtual box
        b. (edge point to the virtual rectangle is inside the virtual box
            or the distance of edge point to the virtual rectangle within tolerance)
    '''
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    center_x = sum(point[0] for point in box) / len(box)
    center_y = sum(point[1] for point in box) / len(box)
    # check the center is inside the virtual box and the edge_conner is nearby the virtual box
    if is_point_inside_box((center_x, center_y), virtual_box):
        # check the edge_conner is nearby the virtual box
        for edge_point in box:
            if not is_point_nearby_box(edge_point, virtual_box, tolerant_distance):
                return False
        return True
    else:
        return False

def is_point_nearby_box(point, box, distance):
    dis = cv2.pointPolygonTest(box, (point[0], point[1]), True)
    if dis < distance:
        return True
    else:
        return False

def is_point_inside_box(point, box):
    if (cv2.pointPolygonTest(box, (point[0], point[1]), False) == -1):
        return False
    else:
        return True

def build_virtual_box(box, extension_ratio):
    """
    Build virtual_box with the following steps:
        1. find the Moments of the box
        2. find the upper point and the bottom point
        3. find the upper left, upper right, bottom left, bottom right
        4. extend the line
    """    
    cy = get_center(box)[1]
    upper, bottom = [], []
    for c in box:
        if c[1] > cy:
            upper.append(c)
        else:
            bottom.append(c)
    upper = sorted(upper, key=lambda u: u[0])
    bottom = sorted(bottom, key=lambda b: b[0])
    upper_left = [int(upper[0][0] + (upper[0][0] - bottom[0][0]) * extension_ratio),
                  int(upper[0][1] + (upper[0][1] - bottom[0][1]) * extension_ratio)]
    upper_right = [int(upper[1][0] + (upper[1][0] - bottom[1][0]) * extension_ratio),
                   int(upper[1][1] + (upper[1][1] - bottom[1][1]) * extension_ratio)]
    bottom_left = [int(bottom[0][0] - (upper[0][0] - bottom[0][0]) * extension_ratio),
                   int(bottom[0][1] - (upper[0][1] - bottom[0][1]) * extension_ratio)]
    bottom_right = [int(bottom[1][0] - (upper[1][0] - bottom[1][0]) * extension_ratio),
                    int(bottom[1][1] - (upper[1][1] - bottom[1][1]) * extension_ratio)]
    virtual_box = np.array([upper_left, upper_right, bottom_right, bottom_left], dtype=np.int32)
    return virtual_box

def get_strip(frame):
    '''
    Get the strip location with the following steps:
        1. for a given img, find the location of qr code by utilizing the tool QR-Code-Extractor 
        2. find approximate position of the black background
        3. find exact position of the black background
        4. get the strip position
    input: img frame
    output: stripes background area squares, and the processed img contains only stripes background

    :param frame:  img frame
    :return strip_squares: the vertices of rectangle that containing strip
    :return strip_img: img containing only the strip
    '''

    # straighten the img based on the qr code angle
    _, _, _, rotated_img = qe.extract(frame, False)
    # get the qr code position
    frame, qr_square, qr_area, rotated_img = qe.extract(rotated_img, False)

    # get the approximate position of the black background to narrow down the search range
    qr_pos = [qr_square[x][0][:] for x in range(len(qr_square))]
    mask = get_background_mask_from_qr_pos(qr_pos)
    # masked the travail part
    masked_image = region_of_interest(frame.copy(), mask)

    # get the exact position of the black background based on the approximate position and qr_area
    bg_img, bg_squares = get_bg_rectangle(masked_image, qr_area)
    bg_area = cv2.contourArea(bg_squares[0])

    # straighten the img based on the strip angle
    strip_img, _, rotate_strip = get_strip_rectangle(bg_img.copy(), bg_area, bg_squares)
    # get the strip position
    strip_img, strip_squares, rotate_strip = get_strip_rectangle(rotate_strip, bg_area, bg_squares)

    return strip_squares, strip_img


def get_litmus_square(strip_img, strip_squares, scale=LITMUS_RATIO):
    '''
    Given the strip img and the vertices of the strip rectangle
    return the exact position of the litmus

    :param strip_img: strip img
    :param strip_squares: vertices of rectangle that containing strip
    :param scale: fixed ratio to locate the litmus part
    :return free_litmus_contour: the vertices of rectangle that containing free litmus
    :return total_litmus_contour: the vertices of rectangle that containing total litmus
    :return free_litmus_masked_image: the masked img of free_litmus
    :return total_litmus_masked_image: the masked img of total_litmus
    '''
    upper_left, upper_right, bottom_right, bottom_left = strip_squares[0]
    free_upper_left = [upper_left[0] - (upper_left[0] - bottom_left[0]) * scale[0],
                       upper_left[1] - (upper_left[1] - bottom_left[1]) * scale[0]]
    free_upper_right = [upper_right[0] - (upper_right[0] - bottom_right[0]) * scale[0],
                        upper_right[1] - (upper_right[1] - bottom_right[1]) * scale[0]]
    free_bottom_left = [upper_left[0] - (upper_left[0] - bottom_left[0]) * scale[1],
                        upper_left[1] - (upper_left[1] - bottom_left[1]) * scale[1]]
    free_bottom_right = [upper_right[0] - (upper_right[0] - bottom_right[0]) * scale[1],
                         upper_right[1] - (upper_right[1] - bottom_right[1]) * scale[1]]
    free_litmus_contour = np.array([free_upper_left, free_upper_right, free_bottom_right, free_bottom_left],
                                   dtype=np.int32)

    total_upper_left = [upper_left[0] - (upper_left[0] - bottom_left[0]) * scale[2],
                        upper_left[1] - (upper_left[1] - bottom_left[1]) * scale[2]]
    total_upper_right = [upper_right[0] - (upper_right[0] - bottom_right[0]) * scale[2],
                         upper_right[1] - (upper_right[1] - bottom_right[1]) * scale[2]]
    total_bottom_left = [upper_left[0] - (upper_left[0] - bottom_left[0]) * scale[3],
                         upper_left[1] - (upper_left[1] - bottom_left[1]) * scale[3]]
    total_bottom_right = [upper_right[0] - (upper_right[0] - bottom_right[0]) * scale[3],
                          upper_right[1] - (upper_right[1] - bottom_right[1]) * scale[3]]
    total_litmus_contour = np.array([total_upper_left, total_upper_right, total_bottom_right, total_bottom_left],
                                    dtype=np.int32)

    return free_litmus_contour, total_litmus_contour


def crop_circle(rectangle_contour, rectangle_img):
    '''
    Given a litmus, find the biggest circle inside this rectangle

    :param rectangle_contour: the vertices of rectangle that containing litmus
    :param rectangle_img: the img
    :return cropped_img: the strip img cropped into circle
    :return average_rgb: the average rgb value of the strip circle
    '''
    radius, cropped_img = crop_circle_inside_square(rectangle_contour, rectangle_img)
    average_rgb = get_rgb_value_of_circle_strip(cropped_img, radius)
    return average_rgb, cropped_img


def crop_circle_inside_square(square, img):
    # get square center
    circle_center = get_center(square)
    circle_center_tuple = (circle_center[0], circle_center[1])
    # find the minimal distance between the center and the square
    radius = int(cv2.pointPolygonTest(square, circle_center_tuple, True))

    # crop the img based on the circle
    crop_img = img.copy()
    cv2.circle(crop_img, circle_center_tuple, radius, (0, 0, 0))
    crop_img = crop_img[circle_center[1] - radius: circle_center[1] + radius,
               circle_center[0] - radius: circle_center[0] + radius]

    return radius, crop_img


def get_rgb_value_of_circle_strip(img, radius):
    try:
        # generate mask
        mask = np.zeros_like(img)
        cv2.circle(mask, (radius, radius), radius, (255, 255, 255), -1)
        dst = cv2.bitwise_and(img, mask)
        # filter black color and fetch color values
        data = []
        for i in range(3):
            channel = dst[:, :, i]
            indices = np.where(channel != 0)[0]
            color = np.mean(channel[indices])
            data.append(int(color))

        # opencv images are in bgr format: blue, green, red
        print("blue, green, red =", data)
        return data
    except:
        return []


def processing_img(frame):
    '''
    For a given img, return it's cropped circle litmus and the average rgb value

    :param frame: input image
    :return free_circle_cropped_img: the free litmus img cropped into circle
    :return free_rgb: the average rgb value of the free litmus
    :return total_circle_cropped_img: the total litmus img cropped into circle
    :return total_rgb: the average rgb value of the total litmus
    '''

    simple_wb_out = white_balance(frame)
    strip_vertices, strip_img = get_strip(simple_wb_out)
    wb_strip = simple_white_balance(strip_img, strip_vertices)
    free_litmus_contour, total_litmus_contour = get_litmus_square(wb_strip, strip_vertices)
    print("free:")
    free_rgb, free_circle_cropped_img = crop_circle(free_litmus_contour, wb_strip)
    print("total:")
    total_rgb, total_circle_cropped_img = crop_circle(total_litmus_contour, wb_strip)
    return free_circle_cropped_img, free_rgb, total_circle_cropped_img, total_rgb, wb_strip
