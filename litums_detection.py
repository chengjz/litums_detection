#%%
#!/usr/bin/env python
import math
import imutils
import cv2
import os
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model
from scipy.misc import imread, imsave, imresize, imshow
# import line_detection

BLUR_VALUE = 3
SQUARE_TOLERANCE = .15
AREA_TOLERANCE = 0.2
LITMUS_AREA_TOLERANCE = 0.05
DISTANCE_TOLERANCE = 0.25
WARP_DIM = 300
SMALL_DIM = 29


# Gaussian smoothing
kernel_size = 3

# Canny Edge Detector
low_threshold = 50
high_threshold = 150


# Hough Transform
rho = 2 # distance resolution in pixels of the Hough grid
theta = 1 * np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15	 # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10 #minimum number of pixels making up a line
max_line_gap = 20	# maximum gap in pixels between connectable line segments


#if they have at least 2 children and no parent, calls itself within the function.. weird didnt know you could do that
def count_children(hierarchy, parent, inner=False):
    if parent == -1:
        return 0
    elif not inner:
        return count_children(hierarchy, hierarchy[parent][2], True)
    return 1 + count_children(hierarchy, hierarchy[parent][0], True) + count_children(hierarchy, hierarchy[parent][2], True)

def has_square_parent(hierarchy, squares, parent):
    if hierarchy[parent][3] == -1:
        return False
    if hierarchy[parent][3] in squares:
        return True
    return has_square_parent(hierarchy, squares, hierarchy[parent][3])

def get_center(c):
    m = cv2.moments(c)
    return [int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])]


def get_angle(p1, p2):
    x_diff = p2[0] - p1[0]
    y_diff = p2[1] - p1[1]
    return math.degrees(math.atan2(y_diff, x_diff))


def get_midpoint(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]


def get_farthest_points(contour, center):
    distances = []
    distances_to_points = {}
    for point in contour:
        # print("point")
        # print(point)
        # print(center)
        point = point[0]
        d = math.hypot(point[0] - center[0], point[1] - center[1])
        distances.append(d)
        distances_to_points[d] = point
    distances = sorted(distances)
    return [distances_to_points[distances[-1]], distances_to_points[distances[-2]]]

def get_close_points(contour, center):
    distances = []
    distances_to_points = {}
    for point in contour:
        # print("point")
        # print(point)
        # print(center)
        # point = point[0]
        d = math.hypot(point[0] - center[0], point[1] - center[1])
        distances.append(d)
        distances_to_points[d] = point
    distances = sorted(distances)
    return [distances_to_points[distances[i]] for i in range(len(distances))]


def rotate_img(img, center, degree):

    M = cv2.getRotationMatrix2D(center, degree, 1)
    rotated = cv2.warpAffine(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated

def line_intersection(line1, line2):
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(x_diff, y_diff)
    if div == 0:
        return [-1, -1]

    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div
    return [int(x), int(y)]


def extend(a, b, length, int_represent=False):
    length_ab = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    if length_ab * length <= 0:
        return b
    result = [b[0] + (b[0] - a[0]) / length_ab * length, b[1] + (b[1] - a[1]) / length_ab * length]
    if int_represent:
        return [int(result[0]), int(result[1])]
    else:
        return result


def extract(frame, debug=False):
    output = frame.copy()

    # Remove noise and unnecessary contours from frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.GaussianBlur(gray, (BLUR_VALUE, BLUR_VALUE), 0)
    edged = cv2.Canny(gray, 30, 200)

    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # out = frame.copy()
    # cv2.drawContours(out, contours, -1, (127, 0, 0), 2)
    # cv2.imwrite("extract_frame_contours.jpg", out)

    squares = []
    square_indices = []

    i = 0
    for c in contours:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) == 4:
            # Determine if quadrilateral is a square to within SQUARE_TOLERANCE
            if area > 25 and 1 - SQUARE_TOLERANCE < math.fabs((peri / 4) ** 2) / area < 1 + SQUARE_TOLERANCE:
                if count_children(hierarchy[0], i) >= 2 and has_square_parent(hierarchy[0], square_indices, i) is False:
                    squares.append(approx)
                    square_indices.append(i)

        i += 1

    main_corners = []
    east_corners = []
    south_corners = []
    tiny_squares = []
    rectangles = []
    angle = 0
    center_point = []

    # Determine if squares are QR codes
    j = 0
    for square in squares:
        j=j+1
        area = cv2.contourArea(square)
        center = get_center(square)
        peri = cv2.arcLength(square, True)
        similar = []
        tiny = []
        for other in squares:
            if square[0][0][0] != other[0][0][0] or square[0][0][1] != other[0][0][1]:

                # Determine if square is similar to other square within AREA_TOLERANCE
                if math.fabs(area - cv2.contourArea(other)) / max(area, cv2.contourArea(other)) <= AREA_TOLERANCE:
                    similar.append(other)
                elif peri / 4 / 2 > cv2.arcLength(other, True) / 4:
                    tiny.append(other)

        if len(similar) >= 2:
            distances = []
            distances_to_contours = {}
            for sim in similar:
                sim_center = get_center(sim)
                d = math.hypot(sim_center[0] - center[0], sim_center[1] - center[1])
                distances.append(d)
                distances_to_contours[d] = sim
            distances = sorted(distances)
            closest_a = distances[-1]
            closest_b = distances[-2]

            if max(closest_a, closest_b) < cv2.arcLength(square, True) * 2.5 and math.fabs(closest_a - closest_b) / max(closest_a, closest_b) <= DISTANCE_TOLERANCE:
                # Determine placement of other indicators (even if code is rotated)
                angle_a = get_angle(center, get_center(distances_to_contours[closest_a]))
                angle_b = get_angle(center, get_center(distances_to_contours[closest_b]))
                angle = angle_b
                # print('angle a ', angle_a, ' angle b ', angle_b)
                if angle_a < angle_b or (angle_b < -90 and angle_a > 0):
                    east = distances_to_contours[closest_a]
                    south = distances_to_contours[closest_b]
                else:
                    east = distances_to_contours[closest_b]
                    south = distances_to_contours[closest_a]
                midpoint = get_midpoint(get_center(east), get_center(south))
                center_point = midpoint
                # Determine location of fourth corner
                # Find closest tiny indicator if possible
                min_dist = 10000
                t = []
                tiny_found = False
                if len(tiny) > 0:
                    for tin in tiny:
                        tin_center = get_center(tin)
                        d = math.hypot(tin_center[0] - midpoint[0], tin_center[1] - midpoint[1])
                        if d < min_dist:
                            min_dist = d
                            t = tin
                    tiny_found = len(t) > 0 and min_dist < peri
                diagonal = peri / 4 * 1.41421

                if tiny_found:
                    # Easy, corner is just a few blocks away from the tiny indicator
                    tiny_squares.append(t)
                    offset = extend(midpoint, get_center(t), peri / 4 * 1.41421)
                else:
                    # No tiny indicator found, must extrapolate corner based off of other corners instead
                    farthest_a = get_farthest_points(distances_to_contours[closest_a], center)
                    farthest_b = get_farthest_points(distances_to_contours[closest_b], center)
                    # Use sides of indicators to determine fourth corner
                    offset = line_intersection(farthest_a, farthest_b)
                    if offset[0] == -1:
                        # Error, extrapolation failed, go on to next possible code
                        continue
                    offset = extend(midpoint, offset, peri / 4 / 7)
                    if debug:
                        cv2.line(output, (farthest_a[0][0], farthest_a[0][1]), (farthest_a[1][0], farthest_a[1][1]), (0, 0, 255), 4)
                        cv2.line(output, (farthest_b[0][0], farthest_b[0][1]), (farthest_b[1][0], farthest_b[1][1]), (0, 0, 255), 4)

                # Append rectangle, offsetting to farthest borders
                rectangles.append([extend(midpoint, center, diagonal / 2, True), extend(midpoint, get_center(distances_to_contours[closest_b]), diagonal / 2, True), offset, extend(midpoint, get_center(distances_to_contours[closest_a]), diagonal / 2, True)])
                east_corners.append(east)
                south_corners.append(south)
                main_corners.append(square)
                # qr_area = area

    codes = []
    i = 0
    if debug:
        # Draw debug information onto frame before ting it
        cv2.drawContours(output, squares, -1, (5, 5, 5), 2)
        cv2.drawContours(output, main_corners, -1, (0, 0, 128), 2)
        cv2.drawContours(output, east_corners, -1, (0, 128, 0), 2)
        cv2.drawContours(output, south_corners, -1, (128, 0, 0), 2)
        cv2.drawContours(output, tiny_squares, -1, (128, 128, 0), 2)
        cv2.imwrite("debug_frame_qr.jpg", output)

    for rect in rectangles:
        i += 1
        # Draw rectangle
        vrx = np.array((rect[0], rect[1], rect[2], rect[3]), np.int32)
        vrx = vrx.reshape((-1, 1, 2))
        cv2.polylines(output, [vrx], True, (0, 255, 255), 1)
        # Warp codes and draw them
        wrect = np.zeros((4, 2), dtype="float32")
        wrect[0] = rect[0]
        wrect[1] = rect[1]
        wrect[2] = rect[2]
        wrect[3] = rect[3]
        dst = np.array([
            [0, 0],
            [WARP_DIM - 1, 0],
            [WARP_DIM - 1, WARP_DIM - 1],
            [0, WARP_DIM - 1]], dtype="float32")
        warp = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(wrect, dst), (WARP_DIM, WARP_DIM))
        # Increase contrast
        warp = cv2.bilateralFilter(warp, 11, 17, 17)
        warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(warp, (SMALL_DIM, SMALL_DIM), 0, 0, interpolation=cv2.INTER_CUBIC)
        _, small = cv2.threshold(small, 100, 255, cv2.THRESH_BINARY)
        codes.append(small)

    try:
        upper_left = get_close_points(rectangles[0], get_center(tiny_squares[0]))
        east_corners_ptr = get_farthest_points(east_corners[0], center_point)
        south_corners_ptr = get_farthest_points(south_corners[0], center_point)
        bottom_right = get_close_points(rectangles[0], get_center(tiny_squares[0]))
        upper_left_center = get_center(main_corners[0])
        upper_left_center = (upper_left_center[0], upper_left_center[1])
    except:
        print("QR code not detected")
        
    
    # print("upper_left_center")
    # print(upper_left_center)

    # print(east_corners_ptr)
    # print(east_corners)
    # print(east_corners)
    # print(east_corners)
    # print(east_corners)
    qr_square = np.array([[upper_left[-1]], [east_corners_ptr[0]], [(bottom_right[0])], [south_corners_ptr[0]]], np.dtype(np.int32))
    # print("qr_square")
    # print(qr_square)
    out = frame.copy()
    # cv2.drawContours(out, [qr_square], -1, (127, 0, 0), 2)
    # cv2.imwrite("frame_qr.jpg", out)
    # print("rectangles")
    # print(np.shape(rectangles))
    # print(rectangles[0][0])
    w = math.hypot(rectangles[0][0][0] - rectangles[0][1][0], rectangles[0][0][1] - rectangles[0][1][1])
    h = math.hypot(rectangles[0][0][0] - rectangles[0][2][0], rectangles[0][0][1] - rectangles[0][2][1])
    rotated_img = rotate_img(frame, upper_left_center, angle)
    # rectangles = sorted(rectangles[0], key=lambda x: cv2.contourArea(x))
    qr_area = w * h
    # cv2.imwrite("frame_rotated.jpg", rotated_img)
    return codes, out, qr_square, angle, qr_area, rotated_img

def mask_area(qr_square, angle):
    rectangles, main_corners, east_corners, south_corners = qr_square
    angle_a = get_angle(rectangles[0][0][:], rectangles[0][1][:]) 
    dist = math.hypot(rectangles[0][0][0] - rectangles[0][1][0], rectangles[0][0][1] - rectangles[0][1][1])
    square = []
    xPos, yPos = getSquareCenter(rectangles[0][:][:])
    # print(xPos, yPos)
    # print(angle)
    # print(rectangles[0][0][:], rectangles[0][1][:])
    # print(rectangles)
    # print(angle_a, dist)
    # print(rectangles[0][:][:])
    # print(main_corners)
    # print(east_corners)
    # print(south_corners)

def get_mask(rectangles):
    x1, x2, x3, x4 = rectangles
    y4 = extend_scale(x3, x4, 1)
    y3 = x4
    y2 = extend_scale(x4, x1, 3)
    y1 = extend_scale(y3, get_midpoint(y2, y4), 1)
    y4 = extend_scale(y1, y4, 0.05)
    y3 = extend_scale(y2, y3, 0.05)
    # print(y1, y2, y3, y4)
    return np.array([[y1, y2, y3, y4]], dtype=np.int32)

def extend_scale(a, b, scale):
    length_ab = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    result = [(int)(b[0] + (b[0] - a[0]) * scale), (int)(b[1] + (b[1] - a[1]) * scale)]
    return result

def getSquarePos(qr_square):
    rectangles, main_corners, east_corners, south_corners = qr_square
    pos = []

def getSquareCenter(rectangles):
    xPos = yPos = 0
    for pos in rectangles:
        xPos += pos[0]
        yPos += pos[1]
    return xPos / len(rectangles), yPos / len(rectangles)

def region_of_interest(img, vertices):
	"""
	Applies an image mask.
	
	Only keeps the region of the image defined by the polygon
	formed from `vertices`. The rest of the image is set to black.
	"""
	#defining a blank mask to start with
	mask = np.zeros_like(img)   
	
	#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
		
	#filling pixels inside the polygon defined by "vertices" with the fill color	
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	
	#returning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

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

	
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
	"""
	`img` should be the output of a Canny transform.
		
	Returns an image with hough lines drawn.
	"""
	lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
	line_img = np.zeros((img.shape, 3), dtype=np.uint8)  # 3-channel RGB image
	# cv2.drawContours(line_img, lines, -1, (128, 0, 0), 2)
    
	return line_img

def get_contours(image, area_threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.GaussianBlur(gray, (BLUR_VALUE, BLUR_VALUE), 0)
    edged = cv2.Canny(gray, 1, 200)

    
    # contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    test = image.copy()
    # cv2.drawContours(test, contours, -1, (128, 0, 0), 2)
    # cv2.imwrite("frame_bg_contours.jpg", test)
    # print("bg_contours")
    # print(area_threshold)
    # for x in contours:
    #     print(cv2.contourArea(x))

    big_squares = []
    big_square_indices = []
    small_squares = []
    small_square_indices = []

    
    i = 0
    arr = []
    
    min_bounding_area = area_threshold * 5
    min_rect = []
    bg_contours = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        # print("rect")
        # print(rect)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # cv2.drawContours(img,[box],0,(0,0,255),2)
        # print("box")
        # print(box)
        bounding_area = cv2.contourArea(box)
        # print("bounding_area")
        # print(bounding_area)
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if bounding_area > area_threshold * 1.16 and bounding_area < min_bounding_area:
            min_bounding_area = bounding_area
            min_rect = rect
            bg_contours = [box]
            # squares.append(c)

    # for c in contours:
    #     x,y,w,h = cv2.boundingRect(c)
    #     bounding_area = w * h
    #     bounding_contours = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], np.dtype(np.int32))
    #     # bounding_contours = [[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]]
    #     # print("bounding_contours")
    #     # print(bounding_contours)
    #     # print(c)
    #     # arr += bounding_contours
    #     # Approximate the contour
    #     peri = cv2.arcLength(c, True)
    #     area = cv2.contourArea(c)
    #     approx = cv2.approxPolyDP(c, 0.03 * peri, True)

    #     # Find all quadrilateral contours
    #     # if len(approx) >= 4:
    #     if bounding_area > area_threshold * 1.16:
    #         big_squares.append(bounding_contours)
    #         big_square_indices.append(i)

    #     if bounding_area <= area_threshold * 1.16:
    #         small_squares.append(bounding_contours)
    #         small_square_indices.append(i)
    #     i += 1

    # big_cntsSorted = sorted(big_squares, key=lambda x: cv2.contourArea(x))
    # # small_cntsSorted = sorted(small_squares, key=lambda x: cv2.contourArea(x))
    # for x in big_cntsSorted:
    #     print(cv2.contourArea(x))
    # print("bg")
    # print(area_threshold * 1.16)
    # for x in small_cntsSorted:
    #     print(cv2.contourArea(x))
    # test = image.copy()
    # cv2.drawContours(test, big_cntsSorted, -1, (128, 0, 0), 2)
    # cv2.imwrite("frame_bg_big_cntsSorted.jpg", test)
    return bg_contours, edged

def get_masked_image(image, area_threshold, flag):
    
    big_cntsSorted, edged = get_contours(image, area_threshold)
    print(big_cntsSorted)
    if flag is True:
        squares = [big_cntsSorted[0]]
    # else:
    #     squares = [small_cntsSorted[-1]]

    masked_image = region_of_interest(image, squares)
    # cv2.drawContours(image, squares, -1, (0, 0, 0), 2)
    return masked_image, squares

def get_bounding_contour(contour):
    return np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))

def get_strip_square(img, bg_area):
    cv2.imwrite("frame_strip_img.jpg", img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # gray = cv2.GaussianBlur(gray, (BLUR_VALUE, BLUR_VALUE), 0)
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    edged = cv2.Canny(gray, 1, 200)
    cv2.imwrite("frame_strip_edged2.jpg", edged)

    
    # contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (128, 0, 0), 2)
    # cv2.imwrite("frame_contours.jpg", img)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    tot_squares = []
    square_indices = []

    # print("contours")
    # contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    # for x in contours:
    #     # print(x)
    #     print(cv2.contourArea(x))

    squares = []
    # rect_to_box_map = {}
    i = 0
    arr = []
    max_bounding_area = 0
    max_rect = []
    bg_contours = []
    print("bounding_area")
    print(bg_area * 0.7 )
    for c in contours:
        rect = cv2.minAreaRect(c)
        # print("rect")
        # print(rect)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # cv2.drawContours(img,[box],0,(0,0,255),2)
        # print("box")
        # print(box)
        bounding_area = cv2.contourArea(box)
        # print("bounding_area")
        # print(bounding_area)
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        print(bounding_area)
        if bounding_area < bg_area * 0.7 and bounding_area > max_bounding_area:
            max_bounding_area = bounding_area
            max_rect = rect
            bg_contours = [box]
            squares.append(c)
            # rect_to_box_map[rect] = rect
            # i += 1

    # print("bg_area_size")
    # print(bg_area * 0.8)
    # print()
    # print("squares")
    squares = sorted(squares, key=lambda x: cv2.contourArea(get_bounding_contour(x)))
    # for x in squares:
    #     # print(x)
    #     print(cv2.contourArea(x))
    # bg_contours = [get_bounding_contour(squares[-1])]

    # strip_image, strip_squares = get_masked_image(bg_image, bg_area * 0.9, False)
    # drawContours = cv2.drawContours(img, bg_contours, -1, (128, 0, 0), 2)
    # cv2.imwrite("frame_drawContours.jpg", drawContours)
    masked_image = region_of_interest(img, bg_contours)
    cv2.imwrite("frame_masked_image1.jpg", masked_image)
    print("rect")
    rect = cv2.minAreaRect(squares[-1])
    bg_area = cv2.contourArea(np.int0(cv2.boxPoints(rect)))
    print(max_bounding_area)
    print(max_rect)
    # print(rect_to_box_map[squares[-1]])
    # rotate_center = (squares[-1][0], squares[-1][1])
    rotated_img = img.copy()
    if max_rect[2] > 20:
        angel = 90 - max_rect[2] 
    elif max_rect[2] < -20:
        angel = max_rect[2] + 90
    else:
        angel = max_rect[2]
    # print(angel)
    M = cv2.getRotationMatrix2D(max_rect[0], angel, 1)
    rotated = cv2.warpAffine(rotated_img, M, rotated_img.shape[1::-1])
    # rotated_img = rotate_img(rotated_img, max_rect[0], max_rect[2])
    # cv2.imwrite("frame_rotated_img.jpg", rotated)
    return masked_image, [squares[-1]], rotated


def simple_white_balance(img, strip_squares):
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

def get_backgroud_square(frame):
    '''
    for a given img frame, find the location of qr code and then locate the stripes background area.
    input: img frame
    output: stripes background area squares, and the processed img contains only stripes background
    '''
    _, _, _, _, _, rotated_img = extract(frame, True)
    codes, frame, qr_square, angle, qr_area, rotated_img = extract(rotated_img, True)
    # mask_area(qr_square, angle)
    # print("qr_square")
    # print(qr_square)
    cv2.imwrite("frame_masked_frame.jpg", frame)
    pos = [qr_square[x][0][:] for x in range(len(qr_square))]
    # print(pos)
    mask = get_mask(pos)
    # print(mask)

    masked_image = region_of_interest(frame, mask)
    cv2.imwrite("frame_masked_image1.jpg", masked_image)
    
    bg_image, bg_squares = get_masked_image(masked_image, qr_area, True)
    bg_area = cv2.contourArea(bg_squares[0])
    cv2.imwrite("frame_bg_image.jpg", bg_image)
    
    _, _, rotate_strip =  get_strip_square(bg_image, bg_area)
    cv2.imwrite("frame_rotated_img.jpg", rotate_strip)
    rotate_strip = cv2.imread("frame_rotated_img.jpg")
    strip_img, strip_squares, rotate_strip =  get_strip_square(rotate_strip, bg_area)

    # print("area")
    # print(qr_area * 0.6)
    # print(cv2.contourArea(strip_squares[0]))
    # masked_image = region_of_interest(strip_img, strip_squares)
    cv2.imwrite("frame_strip_masked_image.jpg", strip_img)
    # cv2.drawContours(frame, strip_squares, -1, (5, 5, 5), 2)
    # cv2.imwrite("frame_strip.jpg", frame)
    # wb_strip = simple_white_balance(strip_img, strip_squares)
    # cv2.imwrite("frame_wb_strip.jpg", wb_strip)

    return strip_squares, strip_img, bg_squares, bg_image

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def get_litmus(img, strip_squares):
    '''
    for a given img frame, locate the litmus area.
    input: img frame
    output: litmus area squares, and the processed img contains only litmus
    '''
    # img = cv2.imread("frame_upper_crop_img.jpg", cv2.IMREAD_GRAYSCALE)
    # _, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    # contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print("squacontoursres")
    # print(len(contours))
    # for x in contours:
    #     print(x)
    #     print(cv2.contourArea(x))

    # cv2.drawContours(img, contours, -1, (128, 0, 0), 2)
    # cv2.imwrite("frame_strip_edge1d.jpg", img)
    # try:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("frame_strip_cvtColor.jpg", gray)
    # except:
    #     gray = img
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    cv2.imwrite("frame_strip_bilateralFilter.jpg", gray)
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    cv2.imwrite("frame_strip_GaussianBlur.jpg", gray)
    # edged = cv2.Canny(gray, 40, 80)
    # edged = auto_canny(gray)
    edged = cv2.Canny(gray, 10, 70)
    cv2.imwrite("frame_strip_edged.jpg", edged)
    area_threshold = cv2.contourArea(strip_squares[0])
    # print("area_threshold", area_threshold)
    # contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    squares = []
    upper = []
    lower = []
    # cv2.drawContours(img, contours, -1, (128, 0, 0), 2)
    # M = cv2.moments(strip_squares[0])
    # print(strip_squares)
    # cX = int(M["m10"] / M["m00"])
    # cY = int(M["m01"] / M["m00"])
    dimensions = img.shape
    # print("dimensions")
    # print(dimensions)
    # print("strip_squares")
    # print(len(strip_squares))
    # # print(cX, cY)
    # print("contours")
    # print(contours)
    # print(len(contours))
    # print(len(contours[0]))
    # print((contours[0][0]))
    arrY = [contours[y][x][0][1] for y in range(len(contours)) for x in range(len(contours[y]))]
    lower_boundY = min(arrY) + (max(arrY) - min(arrY)) * 0.2
    upper_boundY = min(arrY) + (max(arrY) - min(arrY)) * 0.8
    litums_arrX_right = [contours[y][x][0][0] for y in range(len(contours)) for x in range(len(contours[y])) if contours[y][x][0][0] > (dimensions[1] / 2) and contours[y][x][0][1] > lower_boundY and contours[y][x][0][1] < upper_boundY]
    litums_arrX_left = [contours[y][x][0][0] for y in range(len(contours)) for x in range(len(contours[y])) if contours[y][x][0][0] < (dimensions[1] / 2) and contours[y][x][0][1] > lower_boundY and contours[y][x][0][1] < upper_boundY]
    arrX_right = [contours[y][x][0][0] for y in range(len(contours)) for x in range(len(contours[y])) if contours[y][x][0][0] > (dimensions[1] / 2)]
    arrX_left = [contours[y][x][0][0] for y in range(len(contours)) for x in range(len(contours[y])) if contours[y][x][0][0] < (dimensions[1] / 2)]
    arrX = [contours[y][x][0][0] for y in range(len(contours)) for x in range(len(contours[y]))]
    
    left_boundX = min(arrX) + (max(arrX) - min(arrX)) * 0.2
    right_boundX = min(arrX) + (max(arrY) - min(arrX)) * 0.8
    litums_arrY_upper = [contours[y][x][0][1] for y in range(len(contours)) for x in range(len(contours[y])) if contours[y][x][0][1] > (dimensions[0] / 2) and contours[y][x][0][0] > left_boundX and contours[y][x][0][0] < right_boundX]
    litums_arrY_lower = [contours[y][x][0][1] for y in range(len(contours)) for x in range(len(contours[y])) if contours[y][x][0][1] < (dimensions[0] / 2) and contours[y][x][0][0] > left_boundX and contours[y][x][0][0] < right_boundX]
    arrY_upper = [contours[y][x][0][1] for y in range(len(contours)) for x in range(len(contours[y])) if contours[y][x][0][1] > (dimensions[0] / 2)]
    arrY_lower = [contours[y][x][0][1] for y in range(len(contours)) for x in range(len(contours[y])) if contours[y][x][0][1] < (dimensions[0] / 2)]
    
    
    # print(dimensions)
    # print(arrY)
    # print(arrX_right)
    # print(arrX_left)
    # print(arrY)
    print(litums_arrX_right)
    print(litums_arrX_left)
    print(img.shape)
    a = np.mean(arrX_right) - np.mean(arrX_left)
    b = min(arrX_right) - max(arrX_left)
    # print(arrX_right)
    # print(arrX_left)
    # print(a, b)
    if len(litums_arrY_upper) == 0:
        upper = img.shape[0]
    if len(litums_arrY_lower) == 0:
        lower = 0
    # if len(litums_arrY_lower) != 0 or len(litums_arrY_upper) != 0 or ((np.mean(arrY_upper) - np.mean(arrY_lower))) * 0.5 > (min(litums_arrY_upper) - max(litums_arrY_lower)):
    #     lower = (int)(np.mean(arrY_lower))
    #     upper = (int)(np.mean(arrY_upper))
    # elif ((np.mean(arrX_right) - np.mean(arrX_left))) * 0.5 > (min(arrX_right) - max(arrX_left)):
    #     crop_img = img[min(arrY):max(arrY), (int)(np.mean(arrX_left)):(int)(np.mean(arrX_right))]
    else:
        lower = (int)(max(litums_arrY_lower))  if len(litums_arrY_lower) != 0 else 0
        upper = (int)(min(litums_arrY_upper)) if len(litums_arrY_upper) != 0 else 0
    if upper < img.shape[0] * 0.7:
        upper = (int)(np.mean(arrY_upper))
    if lower > img.shape[0] * 0.3:
        lower = (int)(np.mean(arrY_lower))

    # if len(litums_arrX_right) == 0:
    #     right = img.shape[1]
    # if len(litums_arrX_left) == 0:
    #     left = 0
    if (len(litums_arrX_right) != 0 and len(litums_arrX_left) != 0) and ((np.mean(arrX_right) - np.mean(arrX_left))) * 0.7 > (min(litums_arrX_right) - max(litums_arrX_left)):
        left = (int)(np.mean(arrX_left))
        right = (int)(np.mean(arrX_right))
    # elif ((np.mean(arrX_right) - np.mean(arrX_left))) * 0.5 > (min(arrX_right) - max(arrX_left)):
    #     crop_img = img[min(arrY):max(arrY), (int)(np.mean(arrX_left)):(int)(np.mean(arrX_right))]
    else:
        left = (int)(max(litums_arrX_left)) if len(litums_arrX_left) != 0 else 0
        right = (int)(min(litums_arrX_right)) if len(litums_arrX_right) != 0 else img.shape[1]

    crop_img = img[lower:upper, left:right]
    # cv2.imwrite("frame_crop_imgimage.jpg", crop_img)
    
    # for c in contours:
    #     # Approximate the contour
    #     peri = cv2.arcLength(c, True)
    #     area = cv2.contourArea(c)
    #     approx = cv2.approxPolyDP(c, 0.1 * peri, True)
    #     # print(area)
    #     # print(peri, area, len(approx))
    #     # Find all quadrilateral contours
    #     if len(approx) >= 4:
    #         # Determine if quadrilateral is a square to within SQUARE_TOLERANCE
    #         # if area > 25 and 1 - SQUARE_TOLERANCE < math.fabs((peri / 4) ** 2) / area < 1 + SQUARE_TOLERANCE and count_children(hierarchy[0], i) >= 2 and has_square_parent(hierarchy[0], square_indices, i) is False:
    #         # if area > cv2.contourArea(strip_squares[0]) * 0.05:
    #         squares.append(c)

    # squares = sorted(squares, key=lambda x: cv2.contourArea(x))
    # cv2.drawContours(img, contours, -1, (128, 0, 0), 2)
    # cv2.imwrite("frame_squares.jpg", img)   

    # print("squares")
    # print(len(squares))
    # for x in squares:
    #     print(x)
    #     print(cv2.contourArea(x))

    # # cv2.drawContours(img, [squares[0]], -1, (128, 0, 0), 2)
    # # cv2.imwrite("frame_contours.jpg", img)

    # final = []
    # # for c in squares:
    # for square in squares:

    #     area = cv2.contourArea(square)
    #     center = get_center(square)
    #     peri = cv2.arcLength(square, True)
    #     similar = []

    #     for other in squares:
    #         if square[0][0][0] != other[0][0][0] or square[0][0][1] != other[0][0][1]:

    #             # Determine if square is similar to other square within AREA_TOLERANCE
    #             if math.fabs(area - cv2.contourArea(other)) / max(area, cv2.contourArea(other)) <= LITMUS_AREA_TOLERANCE:
    #                 final = [square, other]

    #     # if len(similar) >= 1:
    #     #     final = [square, other]

    # cv2.drawContours(img, squares, -1, (128, 0, 0), 2)
    # cv2.imwrite("frame_contours1.jpg", img)
    
    # final = sorted(final, key=lambda x: cv2.contourArea(x))
    # litmus_image = region_of_interest(img, [contours[0]])
    # # litmus_image, litmus_squares = get_masked_image(img, squares, True)
    # # litmus_area = cv2.contourArea(litmus_squares[0])
    # # cv2.imwrite("frame_bg_image.jpg", bg_image)
    # # print(contours[0])
    return crop_img, [contours[0]]



def filter_white_colors(image):
	# """
	# Filter the image to exclude white pixels
	# """
    # # Filter white pixels
    white_upper_threshold = 225
    white_lower_threshold = 10 
    lower_white = np.array([white_lower_threshold, white_lower_threshold, white_lower_threshold]) 
    upper_white = np.array([white_upper_threshold, white_upper_threshold, white_upper_threshold])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)

    return white_image

def get_upper_lower_litums(frame):
    simple_wb_out = white_balance(frame)
    cv2.imwrite("simple_wb_out.jpg", simple_wb_out)
    # try:
    strip_squares, strip_img, bg_squares, bg_image = get_backgroud_square(simple_wb_out)
    # except:
        # print("backgroud not detected")
    wb_strip = simple_white_balance(strip_img, strip_squares)
    cv2.imwrite("wb_strip.jpg", wb_strip)
    try:
        upper_crop_img, lower_crop_img, crop_img = get_crop_strip(strip_squares, wb_strip)
        lower_crop_litmus_img, litmus_squares = get_litmus(lower_crop_img, strip_squares)
        upper_crop_litmus_img, litmus_squares = get_litmus(upper_crop_img, strip_squares)
    except:
        wb_strip = rotate_img(wb_strip, get_center(strip_squares[0]), 180)
        upper_crop_img, lower_crop_img, crop_img = get_crop_strip(strip_squares, wb_strip)
        lower_crop_litmus_img, litmus_squares = get_litmus(lower_crop_img, strip_squares)
        upper_crop_litmus_img, litmus_squares = get_litmus(upper_crop_img, strip_squares)
        # print("croped strip not detected")
    # try:
    
    # except:
        # print("lower_crop_litmus_img not detected")
    # try:
    
    # except:
        # print("upper_crop_litmus_img not detected")
    # cv2.imwrite("lower_crop_litmus_img.jpg", lower_crop_litmus_img)
    # cv2.imwrite("upper_crop_litmus_img.jpg", upper_crop_litmus_img)
    return lower_crop_litmus_img, upper_crop_litmus_img

# strip_squares, strip_img, bg_squares, bg_image = get_backgroud_square(frame)
# print("strip_img")
# for x in strip_squares:
#     print(x)
#     print(cv2.contourArea(x))

# wb_strip = simple_white_balance(strip_img, strip_squares)
# cv2.imwrite("frame_wb_strip.jpg", wb_strip)

# print("strip_squares")
# print(cv2.contourArea(strip_squares[0]))
# # filtered_wb_strip = filter_white_colors(wb_strip)
# # cv2.imwrite("frame_filtered_wb_strip.jpg", filtered_wb_strip)

# # crop_img = wb_strip[796:803, 725:765]
# # cv2.imwrite("frame_crop_img.jpg", crop_img)

def get_crop_strip(strip_squares, img):
    # minX, minY, maxX, maxY = 
    # print("strip_\n")
    # print(len(np.shape(strip_squares)))
    # print(strip_squares)
    scale = [0.010, 0.075, 0.175, 0.238]
    
    try:
        arrX = [strip_squares[y][x][j][0] for y in range(len(strip_squares)) for x in range(len(strip_squares[y])) for j in range(len(strip_squares[y][x]))]
        arrY = [strip_squares[y][x][j][1] for y in range(len(strip_squares)) for x in range(len(strip_squares[y])) for j in range(len(strip_squares[y][x]))]
    except:
        arrX = [strip_squares[y][x][0] for y in range(len(strip_squares)) for x in range(len(strip_squares[y]))]
        arrY = [strip_squares[y][x][1] for y in range(len(strip_squares)) for x in range(len(strip_squares[y]))]

    # print(min(arrX), max(arrX))
    # print("/n")
    # print(min(arrY), max(arrY))

    # crop_img = img[min(arrY):max(arrY), min(arrX) : (int)(min(arrX) + (max(arrX) - min(arrX)) * 0.1)]
    if (max(arrY) - min(arrY)) > (max(arrX) - min(arrX)):
        bound = [(int)(min(arrY) + (max(arrY) - min(arrY)) * x) for x in scale]
        # upper_bound1 = (int)(min(arrY) + (max(arrY) - min(arrY)) * scale[0])
        # upper_bound2 = (int)(min(arrY) + (max(arrY) - min(arrY)) * scale[1])
        # lower_bound1 = (int)(min(arrY) + (max(arrY) - min(arrY)) * scale[2])
        # lower_bound2 = (int)(min(arrY) + (max(arrY) - min(arrY)) * scale[3])
        upper_arrX = [arrX[i] for i in range(len(arrX)) if arrY[i] > bound[0] and arrY[i] < bound[1]]
        lower_arrX = [arrX[i] for i in range(len(arrX)) if arrY[i] > bound[2] and arrY[i] < bound[3]]
        if (max(upper_arrX) - min(upper_arrX)) < 0.6 *  (max(arrX) - min(arrX)):
            upper_crop_img = img[bound[0] + 1:bound[1] - 2, min(arrX):max(arrX)]
        else :
            upper_crop_img = img[bound[0] :bound[1] - 2, min(upper_arrX):max(upper_arrX)]

        if (max(lower_arrX) - min(lower_arrX)) < 0.6 *  (max(arrX) - min(arrX)):
            lower_crop_img = img[bound[2] :bound[3] - 2, min(arrX):max(arrX)]
        else :
            lower_crop_img = img[bound[2] :bound[3] - 2, min(lower_arrX):max(lower_arrX)]

        # try:
        #     upper_crop_img = img[bound[0] :bound[1] - 2, min(upper_arrX):max(upper_arrX)]
        #     lower_crop_img = img[bound[2] :bound[3] - 2, min(lower_arrX):max(lower_arrX)]
        # except:
        #     upper_crop_img = img[bound[0] + 1:bound[1] - 2, min(arrX):max(arrX)]
        #     lower_crop_img = img[bound[2] :bound[3] - 2, min(arrX):max(arrX)]
    else:
        bound = [(int)(min(arrX) + (max(arrX) - min(arrX))) * x for x in scale]
        # upper_bound1 = (int)(min(arrY) + (max(arrY) - min(arrY)) * 0.006)
        # upper_bound2 = (int)(min(arrY) + (max(arrY) - min(arrY)) * 0.075)
        # lower_bound1 = (int)(min(arrY) + (max(arrY) - min(arrY)) * 0.175)
        # lower_bound2 = (int)(min(arrY) + (max(arrY) - min(arrY)) * 0.238)
        upper_arrY = [arrY[i] for i in range(len(arrY)) if arrX[i] > bound[0] and arrX[i] < bound[1]]
        lower_arrY = [arrY[i] for i in range(len(arrY)) if arrX[i] > bound[2] and arrX[i] < bound[3]]
        if (max(upper_arrY) - min(upper_arrY)) < 0.6 *  (max(arrY) - min(arrY)):
            upper_crop_img = img[min(arrY):max(arrY), bound[0] :bound[1] - 2]
        else :
            upper_crop_img = img[min(upper_arrY):max(upper_arrY), bound[0] :bound[1] - 2]

        if (max(lower_arrY) - min(lower_arrY)) < 0.6 *  (max(arrY) - min(arrY)):
            lower_crop_img = img[min(arrY):max(arrY), bound[2] :bound[3] - 2]
        else :
            lower_crop_img = img[min(lower_arrY):max(lower_arrY), bound[2] :bound[3] - 2]


        # try:
        #     upper_crop_img = img[min(upper_arrY):max(upper_arrY), bound[0] :bound[1] - 2]
        #     lower_crop_img = img[min(lower_arrX):max(lower_arrX), bound[2] :bound[3] - 2]
        # except:
        #     upper_crop_img = img[min(arrY):max(arrY), bound[0] :bound[1] - 2]
        #     lower_crop_img = img[min(arrY):max(arrY), bound[2] :bound[3] - 2]

    crop_img = img[min(arrY):max(arrY), min(arrX):max(arrX)]
    cv2.imwrite("frame_upper_crop_img.jpg", upper_crop_img)
    cv2.imwrite("frame_lower_crop_img.jpg", lower_crop_img)
    cv2.imwrite("frame_crop_img.jpg", crop_img)
    return upper_crop_img, lower_crop_img, crop_img

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

# # def main(frame):
# frame = cv2.imread('0.530.JPG')
# # simple_wb_out = white_balance(frame)
# lower_crop_litmus_img, upper_crop_litmus_img = get_upper_lower_litums(frame)
# cv2.imwrite("lower_crop_litmus_img.jpg", lower_crop_litmus_img)
# cv2.imwrite("upper_crop_litmus_img.jpg", upper_crop_litmus_img)
# # #     # print(len(strip_squares[0]))
#     # print(strip_squares[0][0][0])
#     # print(strip_squares[0][1][0])
#     # print(strip_squares[0][:][:])

# # def getMinMax(squares):
# #     minX = 
# #     for x in squares
# upper_crop_img, lower_crop_img, crop_img = get_crop_strip(strip_squares, wb_strip)

# gray = cv2.cvtColor(bg_image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("frame_gray.jpg", gray)

# # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# gray = cv2.bilateralFilter(gray, 11, 17, 17)
# # cv2.imwrite("frame_gray_bilateralFilter.jpg", gray)
# # gray = cv2.GaussianBlur(gray, (1, 1), 0)
# # cv2.imwrite("frame_gray_GaussianBlur.jpg", gray)

# # cv2.imwrite("frame_gray_edged.jpg", edged)

# # contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
# edged = cv2.Canny(th2, 80, 200)
# cv2.imwrite("frame_th.jpg", edged)
# # debug = False
# # img_contour, contours = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # shape = "undefined"

# # contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # cv2.drawContours(th2, contours, -1, (128, 128, 0), 2)
# # cv2.imwrite("frame_th1.jpg", th2)

# # for contour in contours:
# #     # peri = cv2.arcLength(c, True)
# #     print(contour)
# #     peri = cv2.arcLength(contour, True)
# #     area = cv2.contourArea(contour)
# #     # if (area < cv2.contourArea(strip_squares[0]) * 0.01):
# #         # continue
# #     approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
# #     # epsilon = 0.03 * cv2.arcLength(contour, True)
# #     # approx = cv2.approxPolyDP(contour, epsilon, True)
# #     x, y, w, h = cv2.boundingRect(contour)
# #     cv2.rectangle(th2, (x, y), (x + w, y + h), (0, 0, 255), 2)
# #     cv2.rectangle(th2, (x, y-10), (x + w, y + 10), (0, 0, 255), -1)
# #     font = cv2.FONT_HERSHEY_SIMPLEX
# #     number = ""
# #     if debug:
# #         cv2.drawContours(th2, [contour], 0, (0, 255, 0), 2)

# #         for pt in approx:
# #             cv2.circle(th2, (pt[0][0], pt[0][1]), 5, (255, 0, 0), -1)
# #         number = str(len(approx)) + " "

# #     if len(approx) == 3:
# #         shape = "triangle"
# #     elif len(approx) == 4:
# #         # print(w, h, w / h)
# #         if 0.95 < w / h < 1.05:
# #             shape = "Square"
# #         else:
# #             shape = "Rectangle"
# #     elif len(approx) == 5:
# #         shape = "Pentagon"
# #     else:
# #         shape = "Circle"
# #     cv2.putText(th2, number + shape, (x, y), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
# # cv2.imwrite("frame_th2.jpg", th2)

# img = bg_image
# gray = bg_image
# edges = cv2.Canny(gray,50,150,apertureSize = 3)
# minLineLength = 100
# maxLineGap = 10
# lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
# for x1,y1,x2,y2 in lines[0]:
#     cv2.line(bg_image,(x1,y1),(x2,y2),(0,255,0),2)

# cv2.imwrite('houghlines5.jpg',frame)

# # Apply edge detection method on the image 
# edges = cv2.Canny(bg_image,50,150,apertureSize = 3) 
# cv2.imwrite("frame_th2_edges.jpg", edges)

# # This returns an array of r and theta values 
# lines = cv2.HoughLines(edges,0.5,np.pi/180, 10) 
  
# # The below for loop runs till r and theta values  
# # are in the range of the 2d array 
# for r,theta in lines[0]: 
      
#     # Stores the value of cos(theta) in a 
#     a = np.cos(theta) 
  
#     # Stores the value of sin(theta) in b 
#     b = np.sin(theta) 
      
#     # x0 stores the value rcos(theta) 
#     x0 = a*r 
      
#     # y0 stores the value rsin(theta) 
#     y0 = b*r 
      
#     # x1 stores the rounded off value of (rcos(theta)-1000sin(theta)) 
#     x1 = int(x0 + 1000*(-b)) 
      
#     # y1 stores the rounded off value of (rsin(theta)+1000cos(theta)) 
#     y1 = int(y0 + 1000*(a)) 
  
#     # x2 stores the rounded off value of (rcos(theta)+1000sin(theta)) 
#     x2 = int(x0 - 1000*(-b)) 
      
#     # y2 stores the rounded off value of (rsin(theta)-1000cos(theta)) 
#     y2 = int(y0 - 1000*(a)) 
      
#     # cv2.line draws a line in img from the point(x1,y1) to (x2,y2). 
#     # (0,0,255) denotes the colour of the line to be  
#     #drawn. In this case, it is red.  
#     cv2.line(bg_image,(x1,y1), (x2,y2), (128,128 ,0),2) 
      
# # All the changes made in the input image are finally 
# # written on a new image houghlines.jpg 
# cv2.imwrite('linesDetected.jpg', bg_image)

# # gray = cv2.cvtColor(th3, cv2.COLOR_BGR2GRAY)
# # cv2.imwrite("frame_strip_cvtColor.jpg", gray)
# # gray = cv2.bilateralFilter(th3, 11, 17, 17)
# # gray = cv2.GaussianBlur(gray, (5, 5), 0)
# # cv2.imwrite("frame_strip_GaussianBlur.jpg", gray)
# # edged = cv2.Canny(gray, 10, 200)
# # cv2.imwrite("frame_strip_edged.jpg", edged)

# # contours, hierarchy = cv2.findContours(th3.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# # # contours, _ = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # font = cv2.FONT_HERSHEY_COMPLEX

# # cv2.drawContours(wb_strip, contours, -1, (128, 0, 0), 2)

# # for cnt in contours:
# #     approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
# #     if (cv2.contourArea(cnt) < cv2.contourArea(strip_squares[0]) * 0.1) and len(approx) != 4:
# #         # contours.Remove(cnt)
# #         continue
# #     cv2.drawContours(th3, [approx], 0, (0), 5)
# #     x = approx.ravel()[0]
# #     y = approx.ravel()[1]
#     # if len(approx) == 3:
#     #     cv2.putText(th3, "Triangle", (x, y), font, 1, (0))
#     # if len(approx) == 4:
#     #     cv2.putText(th3, "Rectangle", (x, y), font, 1, (0))
#     # elif len(approx) == 5:
#     #     cv2.putText(th3, "Pentagon", (x, y), font, 1, (0))
#     # elif 6 < len(approx) < 15:
#     #     cv2.putText(th3, "Ellipse", (x, y), font, 1, (0))
#     # else:
#     #     cv2.putText(th3, "Circle", (x, y), font, 1, (0))

# upper_threshold = filter_white_colors(lower_crop_img)
# cv2.imwrite("frame_upper_threshold.jpg", wb_strip)
# lower_crop_litmus_img, litmus_squares = get_litmus(lower_crop_img, strip_squares)
# upper_crop_litmus_img, litmus_squares = get_litmus(upper_crop_img, strip_squares)

# cv2.imwrite("lower_crop_litmus_img.jpg", lower_crop_litmus_img)
# cv2.imwrite("upper_crop_litmus_img.jpg", upper_crop_litmus_img)


# class FilledShape:
#     def __init__(self, img):
#         self.img = img

#     def detect(self, contour, debug):
#         shape = "undefined"
#         epsilon = 0.03 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#         cv2.rectangle(self.img, (x, y-10), (x + w, y + 10), (0, 0, 255), -1)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         number = ""
#         if debug:
#             cv2.drawContours(self.img, [contour], 0, (0, 255, 0), 2)

#             for pt in approx:
#                 cv2.circle(self.img, (pt[0][0], pt[0][1]), 5, (255, 0, 0), -1)
#             number = str(len(approx)) + " "

#         if len(approx) == 3:
#             shape = "triangle"
#         elif len(approx) == 4:
#             # print(w, h, w / h)
#             if 0.95 < w / h < 1.05:
#                 shape = "Square"
#             else:
#                 shape = "Rectangle"
#         elif len(approx) == 5:
#             shape = "Pentagon"
#         else:
#             shape = "Circle"
#         cv2.putText(self.img, number + shape, (x, y), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

#     def preprocessing_image(self):
#         # img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
#         # _, threshold = cv2.threshold(img_gray, 127, 255, 0)
#         # kernel = np.ones((5, 5), np.uint8)
#         # cv2.dilate(threshold, kernel, iterations=1)
#         # threshold = cv2.GaussianBlur(threshold, (15, 15), 0)
#         img_contour, contours, = cv2.findContours(self.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         return threshold, contours

# def capture(frame, debug=False):
#     img_object = FilledShape(frame)
#     threshold, contours = img_object.preprocessing_image()
#     for contour in contours:
#         img_object.detect(contour, debug)
#     cv2.imshow('Threshold', threshold)
#     cv2.imshow('Original', frame)

# # capture(th2)
# #%%
