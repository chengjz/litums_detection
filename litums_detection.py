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
import line_detection

BLUR_VALUE = 3
SQUARE_TOLERANCE = .15
AREA_TOLERANCE = 0.2
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
        point = point[0]
        d = math.hypot(point[0] - center[0], point[1] - center[1])
        distances.append(d)
        distances_to_points[d] = point
    distances = sorted(distances)
    return [distances_to_points[distances[-1]], distances_to_points[distances[-2]]]


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
                angle_a = get_angle(get_center(distances_to_contours[closest_a]), center)
                angle_b = get_angle(get_center(distances_to_contours[closest_b]), center)
                angle = angle_b
                # print('angle a ', angle_a, ' angle b ', angle_b)
                if angle_a < angle_b or (angle_b < -90 and angle_a > 0):
                    east = distances_to_contours[closest_a]
                    south = distances_to_contours[closest_b]
                else:
                    east = distances_to_contours[closest_b]
                    south = distances_to_contours[closest_a]
                midpoint = get_midpoint(get_center(east), get_center(south))
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
                qr_area = area

    codes = []
    i = 0
    if debug:
        # Draw debug information onto frame before ting it
        cv2.drawContours(output, squares, -1, (5, 5, 5), 2)
        cv2.drawContours(output, main_corners, -1, (0, 0, 128), 2)
        cv2.drawContours(output, east_corners, -1, (0, 128, 0), 2)
        cv2.drawContours(output, south_corners, -1, (128, 0, 0), 2)
        cv2.drawContours(output, tiny_squares, -1, (128, 128, 0), 2)

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
       
    qr_square = np.array([main_corners[0][0][:], east_corners[0][3][:], [np.asarray(rectangles[0][2][:], dtype=np.int32)], south_corners[0][1][:]])
    return codes, output, qr_square, angle, qr_area

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
    y1 = extend_scale(x4, get_midpoint(y2, y4), 1)
    y4 = extend_scale(y1, y4, 0.05)
    y3 = extend_scale(y2, y3, 0.05)
    # print(y1, y2, y3, y4)
    return np.array([[y1, y2, y3, y4]], dtype=np.int32)

def extend_scale(a, b, scale):
    length_ab = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    result = [b[0] + (b[0] - a[0]) * scale, b[1] + (b[1] - a[1]) * scale]
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
	line_img = np.zeros((*img.shape, 3), dtype=np.uint8)  # 3-channel RGB image
	cv2.drawContours(line_img, lines, -1, (128, 0, 0), 2)
    
	return line_img

def get_contours(image, area_threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.GaussianBlur(gray, (BLUR_VALUE, BLUR_VALUE), 0)
    edged = cv2.Canny(gray, 1, 200)

    
    # contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    big_squares = []
    big_square_indices = []
    small_squares = []
    small_square_indices = []

    i = 0
    for c in contours:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)

        # Find all quadrilateral contours
        if len(approx) >= 4:
            if area > area_threshold:
                big_squares.append(approx)
                big_square_indices.append(i)

            if area <= area_threshold:
                small_squares.append(approx)
                small_square_indices.append(i)
        i += 1

    big_cntsSorted = sorted(big_squares, key=lambda x: cv2.contourArea(x))
    small_cntsSorted = sorted(small_squares, key=lambda x: cv2.contourArea(x))
    return big_cntsSorted,small_cntsSorted, edged

def get_masked_image(image, area_threshold, flag):
    
    big_cntsSorted, small_cntsSorted, edged = get_contours(image, area_threshold)
    if flag is True:
        squares = [big_cntsSorted[0]]
    else:
        squares = [small_cntsSorted[-1]]

    masked_image = region_of_interest(image, squares)
    cv2.drawContours(image, squares, -1, (0, 0, 0), 2)
    return masked_image, squares

def get_strip_square(img, bg_area):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # gray = cv2.GaussianBlur(gray, (BLUR_VALUE, BLUR_VALUE), 0)
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    edged = cv2.Canny(gray, 1, 200)
    # cv2.imwrite("frame_strip_edged.jpg", edged)

    
    # contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (128, 0, 0), 2)
    # cv2.imwrite("frame_contours.jpg", img)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    tot_squares = []
    square_indices = []

    squares = []
    for c in contours:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if area < bg_area * 0.9:
            squares.append(c)

    # print("squares")
    squares = sorted(squares, key=lambda x: cv2.contourArea(x))
    # for x in squares:
        # print(x)
        # print(cv2.contourArea(x))
    # strip_image, strip_squares = get_masked_image(bg_image, bg_area * 0.9, False)
    cv2.drawContours(img, [squares[-1]], -1, (128, 0, 0), 2)
    masked_image = region_of_interest(img, [squares[-1]])
    # cv2.imwrite("frame_strip_image.jpg", masked_image)
    return masked_image, [squares[-1]]


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
    codes, frame, qr_square, angle, qr_area = extract(frame, True)
    # mask_area(qr_square, angle)
    # print(qr_square)
    pos = [qr_square[x][0][:] for x in range(len(qr_square))]
    # print(pos)
    mask = get_mask(pos)
    # print(mask)

    masked_image = region_of_interest(frame, mask)
    cv2.imwrite("frame_masked_image.jpg", masked_image)
    
    bg_image, bg_squares = get_masked_image(masked_image, qr_area, True)
    bg_area = cv2.contourArea(bg_squares[0])
    cv2.imwrite("frame_bg_image.jpg", bg_image)
    
    strip_img, strip_squares =  get_strip_square(bg_image, bg_area)
    # masked_image = region_of_interest(strip_img, strip_squares)
    cv2.imwrite("frame_strip_masked_image.jpg", strip_img)
    # cv2.drawContours(frame, strip_squares, -1, (5, 5, 5), 2)
    # cv2.imwrite("frame_strip.jpg", frame)
    wb_strip = simple_white_balance(strip_img, strip_squares)
    cv2.imwrite("frame_wb_strip.jpg", wb_strip)

    return strip_squares, wb_strip

def get_litmus(img, strip_squares):
    '''
    for a given img frame, locate the litmus area.
    input: img frame
    output: litmus area squares, and the processed img contains only litmus
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("frame_strip_cvtColor.jpg", gray)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    cv2.imwrite("frame_strip_bilateralFilter.jpg", gray)
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    cv2.imwrite("frame_strip_GaussianBlur.jpg", gray)
    edged = cv2.Canny(gray, 1, 200)
    cv2.imwrite("frame_strip_edged.jpg", edged)
    area_threshold = cv2.contourArea(strip_squares[0])
    # print("area_threshold", area_threshold)
    # contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (128, 0, 0), 2)
    cv2.imwrite("frame_contours.jpg", img)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    squares = []
    for c in contours:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        # print(area)

        # Find all quadrilateral contours
        if len(approx) == 4:
            # Determine if quadrilateral is a square to within SQUARE_TOLERANCE
            # if area > 25 and 1 - SQUARE_TOLERANCE < math.fabs((peri / 4) ** 2) / area < 1 + SQUARE_TOLERANCE and count_children(hierarchy[0], i) >= 2 and has_square_parent(hierarchy[0], square_indices, i) is False:
            if area > cv2.contourArea(strip_squares[0]) * 0.05:
                squares.append(c)
    cv2.drawContours(img, squares, -1, (128, 0, 0), 2)
    # cv2.imwrite("frame_contours.jpg", img)
    litmus_image, litmus_squares = get_masked_image(img, squares, True)
    litmus_area = cv2.contourArea(litmus_squares[0])
    # cv2.imwrite("frame_bg_image.jpg", bg_image)
    return litmus_image, litmus_squares


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

frame = cv2.imread('98.JPG')
strip_squares, wb_strip = get_backgroud_square(frame)
filtered_wb_strip = filter_white_colors(wb_strip)
cv2.imwrite("frame_filtered_wb_strip.jpg", filtered_wb_strip)
litmus_img, litmus_squares = get_litmus(filtered_wb_strip, strip_squares)
cv2.imwrite("frame_litmus.jpg", litmus_img)
#%%
