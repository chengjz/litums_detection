
import cv2
import litums_detection as ld

frame = cv2.imread('test.jpg')

try:
    free_circle_cropped_img, free_rgb, total_circle_cropped_img, total_rgb = ld.processing_img(frame)
    cv2.imwrite("free_circle_cropped_img.jpg", free_circle_cropped_img)
    cv2.imwrite("total_circle_cropped_img.jpg", total_circle_cropped_img)
    print("total_rgb", total_rgb)
    print("free_rgb", free_rgb)
except:
    pass

