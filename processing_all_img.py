import cv2
import os
import litums_detection as ld

def get_all_img():
    total_img = 0
    error = 0

    # Get the list of all files in directory tree at given path
    dirName1 = "../imgs_for_jason/results/crop fail"

    print("dirName1", dirName1)
    low_conc_rgb = {}
    upper_conc_rgb = {}
    for (dirpath, dirnames, filenames) in os.walk(dirName1):
        # if target_dir1 in dirnames:
        print("-------------------")
        print("dirpath")
        print(dirpath)
        print(dirnames)
        print(filenames)

        for fn in filenames:
            total_img += 1
            if fn.find("_crop_litmus_img.jpg") > 0 or fn.find("_freer.jpg") > 0 or fn.find("_total.jpg") > 0:
                continue
            file_Path = os.path.join(dirpath, fn)
            print(file_Path)
            base = os.path.basename(file_Path)
            conc = os.path.splitext(base)[0]
            print("conc")
            print(conc)
            try:
                frame = cv2.imread(file_Path)
                free_circle_cropped_img, free_rgb, total_circle_cropped_img, total_rgb = ld.processing_img(frame)
                cv2.imwrite(os.path.splitext(file_Path)[0] + "_free.jpg", free_circle_cropped_img)
                cv2.imwrite(os.path.splitext(file_Path)[0] + "_total.jpg", total_circle_cropped_img)
            except:
                error += 1
                print("-------------------")
                print(fn)
    return total_img, error, low_conc_rgb, upper_conc_rgb

total_img, error, low_conc_rgb, upper_conc_rgb = get_all_img()
