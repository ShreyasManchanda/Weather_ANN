import os
import cv2
import numpy as np
import shutil

# Source and destination base folders
classes = ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']
src_base = "C:/Users/Shreyas/Desktop/Weather_ANN/ogdataset"
dst_base = "C:/Users/Shreyas/Desktop/Weather_ANN/ogdataset_filtered"

# Create destination folders
for cls in classes:
    os.makedirs(os.path.join(dst_base, cls), exist_ok=True)

def apply_combined_filters(img):
    # Histogram Equalization on Y channel
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # Gaussian Blur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Sharpening
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    img = cv2.filter2D(img, -1, sharpen_kernel)

    return img

# Process all images
for cls in classes:
    src_folder = os.path.join(src_base, cls)
    dst_folder = os.path.join(dst_base, cls)

    for filename in os.listdir(src_folder):
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)

        img = cv2.imread(src_path)
        if img is not None:
            filtered_img = apply_combined_filters(img)
            cv2.imwrite(dst_path, filtered_img)
        else:
            print(f"Failed to read: {src_path}")
