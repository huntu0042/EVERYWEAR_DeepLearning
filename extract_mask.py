import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
  
def main():
  data_path = "./cloth/*.*"
  result_path = "./cloth-mask/"
  data_list = glob(data_path)
  for data_dir in data_list:
    cloth_image = cv2.imread(data_dir, 0)
    cloth_name = data_dir[-12:-4]
    print(cloth_name)

    # 노이즈 제거를 위한

     # 히스토그램 평활화
    equalize = cv2.equalizeHist(cloth_image)
    
    gaussian = cv2.GaussianBlur(equalize, (11, 11), 0)
  
    """
    gaussian = cv2.adaptiveThreshold(cloth_image, 255, 
               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    """
    otsu_thr, otsu_mask = cv2.threshold(gaussian, -1, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, contours, hierarchy = cv2.findContours(otsu_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    image_external = np.zeros(cloth_image.shape, cloth_image.dtype)
    for i in range(len(contours)):
      if hierarchy[0][i][3] == -1:
        cv2.drawContours(image_external, contours, i, 255, -1)
    plt.figure()
    plt.subplot(151)
    plt.axis('off')
    plt.title('original')
    plt.imshow(cloth_image, cmap='gray')

    plt.subplot(152)
    plt.axis('off')
    plt.title('equalize')
    plt.imshow(equalize, cmap='gray')

    plt.subplot(153)
    plt.axis('off')
    plt.title('gaussian')
    plt.imshow(gaussian, cmap='gray')

    plt.subplot(154)
    plt.axis('off')
    plt.title('otsu_mask')
    plt.imshow(otsu_mask, cmap='gray')

    plt.subplot(155)
    plt.axis('off')
    plt.title('image_external')
    plt.imshow(image_external, cmap='gray')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
  main()