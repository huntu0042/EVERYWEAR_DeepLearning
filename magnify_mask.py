import cv2
import numpy as np
from glob import glob

def check_up_mask(mask, c_up=0, switch_up=False):
  height, width = mask.shape
  for i in range(0, height, 1):
    c_up += 1
    for j in range(0, width, 1):
      if not(mask[i][j] == 0 and
             mask[i][j] == 0 and
             mask[i][j] == 0):
        switch_up = True
        break
    if(switch_up):
      break
  only_mask_image = mask[c_up-1:][:]
  magnify_only_mask_image = magnify_mask(only_mask_image)
  magnify_only_mask_image = magnified_mask_trim_down(magnify_only_mask_image)

  empty_place = np.zeros((c_up, width))
  print("empty_place: " + str(empty_place.shape))
  print("mask: " + str(magnify_only_mask_image.shape))

  update_mask = np.vstack([empty_place, magnify_only_mask_image])
  return update_mask


def magnified_mask_trim_down(mask, magnified_size=12):
  devided_size = int(magnified_size / 2)

  trimmed_h_mask = mask[:-(magnified_size+1)][:][:]
  transpose_image = np.transpose(trimmed_h_mask, (1, 0))
  trimmed_w_mask = transpose_image[devided_size-1:-(devided_size+1)][:]
  update_mask = np.transpose(trimmed_w_mask, (1, 0))
  return update_mask

def magnify_mask(mask, magnified_size=12):
  mask_h, mask_w = mask.shape
  update_mask_h = mask_h + magnified_size
  update_mask_w = mask_w + magnified_size

  update_mask = cv2.resize(mask, (update_mask_w, update_mask_h))

  return update_mask

"""
if __name__ == "__main__":
  image_list = glob('./trim_example/*.*')
  result_dir = './trim_example/'
  for image_dir in image_list:
    image_name = image_dir[-12:]
    cloth = cv2.imread(image_dir)
    new_cloth = check_up_black(cloth)
    
    cv2.imshow('img', new_cloth)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite(result_dir + '1.png', new_cloth)
"""