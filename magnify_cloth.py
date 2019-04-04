import cv2
import numpy as np
from glob import glob


def check_up_fg(cloth, magnified_size, c_up=0, switch_up=False):
  height, width, _ = cloth.shape
  for i in range(0, height, 1):
    c_up += 1
    for j in range(0, width, 1):
      if not(cloth[i][j][0] < 100 and 
             cloth[1][j][1] < 100 and 
             cloth[i][j][2] < 100):
        switch_up = True
        break
    if(switch_up):
      break
  only_cloth_image = cloth[c_up-1:][:][:]
  magnify_only_cloth_image = magnify_cloth(only_cloth_image, magnified_size)
  
  
  magnify_only_cloth_image = magnified_cloth_trim_down(magnify_only_cloth_image, magnified_size)
  
  """
  print(c_up)
  print(trimmed_magnify_only_cloth_image.shape)
  """
  empty_place = np.zeros((c_up, width, 3), dtype=np.uint8)
  
  
  print("empty_place: " + str(empty_place.shape))
  print("remain: " + str(magnify_only_cloth_image.shape))
  
  update_cloth = np.vstack([empty_place, magnify_only_cloth_image])
  
  print(update_cloth.shape)
  
  return update_cloth
"""
def check_up(cloth, c_up=0, switch_up=False):
  height, width, _ = cloth.shape
  for i in range(0, height, 2):
    c_up += 2
    for j in range(0, width, 2):
      if not(cloth[i][j][0] > 240 and 
             cloth[1][j][1] > 240 and 
             cloth[i][j][2] > 240):
        switch_up = True
        break
    if(switch_up):
      break
  only_cloth_image = cloth[c_up-1:][:][:]
  magnify_only_cloth_image = magnify_cloth(only_cloth_image)
  trimmed_magnify_only_cloth_image = magnified_cloth_trim_down(magnify_only_cloth_image)


  empty_place = np.zeros((c_up, width, 3))
  for i in range(0, empty_place.shape[0]):
    for j in range(0, empty_place.shape[1]):
      empty_place[i][j][0] = 255
      empty_place[i][j][1] = 255
      empty_place[i][j][2] = 255

  
  update_cloth = np.vstack([empty_place, trimmed_magnify_only_cloth_image])
  print(update_cloth.shape)
  return update_cloth
"""
def magnified_cloth_trim_down(cloth, magnified_size):
  devided_size = int(magnified_size / 2)

  trimmed_h_cloth = cloth[:-(magnified_size+1)][:][:]
  transpose_image = np.transpose(trimmed_h_cloth, (1, 0, 2))
  trimmed_w_cloth = transpose_image[devided_size-1:-(devided_size+1)][:][:]
  update_cloth = np.transpose(trimmed_w_cloth, (1, 0, 2))
  return update_cloth

def magnify_cloth(cloth, magnified_size):
  cloth_h, cloth_w, _ = cloth.shape
  update_cloth_h = cloth_h + magnified_size
  update_cloth_w = cloth_w + magnified_size

  update_cloth = cv2.resize(cloth, (update_cloth_w, update_cloth_h), 
                            interpolation=cv2.INTER_LINEAR_EXACT)

  return update_cloth

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