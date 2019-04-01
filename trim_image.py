import cv2
import numpy as np
def check_up(clothes, c_up=0, switch_up=False):
  height, width, _ = clothes.shape
  remain_height = height // 10
  remain_width = width // 10
  for i in range(0, height, 2):
    c_up +=2
    for j in range(0, width, 2):
      if not(clothes[i][j][0] > 240 and 
             clothes[1][j][1] > 240 and 
             clothes[i][j][2] > 240):
        switch_up = True
        break
    if(switch_up):
      break
  print(remain_height)
  c_up_clothes = clothes[c_up - (remain_height - 1):][:][:]
  cropped_height = c_up - remain_height
  return cropped_height, c_up_clothes

def check_down(clothes, c_down=0, switch_down=False):
  height, width, _ = clothes.shape
  remain_height = height // 10
  remain_width = width // 10
  for i in range(height - 1, 0, -2):
    c_down += 2
    for j in range(width - 1, 0, -2):
      if not(clothes[i][j][0] > 240 and 
             clothes[1][j][1] > 240 and 
             clothes[i][j][2] > 240):
        switch_down = True
        break
    if(switch_down):
      break
  
  print(remain_height)
  c_down_clothes = clothes[:-(c_down) + (remain_height - 1)][:][:]
  cropped_height = c_down - remain_height
  return cropped_height, c_down_clothes

def trim_clothes_width(clothes):
  #trim white sides of target clotes
  """
  c_up_clothes = check_up(clothes)
  c_down_clothes = check_down(c_up_clothes)
  """
  clothes = np.transpose(clothes, (1, 0, 2))
  _, c_left_clothes = check_up(clothes)
  _, c_right_clothes = check_down(c_left_clothes)
  
  temp = np.transpose(c_right_clothes, (1, 0, 2))
  
  return temp

def trim_clothes_height(clothes,
                        resized_height = 256,
                        resized_width = 192):
  height, trimmed_width, _ = clothes.shape
  final_height = 4 * trimmed_width / 3
  c_u, _ = check_up(clothes)
  c_d, _ = check_down(clothes)
  
  if height > final_height:
    if c_u > c_d:
      min_value = c_d
      update_1_cloth = clothes[(c_u - min_value - 1):][:][:]
      
    else:
      min_value = c_u
      update_1_cloth = clothes[:-(c_d) + (min_value - 1)][:][:]

  update_1_height, update_1_width, _ = update_1_cloth.shape
  
  if update_1_height > final_height:
    
    trim_value = int((update_1_height - final_height) / 2)
    
    final_cloth = update_1_cloth[(trim_value - 1):-(trim_value) - 1][:][:]
    update_1_cloth = final_cloth
    print("final_cloth_shape: " + str(final_cloth.shape))
    print("check ratio")
    print(final_cloth.shape[1] / final_cloth.shape[0])
  
  final_cloth = cv2.resize(update_1_cloth, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
  print("check final_cloth.shape: " + str(final_cloth.shape))
  return final_cloth
    
    
  
  
def main():
  clothes = cv2.imread('./trim_example2/102021_1.png')
  new_clothes = trim_clothes_width(clothes)
  new_clothes = trim_clothes_height(new_clothes)
  
  new_clothes = cv2.resize(new_clothes, (192, 256), interpolation=cv2.INTER_AREA)
  cv2.imwrite('./trim_example2/102021_1.png', new_clothes)
  """
  cv2.imshow('img', new_clothes)
  cv2.waitKey()
  cv2.destroyAllWindows()
  """


if __name__ == "__main__":
  main()
    