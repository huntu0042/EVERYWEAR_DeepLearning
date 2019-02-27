from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os

userID = "cherry3"
test_dir = "testdata/"

if __name__=="__main__":
    try:
        os.mkdir(test_dir + userID)
    except:
        pass
    id_dir = test_dir + userID + "/"
    """
        body_images : 사용자가 올린 전신 이미지
        body_pose : 전신으로부터 추출한 포즈
        body_segment : 전신으로부터 추출한 세그먼트
        composed_upper_images : 상의가 합성된 상체 영역 이미지 + mask
        composed_lower_images : 하의가 합성된 전신 영역 이미지 + mask
        upper_images : 전신으로부터 추출한 상체 영역 이미지
        upper_segment : 전신으로부터 추출한 상체 영역 세그먼트
        upper_pose : 전신으로부터 추출한 상체 영역 포즈
        final_upper_images : 상의만 합성된 전신 영역 이미지
        final_images : 상, 하의가 모두 합성된 전신 영역 이미지
        upper_pickle : 상의 합성 시 필요
        body_pickle : 하의 합성 시 필요
        test.txt : 사진정보
    """
    try:
        input_dir = id_dir + "input"
        stage_dir = id_dir + "stage"
        output_dir = id_dir + "output"
        os.mkdir(input_dir)
        os.mkdir(stage_dir)
        os.mkdir(output_dir)
        
        os.mkdir(input_dir + "/body_images")
        os.mkdir(input_dir + "/body_resized")
        os.mkdir(input_dir + "/body_pose")
        os.mkdir(input_dir + "/body_segment")
        
        os.mkdir(input_dir + "/upper_images")
        os.mkdir(input_dir + "/upper_pose")
        os.mkdir(input_dir + "/upper_segment")

        os.mkdir(input_dir + "/upper_pickle")
        os.mkdir(input_dir + "/body_pickle")

        os.mkdir(output_dir + "/composed_images")
        
        os.mkdir(output_dir + "/final_upper_images")
        os.mkdir(output_dir + "/final_lower_images")
        os.mkdir(output_dir + "/final_images")
        
        
    except:
        pass
