---
title : CoCo_version_decument(ver 1.0.0) 
writer : khosungpil
type : Version document(official)
local : soma05
objective : 로컬에서 버전을 업데이트하고 변경 사항 있을 시 문서화 한다.
---

# CoCo version Document #

## ver 0.0.0 (2018.12.17) ##
1. 공식 메뉴얼 작성
2. 필요한 코드만 남기고 전부 백업
<hr>

## ver 0.0.1 (2018.12.21) ##
1. extract_part_1220.py, 이미지 전처리 코드 수정
    - 사진 리사이징 640:480으로 고정
    - segment 리사이징 할 시 확대 리사이징 코드 삭제
    - 변수명 및 함수명 변경
2. image_preprocessing.py
    - Import 변경, 함수 변경

## ver 0.0.2 (2019.2.28) ##
1. 기존의 mask로부터 합성영역을 뗐다면 segment + mask를 통해 합성영역 추출
    - 하얀색으로 경계 생성
    - 나시 -> 반팔의 경우 살이 보일 여지가 있음.
    - cropped_body_segment가 png로 저장되도록 수정
2. image_preprocessing.py
    - 전신에서 바지 label(9번)만을 추출하여 binary image 추출
    - xxxxxx_1.png 로 저장
    
## ver 0.0.3 (2019.3.2) ##
1. post_test_every.py 생성
    - 테스트를 한꺼번에 돌리기 

## ver 0.0.4 (2019.4.1) ##
1. trim_image.py
    - 옷을 사진 사이즈에 맞춰서 자르는 코드
2. 0318_every.py
    - tps로 옷을 warping하는 과정을 제거
    - stage폴더에 GMM으로 warping한 옷이 있을 때 그것을 tps_image로 사용함

## ver 0.0.5 (2019.4.4) ##
1. magnify_cloth.py, magnify_mask.py 추가
    - 옷 이미지를 다소 확대하여 적용시키는 모듈
2. 0404_every.py
    - fg, bg를 각각 확대 후 적용시키도록 메인 수정