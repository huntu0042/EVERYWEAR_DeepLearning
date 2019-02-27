---
title : CoCo_official_menual(ver 1.0) 
writer : khosungpil
type : menual document(official)
local : soma05
objective : 로컬에서 COCO 실행법 메뉴얼
---

# CoCo official manual #
### 2018.12.26 작성

## User Data folder 만들기 ##
: ID를 부여하고 그 ID폴더를 만든 후 하위폴더에 필요한 폴더 생성

1. make_folder.py 열기
2. userID 값 변경
3. 실행
4. testdata 폴더 하위에 ID를 갖는 폴더 생성
5. profile.txt에 자동으로 ID 추가

## Image preprocessing ##
: User data folder에 이미지를 등록하면 이미지 전처리를 수행하여 파일에 저장한다.

1. Image_preprocessing 폴더
2. flask_server_to_deep.py를 실행하여 서버를 열음
3. image_preprocessing.py를 실행, 요청 대기
4. http://127.0.0.1:10008/userupload 접속
5. 사진 등록, userid는 만든 id를 기입하고 imageid를 기입한 후 upload
6. 이미지 전처리 수행 (30초 소요)

## image_preprocessing.py
* write_id()
  - Usage: 
    + text_dir(global): profile.txt에 id값 새로 쓰고 저장
    + input_dir : userId 하위 input폴더의 image.txt에 imageId + "_0.jpg" 새로 쓰고 저장
  - Args: userId, imageId
  - Return: 의미 없음.
  - Update:
    + png 파일을 고려해서 하드코딩말고 파일명 자체를 쓰도록 수정 해야 함.

* evaluate_segment_parsing()
  - Usage:
    + image.txt에 저장 된 이미지 파일명으로 이미지를 불러와서 segment parsing 수행
    + parsing 후 decode_labels()를 통해 mat file 형태로 저장
    + parsing 후 나온 array를 현재 color scale로 "image_id + _vis.png" 형태로 저장
  - Args: userId
  - Return: -
  - Update:
    + input_size와 resize_height, resize_width 구별해서 코딩
    + 현재 output이 644 * 483으로 나옴. size issue 시 고려해야 한다.

* decode_labels()
  - Usage: 
    + segmentation 후의 결과를 batch로 decoding
    + decoding된 segment를 mat file로 저장
    + bw_upper_segment : 상체 영역만 잘라서 binary_map으로 저장하여 이미지로 변환 후 저장
  - Args:
    + mask: argmax(segment label)을 얻고 inference result
    + num_images : batch에서 decoding할 Image의 수
  - Return: np.array(img)
  - Update:
    + resize issue 해결해야 한다.

* _cropping_process_image() : extract_part.py
  - Usage:
    + 사용자 전신 이미지로부터 상체와 하체 일부를 자르는 작업
    + <a href="#load_image">_load_image()</a>
    + <a href="#process_segment_map">process_segment_map()</a>
    + <strike>half: 이미지 너비 절반</strike>
    + crop_size : 이미지 분할 사이즈

    + <a href="#cropping_upper_segment_height">cropping_upper_segment_height()</a>
    + <a href="#image_trim_height">image_trim_height()</a>
    + <a href="#cropping_segment_width">cropping_segment_width()</a>
    + <a href="#image_trim_width">image_trim_width()</a>
    + <a href="#image_resize">image_resize()</a>
  - Args:
    + userId
  - Returns:
  - Updates:
    + half삭제 -> cropping_upper_segment_height()에 half 삭제(2018.12.17)
    + crop_size 부분도 함수 별로 나누는게 좋을 거 같음.

<p id="load_image">
</p>

* _load_image() : _cropping_process_image() : extract_part.py
  - Usage:
    + cv2.imread일 경우 BGR이므로 RGB로 convert 후에 return
  - Args: img_name_dir
  - Returns: img

<p id="process_segment_map">
</p>

* process_segment_map() : _cropping_process_image() : extract_part.py
  - Usage:
    + 644 * 483 segment mat file을 확대
    + matfile을 np.asarray에 복사(참조본이므로 원본이 바뀌면 같이 바뀜)
    + 각 element type에 따라 색깔을 변경할 수 있을 것으로 예상
  - Args:
    + segment: 모델의 전신 segment mat file(644 * 483)
    + height : 모델의 원본 이미지 height
    + width : 모델의 원본 이미지 width
  - Returns: 644 * 483 -> height * width 크기로 확대 된 segment
  - Updates:
    + cv2.INTER_AREA 는 이미지 축소 시 쓰는 보간법, cv2.INTER_CUBIC, cv2.INTER_LINEAR
    + 왜 segment를 키우려 한 지 모름(?)

<p id="cropping_upper_segment_height">
</p>

* cropping_upper_segment_height() : _cropping_process_image() : extract_part.py
  - Usage:
    + 전신 이미지에서 사진의 절반을 기준으로 위에서 밑으로 탐색 시작
    + 하의가 나오면 하의 탐색 완료 후 조금 더 자름 현재는 높이의 1/10만큼 더 자르게 되어 있다.
    + croped_height에서 시작하여 height만큼 탐색 시작
    + check: 절반에서 각 element가 갖는 segment
    + find_pants: 탐색 중 바지가 발견하면 true
    + crop_size: 

  - Args:
    + segment: 모델의 원본 segment(height * width)
    + height: 모델의 원본 height
    + width : 모델의 원본 width
    + <strike>half: 모델의 원본 너비의 가운데</strike>
    + crop_size: 모델의 원본 height / 10

  - Returns:
    + update_seg: 자르고 새롭게 쌓은 segment
    + croped_height: 원본사진을 자른 높이
  - Updates:
    + Argument 중 half를 삭제하고 변수로 만듦(2018.12.17)
    + 고정된 625값에서 다시 탐색 기능 활성화

<p id="image_trim_height">
</p>
    
* Image_trim_height() : _cropping_process_image() : extract.py
  - Usage:
    + segment를 모델의 원본 사이즈에 맞췄기 때문에 세그먼트를 자른 높이만큼 모델의 원본 이미지도 자른다.
  - Args:
    + image_raw
    + cropping_height
    + tag
  - Returns:
    + img_trim: 자르고 새롭게 쌓은 모델 원본 이미지
  - Updates:
    + cropping_upper_segment_height에 추가시켜도 상관없을 기능일 듯

<p id="cropping_segment_width">
</p>

* cropping_segment_width: _cropping_process_image() : extract.py
  - Usage:
    + 모델의 원본에서 자른 높이를 기준으로 비율에 맞춰 너비 축소
    + real_width: image 높이의 3/4(segment 높이랑 같으므로 없애자)
    + cropping_width: 잘라야 하는 너비 1/2 * (segment.shape[1] - real_width)
    + resized_cropping_width: 합성 후에 나오는 이미지가 256 * 192이므로 256*192 때 너비를 자르기 위해 interval을 resizing<br>
    256 : segment.shape[0] = resized_cropping_width : cropping_width(왠지 잘못 된 것 같음)
    + 너비를 자르기 위해 transpose 시킨 후 위 아래를 자른다.
  - Args:
    + image_id
    + segment: 높이를 허벅지까지 자른 update segment
    + croped_height: 허벅지까지 자른 segment의 높이(segment의 높이랑 같음 없애도 됨)
    + interval_upper_dir: 자른 width의 값을 저장하기 위한 문서 경로
    + tag
  - Returns:
    + update_seg: 너비까지 자른 segment
    + cropping_width: 잘라야 하는 너비
  - Updates:
    + 프로세스 자체를 바꿔야 함

<p id="image_trim_width">
</p>

* image_trim_width() : _cropping_process_image() : extract_part.py
  - Usage: 
    + 자른 segment를 토대로 이미지의 너비도 자름
  - Args:
    + image_raw: 모델 원본 이미지
    + cropping_width: 자르는 너비
  - Returns:
    + 자른 이미지 반환

<p id="image_resize"></p>

* image_resize() : _cropping_process_image() : extract_part.py
  - Usage:
    + 원본 이미지를 resizing하기 위함
  - Args:
    + 가로:세로가 3:4일 때 가로가 좀 더 길면 image_trim_width() 수정해야 할 듯
    + 모델 원본 이미지 resizing
  - Returns:
    + resizing 된 이미지










