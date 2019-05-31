---
title : Everywear tensorflow serving
version : 0.0.1
writer : khosungpil
type : Version document(official)
local : soma05
objective : Everywear tensorflow serving 문서
---
## ver 0.0.1 ##
1. 서빙 초안 문서 작성

<hr>

## Saved_model(RM) ##
&nbsp;&nbsp; Refinement Model serving 작업을 진행했으며 카테고리 별로 작업이 진행된 것은 아니고 graph에 input을 넣었을 때 정확하게 output이 나오는 지 테스트 한 상태.
### Args ###
~~~python
builder.add_meta_graph_and_variables(sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
                signature_def_map={
                    "model": tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
                        inputs={"prod_image_holder_bytes": prod_image_holder_bytes,
                                "prod_mask_holder_bytes": prod_mask_holder_bytes,
                                "coarse_image_holder_bytes": coarse_image_holder_bytes,
                                "tps_image_holder_bytes": tps_image_holder_bytes},
                        outputs={"model_image_outputs_bytes": model_image_outputs_bytes,
                                 "select_mask_bytes": select_mask_bytes})
                })
~~~

## Saved_model(EDM) ##
&nbsp;&nbsp; Encoder-Decoder Model을 serving 하기 위해 시도했다. 하지만 **RM**의 경우 input이 모두 image여서 byte_string을 tf.placeholder에 input으로 들어가면 tf.image.decode_png로 변환이 가능하였다. 하지만 **EDM**의 경우 input으로 들어가는 segment_map(mat file, [256,192])과 pose_map(pkl, [256,192,18])이 decoding 시 image가 아니어서 변환되지 않는다. 따라사 이 이슈를 해결하기 위한 다양한 시도들이 있었다.
### Try 1: pkl file -> decode_png ###
* pkl.load() 시 데이터를 잘 읽어온다.
* with tf.gfile.FastGFile을 통해 파일을 열고 f.read()로 binary file(pkl)을 byte_string으로 읽어온 후에 tf.image.decode_png를 진행한다. 이미지가 아니기 때문에 format error가 발생한다.

### Try 2: pkl file -> decode_raw ###
* 가장 가능성 있는 방법 같지만 pkl file에서 데이터가 다르게 바뀌어 나옴.
* with tf.gfile.FastGFile을 통해 파일을 열고 f.read()로 binary file(pkl)을 byte_string으로 읽어온 후에 tf.image.decode_raw를 진행한다. segment의 경우 0번부터 22번까지의 값들로 이루어져 있지만 세자리 수의 값이 포함되어 있고 placeholder에서 reshape 시 49152(256 * 192)개의 데이터가 필요하지만 데이터 수가 부족하다는 error 발생

### Try 3: py_func를 통해 pkl.loads를 saved_model에 포함 ###
* py_func에 내장되는 function은 GraphDef로 직렬화되지 않는다. 따라서 다른 환경에서 모델을 직렬화하고 파라미터를 불러온 다면 tf.py_func는 동작하지 않는다.

### Try 4: csv -> decode_csv ###
* tf.decode_csv는 csv파일을 tensor로 decoding하는 것이 아닌 txt에서 csv파일로 디코딩하는 함수이다.
