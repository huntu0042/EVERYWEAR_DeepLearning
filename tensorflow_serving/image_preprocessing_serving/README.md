---
title : Everywear image-synthesizing tensorflow serving
version : 0.0.1
writer : khosungpil
type : Version document(official)
local : soma05
objective : Everywear image-synthesizing tensorflow serving 문서
---
## ver 0.0.1 ##
1. 서빙 초안 문서 작성

<hr>

## Semantic segment parsing ##
&nbsp;&nbsp; semantic segment 작업을 진행했으며, input이 어떤 사이즈든 3:2를 갖고 있어서 640:480의 텐서를 반환한다.
### Args ###
~~~python
builder = tf.compat.v1.saved_model.builder.SavedModelBuilder("savedmodel")
        builder.add_meta_graph_and_variables(sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
                signature_def_map={
                    "model": tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
                        inputs={"image_holder_bytes": image_holder_bytes},
                        outputs={"segment_output": segment_output_bytes})
                })
~~~

## Semantic segment parsing ##
&nbsp;&nbsp; Segmantic segment parsing Model을 serving 하기 위해 시도했다.

image input으로 사용자 전신 사진을 받는다. 사이즈는 3:2이고 640:480 이상이어야 한다. resizing은 640:480 사이즈로 진행하며, segment도 640*480에 신체 부위별로 labeling 되어있는 4-d tensor를 반환한다.
테스트 진행 시 4-d tensor에서 squeeze를 통해 필요없는 차원을 제거한 후 matfile로 저장하게 된다.
