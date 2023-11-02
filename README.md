# openvino

## Open Model Zoo
https://github.com/openvinotoolkit/open_model_zoo

This repository includes optimized deep learning models and a set of demos to expedite development of high-performance deep learning inference applications. Use these free pre-trained models instead of training your own models to speed-up the development and production deployment process.

- [Intel Pre-Trained Models](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/index.md)
- [Public Pre-Trained Models](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md)
- [Model Downloader](https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/model_tools/README.md) and other automation tools
- [Demos](https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/README.md) that demonstrate models usage with OpenVINO™ Toolkit

### License
Open Model Zoo is licensed under [Apache License Version 2.0](https://github.com/openvinotoolkit/open_model_zoo/blob/master/LICENSE).

### Online Documentation
- [Pre-Trained Models](https://docs.openvino.ai/2023.0/model_zoo.html#)
- [Demos and Samples](https://docs.openvino.ai/2023.0/omz_demos.html)

### Installation
1. Download Open Model Zoo
```sh
git clone --recurse-submodules https://github.com/openvinotoolkit/open_model_zoo.git
```
2. Install Required packages
```sh
sudo apt install libopencv-dev
cd open_model_zoo
python3 -m venv omz_venv
source omz_venv/bin/activate
python3 -m pip install --upgrade pip
pip install openvino openvino-dev
pip install -r demos/requirements.txt
```
3. Modify code to replace GStreamer with V4L2 as webcam backbone
```diff
diff --git a/demos/common/cpp/utils/src/images_capture.cpp b/demos/common/cpp/utils/src/images_capture.cpp
index 8a205fc64..fefcb1463 100644
--- a/demos/common/cpp/utils/src/images_capture.cpp
+++ b/demos/common/cpp/utils/src/images_capture.cpp
@@ -240,7 +240,7 @@ public:
             throw std::runtime_error("readLengthLimit must be positive");
         }
         try {
-            if (cap.open(std::stoi(input))) {
+            if (cap.open(std::stoi(input), cv::CAP_V4L2)) {
                 this->readLengthLimit = loop ? std::numeric_limits<size_t>::max() : readLengthLimit;
                 cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
                 cap.set(cv::CAP_PROP_FRAME_WIDTH, cameraResolution.width);
```
4. Build c++ demo samples
```sh
cd demos
source /opt/intel/openvino_2023.1.0/setupvars.sh
./build_demos.sh -DENABLE_PYTHON=y --build_dir=./
```
* Expected reaults after build
```
open_model_zoo/demos$ tree intel64/Release/ -L 1
├── classification_benchmark_demo
├── crossroad_camera_demo
├── ctcdecode_numpy
├── gaze_estimation_demo
├── human_pose_estimation_demo
├── image_processing_demo
├── interactive_face_detection_demo
├── libgflags_nothreads.a
├── libmodels.a
├── libmonitors.a
├── libmulti_channel_common.a
├── libpipelines.a
├── libutils.a
├── mask_rcnn_demo
├── monitors_extension.so
├── mri_reconstruction_demo
├── multi_channel_face_detection_demo
├── multi_channel_human_pose_estimation_demo
├── multi_channel_object_detection_demo_yolov3
├── noise_suppression_demo
├── object_detection_demo
├── pedestrian_tracker_demo
├── pose_extractor.so
├── security_barrier_camera_demo
├── segmentation_demo
├── smart_classroom_demo
├── social_distance_demo
└── text_detection_demo
```

### Practice #1 - bert_question_answering_demo
1. Go to the demo
```sh
cd demos/bert_question_answering_demo/python
```
2. Download required pretrained models
```sh
omz_downloader --list models.lst --precision FP16
```
3. Run
```sh
python3 bert_question_answering_demo.py \
    --vocab intel/bert-small-uncased-whole-word-masking-squad-0001/vocab.txt \
    --model intel/bert-small-uncased-whole-word-masking-squad-0001/FP16/bert-small-uncased-whole-word-masking-squad-0001.xml \
    --input_names="input_ids,attention_mask,token_type_ids" \
    --output_names="output_s,output_e" \
    --input="https://en.wikipedia.org/wiki/Bert_(Sesame_Street)" \
    -c
```

### Practice #2 - interactive_face_detection_demo
* Refer to the `README.md` from `open_model_zoo/demos/interactive_face_detection_demo/cpp`
```sh
./interactive_face_detection_demo -i 0 \
    -m intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml \
    --mag intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml \
    --mhp intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml \
    --mem intel/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.xml \
    --mlm intel/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002.xml \
    --mam public/anti-spoof-mn3/FP16/anti-spoof-mn3.xml \
    -d CPU
```

### Practice #3 - gaze_estimation_demo

### Practice #4 - monodepth_demo

### Practice #5 - object_detection_demo

### Practice #6 - multi_channel_object_detection_demo_yolov3
* Use more than two camera/video inputs

### Practice #7 - segmentation_demo

### (Optional) Practice #8
* 위 데모 실행을 shell script 로 작성해 실행
* 작성한 shell script 실행 시 device 를 parameter 를 받아 CPU or GPU 로 inference 되도록 수정
```sh
# example
$ run.sh CPU # <<-------- CPU 에서 실행

## Hugging Face + Openvino
https://huggingface.co/docs/optimum/intel/inference
$ run.sh GPU # <<-------- GPU 에서 실행
```
## Hugging Face
https://huggingface.co/docs/optimum/intel/inference
