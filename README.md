# Computer Pointer Controller

Computer Pointer Controller is an application the moves the clicker base on the gaze estimates either from a pre-reocrded video or from a more logical use of the application, a live stream of a webcam.

## Project Set Up and Installation
1. Clone this repository.
2. Install [openvino](https://docs.openvinotoolkit.org/latest/)
3. Run openvino setup.sh
4. inside the repository, run `pip3 install -r requirements.txt`. This downloads the modules(mainly opencv and pyautogui) that the application uses to function.
5. inside the repository, run  `./download.sh`. This downloads the necessary IRs(face_detection, landmark_detection, head_pose_estimation, gaze_estimation). Check the file for security.

## Demo
You can run this from any directory, what's important is to provide either relative or absolute paths of files as parameters.
For this example, we'll be running the application from the root of the repository, and will be providing relative paths.
```
python3 src/main.py -fd intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 -fl intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 -hp intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 -ge intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 -i bin/demo.mp4
```
The following runs the application with CAM as the input, and with visualizations on the intermediate outputs of each model
```
python3 src/main.py -fd intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 -fl intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 -hp intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 -ge intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 -i CAM -flag fd fl hp ge
```
## Documentation
-h params displays the help message for each parameters
```
  -h, --help            show this help message and exit
  -fd FACE_DETECTION_MODEL, --face_detection_model FACE_DETECTION_MODEL
                        Path to Face Detection model xml
  -fl FACE_LANDMARK_MODEL, --face_landmark_model FACE_LANDMARK_MODEL
                        Path to Face Landmark Detection model xml
  -hp HEAD_POSE_MODEL, --head_pose_model HEAD_POSE_MODEL
                        Path to Head Pose Estimation model xml
  -ge GAZE_ESTIMATION_MODEL, --gaze_estimation_model GAZE_ESTIMATION_MODEL
                        Path to Head Pose Estimation model xml
  -i INPUT, --input INPUT
                        Path to image/video file or CAM
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on: CPU, GPU, FPGA
                        or MYRIAD is acceptable(CPU by default)
  -t THRESHOLD, --threshold THRESHOLD
                        Probability threshold for detections filtering(0.5 by
                        default)
  -e EXTENSIONS, --extensions EXTENSIONS
                        Extensions for custom layers support
  -flag VISUALIZATION [VISUALIZATION ...], --visualization VISUALIZATION [VISUALIZATION ...]
                        Example: --flag [fd] [fl] [hp] [ge] (No flag is
                        required,each represents stages of the pipeline)
  ```

## Benchmarks
Model Size(MB)
|Models\Precision   |FP32|FP16|INT8|
|-------------------|----|----|----|
|Face_Detector      |1.9 | NA | NA |
|LandMark_Detector  |0.796|0.424|0.320|
|Head_Pose_Estimator|7.4|3.7|2.1|
|Gaze_Estimator     |7.2|3.7|2.1|

Model Loading Times(s)
|Models\Precision   |FP32|FP16|INT8|
|-------------------|----|----|----|
|Face_Detector      |0.20738983154296875    |NA  |NA  |
|LandMark_Detector  |0.08849477767944336    |0.09293055534362793    |0.12967395782470703    |
|Head_Pose_Estimator|0.10287046432495117    |0.13409686088562012    |0.28405213356018066    |
|Gaze_Estimator     |0.1253974437713623    |0.15026116371154785    |0.32758021354675293    |

Model Predict Times(s)
|Models\Precision   |FP32|FP16|INT8|
|-------------------|----|----|----|
|Face_Detector      |0.041494131088256836    |NA  |NA  |
|LandMark_Detector  |0.0017197132110595703    |0.0014123916625976562    |0.0023717880249023438    |
|Head_Pose_Estimator|0.003668069839477539    |0.003798961639404297    |0.010738611221313477    |
|Gaze_Estimator     |0.002620220184326172    |0.0048520565032958984    |0.002721071243286133    |

## Stand Out Suggestions
### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame.
- The application is not expected to crash base on condition, the crucial part here is that Face Detector is able to detect a face from the input, otherwise the application just proceeds to the next frame. This information is crucial to the moment that the application moves the mouse currently, and is not able to detect a face on the next frame, so it just continues to move the mouse on this instance.
- Also Face Detector is able to provide multiple coordinates for multiple faces, the application currently uses one, whichever is detected first on the CURRENT FRAME, which can be a little bit unexpected when not explained, basically this doesn't mean who ever is there first on the video feed.
- Also, pyautogui crashes when mouse gets too far to the corner, this can be bypassed but is not recommended by pyautogui developer
