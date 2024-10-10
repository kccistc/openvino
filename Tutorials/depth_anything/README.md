# Depth Anything

## How to run
 - Prerequisites:
   ```
   python3 -m venv venv_depth
   source venv_depth/bin/activate
   pip install -r requirements.txt
   ```
 - Run:
   ```
   # For video input
   python3 depth_main.py --input <video file path>
   ```
   ```
   # For camera input
   python3 depth_main.py --input 0 # <- Your video device index such as "/dev/video0"
   ```

## Reference
https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/depth-anything
