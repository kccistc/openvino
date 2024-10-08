import cv2
import numpy as np
import openvino as ov
import time
import argparse
from pathlib import Path
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

# Constants
MODEL_ID = "models/depth_anything_vits14_int8"
OV_DEPTH_ANYTHING_PATH = Path(f"{MODEL_ID}.xml")
DEVICE = 'CPU'
WINDOW_TITLE = 'Depth Anything'
SKIP_FRAMES = 3

class DepthModel:
    """
    This class encapsulates the model loading, processing, and inference logic.
    It includes methods for model initialization, frame preprocessing, inference, and result visualization.
    """

    def __init__(self, model_path, device):
        self.core = ov.Core()
        self.compiled_model = self._load_model(model_path, device)
        self.transform = self._create_transform()

    def _load_model(self, model_path, device):
        """
        Load the OpenVINO model and return the compiled model.
        """
        return self.core.compile_model(model=model_path, device_name=device)

    def _create_transform(self):
        """
        Create a series of transformations for input image preprocessing.
        """
        return Compose(
            [
                Resize(width=518, height=518, resize_target=False, ensure_multiple_of=14,
                       resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    @staticmethod
    def normalize_minmax(data):
        """
        Normalizes the values in `data` between 0 and 1.
        """
        return (data - data.min()) / (data.max() - data.min())

    def preprocess_frame(self, image):
        """
        Preprocesses the input frame for the model inference.
        Converts image to RGB, applies normalization, and resizes.
        """
        h, w = image.shape[:-1]
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        return self.transform({"image": input_image})["image"]

    def infer(self, input_image):
        """
        Perform inference on the input image.
        """
        input_image = np.expand_dims(input_image, 0)  # Reshape for NCHW format.
        return self.compiled_model(input_image)[0]

    def convert_result_to_image(self, result):
        """
        Convert the network result to an RGB image with closer distances in red and farther distances in blue.
        """
        result = result.squeeze(0)
        result = self.normalize_minmax(result)
        result = (result * 255).astype(np.uint8)
        result = cv2.applyColorMap(result, cv2.COLORMAP_JET)
        return result


class VideoStream:
    """
    This class handles the video stream input from a camera or file and processes it for depth visualization.
    """

    def __init__(self, source):
        self.source = source
        self.cap = self._open_video_stream(source)

    @staticmethod
    def _open_video_stream(source):
        """
        Open the video stream from the provided source.
        """
        if source.isdigit():
            return cv2.VideoCapture(int(source))
        else: # Video file
            return cv2.VideoCapture(source)

    def read_frame(self):
        """
        Read a frame from the video stream.
        """
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Error: Can't receive frame (stream end?). Exiting ...")
        return frame

    def release(self):
        """
        Release the video stream and close all OpenCV windows.
        """
        self.cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def display_frame(window_title, frame):
        """
        Display the processed frame in a window.
        """
        cv2.imshow(window_title, frame)


def run_depth_visualization(model: DepthModel, video_stream: VideoStream):
    """
    Main function to capture video, run inference on each frame, and display the depth visualization.
    """
    frame_counter = 0  # Initialize frame counter
    try:
        while True:
            # Capture frame from video stream
            frame = video_stream.read_frame()

            # Increment the frame counter
            frame_counter += 1
            # Skip frames: only process every Nth frame
            if frame_counter % SKIP_FRAMES != 0:
                continue

            # Preprocess the frame and run model inference
            processed_frame = model.preprocess_frame(frame)
            depth_map = model.infer(processed_frame)

            # Convert the model result to a visual image
            result_frame = model.convert_result_to_image(depth_map)

            # Display the frame
            video_stream.display_frame(WINDOW_TITLE, result_frame)

            # Break loop if 'Esc' key is pressed
            if cv2.waitKey(30) == 27:  # 27 is the ASCII code for the escape key
                break

    except RuntimeError as e:
        print(e)
    finally:
        video_stream.release()


def main():
    """
    Main entry point for initializing the depth model and video stream, and running the visualization.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Depth Visualization")
    parser.add_argument('--input', type=str, default="0", help='Input')
    args = parser.parse_args()

    # Initialize the depth model
    depth_model = DepthModel(model_path=OV_DEPTH_ANYTHING_PATH, device=DEVICE)

    # Initialize the video stream
    video_stream = VideoStream(source=args.input)

    # Run depth visualization
    run_depth_visualization(model=depth_model, video_stream=video_stream)


# Check if this script is being run directly
if __name__ == "__main__":
    main()
