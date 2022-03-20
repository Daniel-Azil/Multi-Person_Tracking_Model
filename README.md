# Object Detection and Tracking with YOLO and SORT

This project uses the YOLO model for object detection and the SORT (Simple Online and Realtime Tracking) algorithm for tracking objects in a video.

## Requirements

- Python 3.x
- OpenCV
- cvzone
- Ultralytics YOLO
- SORT

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    ```
2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Place your video file in the `videos_and_images` directory.
2. Place your mask image in the `videos_and_images` directory.
3. Run the script:
    ```sh
    python your_script_name.py
    ```

## Code Explanation

- **Model Initialization:** 
    ```python
    from ultralytics import YOLO
    import cv2
    import cvzone
    import math
    from sort import *

    model = YOLO("model_weights/yolov8n.pt")
    ```

- **Video and Mask Loading:**
    ```python
    cap = cv2.VideoCapture("videos_and_images/people.mp4")
    mask = cv2.imread("videos_and_images/mask_region.png")
    ```

- **Tracker Initialization:**
    ```python
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    ```

- **Detection and Tracking:**
    ```python
    while True:
        success, img = cap.read()
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask_region = cv2.bitwise_and(img, mask)
        image_graphics = cv2.imread("videos_and_images/graphics.png", cv2.IMREAD_UNCHANGED)
        img = cvzone.overlayPNG(img, image_graphics, (750, 60))
        
        results = model(mask_region, stream=True)
        
        detections = np.empty((0, 5))
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                
                conf = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])
                currentClass = className[cls]
                
                if (currentClass == "person") and conf > 0.49:
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_array))
        
        track_result = tracker.update(detections)
        # Additional code for drawing boxes and counting objects
    ```

## License

This project is licensed under the MIT License.
## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
- [SORT: Simple Online and Realtime Tracking](https://github.com/abewley/sort)
