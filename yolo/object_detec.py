# yolo detect predict model=yolov8l.pt source=0 show=true
# for using the yolov8 trained model on your camera

import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np

import time

ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    # zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))

    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=2, text_thickness=4, text_scale=2)


    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        
        frame_count += 1
        elapsed_time = time.time() - start_time

        # Calculate and display FPS
        if elapsed_time >= 1.0:  # Update FPS every 1 second
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(result)

        labels=[]

        for confidence, class_id in zip(detections.confidence, detections.class_id):
            # Retrieve the class name from YOLO model names
            class_name = model.model.names[class_id]
            label = f"{class_name} {confidence:0.2f}"
            labels.append(label)

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        zone.trigger(detections=detections)
        #box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
        frame = zone_annotator.annotate(scene=frame)      
        
        cv2.imshow("yolov8", frame)

        # if esc is pressed, while loop will break
        if(cv2.waitKey(30) == 27):
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()