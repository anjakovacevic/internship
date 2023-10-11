from ultralytics import YOLO
import supervision as sv
import cv2
import argparse
import numpy as np

'''
polygon = np.array([
    [380, 170],
    [380, 750],
    [1920, 750],
    [1920, 170],
])
'''
polygon = np.array([
    [0, 0.1],
    [1, 0.1],
    [0, 0.9],
    [1, 0.9],
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
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    
    model = YOLO('yolov8s.pt') 
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
     
    # initiate annotators
    zone_polygon = (polygon * np.array(args.webcam_resolution)).astype(int)
    # box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))

    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=6, text_thickness=6, text_scale=4)

    while True:
        ret, frame = cap.read()
        
        # detect
        results = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 2]
        #zone.trigger(detections=detections)

        labels=[]

        for confidence, class_id in zip(detections.confidence, detections.class_id):
            # Retrieve the class name from YOLO model names
            class_name = model.model.names[class_id]
            label = f"{class_name} {confidence:0.2f}"
            labels.append(label)

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        zone.trigger(detections=detections)  
        frame = zone_annotator.annotate(scene=frame)
        cv2.imshow("yolov8", frame)

        # if esc is pressed, while loop will break
        if(cv2.waitKey(30) == 27):
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()

# from IPython import display
# display.clear_output()