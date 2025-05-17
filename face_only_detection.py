import cv2
import numpy as np
from ultralytics import YOLO
import time

def main():
    # Initialize YOLO model
    model = YOLO('yolov8n.pt')
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    print("Webcam opened successfully! Press 'q' to quit.")

    # Initialize variables for tracking
    chair_status = {}  # Dictionary to store chair status
    last_update_time = time.time()
    update_interval = 1.0  # Update status every 1 second

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Get current time
        current_time = time.time()

        # Run YOLO detection
        results = model(frame, classes=[0, 56, 67])  # 0: person, 56: chair, 67: cell phone

        # Process detections
        if current_time - last_update_time >= update_interval:
            chair_status.clear()  # Clear previous status
            person_on_chair = False
            phone_detected = False

            # Process each detection
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf < 0.5:  # Confidence threshold
                        continue

                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    if cls == 56:  # Chair
                        chair_status[(x1, y1, x2, y2)] = "Vacant"
                    elif cls == 0:  # Person
                        # Check if person is on any chair
                        for chair_coords in chair_status.keys():
                            if (x1 >= chair_coords[0] and x2 <= chair_coords[2] and
                                y1 >= chair_coords[1] and y2 <= chair_coords[3]):
                                chair_status[chair_coords] = "Occupied"
                                person_on_chair = True
                    elif cls == 67:  # Cell phone
                        phone_detected = True

            # Update chair status based on phone detection
            if person_on_chair:
                for chair_coords in chair_status.keys():
                    if chair_status[chair_coords] == "Occupied":
                        chair_status[chair_coords] = "Not Working" if phone_detected else "Working"

            last_update_time = current_time

        # Draw results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Set color based on class
                if cls == 56:  # Chair
                    color = (0, 255, 0)  # Green
                    label = "Chair"
                elif cls == 0:  # Person
                    color = (255, 0, 0)  # Blue
                    label = "Person"
                else:  # Phone
                    color = (0, 0, 255)  # Red
                    label = "Phone"

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw chair status
        for chair_coords, status in chair_status.items():
            x1, y1, x2, y2 = chair_coords
            status_color = (0, 255, 0) if status == "Vacant" else (0, 0, 255) if status == "Not Working" else (255, 0, 0)
            cv2.putText(frame, status, (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Show the frame
        cv2.imshow('Chair and Activity Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Detection completed.")

if __name__ == "__main__":
    main() 