import cv2

def test_webcam():
    # Initialize video capture
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)  # Try the default camera (usually 0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam opened successfully!")
    print("Press 'q' to quit")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Display the frame
        cv2.imshow('Webcam Test', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam test completed")

if __name__ == "__main__":
    test_webcam() 