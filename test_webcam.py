import cv2
import time
import pyttsx3

def test_webcam():
    print("Testing webcam...")
    
    # Initialize text-to-speech engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam!")
        return
    
    print("Webcam opened successfully!")
    
    # Load face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("Press 'q' to quit")
    
    # Dictionary to track last announcement time
    last_announced = {}
    cooldown = 3  # seconds between announcements
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame!")
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Draw rectangle around faces and announce
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Announce face detection with cooldown
            current_time = time.time()
            if current_time - last_announced.get('face', 0) >= cooldown:
                engine.say("Face detected")
                engine.runAndWait()
                last_announced['face'] = current_time
        
        # Display the number of faces detected
        cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Webcam Test', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed")

if __name__ == "__main__":
    test_webcam() 