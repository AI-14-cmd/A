import cv2
import os
import time

def create_face_directory():
    """Create the known_faces directory if it doesn't exist"""
    if not os.path.exists('known_faces'):
        os.makedirs('known_faces')
        print("Created 'known_faces' directory")
    else:
        print("'known_faces' directory already exists")

def capture_face(name):
    """Capture a face from webcam and save it to known_faces directory"""
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return False
    
    # Load face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print(f"\nCapturing face for {name}")
    print("Press SPACE to capture when ready")
    print("Press ESC to cancel")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangle around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display instructions
        cv2.putText(frame, "Press SPACE to capture", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press ESC to cancel", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow('Capture Face', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # If space is pressed, capture the face
        if key == 32:  # SPACE key
            if len(faces) > 0:
                # Get the first face
                x, y, w, h = faces[0]
                face = frame[y:y+h, x:x+w]
                
                # Save the face image
                filename = f"known_faces/{name}.jpg"
                cv2.imwrite(filename, face)
                print(f"Face captured and saved as {filename}")
                
                # Show success message
                cv2.putText(frame, "Face captured!", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Capture Face', frame)
                cv2.waitKey(1000)  # Show message for 1 second
                
                cap.release()
                cv2.destroyAllWindows()
                return True
            else:
                print("No face detected! Please position your face in the frame.")
        
        # If ESC is pressed, cancel
        elif key == 27:  # ESC key
            print("Capture cancelled")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return False

def main():
    # Create face directory
    create_face_directory()
    
    while True:
        # Get person's name
        name = input("\nEnter person's name (or 'q' to quit): ").strip()
        
        if name.lower() == 'q':
            break
        
        if not name:
            print("Name cannot be empty!")
            continue
        
        # Capture face
        if capture_face(name):
            print(f"Successfully captured face for {name}")
        else:
            print(f"Failed to capture face for {name}")
        
        # Ask if user wants to capture another face
        another = input("\nCapture another face? (y/n): ").lower()
        if another != 'y':
            break
    
    print("\nFace capture session completed!")
    print("You can now run face_recognition.py to test the recognition")

if __name__ == "__main__":
    main() 