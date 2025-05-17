import cv2
import numpy as np
import os
from datetime import datetime

# Create directories for known faces and face detection
if not os.path.exists('known_faces'):
    os.makedirs('known_faces')

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Load the pre-trained face recognition model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

def load_known_faces():
    """Load known faces from the known_faces directory"""
    faces = []
    labels = []
    label_id = 0
    label_map = {}
    
    # Load each image file from the known_faces directory
    for filename in os.listdir('known_faces'):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Load image file
            image_path = os.path.join('known_faces', filename)
            face_image = cv2.imread(image_path)
            
            if face_image is not None:
                # Convert to grayscale
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                
                # Detect faces in the image
                detected_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(detected_faces) > 0:
                    # Get the first face
                    x, y, w, h = detected_faces[0]
                    face = gray[y:y+h, x:x+w]
                    
                    # Resize face to a standard size
                    face = cv2.resize(face, (100, 100))
                    
                    # Get the name from filename (remove extension)
                    name = os.path.splitext(filename)[0]
                    
                    faces.append(face)
                    labels.append(label_id)
                    label_map[label_id] = name
                    label_id += 1
    
    if faces:
        # Train the face recognizer
        face_recognizer.train(faces, np.array(labels))
        print(f"Trained on {len(faces)} faces")
    
    return label_map

def process_frame(frame, label_map):
    """Process a single frame for face detection and recognition"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize face to match training size
        face_roi = cv2.resize(face_roi, (100, 100))
        
        try:
            # Predict the label
            label_id, confidence = face_recognizer.predict(face_roi)
            
            # Get the name from the label map
            name = label_map.get(label_id, "Unknown")
            
            # Draw rectangle around face
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw name label
            cv2.rectangle(frame, (x, y-35), (x+w, y), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{name} ({confidence:.1f})", (x+6, y-6), font, 1.0, (255, 255, 255), 1)
            
        except Exception as e:
            print(f"Error processing face: {e}")
            # Draw rectangle for unknown face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y-35), (x+w, y), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, "Unknown", (x+6, y-6), font, 1.0, (255, 255, 255), 1)
    
    return frame

def main():
    # Load known faces
    print("Loading known faces...")
    label_map = load_known_faces()
    print(f"Loaded {len(label_map)} known faces")
    
    if not label_map:
        print("No faces found in the known_faces directory!")
        print("Please add some face images to the known_faces directory.")
        print("Image files should be named with the person's name (e.g., john.jpg, sarah.png)")
        return
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide video file path
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process the frame
        processed_frame = process_frame(frame, label_map)
        
        # Display the resulting frame
        cv2.imshow('Face Detection', processed_frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 