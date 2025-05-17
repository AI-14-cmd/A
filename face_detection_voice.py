import cv2
import numpy as np
import os
import pyttsx3
import time

def initialize_voice():
    """Initialize text-to-speech engine"""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    return engine

def load_known_faces():
    """Load known faces from the known_faces directory"""
    faces = []
    labels = []
    label_id = 0
    label_map = {}
    
    print("Loading known faces...")
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
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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
                    print(f"Loaded face: {name}")
    
    if faces:
        # Train the face recognizer
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.train(faces, np.array(labels))
        print(f"Trained on {len(faces)} faces")
        return face_recognizer, label_map
    else:
        print("No faces found in the known_faces directory!")
        print("Please add some face images to the known_faces directory.")
        print("Image files should be named with the person's name (e.g., john.jpg, sarah.png)")
        return None, None

def announce_name(name, announced_faces, engine):
    """Announce the name only if it hasn't been announced before"""
    if name not in announced_faces:
        engine.say(f"I see {name}")
        engine.runAndWait()
        announced_faces.add(name)
        return True
    return False

def draw_detection_info(frame, x, y, w, h, name, is_new_detection):
    """Draw detection information around the face"""
    # Draw rectangle around face
    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    # Draw name label with background
    label = f"{name}"
    if is_new_detection:
        label += " - DETECTED!"
    
    # Calculate text size
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.0
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Draw background rectangle for text
    cv2.rectangle(frame, (x, y-40), (x + text_width + 10, y), color, -1)
    
    # Draw text
    cv2.putText(frame, label, (x+5, y-10), font, font_scale, (255, 255, 255), thickness)
    
    # Draw detection symbol if it's a new detection
    if is_new_detection:
        # Draw a green circle with checkmark
        center_x = x + w//2
        center_y = y - 50
        cv2.circle(frame, (center_x, center_y), 15, (0, 255, 0), 2)
        cv2.putText(frame, "✓", (center_x-5, center_y+5), 
                    font, 0.7, (0, 255, 0), 2)

def draw_status_bar(frame, detected_faces):
    """Draw status bar at the top of the frame"""
    # Create a black bar at the top
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
    
    # Draw status text
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, "Face Detection System", (10, 30), font, 1.0, (255, 255, 255), 2)
    
    # Draw number of faces detected
    cv2.putText(frame, f"Faces Detected: {len(detected_faces)}", 
                (frame.shape[1] - 300, 30), font, 1.0, (0, 255, 0), 2)

def main():
    # Initialize voice engine
    engine = initialize_voice()
    
    # Create known_faces directory if it doesn't exist
    if not os.path.exists('known_faces'):
        os.makedirs('known_faces')
        print("Created 'known_faces' directory")
        print("Please add some face images to the known_faces directory")
        return
    
    # Load known faces
    face_recognizer, label_map = load_known_faces()
    
    if face_recognizer is None or label_map is None:
        return
    
    # Initialize webcam
    print("\nInitializing webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam opened successfully!")
    print("Press 'q' to quit")
    
    # Load face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Set to track announced faces
    announced_faces = set()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw status bar
        draw_status_bar(frame, faces)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize face to match training size
            face_roi = cv2.resize(face_roi, (100, 100))
            
            try:
                # Predict the label
                label_id, confidence = face_recognizer.predict(face_roi)
                print(f"Predicted: {label_map.get(label_id, 'Unknown')} (Confidence: {confidence:.2f})")  # Debug info
                # Set a confidence threshold (lower is better)
                threshold = 150  # Increased threshold to allow more matches
                if confidence < threshold:
                    name = label_map.get(label_id, "Unknown")
                    print(f"Recognized as: {name} (Confidence: {confidence:.2f})")
                else:
                    name = "Unknown"
                    print(f"Unknown face (Confidence too high: {confidence:.2f})")
                # Announce the name if it's not "Unknown" and hasn't been announced before
                is_new_detection = False
                if name != "Unknown":
                    is_new_detection = announce_name(name, announced_faces, engine)
                # Draw detection information
                draw_detection_info(frame, x, y, w, h, name, is_new_detection)
            except Exception as e:
                print(f"Error processing face: {e}")
                # Draw detection information for unknown face
                draw_detection_info(frame, x, y, w, h, "Unknown", False)
        
        # Show the frame
        cv2.imshow('Face Recognition', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Face recognition completed")

if __name__ == "__main__":
    main() 