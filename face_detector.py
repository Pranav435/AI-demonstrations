import cv2
import os
import pygame
import time
from gtts import gTTS

# Initialize pygame mixer for playing audio
pygame.mixer.init()

# Load the pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

def announce_message(message):
    # Convert text to speech using gTTS, with faster speech speed (slow=False)
    tts = gTTS(text=message, lang='en', slow=False)
    tts.save("announcement.mp3")  # Save the audio file

    # Play the audio file with pygame
    pygame.mixer.music.load("announcement.mp3")
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        continue

    # Stop and unload the music to release the file
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()
    
    # Delete the audio file after playing to save space
    os.remove("announcement.mp3")

def get_position(x, y, w, h, frame_width, frame_height):
    # Determine horizontal position
    if x + w / 2 < frame_width / 3:
        horizontal = "left"
    elif x + w / 2 > 2 * frame_width / 3:
        horizontal = "right"
    else:
        horizontal = "center"

    # Determine vertical position
    if y + h / 2 < frame_height / 3:
        vertical = "top"
    elif y + h / 2 > 2 * frame_height / 3:
        vertical = "bottom"
    else:
        vertical = "center"

    # Combine horizontal and vertical positions
    if horizontal == "center" and vertical == "center":
        return "center"
    return f"{vertical} {horizontal}"

previous_face_count = 0
previous_positions = []
centered_frame_count = 0  # Counter to ensure stability before capturing
captured = False  # Flag to indicate if a picture has been captured
guidance_provided = None  # Track last guidance given to avoid repeating

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Convert the frame to grayscale (Haar Cascade works with grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the faces and determine their positions
    positions = []
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        position = get_position(x, y, w, h, frame_width, frame_height)
        positions.append(position)

    # Display the frame with rectangles
    cv2.imshow('Face Detection', frame)
    
    # Get the number of faces detected
    num_faces = len(faces)
    
    # Only announce if the number of faces or their positions change
    if num_faces != previous_face_count or positions != previous_positions:
        announce_message(f"{num_faces} faces detected at positions: {', '.join(positions)}")
        previous_face_count = num_faces
        previous_positions = positions

    # Provide guidance for framing if face is off-center
    if num_faces == 1 and not captured:
        face_position = positions[0]
        
        # Provide guidance based on the face position
        if face_position == "center":
            guidance_provided = None
            centered_frame_count += 1
        else:
            if face_position != guidance_provided:  # Give guidance only if it's changed
                if "left" in face_position:
                    announce_message("Move slightly to your right.")
                elif "right" in face_position:
                    announce_message("Move slightly to your left.")
                if "top" in face_position:
                    announce_message("Move down a bit.")
                elif "bottom" in face_position:
                    announce_message("Move up a bit.")
                guidance_provided = face_position
            centered_frame_count = 0  # Reset count if face is off-center

        # Capture image if face is centered for stable frames
        if centered_frame_count == 5:
            announce_message("Hold still, capturing picture.")
            time.sleep(1)  # Brief wait to ensure stillness
            
            # Capture and save the photo
            cv2.imwrite("captured_image.jpg", frame)
            announce_message("Picture captured. Press Enter to continue.")
            captured = True  # Set flag to indicate a picture has been captured
            
            # Wait for the user to press Enter
            input("Press Enter to continue...")
            
            # Reset flags and counters for next capture
            captured = False
            centered_frame_count = 0
    else:
        centered_frame_count = 0  # Reset counter if no face or multiple faces are detected

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
