import cv2
from deepface import DeepFace
from gtts import gTTS
import os

def speak(text):
    """Function to handle TTS output using gTTS."""
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("temp.mp3")
        os.system("start temp.mp3")  # For Windows. Use "afplay temp.mp3" on macOS and "mpg123 temp.mp3" on Linux.
    except Exception as e:
        print(f"TTS Error: {e}")

def estimate_age():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'Enter' to estimate age and 'q' to quit.")
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the live preview window
        cv2.imshow('Age Estimation - Press Enter to Analyze', frame)

        # Wait for key input for 1ms
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # Enter key code
            # Detect faces and estimate age when Enter is pressed
            try:
                predictions = DeepFace.analyze(frame, actions=['age'], enforce_detection=False)

                # Check if predictions is a list, and if so, access the first element
                if isinstance(predictions, list):
                    predictions = predictions[0]

                # Extract age information
                age = predictions.get('age', "N/A")
                age=int(age)-10
                print(f"Estimated Age: {str(age)}")

                # Text-to-Speech output using gTTS
                speak(f"The estimated age is {int(age)} years old")

                # Display the frame with estimated age
                frame_display = frame.copy()  # Create a copy to display
                cv2.putText(frame_display, f"Age: {int(age)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow('Age Estimation - Press Enter to Analyze', frame_display)

            except Exception as e:
                print(f"Error during prediction: {e}")

        elif key == ord('q'):  # Press 'q' to quit
            print("Quitting...")
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Cleanup the audio file after use
    if os.path.exists("temp.mp3"):
        os.remove("temp.mp3")

# Run the age estimation
estimate_age()
