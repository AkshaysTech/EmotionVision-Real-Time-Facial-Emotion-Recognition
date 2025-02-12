import cv2
import os

# Emotions to capture
emotions = ['happy', 'sad', 'angry']

# Directory paths to save images
output_dirs = {emotion: f"data/{emotion}/" for emotion in emotions}

# Create output directories if they don't exist
for emotion_dir in output_dirs.values():
    os.makedirs(emotion_dir, exist_ok=True)

# Function to capture and save images
def capture_images():
    cap = cv2.VideoCapture(0)  # 0 is the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Display the frame
        cv2.imshow('Frame', frame)

        # Check for key press
        key = cv2.waitKey(1)
        if key == ord('q'):  # Press 'q' to exit
            break
        elif key == ord('h'):  # Press 'h' for happy
            save_image(frame, 'happy')
        elif key == ord('s'):  # Press 's' for sad
            save_image(frame, 'sad')
        elif key == ord('a'):  # Press 'a' for angry
            save_image(frame, 'angry')

    cap.release()
    cv2.destroyAllWindows()

# Function to save image to the appropriate directory
def save_image(image, emotion):
    output_dir = output_dirs[emotion]
    num_files = len(os.listdir(output_dir))
    file_path = os.path.join(output_dir, f"{emotion}_{num_files}.jpg")
    cv2.imwrite(file_path, image)
    print(f"Saved: {file_path}")

# Main function
if __name__ == "__main__":
    print("Press 'h' for happy, 's' for sad, 'a' for angry, and 'q' to quit.")
    capture_images()
