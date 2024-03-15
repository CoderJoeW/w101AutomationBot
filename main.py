import numpy as np
import cv2
import pyautogui
import time
import GameState
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the model
model = load_model('wizard101_navigation_model.h5')

def prepare_image(frame, target_size=(256, 256)):
    img = cv2.resize(frame, target_size)  # Resize the image
    img_array = img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the pixel values if your model expects values in [0, 1]
    return img_array

def predict_direction(frame):
    prepared_img = prepare_image(frame)
    predictions = model.predict(prepared_img)
    class_indices = {'left': 0, 'right': 1, 'forward': 2, 'arrived': 3}
    labels = list(class_indices.keys())
    predicted_class = labels[np.argmax(predictions)]
    return predicted_class

def capture_screen():
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def is_in_battle():
    game_state = GameState.GameState()
    screen = capture_screen()
    game_state.update_state(screen)
    return not getattr(game_state, 'spellbook', False)

def is_our_turn():
    game_state = GameState.GameState()
    screen = capture_screen()
    game_state.update_state(screen)
    return not getattr(game_state, 'pass', False)

def main():
    game_state = GameState.GameState()

    while True:
        screen = capture_screen()
        game_state.update_state(screen)

        in_battle = is_in_battle()
        print(f"In Battle: {in_battle}")

        # If not in battle, predict and print the direction
        if not in_battle:
            direction = predict_direction(screen)
            print(f"Direction: {direction}")

        time.sleep(0.5)

if __name__ == "__main__":
    main()
