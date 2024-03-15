import numpy as np
import cv2
import pyautogui
import time
import GameState

def capture_screen():
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def is_in_battle():
    # Essentially if the spellbook is present on the screen
    # We can assume that we are not in a battle
    game_state = GameState.GameState()
    screen = capture_screen()
    game_state.update_state(screen)
    return not getattr(game_state, 'spellbook', False)

def main():
    game_state = GameState.GameState()

    while True:
        screen = capture_screen()
        game_state.update_state(screen)

        in_battle = is_in_battle()
        print(f"In Battle: {in_battle}")

        time.sleep(0.5)

if __name__ == "__main__":
    main()
