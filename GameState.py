import json
import numpy as np
import cv2

class GameState:
    def __init__(self, config_path='gamestates.json'):
        self.states = {}
        self.load_config(config_path)

    def load_config(self, config_path):
        with open(config_path) as f:
            config = json.load(f)
        
        for state in config['states']:
            name = state['name']
            self.states[name] = {
                'template': cv2.imread(state['template_image'], 0),
                'threshold': state['threshold']
            }

    def update_state(self, screen):
        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        
        for name, data in self.states.items():
            if self.match_template(gray_screen, data['template'], data['threshold']):
                setattr(self, name, True)
            else:
                setattr(self, name, False)

    @staticmethod
    def match_template(screen, template, threshold=0.8):
        res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        return len(loc[0]) > 0
