import logging as lg

import cv2
import os

from analysis.analysis_config import Config
from analysis.analysis_state import State
from analysis.logger_config import setup_logging
from analysis.strategy.camera_calibration_strategy import CameraCalibrationStrategy
from analysis.strategy.main_strategy import MainAnalysisStrategy
from sandbox.scene3d.run3d import Run3D

"""
Для быстрой тестировки без вебки
"""
rect_folder = 'C:/Users/vladi/PycharmProjects/Sravat/sandbox/test_photo/test_rect_photo'
chass_folder = 'C:/Users/vladi/PycharmProjects/Sravat/sandbox/test_photo/chess_photo'
shadow_folder = 'C:/Users/vladi/PycharmProjects/Sravat/sandbox/test_photo/test_shadow_photo'
rect_files = [f for f in os.listdir(rect_folder) if f.lower().endswith('.jpg')]
chess_files = [f for f in os.listdir(chass_folder) if f.lower().endswith('.jpg')]
shadow_files = [f for f in os.listdir(shadow_folder) if f.lower().endswith('.jpg')]

camera_calibration = CameraCalibrationStrategy()
for image_name in chess_files:
    print(os.path.join(chass_folder, image_name))
    frame = cv2.imread(os.path.join(chass_folder, image_name))
    camera_calibration(frame)

main_strategy = MainAnalysisStrategy()

# run3d = Run3D(main_strategy._state)
# run3d.setup()

setup_logging()
idx = 1
pidx = -1
image_name = shadow_files[0]

for image_name in rect_files:
    frame = cv2.imread(os.path.join(rect_folder, image_name))
    main_strategy(frame)

while True:
    if pidx == idx:
        cv2.imshow('image', frame)
        # run3d.show()
    else:
        pidx = idx
        image_name = shadow_files[idx]
        frame = main_strategy(cv2.imread(os.path.join(shadow_folder, image_name)))
        # run3d.show()

    key = cv2.waitKey(50) & 0xFF
    if key == ord('q'):
        break
    if key == ord('n'):
        idx += 1