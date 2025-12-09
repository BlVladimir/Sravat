import logging as lg

import cv2
import os

from analysis.analysis_config import Config
from analysis.analysis_state import State
from analysis.logger_config import setup_logging
from analysis.strategy.camera_calibration_strategy import CameraCalibrationStrategy
from analysis.strategy.main_strategy import MainAnalysisStrategy
from rudiments.scene3d.run3d import Run3D

image_folder = '/Users/vblinov/PycharmProjects/Sravat/test_photo'
jpg_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]

Config.load_calibration()
camera_calibration = CameraCalibrationStrategy()
main_strategy = MainAnalysisStrategy()

run3d = Run3D(main_strategy._state)
run3d.setup()

setup_logging()
idx = 0
pidx = -1
image_name = jpg_files[0]
while True:
    if pidx == idx:
        cv2.imshow('image', frame)
        run3d.show()
    else:
        pidx = idx
        image_name = jpg_files[idx]
        frame = main_strategy(cv2.imread(os.path.join(image_folder, image_name)))
        run3d.show()

    key = cv2.waitKey(50) & 0xFF
    if key == ord('q'):
        break
    if key == ord('n'):
        idx += 1