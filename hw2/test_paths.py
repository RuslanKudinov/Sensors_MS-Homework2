import os
import glob

folder = "calibration_images"
print("Содержимое папки:", os.listdir(folder))
print("Полные пути:", glob.glob(os.path.join(folder, "*")))