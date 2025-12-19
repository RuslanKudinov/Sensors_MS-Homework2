import cv2
import os

# Создаем папки вручную
print("Создаю папки...")
os.makedirs("calibration_images", exist_ok=True)
os.makedirs("stereo_images", exist_ok=True)

# 1. Извлекаем кадры из левого видео для монокалибровки
print("\n1. Извлекаю кадры из левого видео...")
cap_left = cv2.VideoCapture("video_left.mp4")
count = 0
frame_num = 0

while True:
    ret, frame = cap_left.read()
    if not ret:
        break
    
    # Берем каждый 10-й кадр
    if frame_num % 10 == 0:
        filename = f"calibration_images/chessboard_{count:03d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Сохранен: {filename}")
        count += 1
    
    frame_num += 1
    
    # Останавливаемся после 50 кадров
    if count >= 100:
        break

cap_left.release()
print(f"Извлечено {count} кадров для монокалибровки")

# 2. Извлекаем синхронные пары для стереокалибровки
print("\n2. Извлекаю синхронные пары...")
cap_left = cv2.VideoCapture("video_left.mp4")
cap_right = cv2.VideoCapture("video_right.mp4")

count = 0
frame_num = 0

while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    
    if not ret_left or not ret_right:
        break
    
    # Берем каждый 10-й кадр
    if frame_num % 10 == 0:
        left_name = f"stereo_images/left_{count:03d}.jpg"
        right_name = f"stereo_images/right_{count:03d}.jpg"
        
        cv2.imwrite(left_name, frame_left)
        cv2.imwrite(right_name, frame_right)
        
        print(f"Сохранена пара {count}: left_{count:03d}.jpg, right_{count:03d}.jpg")
        count += 1
    
    frame_num += 1
    
    # Останавливаемся после 50 пар
    if count >= 50:
        break

cap_left.release()
cap_right.release()
print(f"Извлечено {count} пар для стереокалибровки")

print("\n✅ Всё готово!")
print("Папка 'calibration_images/' - для задания 1")
print("Папка 'stereo_images/' - для задания 2")