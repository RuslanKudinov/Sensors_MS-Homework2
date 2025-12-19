import cv2
import numpy as np
import glob
import os

# Параметры шахматной доски
CHESSBOARD_SIZE = (9, 6)  # Внутренние углы (клеток-1)
SQUARE_SIZE = 2.5  # Размер клетки в мм (или любых единицах)

# Подготовка точек в 3D: (0,0,0), (30,0,0), (60,0,0), ..., (240,150,0)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Массивы для хранения точек
objpoints = []  # 3D точки в реальном мире
imgpoints = []  # 2D точки на изображении

# Папка с изображениями
image_folder = "calibration_images"
images = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))

print(f"Найдено изображений: {len(images)}")

for fname in images:
    print(f"Обрабатывается: {os.path.basename(fname)}")
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ищем углы шахматной доски
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        objpoints.append(objp)

        # Уточняем положение углов
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners_refined)

        # Отображение углов (опционально)
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners_refined, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(300)
    else:
        print(f"Углы не найдены на {os.path.basename(fname)}")

cv2.destroyAllWindows()

print(f"Успешно обработано кадров: {len(objpoints)}")
if len(objpoints) < 10:
    print("Слишком мало удачных кадров. Нужно больше 10 для хорошей калибровки.")
    exit()

# Калибровка камеры
print("Калибруем камеру...")
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n=== Внутренние параметры камеры (матрица K) ===")
print(K)

print("\n=== Коэффициенты дисторсии ===")
print(dist)

print(f"\nСредняя ошибка репроекции (пиксели): {ret}")

# Сохраняем параметры в файл
np.savez("camera_params.npz", K=K, dist=dist, ret=ret)
print("Параметры сохранены в camera_params.npz")
