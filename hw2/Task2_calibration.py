import cv2
import numpy as np
import glob
import os

# Параметры шахматной доски
CHESSBOARD_SIZE = (9, 6)  # внутренние углы
SQUARE_SIZE = 2.5  # размер клетки в мм

# Подготовка 3D точек
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Массивы для точек
objpoints = []  # 3D точки
imgpoints_left = []  # 2D точки левой камеры
imgpoints_right = []  # 2D точки правой камеры

# Загрузка изображений
left_images = sorted(glob.glob("stereo_images/left_*.jpg"))
right_images = sorted(glob.glob("stereo_images/right_*.jpg"))

print(f"Найдено {len(left_images)} левых и {len(right_images)} правых изображений")

if len(left_images) != len(right_images):
    print("Количество левых и правых изображений не совпадает!")
    exit()

# Обработка пар изображений
for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
    print(f"Обработка пары {i + 1}: {os.path.basename(left_path)} / {os.path.basename(right_path)}")

    imgL = cv2.imread(left_path)
    imgR = cv2.imread(right_path)

    if imgL is None or imgR is None:
        print(f"   Пропуск: не удалось загрузить изображения")
        continue

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Поиск углов на левом изображении
    retL, cornersL = cv2.findChessboardCorners(grayL, CHESSBOARD_SIZE, None)
    # Поиск углов на правом изображении
    retR, cornersR = cv2.findChessboardCorners(grayR, CHESSBOARD_SIZE, None)

    if retL and retR:
        objpoints.append(objp)

        # Уточнение углов
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cornersL_refined = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR_refined = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        imgpoints_left.append(cornersL_refined)
        imgpoints_right.append(cornersR_refined)

        # Визуализация
        cv2.drawChessboardCorners(imgL, CHESSBOARD_SIZE, cornersL_refined, retL)
        cv2.drawChessboardCorners(imgR, CHESSBOARD_SIZE, cornersR_refined, retR)
        combined = np.hstack((imgL, imgR))
        cv2.imshow('Chessboard corners', combined)
        cv2.waitKey(300)
    else:
        print(f"   Углы не найдены на одной из камер")

cv2.destroyAllWindows()

print(f"\nУспешно обработано пар: {len(objpoints)}")

if len(objpoints) < 15:
    print("Слишком мало удачных пар. Нужно минимум 15-20 для хорошей калибровки.")
    exit()

# ================= 1. КАЛИБРОВКА ОТДЕЛЬНЫХ КАМЕР =================
print("\nКалибровка левой камеры...")
retL, K1, dist1, rvecsL, tvecsL = cv2.calibrateCamera(
    objpoints, imgpoints_left, grayL.shape[::-1], None, None
)

print("Калибровка правой камеры...")
retR, K2, dist2, rvecsR, tvecsR = cv2.calibrateCamera(
    objpoints, imgpoints_right, grayR.shape[::-1], None, None
)

print("\n" + "=" * 60)
print("ВНУТРЕННИЕ ПАРАМЕТРЫ ЛЕВОЙ КАМЕРЫ")
print("=" * 60)
print(f"K1 =\n{np.array2string(K1, precision=3, suppress_small=True)}")
print(f"dist1 = {dist1.ravel()}")

print("\n" + "=" * 60)
print("ВНУТРЕННИЕ ПАРАМЕТРЫ ПРАВОЙ КАМЕРЫ")
print("=" * 60)
print(f"K2 =\n{np.array2string(K2, precision=3, suppress_small=True)}")
print(f"dist2 = {dist2.ravel()}")

# ================= 2. СТЕРЕОКАЛИБРОВКА =================
print("\n" + "=" * 60)
print("СТЕРЕОКАЛИБРОВКА")
print("=" * 60)

flags = 0
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

ret, K1_new, dist1_new, K2_new, dist2_new, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    K1, dist1, K2, dist2, grayL.shape[::-1],
    criteria=criteria, flags=flags
)

print(f"Ошибка стереокалибровки: {ret:.4f} пикселей")
print(f"\nМатрица поворота R (правая камера относительно левой):\n{R}")
print(f"\nВектор сдвига T (мм):\n{T.ravel()}")
print(f"\nБазис стереосистемы: {np.linalg.norm(T):.1f} мм")

# ================= 3. СТЕРЕОРЕКТИФИКАЦИЯ =================
print("\n" + "=" * 60)
print("СТЕРЕОРЕКТИФИКАЦИЯ")
print("=" * 60)

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1_new, dist1_new, K2_new, dist2_new,
    grayL.shape[::-1], R, T, alpha=0
)

print(f"Матрица репроекции Q:\n{np.array2string(Q, precision=3, suppress_small=True)}")

# ================= 4. СОХРАНЕНИЕ ПАРАМЕТРОВ =================
np.savez("stereo_params.npz",
         K1=K1_new, dist1=dist1_new,
         K2=K2_new, dist2=dist2_new,
         R=R, T=T,
         R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
         roi1=roi1, roi2=roi2
         )

print("\nПараметры сохранены в 'stereo_params.npz'")

# ================= 5. ВИЗУАЛИЗАЦИЯ РЕКТИФИКАЦИИ =================
# Загружаем тестовую пару
imgL_test = cv2.imread(left_images[0])
imgR_test = cv2.imread(right_images[0])

# Карты преобразования для ректификации
map1x, map1y = cv2.initUndistortRectifyMap(K1_new, dist1_new, R1, P1, grayL.shape[::-1], cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2_new, dist2_new, R2, P2, grayR.shape[::-1], cv2.CV_32FC1)

# Применяем ректификацию
imgL_rect = cv2.remap(imgL_test, map1x, map1y, cv2.INTER_LINEAR)
imgR_rect = cv2.remap(imgR_test, map2x, map2y, cv2.INTER_LINEAR)

# Рисуем горизонтальные линии для проверки
for i in range(0, imgL_rect.shape[0], 50):
    cv2.line(imgL_rect, (0, i), (imgL_rect.shape[1], i), (0, 255, 0), 1)
    cv2.line(imgR_rect, (0, i), (imgR_rect.shape[1], i), (0, 255, 0), 1)

combined_rect = np.hstack((imgL_rect, imgR_rect))
cv2.imwrite("rectified_pair.jpg", combined_rect)
print("Ректифицированная пара сохранена в 'rectified_pair.jpg'")

# ================= 6. DISPARITY MAP =================
print("\n" + "=" * 60)
print("ПОСТРОЕНИЕ КАРТЫ ГЛУБИНЫ")
print("=" * 60)

# Конвертируем в grayscale
grayL_rect = cv2.cvtColor(imgL_rect, cv2.COLOR_BGR2GRAY)
grayR_rect = cv2.cvtColor(imgR_rect, cv2.COLOR_BGR2GRAY)

# Настройки StereoSGBM
window_size = 7
min_disp = 0
num_disp = 128 - min_disp

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

print("Вычисляем disparity map...")
disparity = stereo.compute(grayL_rect, grayR_rect).astype(np.float32) / 16.0

# Нормализуем для визуализации
disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

cv2.imwrite("disparity_map.jpg", disp_vis)
print("Карта глубины сохранена в 'disparity_map.jpg'")

# ================= 7. ВОССТАНОВЛЕНИЕ 3D =================
print("\n" + "=" * 60)
print("ВОССТАНОВЛЕНИЕ 3D ТОЧЕК")
print("=" * 60)

points_3D = cv2.reprojectImageTo3D(disparity, Q)

# Пример: расстояние до центральной точки
h, w = disparity.shape
center_x, center_y = w // 2, h // 2
if disparity[center_y, center_x] > 0:
    Z = points_3D[center_y, center_x][2]
    print(f"Расстояние до центральной точки: {Z / 10:.1f} см")
else:
    print("В центральной точке нет данных о глубине")

# Сохраняем результаты
np.savez("stereo_results.npz",
         disparity=disparity,
         points_3D=points_3D
         )

print("\n" + "=" * 60)
print("СТЕРЕОКАЛИБРОВКА ЗАВЕРШЕНА!")
print("=" * 60)
print("Созданы файлы:")
print("1. stereo_params.npz - все параметры калибровки")
print("2. rectified_pair.jpg - ректифицированные изображения")
print("3. disparity_map.jpg - карта глубины")
print("4. stereo_results.npz - disparity и 3D точки")