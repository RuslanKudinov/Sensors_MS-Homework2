import cv2
import numpy as np

# ================= 1. ЗАГРУЗКА ПАРАМЕТРОВ КАЛИБРОВКИ =================
data = np.load("camera_params.npz")
K = data['K']
dist = data['dist']
print("Матрица K:\n", K)
print("Коэффициенты дисторсии:\n", dist)

# ================= 2. ЗАГРУЗКА ИЗОБРАЖЕНИЯ =================
img_path = "card.jpg"  # <-- замени на своё имя файла
img = cv2.imread(img_path)
if img is None:
    print("Ошибка загрузки изображения!")
    exit()

# Убираем дисторсию
h, w = img.shape[:2]
newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
undistorted = cv2.undistort(img, K, dist, None, newK)

# ================= 3. НАХОЖДЕНИЕ ДОСКИ =================
CHESSBOARD_SIZE = (9, 6)  # внутренние углы
SQUARE_SIZE = 30.0  # мм

gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

if not ret:
    print("Шахматная доска не найдена!")
    exit()

# Уточняем углы
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

# Отображаем углы доски
img_with_corners = undistorted.copy()
cv2.drawChessboardCorners(img_with_corners, CHESSBOARD_SIZE, corners, ret)
cv2.imwrite("board_corners.jpg", img_with_corners)
print("Углы доски найдены и сохранены в board_corners.jpg")

# ================= 4. ПОЛОЖЕНИЕ КАМЕРЫ ОТНОСИТЕЛЬНО ДОСКИ =================
# 3D точки доски в мм (плоскость Z=0)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# solvePnP для получения rvec, tvec
ret, rvec, tvec = cv2.solvePnP(objp, corners, newK, None)
print("\nПоложение камеры найдено")
print("rvec (радианы):\n", rvec.ravel())
print("tvec (мм):\n", tvec.ravel())

# Преобразуем rvec в матрицу поворота
R, _ = cv2.Rodrigues(rvec)


# ================= 5. РУЧНАЯ РАЗМЕТКА УГЛОВ КАРТЫ =================
def select_points(image, title="Выберите 4 угла карты (по часовой стрелке)"):
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image, (x, y), 7, (0, 255, 0), -1)
            cv2.imshow(title, image)

    cv2.imshow(title, image)
    cv2.setMouseCallback(title, click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return np.array(points, dtype=np.float32)


print("\n=== РУЧНАЯ РАЗМЕТКА ===")
print("Щёлкни по 4 углам карты-пропуска (по часовой стрелке, начиная с левого верхнего)")
img_for_selection = undistorted.copy()
pixel_points = select_points(img_for_selection, "Выберите 4 угла карты")

if len(pixel_points) != 4:
    print("Нужно выбрать ровно 4 точки!")
    exit()

# ================= 6. ПРЕОБРАЗОВАНИЕ ПИКСЕЛЕЙ В 3D =================
card_points_3d = []
for uv in pixel_points:
    # 1. Нормализованные координаты
    uv_homo = np.array([uv[0], uv[1], 1.0])
    K_inv = np.linalg.inv(newK)
    ray_cam = K_inv @ uv_homo  # луч в системе камеры

    # 2. Преобразуем луч в систему координат доски
    ray_world = R.T @ ray_cam  # поворот обратно в систему доски

    # 3. Положение камеры в системе доски
    tvec_world = -R.T @ tvec.ravel()

    # 4. Луч в системе доски: X = tvec_world + λ * ray_world
    # Пересечение с плоскостью Z=0: (tvec_world_z + λ * ray_world_z) = 0
    lambda_val = -tvec_world[2] / ray_world[2]

    # 5. 3D точка в системе доски
    point_3d = tvec_world + lambda_val * ray_world
    card_points_3d.append(point_3d)

card_points_3d = np.array(card_points_3d)  # в мм, в системе координат доски

# ================= 7. ВЫЧИСЛЕНИЕ РАЗМЕРОВ =================
# Порядок точек: [левый верхний, правый верхний, правый нижний, левый нижний]
width1 = np.linalg.norm(card_points_3d[0] - card_points_3d[1])  # верхняя сторона
width2 = np.linalg.norm(card_points_3d[3] - card_points_3d[2])  # нижняя сторона
height1 = np.linalg.norm(card_points_3d[0] - card_points_3d[3])  # левая сторона
height2 = np.linalg.norm(card_points_3d[1] - card_points_3d[2])  # правая сторона

avg_width = (width1 + width2) / 2.0
avg_height = (height1 + height2) / 2.0

print("\n" + "=" * 50)
print("РЕЗУЛЬТАТЫ ИЗМЕРЕНИЯ КАРТЫ")
print("=" * 50)
print(f"Ширина карты: {avg_width:.1f} мм")
print(f"Высота карты: {avg_height:.1f} мм")

# Размер карты ~ 85.6 × 54.0 мм (ID-1)
real_width = 85
real_height = 55
print(f"\nОжидаемый размер: {real_width} × {real_height} мм")

# ================= 8. ВИЗУАЛИЗАЦИЯ =================
result_img = undistorted.copy()
for i, pt in enumerate(pixel_points):
    cv2.circle(result_img, tuple(pt.astype(int)), 8, (0, 255, 0), -1)
    cv2.putText(result_img, str(i + 1), tuple(pt.astype(int) + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

cv2.polylines(result_img, [pixel_points.astype(int)], True, (0, 0, 255), 3)

# Добавляем текст с размерами
cv2.putText(result_img, f"Width: {avg_width:.1f} mm", (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
cv2.putText(result_img, f"Height: {avg_height:.1f} mm", (50, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

cv2.imwrite("measured_card.jpg", result_img)
print("\nРезультат сохранён в 'measured_card.jpg'")

cv2.imshow("Result", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()