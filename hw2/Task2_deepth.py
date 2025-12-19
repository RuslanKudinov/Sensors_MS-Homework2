import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("ПОСТРОЕНИЕ КАРТЫ ГЛУБИНЫ ДЛЯ СТЕРЕОПАРЫ")
print("=" * 60)

# ================= 1. ЗАГРУЗКА ПАРАМЕТРОВ =================
try:
    data = np.load("stereo_params.npz")
    print("Параметры стереокалибровки загружены")
except:
    print("ОШИБКА: Файл stereo_params.npz не найден!")
    print("Сначала выполни стереокалибровку (stereo_calibration.py)")
    exit()

# Извлекаем параметры
K1 = data['K1']
dist1 = data['dist1']
K2 = data['K2']
dist2 = data['dist2']
R1 = data['R1']
R2 = data['R2']
P1 = data['P1']
P2 = data['P2']
Q = data['Q']

print(f"Размер изображений при калибровке: {data['roi1'] if 'roi1' in data else 'не указан'}")

# ================= 2. ЗАГРУЗКА СТЕРЕОПАРЫ =================
imgL = cv2.imread("Andrew.jpg")
imgR = cv2.imread("Andrew2.jpg")

if imgL is None or imgR is None:
    print("ОШИБКА: Не удалось загрузить Andrew.jpg или Andrew2.jpg")
    print("Убедись, что файлы находятся в той же папке, что и скрипт")
    exit()

print(f"Левый снимок: {imgL.shape}")
print(f"Правый снимок: {imgR.shape}")

# Если изображения слишком большие, уменьшим для скорости
if imgL.shape[1] > 1200:
    scale = 1200 / imgL.shape[1]
    new_width = int(imgL.shape[1] * scale)
    new_height = int(imgL.shape[0] * scale)
    imgL = cv2.resize(imgL, (new_width, new_height))
    imgR = cv2.resize(imgR, (new_width, new_height))
    print(f"Изображения уменьшены до: {imgL.shape}")

h, w = imgL.shape[:2]

# ================= 3. РЕКТИФИКАЦИЯ =================
print("\nРектификация изображений...")
map1x, map1y = cv2.initUndistortRectifyMap(K1, dist1, R1, P1, (w, h), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, dist2, R2, P2, (w, h), cv2.CV_32FC1)

imgL_rect = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
imgR_rect = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

# Сохраняем ректифицированные изображения для проверки
cv2.imwrite("rectified_left.jpg", imgL_rect)
cv2.imwrite("rectified_right.jpg", imgR_rect)

# ================= 4. ПОСТРОЕНИЕ DISPARITY MAP =================
print("Построение карты смещений (disparity map)...")
grayL = cv2.cvtColor(imgL_rect, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR_rect, cv2.COLOR_BGR2GRAY)

# Настройки для SGBM (подбираются экспериментально)
window_size = 5
min_disp = 0
num_disp = 128  # должно делиться на 16

# Создаем StereoSGBM объект
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=5,
    uniquenessRatio=15,
    speckleWindowSize=50,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Вычисляем disparity
disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

# Убираем шум (значения < min_disp)
disparity[disparity < min_disp] = min_disp

print(f"Disparity map: {disparity.shape}, min={disparity.min():.1f}, max={disparity.max():.1f}")

# ================= 5. ПРЕОБРАЗОВАНИЕ В ГЛУБИНУ =================
print("Преобразование disparity в глубину (расстояние)...")
# Используем матрицу Q для преобразования disparity в 3D
points_3D = cv2.reprojectImageTo3D(disparity, Q)

# Глубина = Z-координата (расстояние от камеры)
depth = points_3D[:, :, 2]

# Заменяем отрицательные и нулевые значения
depth[depth <= 0] = np.nan

print(f"Глубина: min={np.nanmin(depth):.1f} мм, max={np.nanmax(depth):.1f} мм")

# ================= 6. ВИЗУАЛИЗАЦИЯ КАРТЫ ГЛУБИНЫ =================
print("Создание визуализации карты глубины...")

# Вариант 1: Псевдоцветная карта глубины (лучше для анализа)
depth_normalized = cv2.normalize(depth, None, alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

# Заменяем черные области (нет данных) на серый
depth_colored[depth_normalized == 0] = [128, 128, 128]

# Вариант 2: Градации серого (просто глубина)
depth_gray = cv2.normalize(depth, None, alpha=0, beta=255,
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# ================= 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ =================
# Основной результат - цветная карта глубины
cv2.imwrite("deepth_stereo.jpg", depth_colored)
print("Карта глубины сохранена как 'deepth_stereo.jpg'")

# Дополнительные файлы для анализа
cv2.imwrite("depth_gray.jpg", depth_gray)
cv2.imwrite("disparity_map.jpg",
            cv2.normalize(disparity, None, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))

# ================= 8. ИНФОРМАЦИЯ О ГЛУБИНЕ =================
print("\n" + "=" * 60)
print("ИНФОРМАЦИЯ О КАРТЕ ГЛУБИНЫ")
print("=" * 60)

# Средняя глубина (без учета NaN)
valid_depth = depth[~np.isnan(depth)]
if len(valid_depth) > 0:
    print(f"Средняя глубина: {np.mean(valid_depth) / 10:.1f} см")
    print(f"Минимальная глубина: {np.min(valid_depth) / 10:.1f} см")
    print(f"Максимальная глубина: {np.max(valid_depth) / 10:.1f} см")

    # Глубина в центре кадра
    center_y, center_x = h // 2, w // 2
    center_depth = depth[center_y, center_x]
    if not np.isnan(center_depth):
        print(f"Глубина в центре кадра: {center_depth / 10:.1f} см")
    else:
        print("В центре кадра нет данных о глубине")
else:
    print("Нет валидных данных о глубине!")

# ================= 9. ВИЗУАЛИЗАЦИЯ =================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Исходные изображения
axes[0, 0].imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Andrew.jpg (left)")
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Andrew2.jpg (right)")
axes[0, 1].axis('off')

# Ректифицированные
axes[0, 2].imshow(cv2.cvtColor(imgL_rect, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Rectified left")
axes[0, 2].axis('off')

# Disparity
disp_vis = axes[1, 0].imshow(disparity, cmap='jet')
axes[1, 0].set_title("Disparity Map")
axes[1, 0].axis('off')
plt.colorbar(disp_vis, ax=axes[1, 0], fraction=0.046, pad=0.04)

# Карта глубины (цветная)
axes[1, 1].imshow(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Depth Map (colored)")
axes[1, 1].axis('off')

# Гистограмма глубины
axes[1, 2].hist(valid_depth / 10, bins=50, edgecolor='black')
axes[1, 2].set_title("Depth Histogram")
axes[1, 2].set_xlabel("Depth (cm)")
axes[1, 2].set_ylabel("Frequency")
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("depth_analysis.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("ВСЕ ФАЙЛЫ СОХРАНЕНЫ:")
print("=" * 60)
print("1. deepth_stereo.jpg - цветная карта глубины (основной результат)")
print("2. depth_gray_stereo.jpg - карта глубины в градациях серого")
print("3. disparity_map_stereo.jpg - карта смещений (disparity)")
print("4. rectified_left/right_stereo.jpg - ректифицированные изображения")
print("5. depth_analysis_stereo.png - полный анализ с графиками")
print("\nГотово!")