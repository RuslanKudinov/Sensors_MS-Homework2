import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# Устанавливаем trust_repo=True чтобы избежать предупреждений
model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
model.eval()

# Используем GPU если есть
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используем устройство: {device}")
model.to(device)

# Получаем трансформации
midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
transform = midas_transforms.small_transform

# Загружаем изображение (поменяй путь)
img = cv2.imread('Andrew.jpg')
if img is None:
    print("Ошибка: не могу загрузить изображение!")
    print("Проверь: 1) Файл существует 2) Правильный путь 3) Расширение .jpg/.png")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"Изображение загружено: {img.shape[1]}x{img.shape[0]}")

# Применяем трансформацию КОРРЕКТНО
input_batch = transform(img_rgb).to(device)

# Убираем лишнюю размерность если есть (фикс ошибки)
if input_batch.dim() == 5:
    input_batch = input_batch.squeeze(0)  # [1, C, H, W] -> [C, H, W]
elif input_batch.dim() == 3:
    input_batch = input_batch.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

print(f"Размер входного тензора: {input_batch.shape}")

# Получаем карту глубины
with torch.no_grad():
    prediction = model(input_batch)

    # Интерполируем к исходному размеру изображения
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),  # добавляем размерность канала
        size=img.shape[:2],
        mode='bicubic',
        align_corners=False
    ).squeeze()  # убираем лишние размерности

# Конвертируем в numpy
depth_map = prediction.cpu().numpy()

# Нормализуем для отображения
depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

# Визуализация
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title('Оригинал')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(depth_normalized, cmap='plasma')
plt.title('Цветная карта глубины')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(depth_normalized, cmap='gray')
plt.title('Черно-белая карта глубины')
plt.axis('off')

plt.tight_layout()
plt.show()

# Сохраняем результат
# 1. Сохраняем оригинал
cv2.imwrite('original.jpg', img)

# 2. Сохраняем цветную карту глубины
depth_colored = (depth_normalized * 255).astype(np.uint8)
depth_colored_color = cv2.applyColorMap(depth_colored, cv2.COLORMAP_PLASMA)
cv2.imwrite('depth_plasma.jpg', depth_colored_color)

# 3. Сохраняем черно-белую карту глубины
depth_gray = (depth_normalized * 255).astype(np.uint8)
cv2.imwrite('depth_gray.jpg', depth_gray)

print("\n✅ Готово! Сохранены файлы:")
print("   original.jpg - исходное изображение")
print("   depth_plasma.jpg - цветная карта глубины")
print("   depth_gray.jpg - черно-белая карта глубины")
