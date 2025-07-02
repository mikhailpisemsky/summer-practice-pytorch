import torch
import time

print("--Задание 3: Сравнение производительности CPU vs CUDA--")
# 3.1 Подготовка данных
# Создайте большие матрицы размеров:
# - 64 x 1024 x 1024
matrix_64x1024x1024 = torch.randn(64, 1024, 1024)
# - 128 x 512 x 512
matrix_128x512x512 = torch.randn(128, 512, 512)
# - 256 x 256 x 256
matrix_256x256x256 = torch.randn(256, 256, 256)
# Заполните их случайными числами
print()

# 3.2 Функция измерения времени
device1 = torch.device("cuda")
device2 = torch.device("cpu")
# Создайте функцию для измерения времени выполнения операций
def measure_time(operation, x_cuda, x_cpu):
    # Используйте torch.cuda.Event() для точного измерения на GPU
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        operation(x_cuda)
        end.record()
        torch.cuda.synchronize()
        cuda = start.elapsed_time(end)
    else:
        cuda = None

    # Используйте time.time() для измерения на CPU
    start = time.time()
    operation(x_cpu)
    cpu = (time.time() - start) * 1000

    return cpu, cuda

#3.3 Сравнение операций
# Сравните время выполнения следующих операций на CPU и CUDA:
# - Матричное умножение (torch.matmul)
# - Поэлементное сложение
# - Поэлементное умножение
# - Транспонирование
# - Вычисление суммы всех элементов

# Для каждой операции:
# 1. Измерьте время на CPU
# 2. Измерьте время на GPU (если доступен)
# 3. Вычислите ускорение (speedup)
# 4. Выведите результаты в табличном виде

operations = {
    "Матричное умножение": lambda x: torch.matmul(x, x.transpose(-1, -2)),
    "Сложение": lambda x: x+x,
    "Умножение": lambda x: x*x,
    "Транспонирование": lambda x: x.T,
    "Сумма элементов": lambda x: x.sum(),
}

matrix_64x1024x1024_cuda = matrix_64x1024x1024.to(device1)
matrix_64x1024x1024_cpu = matrix_64x1024x1024.to(device2)
matrix_128x512x512_cuda = matrix_128x512x512.to(device1)
matrix_128x512x512_cpu = matrix_128x512x512.to(device2)
matrix_256x256x256_cuda = matrix_256x256x256.to(device1)
matrix_256x256x256_cpu = matrix_256x256x256.to(device2)

matrices = [(matrix_64x1024x1024_cuda, matrix_64x1024x1024_cpu),
 (matrix_128x512x512_cuda, matrix_128x512x512_cpu),
  (matrix_256x256x256_cuda, matrix_256x256x256_cpu)]

print(f"{'Операция':20} | {'CPU (мс)':>8} | {'GPU (мс)':>8} | {'Ускорение':>10}")

for matrix in matrices:
  for name, function in operations.items():
      cpu, gpu = measure_time(function, matrix[0], matrix[1])
      speedup = (cpu / gpu) if gpu else None
      print(f"{name:20} | {cpu:8.1f} | {gpu or 0:8.1f} | {speedup or '-':>10}")
