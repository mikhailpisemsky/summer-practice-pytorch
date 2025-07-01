import torch

print("--Задание 1: Создание и манипуляции с тензорами--")
print("__1.1 Создание тензоров__")
# Создайте следующие тензоры:
# - Тензор размером 3x4, заполненный случайными числами от 0 до 1
tensor1 = torch.rand(3, 4, dtype=torch.float32)
# - Тензор размером 2x3x4, заполненный нулями
tensor2 = torch.zeros(2, 3, 4, dtype=torch.int32)
# - Тензор размером 5x5, заполненный единицами
tensor3 = torch.ones(5, 5, dtype=torch.int32)
# - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
tensor4 = torch.arange(16, dtype=torch.int32).reshape(4, 4)

print("Тензор размером 3x4, заполненный случайными числами от 0 до 1:", tensor1, sep="\n")
print("Тензор размером 2x3x4, заполненный нулями:", tensor2, sep="\n")
print("Тензор размером 5x5, заполненный единицами:", tensor3, sep="\n")
print("Тензор размером 4x4 с числами от 0 до 15:", tensor4, sep="\n")
print()

print("__1.2. Операции с тензорами__")
# Дано: тензор A размером 3x4 и тензор B размером 4x3
A = torch.rand(3, 4)
B = torch.rand(4, 3)
# Выполните:
# - Транспонирование тензора A
A_transposed = A.T
# - Матричное умножение A и B
A_B_matrix_multiplication = torch.matmul(A, B)
# - Поэлементное умножение A и транспонированного B
A_B_element_multiplication = A * B.T
# - Вычислите сумму всех элементов тензора A
A_element_sum = A.sum()

print("Тензор A размером 3x4:", A, sep="\n")
print("Тензор B размером 4x3:", B, sep="\n")
print("Транспанированный тензор A:", A_transposed, sep="\n")
print("Матричное умножение A и B:", A_B_matrix_multiplication, sep="\n")
print("Поэлементное умножение A и транспонированного B:", A_B_element_multiplication, sep="\n")
print("Сумма всех элементов тензора A:", A_element_sum, sep="\n")
print()

print("__1.3 Индексация и срезы__")
# Создайте тензор размером 5x5x5
tensor5 = torch.rand(5, 5, 5)
# Извлеките:
# - Первую строку
first_row = tensor5[0, 0, :]
# - Последний столбец
last_column = tensor5[:, :, -1]
# - Подматрицу размером 2x2 из центра тензора
center_matrix = tensor5[2:4, 2:4, 2:4]
# - Все элементы с четными индексами
even_index = tensor5[::2, ::2, ::2]

print("Тензор размером 5x5x5:", tensor5, sep="\n")
print("Первая строка:", first_row, sep="\n")
print("Последний столбец:", last_column, sep="\n")
print("Подматрица размером 2x2 из центра тензора:", center_matrix, sep="\n")
print("Все элементы с четными индексами:", even_index, sep="\n")
print()

print("__1.4 Работа с формами__")
# Создайте тензор размером 24 элемента
tensor6 = torch.rand(24)
# Преобразуйте его в формы:
# - 2x12
tensor6_2x12 = tensor6.view(2, 12)
# - 3x8
tensor6_3x8 = tensor6.view(3, 8)
# - 4x6
tensor6_4x6 = tensor6.view(4, 6)
# - 2x3x4
tensor6_2x3x4 = tensor6.view(2, 3, 4)
# - 2x2x2x3
tensor6_2x2x2x3 = tensor6.view(2, 2, 2, 3)

print("Тензор размером 24 элемента", tensor6, sep="\n")
print("Тензор в формате 2x12:", tensor6_2x12, sep="\n")
print("Тензор в формате 3x8:", tensor6_3x8, sep="\n")
print("Тензор в формате 4x6:", tensor6_4x6, sep="\n")
print("Тензор в формате 2x3x4:", tensor6_2x3x4, sep="\n")
print("Тензор в формате 2x2x2x3:", tensor6_2x2x2x3, sep="\n")