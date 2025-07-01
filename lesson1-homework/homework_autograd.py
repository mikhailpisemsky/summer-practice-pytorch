import torch

print("--Задание 2: Автоматическое дифференцирование--")
print("__2.1 Простые вычисления с градиентами__")
# Создайте тензоры x, y, z с requires_grad=True
x = torch.tensor(9, dtype=torch.float32, requires_grad=True)
y = torch.tensor(13, dtype=torch.float32, requires_grad=True)
z = torch.tensor(7, dtype=torch.float32, requires_grad=True)
print("Переменные:", f"x={x}, y={y}, z={z}", sep="\n")
# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f = x**2 + y**2 + z**2 + 2*x*y*z
# Найдите градиенты по всем переменным
f.backward()
dx = x.grad
dy = y.grad
dz = z.grad
print("Градиенты по всем переменным:", f"df/dx={dx}, df/dy={dy}, df/dz={dz}", sep="\n")
# Проверьте результат аналитически
# df/dx=2x+2yz=200
# df/dy=2y+2xz=152
# df/dz=2z+2xy=248
print()

print("__2.2 Градиент функции потерь__")
# Реализуйте функцию MSE (Mean Squared Error):
# MSE = (1/n) * Σ(y_pred - y_true)^2
def mse(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)
x = torch.randn(10, requires_grad=False)
print(f"x={x}")
y_true = 2*x + torch.randn(10) * 0.1
w = torch.tensor(0, dtype=torch.float32 ,requires_grad=True)
b = torch.tensor(0, dtype=torch.float32 ,requires_grad=True)
# где y_pred = w * x + b (линейная функция)
y_pred = w * x + b
loss = mse(y_pred, y_true)
loss.backward()
# Найдите градиенты по w и b
print("Градиент функции потерь:", f"dMSE/dw={w.grad}, dMSE/db={b.grad}", sep="\n")
print()

print("__2.3 Цепное правило__")
# Реализуйте составную функцию: f(x) = sin(x^2 + 1)
x = torch.tensor(7, dtype=torch.float32, requires_grad=True)
f = torch.sin(x**2 + 1)
f.backward()
# Найдите градиент df/dx
print(f"Градиент df/dx={x.grad}")
# Проверьте результат с помощью torch.autograd.grad
grad = torch.autograd.grad(torch.sin(x**2 + 1), x)[0]
print(f"Значение градиента через torch.autograd.grad:", grad.item())
