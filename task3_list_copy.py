"""Задание 3: ссылка vs копия списка."""

print("--- Ссылка на тот же объект ---")
x = [1, 2, 3]
y = x                     # ссылка на тот же объект
print(f"id(x) = {id(x)}")
print(f"id(y) = {id(y)}")
print(f"x is y: {x is y}")   # True — это один и тот же объект

y.append(4)
print(f"После y.append(4): x = {x}, y = {y}")
# Оба изменились!

print("\n--- Копия списка ---")
a = [1, 2, 3]
b = a.copy()              # создаём НОВЫЙ объект
print(f"id(a) = {id(a)}")
print(f"id(b) = {id(b)}")
print(f"a is b: {a is b}")   # False — разные объекты

b.append(4)
print(f"После b.append(4): a = {a}, b = {b}")
# a не изменилось!

print("\n--- Копия среза ---")
c = [1, 2, 3]
d = c[:]                  # ещё один способ копирования
d.append(99)
print(f"c = {c}, d = {d}")
