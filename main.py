# Модель: Власний код (узгоджено з викладачем)
# Автор: Шарапов Валерій Валерійович, група АІ-235

class GradientDescentOptimizer:
    """
    Клас для знаходження локального мінімуму функції
    за допомогою методу градієнтного спуску.
    """
    def __init__(self, learning_rate=0.1, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize(self, func, grad_func, start_x):
        x = start_x
        history = [(x, func(x))]

        for i in range(self.max_iterations):
            grad = grad_func(x)
            new_x = x - self.learning_rate * grad

            # Перевірка критерію зупинки (якщо крок став занадто малим)
            if abs(new_x - x) < self.tolerance:
                print(f"[*] Збіжність досягнута на ітерації {i+1}")
                x = new_x
                history.append((x, func(x)))
                break

            x = new_x

            # Зберігаємо історію кожні 10 ітерацій для компактності
            if (i + 1) % 10 == 0:
                history.append((x, func(x)))

        return x, history

if __name__ == "__main__":
    # Цільова функція f(x) = x^2 - 4x + 4 (парабола з мінімумом у точці x = 2)
    def objective_function(x):
        return x**2 - 4*x + 4

    # Похідна цільової функції f'(x) = 2x - 4
    def gradient_function(x):
        return 2*x - 4

    # Налаштування оптимізатора
    optimizer = GradientDescentOptimizer(learning_rate=0.15, max_iterations=100)
    start_point = 0.0

    print("--- Власний код: Градієнтний спуск ---")
    print(f"Початкова точка: x = {start_point}\n")

    # Запуск процесу оптимізації
    min_x, steps = optimizer.optimize(objective_function, gradient_function, start_point)

    print("\n--- Результати ---")
    print(f"Знайдений мінімум: x = {min_x:.6f}")
    print(f"Значення функції в мінімумі: f(x) = {objective_function(min_x):.6f}")
    print(f"Кількість збережених кроків в історії: {len(steps)}")