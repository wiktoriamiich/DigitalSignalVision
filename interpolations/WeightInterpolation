import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# generowanie wartosci x od 0 do 2pi
x = np.linspace(0, 2 * np.pi, 1000)  # Domyślnie 100 punktów
y = np.sin(x)

# generowanie wezlow
x_wezly = np.linspace(0, 2 * np.pi, 10)
y_wezly = np.sin(x_wezly)

points = ax.scatter(x_wezly, y_wezly, marker="o")
curve = ax.plot(x, y)


def average_interpolation(x, x_wezly, y_wezly):
    yi = np.zeros_like(x)
    for i, xi in enumerate(x):
            idx = np.argsort(np.abs(x_wezly - xi))
            idx_A, idx_B = idx[0], idx[1]

            distance_A_B = np.abs(x_wezly[idx_B] - x_wezly[idx_A])
            distance_B_x = np.abs(xi - x_wezly[idx_B])
            distance_A_x = np.abs(xi - x_wezly[idx_A])
            wagaA = distance_B_x / distance_A_B
            wagaB = distance_A_x / distance_A_B

            if xi == x_wezly[idx_A]:
                # Gdy xi jest równy jednemu z x_nodes
                yi[i] = y_wezly[idx_A]
            else:
                # Obliczanie odległości i wag
                yi[i] = wagaA * y_wezly[idx_A] + wagaB * y_wezly[idx_B]
    return yi


interpolated_y = average_interpolation(x, x_wezly, y_wezly)

plt.plot(x, y, label='sin(x)', color='blue')
plt.show()# Oryginalna funkcja
plt.scatter(x_wezly, y_wezly, color='red', marker='o', label='Węzły interpolacji')  # Węzły
plt.plot(x, interpolated_y, label='Interpolacja', color='green')  # Wynik interpolacji

# Dodanie legendy, tytułu i etykiet
plt.title('Interpolacja za pomocą średniej ważonej')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Wyświetlenie wykresu
plt.show()
