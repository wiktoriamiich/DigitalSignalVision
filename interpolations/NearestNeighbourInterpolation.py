import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

#generowanie wartosci x od 0 do 2pi
x = np.linspace(0, 2 * np.pi, 1000)  # Domyślnie 100 punktów
y = np.sin(x)

x_wezly = np.linspace(0, 2 * np.pi, 10)
y_wezly = np. sin(x_wezly)

points = ax.scatter(x_wezly, y_wezly, marker="o")
curve = ax.plot(x,y)

def nearest_neighbor_interpolation(x, x_wezly, y_wezly):
    y = np.zeros_like(x)
    for i, xi in enumerate(x):
        # Znajdź indeks najbliższego węzła
        idx = np.argmin(np.abs(x_wezly - xi))
        y[i] = y_wezly[idx]
    return y

interpolated_y_nn = nearest_neighbor_interpolation(x, x_wezly, y_wezly)


plt.plot(x, y, label='sin(x)', color='blue')
plt.scatter(x_wezly, y_wezly, color='red', marker='o', label='Węzły interpolacji')
plt.plot(x, interpolated_y_nn,'-', label='Interpolacja NN', color='purple')
plt.title('Interpolacja metodą najbliższego sąsiada')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
