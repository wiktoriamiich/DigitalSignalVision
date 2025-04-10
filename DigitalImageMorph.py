import cv2
import numpy as np

image = cv2.imread('piesek.jpg')
wysokosc, szerokosc, c = image.shape

srednia = np.ones((2, 2), dtype=np.float32) / 4.0
zmniejszony_obraz = np.zeros((wysokosc // 2, szerokosc // 2, 3), dtype=np.uint8)
powiekszony_obraz = np.zeros((wysokosc, szerokosc, 3), dtype=np.uint8)

for i in range(0, wysokosc - 1, 2):
    for j in range(0, szerokosc - 1, 2):
        obszar = image[i:i + 2, j:j + 2]
        zmniejszony_obraz[i // 2, j // 2] = image[i, j]

# Copy pixels to the bigger image
for k in range(wysokosc // 2):
    for l in range(szerokosc // 2):
        powiekszony_obraz[k * 2][l * 2] = zmniejszony_obraz[k][l]

# Interpolate using nearest neigbor method

cv2.imwrite('../bigger_image1.png', powiekszony_obraz)

# Approximate row by row
for k in range(0, wysokosc - 1, 2):
    for l in range(1, szerokosc - 1, 2):
        for kolor in range(3):
            powiekszony_obraz[k][l][kolor] = powiekszony_obraz[k][l - 1][kolor] / 2 + powiekszony_obraz[k][l + 1][
                kolor] / 2

# Approximate column by column

for l in range(0, szerokosc - 1):
    for k in range(1, wysokosc - 1, 2):
        for kolor in range(3):
            powiekszony_obraz[k][l][kolor] = powiekszony_obraz[k - 1][l][kolor] / 2 + powiekszony_obraz[k + 1][l][
                kolor] / 2

a1 = 45
a_rad = np.radians(a1)
a2 = -45
srodek_obrazu = (wysokosc / 2, szerokosc / 2)

szerokosc2 = int(szerokosc * np.sin(a_rad) + wysokosc * np.cos(a_rad))
wysokosc2 = int(wysokosc * np.sin(a_rad) + szerokosc * np.cos(a_rad))

# macierz_obrotu1 = cv2.getRotationMatrix2D(srodek_obrazu, a1, 1)
# macierz_obrotu2 = cv2.getRotationMatrix2D(srodek_obrazu, a2, 1)

def rotate_image(image, angle):
    # Wymiary obrazu i środek
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2

    # Macierz obrotu
    cos_theta = np.cos(np.radians(angle))
    sin_theta = np.sin(np.radians(angle))
    rotation_matrix = np.array([[cos_theta, -sin_theta, cx - cx*cos_theta + cy*sin_theta],
                                [sin_theta, cos_theta, cy - cx*sin_theta - cy*cos_theta]])

    # Tworzenie nowego obrazu
    rotated_image = np.zeros_like(image)

    # Iteracja po pikselach nowego obrazu i wypełnianie wartości z oryginalnego obrazu
    for y in range(h):
        for x in range(w):
            new_x, new_y = np.dot(rotation_matrix, np.array([x, y, 1])).astype(int)
            if 0 <= new_x < w and 0 <= new_y < h:
                rotated_image[y, x] = image[new_y, new_x]

    return rotated_image

#obrocony_obraz1 = cv2.warpAffine(image, macierz_obrotu1, (szerokosc, wysokosc))
#obrocony_obraz2 = cv2.warpAffine(obrocony_obraz1, macierz_obrotu2, (szerokosc, wysokosc))
obrocony_obraz1 = rotate_image(image, 45)
obrocony_obraz2= rotate_image(obrocony_obraz1,45)

sum_b1 = sum_g1 = sum_r1 = 0
for row in range(wysokosc):
    for column in range(szerokosc):
        sum_b1 += powiekszony_obraz[row][column][0]
        sum_g1 += powiekszony_obraz[row][column][1]
        sum_r1 += powiekszony_obraz[row][column][2]

mean11 = (sum_b1 + sum_g1 + sum_r1) / (500 * 500 * 3)

for row in range(wysokosc):
    for column in range(szerokosc):
        sum_b1 += image[row][column][0]
        sum_g1 += image[row][column][1]
        sum_r1 += image[row][column][2]

mean21 = (sum_b1 + sum_g1 + sum_r1) / (500 * 500 * 3)

print(mean21 - mean11)

roznica1 = obrocony_obraz2 - image

sum_b2 = sum_g2 = sum_r2 = 0
for row in range(wysokosc):
    for column in range(szerokosc):
        sum_b2 += obrocony_obraz1[row][column][0]
        sum_g2 += obrocony_obraz1[row][column][1]
        sum_r2 += obrocony_obraz1[row][column][2]

mean12 = (sum_b2 + sum_g2 + sum_r2) / (500 * 500 * 3)

for row in range(wysokosc):
    for column in range(szerokosc):
        sum_b2 += obrocony_obraz2[row][column][0]
        sum_g2 += obrocony_obraz2[row][column][1]
        sum_r2 += obrocony_obraz2[row][column][2]

mean22 = (sum_b2 + sum_g2 + sum_r2) / (500 * 500 * 3)

print(mean22 - mean11)

roznica2 = obrocony_obraz2 - image

cv2.imshow('Oryginalny obraz', image)
cv2.imshow('Zmniejszony obraz', zmniejszony_obraz)
cv2.imshow('Powiekszony_obraz', powiekszony_obraz)
cv2.imshow('Obrocony obraz o 45', obrocony_obraz1)
cv2.imshow('Obrocony obraz o -45', obrocony_obraz2)
cv2.imshow('Roznica przy powiekszaniu', roznica1)
cv2.imshow('Roznica przy obracaniu', roznica2)
cv2.waitKey(0)
