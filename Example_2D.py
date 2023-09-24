from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from MDM import MDM


# Функция отрисовки двух исходных множеств, приближенного решения и линии жесткого отделения, если это возможно
def paint(p1, p2, pts):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.xlim(0, 15)
    plt.ylim(0, 15)

    ax.plot(p1[:, 0], p1[:, 1], '.', color='g')
    ax.plot(p2[:, 0], p2[:, 1], '.', color='b')

    ax.plot(pts[0][0], pts[0][1], '.', color='r')
    ax.plot(pts[1][0], pts[1][1], '.', color='r')

    if round(pts[0][0] - pts[1][0], 2) != 0 or round(pts[0][1] - pts[1][1], 2) != 0:
        mid = ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)
        if pts[0][0] == pts[1][0]:
            ax.plot((0, 100), (mid[1], mid[1]), color='k')
        elif pts[0][1] == pts[1][1]:
            ax.plot((mid[0], mid[0]), (0, 100), color='k')
        else:
            k = - (pts[1][1] - pts[0][1]) / (pts[1][0] - pts[0][0])
            k1 = 1 / k
            b = mid[1] - mid[0] * k1
            ax.plot((0, 100), (b, k1 * 100 + b), color='k')

    plt.show()


if __name__ == '__main__':
    x, y = make_blobs(n_samples=450, n_features=2, centers=2, center_box=(-5, 5))  # Генерация исходных множеств
    pt = MDM(x[y == 1] + 10, x[y == 0] + 10, 0.00000000001)
    paint(x[y == 1] + 10, x[y == 0] + 10, pt)
