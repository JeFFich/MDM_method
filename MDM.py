import numpy as np


# Рассчёт текущего приближения
def getting_w(p1, p2, u):
    s1 = p1.shape[0]
    s2 = p2.shape[0]
    w = np.zeros(p1.shape[1])
    for i in range(s1):
        w += u[i] * p1[i]
    for i in range(s2):
        w -= u[s1 + i] * p2[i]
    return w


# Рассчёт первого приближения
def first_iter(p1, p2):
    s1 = p1.shape[0]
    s2 = p2.shape[0]
    u = np.array([1 / s1 for _ in range(s1)] + [1 / s2 for _ in range(s2)])
    return u, getting_w(p1, p2, u)


# Рассчёт соелующего направления
def delta(pts, u, w):
    D = -np.inf
    M = -1
    m = -1
    d = np.inf
    for i in range(pts.shape[0]):
        cur = np.dot(pts[i], w)
        if u[i] > 0 and D < cur:
            M = i
            D = cur
        if cur < d:
            m = i
            d = cur
    return M, m


# Сам МДМ-метод
def MDM(p1, p2, e):
    S = p1.shape[0]
    u, w = first_iter(p1, p2)
    M1, m1 = delta(p1, u[:S], w)
    M2, m2 = delta(p2, u[S:], -1 * w)
    d1 = np.dot(p1[M1] - p1[m1], w)
    d2 = np.dot(p2[M2] - p2[m2], -1 * w)

    # Условие остановки итераций
    while max(d1, d2) > e:
        if d1 > d2:
            t = min(1, d1 / (u[M1] * np.dot(p1[M1] - p1[m1], p1[M1] - p1[m1])))
            u[m1] += t * u[M1]
            u[M1] -= t * u[M1]
        else:
            t = min(1, d2 / (u[M2 + S] * np.dot(p2[M2] - p2[m2], p2[M2] - p2[m2])))
            u[m2 + S] += t * u[M2 + S]
            u[M2 + S] -= t * u[M2 + S]

        w = getting_w(p1, p2, u)
        M1, m1 = delta(p1, u[:S], w)
        M2, m2 = delta(p2, u[S:], -1 * w)
        d1 = np.dot(p1[M1] - p1[m1], w)
        d2 = np.dot(p2[M2] - p2[m2], -1 * w)

    # Возврат к точкам в абсолютном выражении (по представлению линейной оболочки)
    point1 = np.zeros(p1.shape[1])
    for i in range(S):
        point1 += u[i] * p1[i]
    point2 = np.zeros(p2.shape[1])
    for i in range(p2.shape[0]):
        point2 += u[i + S] * p2[i]
    return point1, point2
