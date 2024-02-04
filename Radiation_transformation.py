import numpy as np

def affine_fit(from_pts, to_pts):
    q = from_pts
    p = to_pts
    if len(q) != len(p) or len(q) < 1:
        print("The number of source points and target points must be the same.")
        return False

    dim = len(q[0])
    if len(q) < dim:
        print("The number of points is less than the dimensionality.")
        return False

    c = [[0.0 for a in range(dim)] for i in range(dim + 1)]
    for j in range(dim):
        for k in range(dim + 1):
            for i in range(len(q)):
                qt = list(q[i]) + [1]
                c[k][j] += qt[k] * p[i][j]


    Q = [[0.0 for a in range(dim)] + [0] for i in range(dim + 1)]
    for qi in q:
        qt = list(qi) + [1]
        for i in range(dim + 1):
            for j in range(dim + 1):
                Q[i][j] += qt[i] * qt[j]

    def gauss_jordan(m, eps=1.0 / (10 ** 10)):
        (h, w) = (len(m), len(m[0]))
        for y in range(0, h):
            maxrow = y
            for y2 in range(y + 1, h):
                if abs(m[y2][y]) > abs(m[maxrow][y]):
                    maxrow = y2
            (m[y], m[maxrow]) = (m[maxrow], m[y])
            if abs(m[y][y]) <= eps:
                return False
            for y2 in range(y + 1, h):
                c = m[y2][y] / m[y][y]
                for x in range(y, w):
                    m[y2][x] -= m[y][x] * c
        for y in range(h - 1, 0 - 1, -1):
            c = m[y][y]
            for y2 in range(0, y):
                for x in range(w - 1, y - 1, -1):
                    m[y2][x] -= m[y][x] * m[y2][y] / c
            m[y][y] /= c
            for x in range(h, w):
                m[y][x] /= c
        return True

    M = [Q[i] + c[i] for i in range(dim + 1)]
    if not gauss_jordan(M):
        print("Error, the source points and target points may be collinear.")
        return False

    class transformation:
        def To_Str(self):
            # res = ""
            P = []
            H = []
            for j in range(dim):
                # str = "x%d' = " % j                                                     # j
                for i in range(dim):
                    # str += "x%d * %f + " % (i, M[i][j + dim + 1])
                    P.append(M[i][j + dim + 1])
                # str += "%f" % M[dim][j + dim + 1]
                P.append(M[dim][j + dim + 1])
                # res += str + "\n"

            H.append([P[0], P[1], P[2]])
            H.append([P[3], P[4], P[5]])
            return H

        def transform(self, pt):
            res = [0.0 for a in range(dim)]
            for j in range(dim):
                for i in range(dim):
                    res[j] += pt[i] * M[i][j + dim + 1]
                res[j] += M[dim][j + dim + 1]
            return res

    return transformation()

def test(from_pt, to_pt):

    trn = affine_fit(from_pt, to_pt)

    if trn:
        # print(trn.To_Str())
        err = 0.0
        for i in range(len(from_pt)):
            fp = from_pt[i]
            tp = to_pt[i]
            t = trn.transform(fp)
            # print("%s => %s ~= %s" % (fp, tuple(t), tp))
            err += ((tp[0] - t[0]) ** 2 + (tp[1] - t[1]) ** 2) ** 0.5

        # print("拟合误差 = %f" % err)
    return trn, err

# if __name__ == "__main__":
#     test()