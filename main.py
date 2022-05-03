import matplotlib.pyplot as plt
import math
import time
from matrix import Matrix


def norm(vec):
    N = len(vec)
    n = 0  # norm
    for i in range(N):
        n += vec[i]**2
    return math.sqrt(n)


def residual(matrix, vec_x, vec_b):
    N = len(vec_x)
    product = matrix*vec_x
    res = [(product[i] - vec_b[i]) for i in range(N)]
    return res


def Jacobi_method(A, b, stop):
    x = [1 for _ in range(A.N)]
    res = [1 for _ in range(A.N)]
    prevx = [1 for _ in range(A.N)]
    iterations = 0
    start = time.time()
    while norm(res) > stop:
        for i in range(A.N):
            series_sum = 0
            for j in range(i):
                series_sum += A.data[i][j] * prevx[j]
            for j in range(i + 1, A.N):
                series_sum += A.data[i][j] * prevx[j]
            x[i] = (b[i] - series_sum) / A.data[i][i]
        prevx = [x[i] for i in range(A.N)]
        res = residual(A, x, b)
        iterations += 1
    end = time.time()
    return x, iterations, end-start


def Gauss_Seidel_method(A, b, stop):
    x = [1 for _ in range(A.N)]
    res = [1 for _ in range(A.N)]
    prevx = [1 for _ in range(A.N)]
    iterations = 0
    start = time.time()
    while norm(res) > stop:
        for i in range(A.N):
            series_sum = 0
            for j in range(i):
                series_sum += A.data[i][j] * x[j]
            for j in range(i + 1, A.N):
                series_sum += A.data[i][j] * prevx[j]
            x[i] = (b[i] - series_sum)/A.data[i][i]
        prevx = [x[i] for i in range(A.N)]
        res = residual(A, x, b)
        iterations += 1
    end = time.time()
    return x, iterations, end-start


def create_LU_matrices(matrix):
    U = matrix.get_copy()
    L = Matrix(matrix.N)
    for i in range(matrix.N-1):
        for j in range(i+1, matrix.N):
            L.data[j][i] = U.data[j][i]/U.data[i][i]
            for k in range(i, matrix.N):
                U.data[j][k] = U.data[j][k] - L.data[j][i]*U.data[i][k]
    return L, U


def LU_decomposition(A, b):
    x = [0 for _ in range(A.N)]
    y = [0 for _ in range(A.N)]
    start = time.time()
    L, U = create_LU_matrices(A)
    for i in range(A.N):
        series_sum = 0
        for j in range(i):
            series_sum += L.data[i][j]*y[j]
        y[i] = (b[i] - series_sum)
    for i in range(A.N-1, -1, -1):
        series_sum = 0
        for j in range(i+1, A.N):
            series_sum += U.data[i][j] * x[j]
        x[i] = (y[i] - series_sum)/U.data[i][i]
    res = residual(A, x, b)
    end = time.time()
    return x, y, end-start, norm(res)


def main():
    # A
    e = 8
    f = 4
    a1 = 5 + e
    a2 = a3 = -1
    N = 965
    b = [math.sin(i * (f + 1)) for i in range(N)]
    A = Matrix(N, a1, a2, a3)

    # B
    stop = 10**(-9)
    gs_result, iterations, elapsed_time = Gauss_Seidel_method(A, b, stop)
    print("Gauss-Seidel method, number of iterations: " + str(iterations) +
          ". Elapsed time: " + str(elapsed_time) + "s.")
    j_result, iterations, elapsed_time = Jacobi_method(A, b, stop)
    print("Jacobi method, number of iterations: " + str(iterations) +
          ". Elapsed time: " + str(elapsed_time) + "s.")

    # # C
    # a1 = 3
    # a2 = a3 = -1
    # f = 4
    # N = 965
    # stop = 10**(-9)
    # A = Matrix(N, a1, a2, a3)
    # b = [math.sin(i * (f + 1)) for i in range(N)]
    # gs_result, iterations, elapsed_time = Gauss_Seidel_method(A, b, stop)
    # print("Gauss-Seidel method, number of iterations: " + str(iterations) +
    #       ". Elapsed time: " + str(elapsed_time) + "s.")
    # j_result, iterations, elapsed_time = Jacobi_method(A, b, stop)
    # print("Jacobi method, number of iterations: " + str(iterations) +
    #       ". Elapsed time: " + str(elapsed_time) + "s.")

    # # D
    # x, y, duration, norm_res = LU_decomposition(A, b)
    # print("LU decomposition, elapsed time: "+str(duration) + ". Norm of residuals: " + str(norm_res))

    # #E
    # plt.rcParams['figure.figsize'] = [15, 5]
    # plt.figure(num='Correlation between duration of algorithms and the number of unknowns')
    # plt.title('Correlation between duration of algorithms and the number of unknowns')
    # plt.xlabel("Number of unknowns")
    # plt.ylabel("Duration [s]")
    # e = 8
    # f = 4
    # a1 = 5 + e
    # a2 = a3 = -1
    # N = [100, 500, 1000, 2000, 3000]
    # stop = 10 ** (-9)
    # duration = [[0 for _ in range(len(N))] for _ in range(3)]
    # for i, size in enumerate(N):
    #     A = Matrix(size, a1, a2, a3)
    #     b = [math.sin(j * (f + 1)) for j in range(size)]
    #     _, _, duration[0][i] = Gauss_Seidel_method(A, b, stop)
    #     _, _, duration[1][i] = Jacobi_method(A, b, stop)
    #     _, _, duration[2][i], _ = LU_decomposition(A, b)
    # plt.plot(N, duration[0], label="Gauss-Seidel method")
    # plt.plot(N, duration[1], label="Jacobi method")
    # plt.plot(N, duration[2], label="LU decomposition")
    # plt.legend(loc="upper left", fancybox=True, shadow=True, borderpad=1)
    # plt.show()


main()
