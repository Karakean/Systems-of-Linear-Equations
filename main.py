import matplotlib.pyplot as plt
import math
import time


def print_matrix(matrix):
    for row in matrix:
        for value in row:
            print(str(round(value, 2))+'\t\t', end='')
        print("")
    return


def create_system_matrix(N, a1, a2, a3):
    A = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        A[i][i] = a1
        if i+1 < N:
            A[i][i + 1] = a2
            A[i+1][i] = a2
        if i+2 < N:
            A[i][i+2] = a3
            A[i+2][i] = a3
    return A


def negative_matrix(matrix):
    N = len(matrix)
    for i in range(N):
        for j in range(N):
            matrix[i][j] = -matrix[i][j]
    return matrix


def negative_vector(vector):
    N = len(vector)
    for i in range(N):
        vector[i] = -vector[i]
    return vector


def add_matrices(m1, m2):
    N = len(m1)
    for i in range(N):
        for j in range(N):
            m1[i][j] += m2[i][j]
    return m1


def copy_matrix(src_matrix):
    N = len(src_matrix)
    dst_matrix = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            dst_matrix[i][j] = src_matrix[i][j]
    return dst_matrix


def multiply_matrix_by_vector(matrix, vector):
    N = len(vector)
    vtr = [0 for _ in range(N)]  # vector to return
    for i in range(N):
        for j in range(N):
            vtr[i] += matrix[i][j] * vector[j]
    return vtr


def norm(vec):
    N = len(vec)
    norm = 0
    for i in range(N):
        norm += vec[i]**2
    return math.sqrt(norm)


def residual(matrix, vecx, vecb):
    N = len(vecx)
    tmp = multiply_matrix_by_vector(matrix, vecx)
    res = [(tmp[i] - vecb[i]) for i in range(N)]
    return res


def Jacobi_method(A, b, stop):
    N = len(A)
    x = [1 for _ in range(N)]
    res = [1 for _ in range(N)]
    prevx = [1 for _ in range(N)]
    iterations = 0
    start = time.time()
    while norm(res) > stop:
        for i in range(N):
            series_sum = 0
            for j in range(i):
                series_sum += A[i][j] * prevx[j]
            for j in range(i + 1, N):
                series_sum += A[i][j] * prevx[j]
            x[i] = (b[i] - series_sum) / A[i][i]
        prevx = [x[i] for i in range(N)]
        res = residual(A, x, b)
        iterations += 1
    end = time.time()
    return x, iterations, end-start


def Gauss_Seidl_method(A, b, stop):
    N = len(A)
    x = [1 for _ in range(N)]
    res = [1 for _ in range(N)]
    prevx = [1 for _ in range(N)]
    iterations = 0
    start = time.time()
    while norm(res) > stop:
        for i in range(N):
            series_sum = 0
            for j in range(i):
                series_sum += A[i][j] * x[j]
            for j in range(i + 1, N):
                series_sum += A[i][j] * prevx[j]
            x[i] = (b[i] - series_sum)/A[i][i]
        prevx = [x[i] for i in range(N)]
        res = residual(A, x, b)
        iterations += 1
    end = time.time()
    return x, iterations, end-start


def create_triangular_matrix(src_matrix, isLower):
    N = len(src_matrix)
    matrix = [[0 for _ in range(N)] for _ in range(N)]
    if isLower:
        for i in range(1, N):
            for j in range(i):
                matrix[i][j] = src_matrix[i][j]
    else:
        for i in range(1, N):
            for j in range(i):
                matrix[j][i] = src_matrix[j][i]
    return matrix


def create_diag(matrix):
    N = len(matrix)
    diag = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        diag[i][i] = matrix[i][i]
    return diag


def create_identity_matrix(N):
    matrix = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        matrix[i][i] = 1
    return matrix


def create_LU_matrices(A):
    N = len(A)
    U = copy_matrix(A)
    L = create_identity_matrix(N)
    for i in range(N-1):
        for j in range(i+1, N):
            L[j][i] = U[j][i]/U[i][i]
            for k in range(i, N):
                U[j][k] = U[j][k] - L[j][i]*U[i][k]
    return L, U


def LU_decomposition(N, a1, a2, a3, f):
    b = [math.sin(i * (f + 1)) for i in range(N)]
    A = create_system_matrix(N, a1, a2, a3)
    x = [0 for _ in range(N)]
    y = [0 for _ in range(N)]
    start = time.time()
    L, U = create_LU_matrices(A)
    for i in range(N):
        series_sum = 0
        for j in range(i):
            series_sum += L[i][j]*y[j]
        y[i] = (b[i] - series_sum)/L[i][i]
    for i in range(N-1, -1, -1):
        series_sum = 0
        for j in range(i+1, N):
            series_sum += U[i][j] * x[j]
        x[i] = (y[i] - series_sum)/U[i][i]
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
    A = create_system_matrix(N, a1, a2, a3)

    # B
    stop = 10**(-9)
    gs_result, iterations, elapsed_time = Gauss_Seidl_method(A, b, stop)
    print("Number of iterations with Gauss-Seidl method: " + str(iterations) + ". Elapsed time: " + str(elapsed_time))
    j_result, iterations, elapsed_time = Jacobi_method(A, b, stop)
    print("Number of iterations with Jacobi method: " + str(iterations) + ". Elapsed time: " + str(elapsed_time))

    #C
    # a1 = 3
    # a2 = a3 = -1
    # N = 965
    # A = create_system_matrix(N, a1, a2, a3)
    # gs_result, iterations, elapsed_time = Gauss_Seidl_method(A, b, stop)
    # print("Number of iterations with Gauss-Seidl method: " + str(iterations) + ". Elapsed time: " + str(elapsed_time))
    # j_result, iterations, elapsed_time = Jacobi_method(A, b, stop)
    # print("Number of iterations with Jacobi method: " + str(iterations) + ". Elapsed time: " + str(elapsed_time))

    #D
    # x, y, duration, norm_res = LU_decomposition(N, a1, a2, a3, f)
    # print(norm_res)


main()
