'''
@Description: code for LCFS-PML,
@Author: Qingqi Han,
@Date: 2024.09.06,
'''
import numpy as np
import time
from numpy.linalg import norm
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# ||X(W+S)-L||_F^2 + alpha* ||XW- (L*T)||_F^2 + beta*||W||2,1 + gamma* ||S||_1^2

eps=np.finfo(np.float64).eps

def stepsize_W(W,grad_W,X,L,S,Ln,alpha,beta):
    c = 0.1
    stepsize = 1
    Wn= W- stepsize *grad_W
    oldobj= norm(X@(S+W)-L,'fro')**2 + alpha *norm(X@W-Ln,'fro')**2 + beta*np.sum(np.sqrt(np.sum(W * W, 1)))
    newobj= norm(X@(S+Wn)-L,'fro')**2 + alpha *norm(X@Wn-Ln,'fro')**2 + beta*np.sum(np.sqrt(np.sum(Wn * Wn, 1)))
    if newobj - oldobj > c* np.sum(np.sum(np.multiply(grad_W, (Wn - W)))):
        while True:
            stepsize =stepsize * 0.1
            Wn = W - stepsize * grad_W
            newobj = norm(X@(S+Wn)-L,'fro')**2 + alpha *norm(X@Wn-Ln,'fro')**2 + beta*np.sum(np.sqrt(np.sum(Wn * Wn, 1)))
            if newobj - oldobj <= c * np.sum(np.sum(np.multiply(grad_W, (Wn - W)))):
                break
    return stepsize

def admm_optimize_S(S, X, W, L, gamma, rho=10.0, max_iter=100, tolerance=1e-4):
  """
    ||X(W+S)-L||_F^2 + gamma * ||S||_1
    L(S,Z,U)= ||X(W+S)-L||_F^2 + gamma * ||Z||_1 + U(S-Z) + rho/2 * || S-Z||_F^2
    """
    # 定义子函数
    def shrink(x, kappa):
        return np.sign(x) * np.maximum(np.abs(x) - kappa, 0.)

    def update_S(X, W, L, Z, U, rho):
        #
        XTX = X.T @ X
        M = 2 *  XTX + rho * np.eye(XTX.shape[0])
        V = 2 * X.T @ L - 2 *  X.T @ X @ W + rho * Z - U
        return np.linalg.solve(M, V)

    def update_Z(S, U, gamma, rho):
        return shrink(S + U / rho, gamma / rho)

    def update_Y(U, S, Z, rho):
        return U + rho * (S - Z)

    # ADMM 循环
    U=np.random.randn(S.shape[0],S.shape[1])
    Z=np.random.randn(S.shape[0],S.shape[1])
    for iteration in range(max_iter):
        S_prev = S.copy()
        Z_prev = Z.copy()

        S = update_S(X, W, L, Z, U,  rho)
        Z = update_Z(S, U, gamma, rho)
        U = update_Y(U, S, Z, rho)

        # 计算原始残差和对偶残差
        primal_residue = np.linalg.norm(S - Z)
        dual_residue = np.linalg.norm(Z - Z_prev)

        if primal_residue < tolerance and dual_residue < tolerance:
            break

    return S

def relax_l21(W):
  '''
    计算对矩阵计算L2，1范式的松弛矩阵
    :param W:
    :return:
    '''
    dv,q=W.shape
    C=np.zeros((dv,dv))       #初始化对角矩阵C
    # 计算对角元素
    for i in range(dv):
        C[i,i]= 1/(0.5*np.sqrt(np.sum(W[i] * W[i])+eps))
    return C

def average_distances(X, dist, L, K):
  """
    X: np.ndarray,  (n, d)
    dist: np.ndarray, (n, n)
    L: np.ndarray,  (n, c)
    K: int, 
    """
    n, d = X.shape
    n, c = L.shape

    result_matrix = np.zeros((n, c))

    for label_index in range(c):
        positive_indices = np.where(L[:, label_index] == 1)[0]

        distances = []
        for idx in positive_indices:
            dist_to_positives = dist[idx, positive_indices]
            dist_to_positives = dist_to_positives[dist_to_positives > 0]

            if len(dist_to_positives) >= K:
                nearest_k_distances = np.sort(dist_to_positives)[:K]
            else:
                nearest_k_distances = dist_to_positives

            if len(nearest_k_distances) > 0:
                avg_distance = np.mean(nearest_k_distances)
            else:
                avg_distance = 0 

            distances.append(avg_distance)

        if len(distances) > 0:
            max_distance = max(distances)
            if max_distance > 0:
                normalized_distances = np.array(distances) / max_distance
            else:
                normalized_distances = np.zeros_like(distances)

            for i, idx in enumerate(positive_indices):
                result_matrix[idx, label_index] = normalized_distances[i]

    return result_matrix

def clustercenter_distances(X, L, n_clusters):
  """
    X: np.ndarray, (n, d)
    L: np.ndarray,  (n, c)
    n_clusters: int,
    """
    n, d = X.shape
    n, c = L.shape

    result_matrix = np.zeros((n, c))

    for label_index in range(c):
        positive_indices = np.where(L[:, label_index] == 1)[0]
        if len(positive_indices) == 0:
            continue

        X_pos = X[positive_indices, :]

        kmeans = KMeans(n_clusters=min(n_clusters, len(X_pos)), random_state=0)
        kmeans.fit(X_pos)

        _, distances = pairwise_distances_argmin_min(X_pos, kmeans.cluster_centers_)

        if len(distances) > 0:
            max_distance = np.max(distances)
            if max_distance > 0:
                normalized_distances = distances / max_distance
            else:
                normalized_distances = np.zeros_like(distances)
            for i, idx in enumerate(positive_indices):
                result_matrix[idx, label_index] = normalized_distances[i]

    return result_matrix

def compute_confidence_matrix(d_avg, d_C, L, lambda_param):
  """
    d_avg: np.ndarray, (n, c)
    d_C: np.ndarray, (n, c)
    L: np.ndarray, (n, c)
    lambda_param: float,
    """
    n, c = L.shape
    T = np.ones((n, c))

    for i in range(n):
        for l in range(c):
            if L[i, l] == 1: 
                numerator = d_avg[i, l] * d_C[i, l]
                denominator = lambda_param * d_avg[i, l] + d_C[i, l]
                T[i, l] = np.exp(- (1 + lambda_param) * (numerator / denominator))
            else:
                T[i, l] = 1
    return T

def update_label_matrix(A, L, theta):
  """
    A: np.ndarray, (n, c)
    L: np.ndarray,  (n, c)
    theta: float, 
    """
    updated_A = np.zeros_like(A)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if L[i, j] == 0:
                updated_A[i, j] = 0
            elif A[i, j] > theta and L[i, j] == 1:
                updated_A[i, j] = 1
            else:
                updated_A[i, j] = 0

    return updated_A

def LCFS_PML(X,L,alpha,beta,gamma,lambda_,theta,K,n_clusters):
  '''
    ||X(W+S)-L||_F^2 + alpha* ||XW- (L*T)||_F^2 + beta*||W||2,1 + gamma* ||S||_1
    :param X: n*d ,feature matrix
    :param L: d*c ,partial label matrix
    :param alpha: float
    :param beta: float
    :param gamma: float
    :param theta: float ,threshold
    :param K: int, 
    :param n_clusters: int,
    :return:
    '''
    n,l= L.shape
    d=X.shape[1]

    maxiter=500 

    # 参数初始化
    W = np.random.rand(d, l)           
    S =  np.random.rand(d, l)          
    C = relax_l21(W)

    dist = squareform(pdist(X, 'euclidean'))
    d_avg = average_distances(X, dist, L, K=K)
    d_C = clustercenter_distances(X, L, n_clusters=n_clusters)
    T = compute_confidence_matrix(d_avg, d_C, L, lambda_)

    Ln = update_label_matrix(L*T, L, theta=theta)

    iteration=0
    obj_save=[]
    
    obj = norm(X@(S+W)-L,'fro')**2 + alpha *norm(X@W-Ln,'fro')**2 + beta*np.sum(np.sqrt(np.sum(W * W, 1))) + gamma*np.sum(np.abs(S))
    obj_save.append(obj)

    print('the object value in iter {} is {}\n'.format(iteration, obj_save[iteration]))

    iteration = 1
    while iteration < maxiter:
        
        grad_W = 2*alpha*X.T@(X@W-Ln) + 2*X.T@(X@(S+W)-L)  + 2*beta*C@W
        W = W - stepsize_W(W,grad_W,X,L,S,Ln,alpha,beta) * grad_W

        S = admm_optimize_S(S, X, W, L, gamma, rho=1.0, max_iter=100, tolerance=1e-4)

        obj_tmp = norm(X@(S+W)-L,'fro')**2 + alpha *norm(X@W-Ln,'fro')**2 + beta*np.sum(np.sqrt(np.sum(W * W, 1))) + gamma*np.sum(np.abs(S))
        obj_save.append(obj_tmp)
        print('the object value in iter {} is {}\n'.format(iteration, obj_save[iteration]))

        if (iteration > 2) and (abs(obj_save[iteration] - obj_save[iteration - 1]) / abs(obj_save[iteration]) < 1e-2):  # 判断是否达到退出条件
            print('break')
            break

        iteration+= 1

        w_2 = norm(W, ord=2, axis=1)
        f_idx = np.argsort(-w_2).tolist()
        Xn = X[:,f_idx[:int(0.5*d)]]

        dist = squareform(pdist(Xn, 'euclidean'))
        d_avg = average_distances(Xn, dist, L, K=K)
        d_C = clustercenter_distances(Xn, L, n_clusters=n_clusters)
        T = compute_confidence_matrix(d_avg, d_C, L, lambda_)

        Ln = update_label_matrix(L * T, L, theta=theta)
        C = relax_l21(W)

    w_2 = norm(W, ord=2, axis=1)
    f_idx = np.argsort(-w_2).tolist()

    return f_idx




