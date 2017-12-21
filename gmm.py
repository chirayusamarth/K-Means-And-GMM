import json
import random
import numpy as np


def gmm_clustering(X, K):
    """
    Train GMM with EM for clustering.

    Inputs:
    - X: A list of data points in 2d space, each elements is a list of 2
    - K: A int, the number of total cluster centers

    Returns:
    - mu: A list of all K means in GMM, each elements is a list of 2
    - cov: A list of all K covariance in GMM, each elements is a list of 4
            (note that covariance matrix is symmetric)
    """

    # Initialization:
    pi = []
    mu = []
    cov = []
    for k in range(K):
        pi.append(1.0 / K)
        mu.append(list(np.random.normal(0, 0.5, 2)))
        temp_cov = np.random.normal(0, 0.5, (2, 2))
        temp_cov = np.matmul(temp_cov, np.transpose(temp_cov))
        cov.append(list(temp_cov.reshape(4)))

    ### you need to fill in your solution starting here ###
    #print(pi)
    #print(mu)
    #print(cov)
    np_mu = np.asarray(mu)
    np_X = np.asarray(X)
    np_cov= np.asarray(cov).reshape((3,2,2))
    
    # Run 100 iterations of EM updates
    for t in range(100):
        logPX = []
        for k in range(K):
            logpx = []
            logwk = np.log(pi[k])
            logmiddleterm = np.log(2 * np.pi * np.sqrt(np.absolute(np.linalg.det(np_cov[k]))))
            #print(logmiddleterm)
            x_minus_mu = []
            for i in range(len(X)):
                x_minus_mu1 = 0.5 * np.matmul(np.matmul((np_X[i] - np_mu[k]).transpose(), np.linalg.inv(np_cov[k])), (np_X[i] - np_mu[k]))
                x_minus_mu.append(x_minus_mu1)
            for x_minus_mu1 in x_minus_mu:
                logpx1 = logwk - logmiddleterm - x_minus_mu1
                logpx.append(logpx1)
            logPX.append(logpx)
        
        # Subtracting from maxx to prevent overflow
        for i in range(len(logPX[0])):
            maxx= np.finfo('d').min
            for l in range(len(logPX)):
                maxx = np.maximum(maxx, logPX[l][i])
                
            for l in range(len(logPX)):
                logPX[l][i] = logPX[l][i] - maxx

        # Computing posterior prob
        for i in range(len(logPX[0])):
            sum_px = 0
            for l in range(len(logPX)):
                sum_px += np.exp(logPX[l][i])
            for l in range(len(logPX)):
                logPX[l][i] = np.exp(logPX[l][i]) / sum_px
        #        print(logPX[l][i])
        #print(logPX)

        

        # M-step: Update parameters
        N = 0
        for l in range(len(logPX)):
            for i in range(len(logPX[l])):
                N += logPX[l][i]
        #print(N)

        for l in range(len(logPX)):
            # Update pi
            Nc = 0
            for i in range(len(logPX[l])):
                Nc += logPX[l][i]
            pi[l] = Nc / N

            #Update mu
            Ncx1 = 0
            Ncx2 = 0
            for i in range(len(logPX[l])):
                Ncx1 += (logPX[l][i] * np_X[i][0])
                Ncx2 += (logPX[l][i] * np_X[i][1])
            np_mu[l][0] = Ncx1 / Nc
            np_mu[l][1] = Ncx2 / Nc

            #Update cov
            Ncx_minus_mu = 0
            
            for i in range(len(logPX[l])):
                x_i = np_X[i].reshape((np_X[i].shape[0], 1))
                mu_l = np_mu[l].reshape((np_mu[l].shape[0], 1))
                Ncx_minus_mu += logPX[l][i] * (np.matmul((x_i - mu_l) , (x_i - mu_l).transpose()))
            np_cov[l] = Ncx_minus_mu / Nc
            
        #print(pi)
        #print(np_mu)
        #print(np_cov)

    mu = np_mu.tolist()
    cov = np_cov.reshape((3,4)).tolist()

    return mu, cov


def main():
    # load data
    with open('hw4_blob.json', 'r') as f:
        data_blob = json.load(f)

    mu_all = {}
    cov_all = {}

    print('GMM clustering')
    for i in range(5):
        np.random.seed(i)
        mu, cov = gmm_clustering(data_blob, K=3)
        mu_all[i] = mu
        cov_all[i] = cov

        print('\nrun' + str(i) + ':')
        print('mean')
        print(np.array_str(np.array(mu), precision=4))
        print('\ncov')
        print(np.array_str(np.array(cov), precision=4))

    with open('gmm.json', 'w') as f_json:
        json.dump({'mu': mu_all, 'cov': cov_all}, f_json)


if __name__ == "__main__":
    main()