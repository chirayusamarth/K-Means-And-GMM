# K-Means And Gaussian Mixture Models

# Dataset

In this project, we will use 2-dimensional synthetic data hw4 circle.json and hw4 blob.json to
perform clustering and density estimation. 

# Cautions

Please do not import packages that are not listed in the provided code. Follow the instructions
in each section strictly to code up your solutions. Do not change the output format. Do
not modify the code unless we instruct you to do so. Please do not change the random seeds set in our template code. Detailed implementation instructions are provided in the python script. 

# K-Means

Implement K-means using random initialization for cluster centers. Run the algorithm on
2 two-dimensional synthetic datasets hw4 circle.json and hw4 blob.json with different values of K ∈ {2, 3, 5}. The algorithm should run until none of the cluster assignments are changed. Then, run kmeans.sh, which will generate a kmeans.json file containing cluster center and the cluster assigned to each data point with all values of K.


# Gaussian Mixture Models

Implement an EM algorithm to fit a Gaussian mixture model on the hw4 blob.json dataset. Finish the implementation of the function gmm clustering in gmm.py. Then, run gmm.sh. It will run 5 rounds of your EM algorithm with the number of components K = 3 and generate a
gmm.json file containing the mean and co-variance matrices of all the three Gaussian components.




