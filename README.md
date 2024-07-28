# Machine Learning Course Assignments Spring 2024

This repository contains a collection of assignments completed as part of the graduate level Machine Learning course offered at University of Tehran. Each assignment aims to explore various aspects of machine learning, including algorithm implementation, data preprocessing, model evaluation, and real-world application scenarios. The assignments are designed to enhance practical understanding and application of core machine learning concepts, utilizing popular frameworks such as Python, Pytorch, and scikit-learn. Users can navigate through the individual folders for specific assignments to find code, reports which are written in Latex markdown language, and relevant datasets. This repository serves as a reflective overview of the learning journey and practical skills gained throughout the course.

## Assignment 1: Bayesian Decision Theory  

In this assignment, we delve into Bayesian Decision Theory, a fundamental framework that combines statistical inference with decision-making under uncertainty. The main objective is to illustrate how probabilities can be used to make optimal decisions based on observed evidence. We begin by defining priors, likelihoods, and posteriors, using Bayes' theorem to update our beliefs in light of new data. Through practical examples and simulations, we explore concepts such as loss functions and utility, which guide the decision-making process. By applying Bayesian methods to a chosen dataset, we demonstrate how this approach can effectively address classification problems while accommodating uncertainty. This assignment strengthens our understanding of probabilistic reasoning and the advantages of incorporating prior knowledge into decision-making processes.


## Assignment 2: K-Nearest Neighbors, Parzen Density Estimation, and Regression  

In this assignment, we explore three powerful techniques in machine learning: K-Nearest Neighbors (KNN), Parzen density estimation, and regression analysis. We start with KNN, a simple yet effective classification algorithm that predicts the class of a data point based on the majority class among its K nearest neighbors in the feature space. Next, we delve into Parzen density estimation, a non-parametric method used to estimate the probability density function of a random variable. This approach enables us to model complex distributions without assuming a specific underlying structure. We then apply regression analysis to establish relationships between dependent and independent variables, demonstrating both linear and non-linear regression techniques. Through practical implementations on selected datasets, we highlight the strengths and limitations of each method while showcasing their applicability in real-world prediction tasks. By the end of this assignment, we gain a comprehensive understanding of these essential supervised and unsupervised learning techniques.


## Assignment 3: Decision Trees and Ensemble Methods  

In this assignment, we investigate Decision Trees with a focus on the ID3 algorithm and Ensemble Methods, specifically Adaptive Boosting (AdaBoost). We begin by implementing the ID3 algorithm from scratch to understand how decision trees recursively partition the feature space based on information gain, effectively creating a model that makes predictions by traversing the tree structure. Through practical examples, we discuss how to handle different types of data and the importance of tuning parameters to avoid overfitting. Following this, we explore Adaptive Boosting, an ensemble technique that combines multiple weak classifiers to form a strong predictive model. By constructing AdaBoost from scratch, we illustrate how it sequentially focuses on misclassified instances, thereby improving overall accuracy. This assignment emphasizes the strengths and complementary nature of these techniques, showcasing how decision tree models can be enhanced through ensemble methods to achieve better classification performance on complex datasets.


## Assignment 4: Support Vector Machines and Neural Networks

In this assignment, we explore Support Vector Machines (SVM) and Neural Networks, with a particular emphasis on implementing the LeNet architecture using PyTorch. We begin by elucidating the principles of SVM, a powerful classification technique that seeks to find the optimal hyperplane that separates data points of different classes while maximizing the margin between them. Following this theoretical foundation, we shift our focus to neural networks, specifically convolutional neural networks (CNNs), which excel in image classification tasks. We implement the LeNet CNN architecture from scratch using PyTorch, detailing the key components such as convolutional layers, activation functions, pooling, and fully connected layers. Through hands-on training on benchmark datasets like MNIST, we discuss the importance of hyperparameter tuning, loss functions, and optimization techniques in achieving robust model performance. This assignment reinforces our understanding of both SVM and neural networks, illustrating the unique advantages of each method in the context of machine learning applications.


## Assignment 5: Mixture Models and Principal Component Analysis  

In this assignment, we delve into Mixture Models and Principal Component Analysis (PCA), exploring their roles in data modeling and dimensionality reduction. We start with Mixture Models, which allow us to represent data distributions as a combination of multiple probability distributions, often unveiling hidden structures within the data. Through real-world examples, we demonstrate how Gaussian Mixture Models (GMM) can be used for clustering and density estimation. Next, we focus on PCA, a widely-used technique for reducing dimensionality while preserving as much variance as possible in the dataset; we implement PCA from scratch and discuss its applications in noise reduction and feature extraction. By applying PCA to a selected dataset, we highlight its effectiveness in simplifying data while maintaining its essential characteristics, thereby gaining valuable insights into how both Mixture Models and PCA can be leveraged to enhance data analysis and visualization. This assignment strengthens our understanding of these foundational statistical techniques and their practical implications in machine learning.