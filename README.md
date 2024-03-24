# Ensemble-Methods
Bagging,boosting, and stacking
What's the diversity in predictve models?
1. build models on different data samples
2. build models using different feature sets
3. build models using different learning algorithm
4. Incorporate some randomness into model building
   
What's the approaches to ensemple-building?
1. Data-centric(same base algorithm, different training data)
2. Model-centric(different base algorithms, same traning data)

Where error comes from in a model?
1. Bias: as the complexity increases, bias will decrease 
2. Variance: as the complexity increase, variance will increase

<img width="464" alt="image" src="https://github.com/MichaelDataM/Ensemble-Methods/assets/145874767/e742d16f-eeb4-49a5-b0cd-65567a70fbaa">

What's bagging?
Bagging tends to reduce overfitting by reducing variance, by building several models at the same time and use mojority voting to generate the final result. 
For each model, bagging use part of the training data with replacement.

What's random forest and what's the con?
Decision Tree are unstable and tend to be overfitting. Small change in input data will impact the structure of the tree. Random forest can solve this problem.
Random Forest will build n decision trees. In each decision tree, random forest just use part of the training data with replacement and part of the features.

What's Boosting and what's the con?
Boosting tends to decrease the insample error. Boosting will train n models sequenqially. In each iteration, build model on the sample with replacement from data.
Evaluate the model on the original training data. Increase the weights of data points on which the current model makes misclassfications and decrease the wights of other data points.
