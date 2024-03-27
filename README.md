#### Summary

An Extra Trees classification model using ScikitLearn and resampling methods for the critical task of credit card fraud detection, showcasing outstanding performance with a recall score surpassing 0.86 across all classes.


#### DataSet 
The dataset used here for credit card fraud detection is from the following Kaggle URL :

https://www.kaggle.com/mlg-ulb/creditcardfraud

- Time : Number of seconds elapsed between this transaction and the first transaction in the dataset
- V1 to V28 : may be result of a PCA Dimensionality reduction to protect user identities and sensitive features
- Amount : Transaction amount
- Class : 1 for fraudulent transactions, 0 otherwise


#### Conclusions from EDA:
- "Class" does not appear to be correlated with "Time" nor "Amount". 
- The other variables, especially V1 to V18, appear to have some amounts of (positive and negative) correlation with "Class".


#### Description of logical steps/flow of the pipeline

1. The raw data is a .csv file, creditcard_data.csv.
2. Data extraction from .csv files are performed by the script data_extraction.py, giving a pandas dataframe as an output.
3. The dataframe is fed into algo.py containing machine learning algorithms which print classification reports for the classification task and outputs the classifier for each algorithm in this script. 


#### Choice of models and evaluations

- This is a classification problem with an imbalanced binary target class, and the Extra-Trees classifier is used here.
We can also attempt to frame this as an anomaly detection problem, and the Isolation-Forest algorithm is used.

- Nonetheless, due to the severe underrepresentation of the minority class (<1% of the dataset), stratification on the target class is used when splitting the data into training and testing sets. Synthetic data generation using SMOTE is also attempted, on both algorithms mentioned above. Since this is a highly imbalanced dataset, and consequences of false negatives are much higher, we can look at the recall score where the evaluations are as follows:

1. isolation-forest, w/ SMOTE and w/o SMOTE:
    A recall score of 0 for the majority class with and without resampling, while for the minority positive class, 0.82 with resampling and 0.69 without.
2. extra-trees, w/ SMOTE:
    Performs well, with a recall score of 1.0 and 0.86 for the majority and minority class, respectively.
3. extra-trees, w/0 SMOTE:
    Slightly worse performance than using SMOTE, with a recall score of 1.0 and 0.83 for the majority and minority class, respectively.
    
The extra-trees model with resampling has the highest recall scores amongst the models used.

As expected from EDA, neither "Time" nor "Amount" is the most important feature in the extra-trees algorithm. Nonetheless, the interpretability of the other more important features (those from V1 to V28) is unclear, since these are already preprocessed, so we shall not comment further on that. 

