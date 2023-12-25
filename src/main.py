import pandas as pd
from data_extraction import extract_data

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from algo import isolation_forest_classifier, extra_trees_classifier



def main():
    df = extract_data()
    
    outlier_percent = df['Class'].value_counts(normalize=True).get(1, 0) 

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,random_state=42)

    smote = SMOTE(sampling_strategy=0.01, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    
    clf_if, report_if, cm_if = isolation_forest_classifier(X_resampled, y_resampled, X_test, y_test, outlier_percent)

    print("Classification Report, iForest (w/ SMOTE):\n", report_if)
    print("Confusion Matrix, iForest (w/ SMOTE):\n", cm_if)
    
    clf_if, report_if, cm_if = isolation_forest_classifier(X_train, y_train, X_test, y_test, outlier_percent)

    print("Classification Report, iForest (w/o SMOTE):\n", report_if)
    print("Confusion Matrix, iForest (w/ SMOTE):\n", cm_if)


    clf_et, report_et, cm_et = extra_trees_classifier(X_resampled, y_resampled, X_test, y_test)
    print("Classification Report, extraTree (w/ SMOTE):\n", report_et)
    print("Confusion Matrix, extraTree (w/ SMOTE):\n", cm_et)
    
    clf_et, report_et, cm_et = extra_trees_classifier(X_train, y_train, X_test, y_test)
    print("Classification Report, extraTree (w/o SMOTE):\n", report_et)
    print("Confusion Matrix, extraTree (w/o SMOTE):\n", cm_et)
    
    
    
    
    return


if __name__ == "__main__":
    main()