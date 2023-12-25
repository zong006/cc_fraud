from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def isolation_forest_classifier(X_train, y_train, X_test, y_test, outlier_percent):

    from sklearn.ensemble import IsolationForest
    
    clf = IsolationForest(n_estimators=100, contamination=outlier_percent, random_state=42, bootstrap=True)
    clf.fit(X_train.values)
    y_pred = clf.predict(X_test.values)
    y_pred_binary = [1 if pred == 1 else 0 for pred in y_pred]

    report = classification_report(y_test, y_pred_binary)
    cm = confusion_matrix(y_test, y_pred_binary)

    return clf, report, cm



def extra_trees_classifier(X_train, y_train, X_test, y_test):
    
    from sklearn.ensemble import ExtraTreesClassifier

    clf = ExtraTreesClassifier(n_estimators=100, random_state=42, bootstrap=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    
    return clf, report, cm