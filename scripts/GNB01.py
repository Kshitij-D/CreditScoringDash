def run():
        import numpy as np
        import pandas as pd
        import seaborn
        import matplotlib.pyplot as pyplot
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
        from sklearn.naive_bayes import GaussianNB

        df = pd.read_table("./data/australian.csv", sep='\s+', header=None)
        y = df[14]
        X = df.drop(columns = 14)
        y.value_counts()
        # Split features and target into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y, test_size = 0.4)
        
        # Instantiate the Classifiers
        
        
        clf = GaussianNB()
        
        clf.fit(X_train, y_train)
        # Make predictions for the test set
        y_pred_test = clf.predict(X_test)


        # View accuracy score
        
        print(classification_report(y_test, y_pred_test))

        clf_probs = clf.predict_proba(X_test)
        # keep probabilities for the positive outcome only
        clf_probs = clf_probs[:, 1]
        # calculate scores
        clf_auc = roc_auc_score(y_test, clf_probs)
        # summarize scores
        print('Gaussian Naive Bayes: ROC AUC=%.3f' % (clf_auc))
        print("accuracy_score is %.3f" % (accuracy_score(y_test, y_pred_test, normalize=True)))
        # calculate roc curves
        clf_fpr, clf_tpr, _ = roc_curve(y_test, clf_probs)
        # plot the roc curve for the model
        pyplot.plot(clf_fpr, clf_tpr, marker='.', label='Gaussian Naive Bayes')
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()
 
 
