def run():
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        dataset = pd.read_table("/home/kshitij/PS/data/australian.csv", sep='\s+', header=None)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
        from xgboost import XGBClassifier
        classifier = XGBClassifier(random_state=1)
        classifier.fit(X_train, y_train)

        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
        y_pred = classifier.predict(X_test)
        accuracy_score(y_test, y_pred)
        print(classification_report(y_test, y_pred))

        XGB_prob=classifier.predict_proba(X_test)
        XGB_prob = XGB_prob[:, 1]

        XGB_auc=roc_auc_score(y_test,XGB_prob)
        print('XGB: ROC AUC=%.3f' % XGB_auc)
        print("accuracy_score is %.3f" % (accuracy_score(y_test, y_pred, normalize=True)))

        XGB_fpr,XGB_tpr,_=roc_curve(y_test,XGB_prob)
        plt.plot(XGB_fpr, XGB_tpr, marker='.', label='XGBoost')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.legend()
        plt.show()
