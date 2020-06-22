def run(): 
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

    df = pd.read_table("./data/australian.csv", sep='\s+', header=None)


    X=df.iloc[:,:-1].values
    y=df.iloc[:,-1].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 1)

    bg = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=20)
    bg.fit(X_train, y_train)

    y_pred = bg.predict(X_test)
    accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))

    bagdt_prob=bg.predict_proba(X_test)

    bagdt_prob=bagdt_prob[:,1]

    bagdt_auc=roc_auc_score(y_test,bagdt_prob)
    print('BAGGING DT: ROC AUC=%.3f' % bagdt_auc)
    print("accuracy_score is %.3f" % (accuracy_score(y_test, y_pred, normalize=True)))

    BAG_fpr,BAG_tpr,_=roc_curve(y_test,bagdt_prob)
    plt.plot(BAG_fpr, BAG_tpr, marker='.', label='BAGGING DT')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend()
    plt.show()
