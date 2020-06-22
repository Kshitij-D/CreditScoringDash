#!/usr/bin/env python

from scripts import rf01
from scripts import xgb01
from scripts import adaB_svm
from scripts import adaB_DT
from scripts import bag_DT
from scripts import gbdt
from scripts import eXtc
from scripts import S_rf_adaB_svm
from scripts import S_xgb_gbdt
from scripts import GNB01
from scripts import MLP01
from scripts import knn01
from scripts import S_GNB_MLP_KNN
from scripts import V_abdt_rf_mlp
from scripts import V_GNB_eXtc_knn
from scripts import V_rf_adaB_DT_MLP_GNB

print("Combining Method : Default\n")

print("Data Used is data/australian.csv\n")

print("0  : AdaBoost + Decision Trees")
print("1  : AdaBoost + SVM")
print("2  : XgBoost + Decision Trees")
print("3  : Random_Forest")
print("4  : Bagged Decision Tress")
print("5  : Gradient Boosted Trees")
print("6  : Extra Trees(Random_Forest)")
print("7  : Stacked_LogR(RandF + AdaBoost_SVM)")
print("8  : Stacked_LogR(XgBoost + Gradient Boosted Trees)")
print("9  : Gaussian Naive Bayes")
print("10 : Multi-layer Perceptron")
print("11 : K-NN")
print("12 : Stacked_LogR(GaussianNB + KNN + MLP)")
print("13 : Voting(AdaBoost_DT + Random_Forest + MLP)")
print("14 : Voting(KNN + GaussianNB + ExtraTrees)")
print("15 : Voting(Random_Forest + AdaBoostDT + MLP + GNB)")

print("\n")

print("TO_EXIT : -1 \n")

print("\n")

ans = 0;

while ans != -1 :

    ans = int(input("ENTER_YOUR_SELECTION : \n"))
    
    if ans == 15:
        print("\nVoting(Random_Forest + AdaBoostDT + MLP + GNB) : 15")
        cc = V_rf_adaB_DT_MLP_GNB.run()
    
    if ans == 14:
        print("\nVoting(KNN + GaussianNB + ExtraTrees) : 14")
        cc = V_GNB_eXtc_knn.run()
    
    if ans == 13:
        print("\nVoting(Multi-layer Perceptron + Random_Forest + MLP) : 13")
        cc = V_abdt_rf_mlp.run()
    
    if ans == 12:
        print("\nStacked_LogR(GaussianNB + KNN + MLP) : 12")
        cc = S_GNB_MLP_KNN.run()
    
    if ans == 11:
        print("\nK-NN : 11")
        cc = knn01.run()
    
    if ans == 10:
        print("\nMulti-layer Perceptron : 10")
        cc = MLP01.run()
    
    if ans == 9:
        print("\nGaussian Naive Bayes : 9")
        cc = GNB01.run()
    
    if ans == 8:
        print("\nStacked_LogR(XgBoost + Gradient Boosted Trees): 8")
        cc = S_xgb_gbdt.run()
    
    if ans == 7:
        print("\nStacked(RandF + AdaBoost_SVM): 7")
        cc = S_rf_adaB_svm.run()
    
    if ans == 6:
        print("\nExtra Trees(Random_Forest) : 6")
        cc = eXtc.run()   

    if ans == 5:
        print("\nGradient Boosted Trees : 5")
        cc = gbdt.run()   
        
    if ans ==4:
        print("\nBagged Decsion Tress : 4")
        cc = bag_DT.run()    

    if ans == 3:
        print("\nRandom_Forest : 3")
        cc = rf01.run()
    elif ans == 2:
        print("\nXgBoost : 2")
        xgb01.run()
    elif ans == 1:
        print("\nAdaBoost + SVM : 1")
        adaB_svm.run()
    elif ans == 0:
        print("\nAdaBoost + Decision Trees : 0")
        adaB_DT.run()
        
    print("\n\n")
