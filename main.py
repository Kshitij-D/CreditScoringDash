#!/usr/bin/env python

from scripts import rf01
from scripts import xgb01
from scripts import adaB_svm
from scripts import adaB_DT
from scripts import bag_DT


print("Combining Method : Default\n")

print("Data Used is data/australian.csv\n")

print("-1 : TO_EXIT \n")

print("0 : AdaBoost + Decision Trees")
print("1 : AdaBoost + SVM")
print("2 : XgBoost + Decision Trees")
print("3 : Random_Forest")
print("4 : Bagged Decision Tress\n")

ans = 0;

while ans != -1 :

    ans = int(input("ENTER_YOUR_SELECTION : \n"))



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
