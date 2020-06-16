#!/usr/bin/env python
import sys
sys.path.append('/home/kshitij/PS/scripts/')
import rf01
import xgb01
import adaB_svm
import adaB_DT

print("Combining Method : Default\n")

print("Data Used is data/australian.csv\n")

print("AdaBoost + Decision Trees : 0")
print("AdaBoost + SVM : 1")
print("XgBoost : 2")
print("Random_Forest : 3 \n")
ans = int(input("ENTER_YOUR_SELECTION : \n"))

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
