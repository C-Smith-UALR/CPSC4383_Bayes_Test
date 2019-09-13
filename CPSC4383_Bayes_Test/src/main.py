#CPSC 4383
#Clark L. Smith
#September 12, 2019
#A program that uses Naive Bayes to predict conditional probability of developing diabetes. 

import csv
import random
import os

def loadCsv(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
        return dataset

filename = '/Users/The_Don_Account/eclipse-workspace/CPSC4383_Bayes_Test/pima-indians-diabetes.data.csv'
dataset = loadCsv(filename)
print('Loaded data file {0} with {1} rows').format(filename, len(dataset))  

#dir_path = os.path.dirname(os.path.realpath(__file__))  
#print(dir_path)

myInt=66
print(myInt)