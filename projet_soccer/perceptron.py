# -*- coding: utf-8 -*-


# Importations nécessaires pour l'ensemble des fonctions de ce fichier:
import pandas as pd
import matplotlib.pyplot as plt

import math
import random as rd
import numpy as np


class LabeledSet:  
    
    def __init__(self, input_dimension):
        self.input_dimension = input_dimension
        self.nb_examples = 0
    
    def addExample(self,vector,label):
        if (self.nb_examples == 0):
            self.x = np.array([vector])
            self.y = np.array([label])
        else:
            self.x = np.vstack((self.x, vector))
            self.y = np.vstack((self.y, label))
        
        self.nb_examples = self.nb_examples + 1
    
    #Renvoie la dimension de l'espace d'entrée
    def getInputDimension(self):
        return self.input_dimension
    
    #Renvoie le nombre d'exemples dans le set
    def size(self):
        return self.nb_examples
    
    #Renvoie la valeur de x_i
    def getX(self, i):
        return self.x[i]
        
    
    #Renvouie la valeur de y_i
    def getY(self, i):
        return(self.y[i])



class Classifier:
    def __init__(self, input_dimension):
        raise NotImplementedError("Please Implement this method")
    
    #Permet de calculer la prediction sur x => renvoie un score
    def predict(self, x):
        raise NotImplementedError("Please Implement this method")
    
    #Permet d'entrainer le modele sur un ensemble de données
    def train(self, labeledSet):
        raise NotImplementedError("Please Implement this method")
    
    #Permet de calculer la qualité du système 
    def accuracy(self, dataset):
        nb_ok = 0
        for i in range(dataset.size()):
            output, acc = self.predict(dataset.getX(i))
            if (output * dataset.getY(i) > 0):
                nb_ok = nb_ok + 1
        acc = nb_ok / (dataset.size() * 1.0)
        return acc


class Perceptron(Classifier):
    
    def __init__(self, input_dimension, learning_rate, nombre_iterations):
        self.dim=input_dimension
        self.lg_rate=learning_rate
        self.nbr_it=nombre_iterations
        self.w=np.random.rand(self.dim+1)
        self.labeledset=None
    
    def train(self, labeledset):
        
        self.labeledset=labeledset
        list_acc=[]
        list_it=[]
        ls=np.arange(labeledset.size())
        np.random.shuffle(ls)
        
        for i in range(self.nbr_it):
            
            for j in ls:
                vote,acc_plus = self.predict(self.labeledset.getX(j))
                if(self.labeledset.getY(j)*vote<0):
                    
                    self.w = self.w + self.lg_rate*self.labeledset.getY(j)*np.hstack((self.labeledset.getX(j),[-1]))

            
            if(i%10==0):
                list_acc.append(self.accuracy(self.labeledset))
                list_it.append(i)
                
        plt.subplot(121)
        plt.plot(list_it,list_acc)
    
        
    def predict(self, x):
        
        scalar = np.dot(self.w, np.hstack((x,[-1])))

        if(scalar >= 0):
            return 1, 0     
        return -1, 0
        


def tirage(VX, m, r):
    
    if (not r):
        return rd.sample(VX, m)
    else:
        return [rd.choice(VX) for i in range(0,m)]



def echantillonLS(X, m, r):
    
    ls = LabeledSet(X.getInputDimension())
    indexes = [i for i in range(0, X.size())]
    sample = tirage(indexes, m, r)

    for i in sample:
        ls.addExample(X.getX(i), X.getY(i))
    
    return ls


        

class ClassifierBaggingPerceptron(Classifier):

    def __init__(self, input_dimension, nbperc, percent, r=True):

        self.input_dimension = input_dimension
        self.nbperc = nbperc
        self.percent = percent
        self.r = r
        self.listPerceptrons = None
        self.labeledset = None
        
    def train(self, labeledset):
        
        listPerceptrons = []
        m = math.ceil(labeledset.size()* self.percent)
        for i in range(0, self.nbperc):
            
            sample = echantillonLS(labeledset, m, self.r)
            t = Perceptron(self.input_dimension, 0.03, 100)
            t.train(sample)
            listPerceptrons.append(t)
            
        self.listPerceptrons = listPerceptrons
        self.labeledset = labeledset
        
    def predict(self, x):
        vote = 0
        nb_plus = 0

        for perc in self.listPerceptrons:
            pred, acc_plus = perc.predict(x)
            vote += pred
            if(pred>0):
                nb_plus+=1

        
        if vote >= 0:
            return +1,  nb_plus/(len(self.listPerceptrons)*1.0)
        return -1,  nb_plus/(len(self.listPerceptrons)*1.0)
    
    def outOfBagsError(self):
        """Retourne une estimation de l'erreur du classifieur par la méthode Out Of Bags"""
        nb_ko = 0
        for i in range(0, self.labeledset.size()):
            
            #Pour chaque exemple xi,yi, déterminer les perceptrons qui ne contiennent pas xi,yi
            xi, yi = self.labeledset.getX(i), self.labeledset.getY(i)
            oobPerceptrons = [perc for perc in self.listPerceptrons if xi not in perc.labeledset.x]
            
            #Faire une prediction sur ces perceptrons
            vote = 0
            for perc in oobPerceptrons:
                v, acc_plus = perc.predict(xi)
                vote+=v

            if vote * yi < 0: #mauvais résultat
                nb_ko += 1
               
        return nb_ko / (self.labeledset.size()* 1.0)


class KNN(Classifier):
    
    def __init__(self,input_dimension,k):
        
        self.input_dimension=input_dimension
        self.k=k
        self.labeledset=None
        
    def predict(self,x):
        
        distances=[]
        
        for i in range(0,self.labeledset.size()):
            
                
            dist=np.linalg.norm(self.labeledset.getX(i)-x)
            
            distances.append((dist,self.labeledset.getY(i)))
         
        distances.sort()
        score=0

        for i in range(0,self.k):
            if(distances[i][1]==1):
                score+=1
            else:
                score-=1

        if(score>=0):
            return 1, 0
        else:
            return -1, 0

        
    def train(self, labeledset):
        self.labeledset=labeledset
        
    def setK(self, k):
        self.k=k



class ClassifierBaggingKNN(Classifier):

    def __init__(self, input_dimension, nbknn, percent, r=True):

        self.input_dimension = input_dimension
        self.nbknn = nbknn
        self.percent = percent
        self.r = r
        self.listKnns = None
        self.labeledset = None
        
    def train(self, labeledset):
        
        listKnns = []
        m = math.ceil(labeledset.size()* self.percent)
        for i in range(0, self.nbknn):
            
            sample = echantillonLS(labeledset, m, self.r)
            t=KNN(self.input_dimension, 3)
            t.train(sample)
            listKnns.append(t)
            
        self.listKnns = listKnns
        self.labeledset = labeledset
        
    def predict(self, x):
        vote = 0
        nb_plus = 0

        for knn in self.listKnns:
            pred, acc_plus = knn.predict(x)
            vote += pred
            if(pred>0):
                nb_plus+=1


        if vote >= 0:
            return 1, nb_plus/(len(self.listKnns)*1.0)
        return -1, nb_plus/(len(self.listKnns)*1.0)
    
    def outOfBagsError(self):
        """Retourne une estimation de l'erreur du classifieur par la méthode Out Of Bags"""
        nb_ko = 0
        for i in range(0, self.labeledset.size()):
            
            #Pour chaque exemple xi,yi, déterminer les perceptrons qui ne contiennent pas xi,yi
            xi, yi = self.labeledset.getX(i), self.labeledset.getY(i)
            oobKnns = [knn for knn in self.listKnns if xi not in knn.labeledset.x]
            
            #Faire une prediction sur ces perceptrons
            vote = 0
            for knn in oobKnns:
                v, acc_plus = knn.predict(xi)
                vote+=v
            
            if vote * yi < 0: #mauvais résultat
                nb_ko += 1
               
        return nb_ko / (self.labeledset.size()* 1.0)



class ClassifierMatch(Classifier):

    def __init__(self, input_dimension, error):

        self.input_dimension = input_dimension
        self.error=error
        self.baggKNN=None
        self.baggPerc=None
        
    def train(self, labeledset):
        


        self.baggKNN=ClassifierBaggingKNN(self.input_dimension, 30,0.2,True)
        self.baggPerc=ClassifierBaggingPerceptron(self.input_dimension, 30, 0.2, True)

        self.baggKNN.train(labeledset)
        self.baggPerc.train(labeledset)
        
    def predict(self, x, home_team_name, away_team_name):
        
        voteKNN, acc_KNN = self.baggKNN.predict(x)
        votePerc, acc_Perc = self.baggPerc.predict(x)

        acc_plus=(acc_KNN+acc_Perc)/2.0

        if(acc_plus<(0.5-self.error)):
            return -1,str(away_team_name)+" win with "+str(1.0-acc_plus)+" of chance !"
        elif(acc_plus>(0.5+self.error)):
            return 1,str(home_team_name)+" win with "+str(acc_plus)+" of chance !"
        else:
            return 0,"Draw !"
