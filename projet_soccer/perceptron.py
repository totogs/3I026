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
            output = self.predict(dataset.getX(i))

            if (output == dataset.getY(i)):
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

        ls=np.arange(labeledset.size())
        np.random.shuffle(ls)
        
        for i in range(self.nbr_it):
            
            for j in ls:
                vote = self.predict(self.labeledset.getX(j))
                if(self.labeledset.getY(j)*vote<0):
                    
                    self.w = self.w + self.lg_rate*self.labeledset.getY(j)*np.hstack((self.labeledset.getX(j),[-1]))

    
        
    def predict(self, x):
        
        scalar = np.dot(self.w, np.hstack((x,[-1])))

        if(scalar >= 0):
            return 1   
        return -1
        


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
            pred = perc.predict(x)
            vote += pred
            if(pred>0):
                nb_plus+=1

        self.acc_home=nb_plus/(len(self.listPerceptrons)*1.0)
        self.acc_away=1.0-self.acc_home

        if vote >= 0:
            return +1
        return -1
    
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
                v= perc.predict(xi)
                vote+=v

            if vote == yi < 0: #mauvais résultat
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
        nb_home=0
        nb_away=0
        nb_draw=0

        for i in range(0,self.k):
            if(distances[i][1]==1):
                nb_home+=1
            elif(distances[i][1]==-1):
                nb_away+=1
            else:
                nb_draw+=1

        self.acc_home=nb_home/(self.k*1.0)
        self.acc_away=nb_away/(self.k*1.0)
        self.acc_draw=nb_draw/(self.k*1.0)

        if(nb_home>nb_away and nb_home>nb_draw):
            return 1
        if(nb_away>nb_home and nb_away>nb_draw):
            return -1
        else:
            return 0

        
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
            t=KNN(self.input_dimension, 10)
            t.train(sample)
            listKnns.append(t)
            
        self.listKnns = listKnns
        self.labeledset = labeledset
        
    def predict(self, x):

        nb_home=0
        nb_away=0
        nb_draw=0

        for knn in self.listKnns:
            pred = knn.predict(x)
            nb_home+=knn.acc_home
            nb_away+=knn.acc_away
            nb_draw+=knn.acc_draw

        self.acc_home=nb_home/(len(self.listKnns)*1.0)#Calcul du pourcentage de vote pour que l'equipe à domicile gagne
        self.acc_away=nb_away/(len(self.listKnns)*1.0)#Calcul du pourcentage de vote pour que l'equipe en extérieur gagne
        self.acc_draw=nb_draw/(len(self.listKnns)*1.0)#Calcul du pourcentage de vote pour un match nul

        if(nb_home>nb_away and nb_home>nb_draw):
            return 1
        if(nb_away>nb_home and nb_away>nb_draw):
            return -1
        else:
            return 0


    
    def outOfBagsError(self):
        """Retourne une estimation de l'erreur du classifieur par la méthode Out Of Bags"""
        nb_ko = 0
        for i in range(0, self.labeledset.size()):
            
            #Pour chaque exemple xi,yi, déterminer les perceptrons qui ne contiennent pas xi,yi
            xi, yi = self.labeledset.getX(i), self.labeledset.getY(i)
            oobKnns = [knn for knn in self.listKnns if xi not in knn.labeledset.x]
            
            #Faire une prediction sur ces knn
            nb_home=0
            nb_away=0
            nb_draw=0
            for knn in oobKnns:
                v = knn.predict(xi)
                nb_home+=knn.acc_home
                nb_away+=knn.acc_away
                nb_draw+=knn.acc_draw
            
            if(nb_home>nb_away and nb_home>nb_draw):
                if(yi!=1):
                    nb_ko+=1
            if(nb_away>nb_home and nb_away>nb_draw):
                if(yi!=-1):
                    nb_ko+=1
            else:
                if(yi!=0):
                    nb_ko+=1
               
        return nb_ko / (self.labeledset.size()* 1.0)



class ClassifierMatch(Classifier):

    def __init__(self, input_dimension):

        self.input_dimension = input_dimension
        self.baggKNN=None
        self.baggPerc=None
        
    def train(self, labeledset):
        


        self.baggKNN=ClassifierBaggingKNN(self.input_dimension, 30,0.3,True)
        self.baggPerc=ClassifierBaggingPerceptron(self.input_dimension, 30, 0.3, True)

        self.baggKNN.train(labeledset)
        self.baggPerc.train(labeledset)
        
    def predict(self, x, home_team_name="home", away_team_name="away"):
        
        voteKNN = self.baggKNN.predict(x)
        votePerc = self.baggPerc.predict(x)

        #pourcentage de chance que l'équipe à domicile gagne
        self.acc_home=(self.baggKNN.acc_home+self.baggPerc.acc_home)/2
        #pourcentage de chance que l'équipe en exterieur gagne
        self.acc_away=(self.baggKNN.acc_away+self.baggPerc.acc_away)/2
        #pourcentage de chance qu'il y ait un match nul
        self.acc_draw=1-(self.acc_home+self.acc_away)

        #Si le pourcentage de chance que l'equipe en extérieur gagne est supérieur aux autres on renvoit -1
        if(self.acc_away>self.acc_home and self.acc_away>self.acc_draw):
            print(str(away_team_name)+" win vs "+str(home_team_name)+" (home win chance: "+str(self.acc_home)+", away win chance: "+str(self.acc_away)+", draw chance:"+str(self.acc_draw)+")")
            return -1
        #Si le pourcentage de chance que l'equipe à domicile gagne est supérieur aux autres on renvoit 1
        elif(self.acc_home>self.acc_away and self.acc_home>self.acc_draw):
            print(str(home_team_name)+" win vs "+str(away_team_name)+" (home win chance: "+str(self.acc_home)+", away win chance: "+str(self.acc_away)+", draw chance:"+str(self.acc_draw)+")")
            return 1
        #Si le pourcentage de chance qu'il y ait un match nul est supérieur aux autres on renvoit 0
        else:
            print( "Draw !  (home win chance: "+str(self.acc_home)+", away win chance: "+str(self.acc_away)+", draw chance: "+str(self.acc_draw)+")")
            return 0