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
                
                if(self.labeledset.getY(j)*self.predict(self.labeledset.getX(j))<0):
                    
                    self.w = self.w + self.lg_rate*self.labeledset.getY(j)*np.hstack((self.labeledset.getX(j),[-1]))

            
            if(i%10==0):
                list_acc.append(self.accuracy(self.labeledset))
                list_it.append(i)
                
        plt.subplot(121)
        plt.plot(list_it,list_acc)
    
        
    def predict(self, x):
        
        scalar = np.dot(self.w, np.hstack((x,[-1])))

        if(scalar >= 0):
            return 1     
        return -1
        

        


def classe_majoritaire(labeledset):
    
    cpt=0
    
    for i in range(labeledset.size()):
        if(labeledset.getY(i)==1):
            cpt+=1
            
    if(cpt>=labeledset.size()/2):
        return 1
    else:
        return -1



import math

def shannon(P):
    k=len(P)
    HsP=0
    if(k==1):
        return 0.0
    
    for pi in P:
        if(pi>0.0):
            HsP+=pi*math.log(pi,k)
        
    return -HsP



def entropie(dataset):
    
    card = dataset.size()
    dimension = dataset.getInputDimension()
    
    distrib = dict()
    
    for i in range(dataset.size()):
        yi=dataset.getY(i)[0]
        
        if(yi in distrib):
            distrib[yi] +=1
        else:
            distrib[yi] = 1
        
    sha = [float(val)/card for val in distrib.values()]
    
    return shannon(sha)



def discretise(LSet, col):
    """ LabelledSet * int -> tuple[float, float]
        col est le numéro de colonne sur X à discrétiser
        rend la valeur de coupure qui minimise l'entropie ainsi que son entropie.
    """
    # initialisation:
    min_entropie = 1.1  # on met à une valeur max car on veut minimiser
    min_seuil = 0.0     
    # trie des valeurs:
    ind= np.argsort(LSet.x,axis=0)
    
    # calcul des distributions des classes pour E1 et E2:
    inf_plus  = 0               # nombre de +1 dans E1
    inf_moins = 0               # nombre de -1 dans E1
    sup_plus  = 0               # nombre de +1 dans E2
    sup_moins = 0               # nombre de -1 dans E2       
    # remarque: au départ on considère que E1 est vide et donc E2 correspond à E. 
    # Ainsi inf_plus et inf_moins valent 0. Il reste à calculer sup_plus et sup_moins 
    # dans E.
    for j in range(0,LSet.size()):
        if (LSet.getY(j) == -1):
            sup_moins += 1
        else:
            sup_plus += 1
    nb_total = (sup_plus + sup_moins) # nombre d'exemples total dans E
    
    # parcours pour trouver le meilleur seuil:
    for i in range(len(LSet.x)-1):
        v_ind_i = ind[i]   # vecteur d'indices
        courant = LSet.getX(v_ind_i[col])[col]
        lookahead = LSet.getX(ind[i+1][col])[col]
        val_seuil = (courant + lookahead) / 2.0;
        # M-A-J de la distrib. des classes:
        # pour réduire les traitements: on retire un exemple de E2 et on le place
        # dans E1, c'est ainsi que l'on déplace donc le seuil de coupure.
        if LSet.getY(ind[i][col])[0] == -1:
            inf_moins += 1
            sup_moins -= 1
        else:
            inf_plus += 1
            sup_plus -= 1
        # calcul de la distribution des classes de chaque côté du seuil:
        nb_inf = (inf_moins + inf_plus)*1.0     # rem: on en fait un float pour éviter
        nb_sup = (sup_moins + sup_plus)*1.0     # que ce soit une division entière.
        # calcul de l'entropie de la coupure
        val_entropie_inf = shannon([inf_moins / nb_inf, inf_plus  / nb_inf])
        val_entropie_sup = shannon([sup_moins / nb_sup, sup_plus  / nb_sup])
        val_entropie = (nb_inf / nb_total) * val_entropie_inf + (nb_sup / nb_total) * val_entropie_sup
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (min_entropie > val_entropie):
            min_entropie = val_entropie
            min_seuil = val_seuil
    return (min_seuil, min_entropie)



def divise(LSet,att,seuil):
    
    Linf = LabeledSet(LSet.getInputDimension());
    Lsup = LabeledSet(LSet.getInputDimension());
    
    for i in range(len(LSet.x)):
        
        if(LSet.getX(i)[att]<seuil):
            
            Linf.addExample(LSet.getX(i),LSet.getY(i))
            
        else:
            
            Lsup.addExample(LSet.getX(i), LSet.getY(i))
            
        
    return Linf, Lsup


import graphviz as gv

class ArbreBinaire:
    def __init__(self):
        self.attribut = None   # numéro de l'attribut
        self.seuil = None
        self.inferieur = None # ArbreBinaire Gauche (valeurs <= au seuil)
        self.superieur = None # ArbreBinaire Gauche (valeurs > au seuil)
        self.classe = None # Classe si c'est une feuille: -1 ou +1
        
    def est_feuille(self):
        #rend True si l'arbre est une feuille
        return self.seuil == None
    
    def ajoute_fils(self,ABinf,ABsup,att,seuil):
        #ABinf, ABsup: 2 arbres binaires
        #att: numéro d'attribut
        #seuil: valeur de seuil
        
        self.attribut = att
        self.seuil = seuil
        self.inferieur = ABinf
        self.superieur = ABsup
    
    def ajoute_feuille(self,classe):
        #classe: -1 ou + 1
        
        self.classe = classe
        
    def classifie(self,exemple):
        #exemple : numpy.array
        #rend la classe de l'exemple: +1 ou -1
        
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] <= self.seuil:
            return self.inferieur.classifie(exemple)
        return self.superieur.classifie(exemple)
    
    def to_graph(self, g, prefixe='A'):
        #construit une représentation de l'arbre pour pouvoir l'afficher
        
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.attribut))
            self.inferieur.to_graph(g,prefixe+"g")
            self.superieur.to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))
        
        return g

def construit_AD(LSet, eps):
    
    AD = ArbreBinaire()
    

    if(entropie(LSet)<=eps):
        AD.ajoute_feuille(classe_majoritaire(LSet))
    else:
        
        entropie_min=10000
        seuil_min=1000000
        imin=0
        for i in range(LSet.getInputDimension()):
            
            seuil , entrp = discretise(LSet,i)
            
            if(entropie_min>entrp):
                entropie_min=entrp
                seuil_min=seuil
                imin=i
          
        Linf, Lsup = divise(LSet, imin, seuil_min)
        
        AD.ajoute_fils(construit_AD(Linf, eps), construit_AD(Lsup, eps), imin, seuil_min)
        
    return AD


class ArbreDecision(Classifier):
    # Constructeur
    def __init__(self,epsilon):
        # valeur seuil d'entropie pour arrêter la construction
        self.epsilon= epsilon
        self.racine = None
    
    # Permet de calculer la prediction sur x => renvoie un score
    def predict(self,x):
        # classification de l'exemple x avec l'arbre de décision
        # on rend 0 (classe -1) ou 1 (classe 1)
        classe = self.racine.classifie(x)
        if (classe == 1):
            return(1)
        else:
            return(-1)
    
    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,set):
        # construction de l'arbre de décision 
        self.set=set
        self.racine = construit_AD(set,self.epsilon)

    # Permet d'afficher l'arbre
    def plot(self):
        gtree = gv.Digraph(format='png')
        return self.racine.to_graph(gtree)

