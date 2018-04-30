# -*- coding: utf-8 -*-


# Importations nécessaires pour l'ensemble des fonctions de ce fichier:
import pandas as pd
import matplotlib.pyplot as plt

import math
import random as rd

# Normalisation des données :

def normalisation(df):
    
    norm=df.copy()
    
    
    for col in norm.columns:
        
            mini=norm[col].min()
            maxi=norm[col].max()
            
            norm[col] = (norm[col]-mini)/(maxi-mini)
            
    return norm

# -------
# Fonctions distances

def dist_vect(df1, df2):
    
    dist=0
    
    for i in range(0,len(df1)):
        
        dist+=(df1.iloc[i]-df2.iloc[i])**2
        
    return dist**(0.5)

# -------
# Calculs de centroïdes :

def centroide(df):
    
    centr=df.copy()
        
    return pd.DataFrame(centr.mean()).T

# -------
# Inertie des clusters :

def inertie_cluster(df):
    
    centroid = centroide(df)
    inertie=0
    
    for i in range(0,len(df)):
        
        inertie+=dist_vect(centroid.iloc[0], df.iloc[i])**2
        
    return inertie

# -------
# Algorithmes des K-means :


def initialisation(k,df):
    
    kdf = pd.DataFrame(columns=list(df.columns))
    
    for i in range(0,k):
        
        index=rd.randint(0,len(df)-1)
        kdf.loc[index] = df.iloc[index]
        
    
    return kdf

# -------
def plus_proche(ex, df):
    
    minvalue=100000000
    mini=-1
    
    for i in range(0,len(df)):
   
        val=dist_vect(df.iloc[i],ex)
        
        if(minvalue>val):
            minvalue=val
            mini=i
            
            
    return mini

# -------
def affecte_cluster(dfTrain, centroides):
    
    matrice = dict()
    for i in range(len(centroides)):
        matrice[i]=[]
    
    for i in range(len(dfTrain)):
        
        icentr=plus_proche(dfTrain.loc[i], centroides )
        matrice[icentr].append(i)
        
        
    
    return matrice

# -------
def nouveaux_centroides(df, da):
    
    kdf = pd.DataFrame(columns=list(df.columns))
    
    for k, liste in da.items():
        
        mean=0
        for i in range(len(liste)):
            
            mean+=df.iloc[liste[i]]
      

        kdf.loc[k]=mean/len(liste)
        
    return kdf

# -------
def inertie_globale(df, da):
    
    new_centr=nouveaux_centroides(df,da)
    
    inertie=0
    
    for k, liste in da.items():
        df_cluster = pd.DataFrame(columns=list(df.columns))
        
        
        df_cluster=df.iloc[liste]
        
        inertie+=inertie_cluster(df_cluster)
    
    return inertie

# -------


def kmoyennes(k, df, eps, iter_max):
    
    centroid=initialisation(k,df)
    inertie=inertie_cluster(df)
    
    i=0
    while(i<iter_max):
        da=affecte_cluster(df,centroid)
        centroid=nouveaux_centroides(df,da)
        new_inertie=inertie_globale(df,da)
        print("Iteration ",i," Inertie : ",new_inertie," Difference : ",math.fabs(inertie-new_inertie))
        
        if(math.fabs(inertie-new_inertie)<eps):
            break
            
        inertie=new_inertie
        i+=1
    
    return centroid, da

# -------
# Affichage :
def affiche_resultat(df, centroid, da):
    
    cmap = plt.cm.get_cmap("hsv", len(centroid)+1)

    for k, liste in da.items():
        
        df_cluster = pd.DataFrame(columns=list(df.columns))

        for i in range(len(liste)):
            df_cluster.loc[i]=df.iloc[liste[i]]
        
        plt.scatter(df_cluster['X'],df_cluster['Y'],color=cmap(k))
    
    plt.scatter(centroid['X'],centroid['Y'],color='r',marker='x')

# -------
