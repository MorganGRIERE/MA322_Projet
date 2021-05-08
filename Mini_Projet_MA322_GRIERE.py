# -*- coding: utf-8 -*-
"""
@author: Morgan GRIERE
@classe : 3PSC2
@matiere: Ma322
"""

#-------------------------------------------
#imports
#-------------------------------------------

import math as m
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sc

#-------------------------------------------
#définitions du programme
#-------------------------------------------

#-------------------------------------------
#3.2 Présentation de benchmark utilisé

# méthode de résolution

def Mat(a,b,K,f,N):
    h = (b-a)/(N-1)                     # N points donc N-1 intervalles
    t = np.linspace(a,b,N)
    x = np.linspace(a,b,N)
    A = np.zeros((N,N))
    F = np.zeros(N)
    for i in range (N):
        F[i] = f(t[i])
        for j in range (N):
            if (j == 0) or (j == N-1):
                A[i,j] = K(x[i],t[j])
            else:
                A[i,j] = 2*K(x[i],t[j])
    M = np.eye(N)- (h/2)*A
    F = F.T
    return t,F,M

# fonction des données de l'énoncé

def F_benchmark(x):
    return (m.cos((m.pi*x)/2) -(8/m.pi))

def K_benchmark(x,t):
    return 2

# Calcul valeur exacte

def Valeur_exacte_benchmark(x):
    U_exacte = []
    for i in  x:
        valeur = m.cos(m.pi*i/2)
        U_exacte.append(valeur)
    return U_exacte

# Calcul de l'erreur

def Erreur_benchmark(U,V):
    return np.linalg.norm(U-V,2)   

#-------------------------------------------
#3.3 Équation de Love en électrostatique

def K_Love(x,t):
    return ((1/m.pi)*(1/(1+(x-t)**2)))

def f_Love(x):
    return 1


#-------------------------------------------
#4. Circuit RLC

# fonction F(t,Y)

def rlcprim(Y,t):
    C = 10**(-6)
    R = 3
    L = 0.5
    e = 10
    M = np.array([(1/L)*(e-Y[1]-R*Y[0]),(1/C)*Y[0]])
    return M

# méthodes de résolution

def Rung_Kutta_4(f,t,y0,h):
    Ye =  np.zeros(shape=(len(t),2))
    y = y0
    for n in range (0,len(t)):
        Ye[n,0] = y[0]
        Ye[n,1] = y[1]
        k1 = f(y,t[n])
        k2 = f(y+(h/2)*k1,t[n]+(h/2))
        k3 = f(y+(h/2)*k2,t[n]+(h/2))
        k4 = f(y+h*k3,t[n]+h)
        y = y + (h/6)*(k1+2*k2+2*k3+k4)
    return Ye

def methode_odeint(F,Y0,t):
    Yode = sc.odeint(F, Y0,t)
    return Yode


#-------------------------------------------
#5. Moteur à courant continu

def vecteur_U(t):
    U =[]
    for i in t:
        if (i>=10 and i<=50):
            U.append(5)
        else:
            U.append(0)
    return U

def moteurCC(Y,t):
    R = 5
    L = 50*10**(-3)
    Ke = 0.2
    Kc = 0.1
    Fm = 0.01
    Jm = 0.05
    if (t>=10 and t<=50):
            U = 5
    else:
            U = 0
    M = np.array([(1/L)*(U-R*Y[0]-Ke*Y[1]),(1/Jm)*(Kc*Y[0]-Fm*Y[1])])
    return M


#-------------------------------------------
#6. Mouvement d'une fusée

def fusee(Y,t) :
    D = 4
    a0 = 8*10**3
    g = 9.81
    k0 = 0.1
    u = 2*10**3
    
    Yprime = np.zeros(3)
    if (Y[1] < 80) :
        Y[1] = 80
        D = 0
    
    Yprime[0] = D*u/Y[1] -g -k0*m.exp(-Y[2]/a0)*Y[0]**2/Y[1]
    Yprime[1] = -D
    Yprime[2] = Y[0]
    
    return Yprime


#-------------------------------------------
#7. Modèle proie-prédateur

def proies(N) :
    a1 = 3
    Liste_proies = []
    for i in range (0,N):
        if i == 0 :
            Liste_proies.append(5)
        else:
            Liste_proies.append(Liste_proies[i-1]*a1)
    return Liste_proies

def predateurs(N) :
    a2 = 2
    Liste_predateurs = []
    for i in range (0,N) :
        if i == 0:
            Liste_predateurs.append(3)
        else:
            if Liste_predateurs[i-1]<0 :
                Liste_predateurs.append(Liste_predateurs[i-1]*a2)
            else:
                Liste_predateurs.append(Liste_predateurs[i-1]*-a2)
    return Liste_predateurs

def proie_predateur(Y,t):
    a1 = 3      #3 ou 11
    b1 = 1    #1 ou 4
    a2 = 2      #2 ou 12
    b2 = 1      #1 ou 0.8
    return np.array([a1*Y[0] - b1*Y[0]*Y[1],-a2*Y[1]+b2*Y[0]*Y[1]])

# Méthode de résolution

def Euler_Exp(f,y0,h,t):
    Ye =  np.zeros(shape=(len(t),2))
    y = y0
    for n in range (0,len(t)):
        Ye[n,0] = y[0]
        Ye[n,1] = y[1]
        y = y + h*f(y,t[n])
    return Ye


#-------------------------------------------
# programme
#-------------------------------------------


#-------------------------------------------
#3.2 Présentation de benchmark utilisé

# Résolution avec la fonction Mat

t,F,M = Mat(-1,1,K_benchmark,F_benchmark,10)
U = np.linalg.inv(M)@F

#Solution excate

U_exacte = Valeur_exacte_benchmark(t)

# Graphique

plt.figure(1)
plt.plot(t, U, label="Solution Approchée")
plt.plot(t, U_exacte, label="Solution Exacte")
plt.title("Résolution de l'équation intégrale avec la présentation de benchmark")
plt.xlabel("Temps (s)")
plt.ylabel("Valeur de u")
plt.grid()
plt.legend(loc = "upper right")
plt.show()

# Erreur

V = U.T
U = np.array(U_exacte).T
erreur = Erreur_benchmark(U,V)
print("\nErreur || U - V ||2 : ", erreur)

#-------------------------------------------
#3.3 Équation de Love en électrostatique

# Résolution avec la fonction Mat

t_Love, F_Love, M_Love = Mat(-1,1,K_Love,f_Love,10)
U_Love = np.linalg.inv(M_Love)@F_Love

# Graphique

plt.figure(2)
plt.plot(t_Love, U_Love, label="Solution Approchée")
plt.title("Résolution numérique de l'équation de Love")
plt.xlabel("Temps (s)")
plt.ylabel("Valeur de u")
plt.grid()
plt.legend(loc = "upper right")
plt.show()


#-------------------------------------------
#4. Circuit RLC

# données initiales

a_RLC = 0
b_RLC = 2
N_RLC = 1000 #200
t_RLC = np.linspace(a_RLC ,b_RLC ,N_RLC)   
Y0 = np.array([0,0])  
h_RLC = (b_RLC - a_RLC)/N_RLC

# Résolutions

liste_RK4_RLC = Rung_Kutta_4(rlcprim ,t_RLC ,Y0 ,h_RLC)
liste_Yode_RLC = methode_odeint(rlcprim ,Y0, t_RLC)

#Graphes intensité du circuit 

plt.figure(2)
plt.plot(t_RLC,liste_Yode_RLC[:,0],label='Odeint')
plt.title("Intensité du système RLC") 
plt.xlabel('Temps (s)')
plt.ylabel('Intensité i(t) (A)')
plt.grid()
plt.legend()
plt.show()

plt.figure(3)
plt.plot(t_RLC,liste_RK4_RLC[:,0],label='Runge Kutta')
plt.title("Intensité du système RLC") 
plt.xlabel('Temps (s)')
plt.ylabel('Intensité i(t) (A)')
plt.grid()
plt.legend()
plt.show()

#Graphes tension du circuit 

plt.figure(4)
plt.plot(t_RLC,liste_Yode_RLC[:,1],label='Odeint')
plt.title("Tension du système RLC") 
plt.xlabel('Temps (s)')
plt.ylabel('Tension s(t) (V)')
plt.grid()
plt.legend()
plt.show()

plt.figure(5)
plt.plot(t_RLC,liste_RK4_RLC[:,1],label='Runge Kutta')
plt.title("Tension du système RLC") 
plt.xlabel('Temps (s)')
plt.ylabel('Tension s(t) (V)')
plt.grid()
plt.legend()
plt.show()


#-------------------------------------------
#5. Moteur à courant continu

# données de l'énoncé

a_MCC = 0
b_MCC = 80
N_MCC = 8000            # pas de 0.01
Y0_MCC = [0,0]
Kc = 0.1

# calcul la tension u(t)

t_MCC = np.linspace(a_MCC ,b_MCC ,N_MCC)
liste_U_MCC = vecteur_U(t_MCC)

# Représentation de la tension

plt.figure(6)
plt.plot(t_MCC,liste_U_MCC,label="Tension")
plt.title("Evolution de la tension")
plt.xlabel("Temps (s)")
plt.ylabel("u(t) (V)")
plt.legend()
plt.grid()
plt.show()

# Résolution de l'équation intégrale avec Odeint

liste_Yode_MCC = methode_odeint(moteurCC ,Y0_MCC , t_MCC)

# Couple moteur

Cm_CC = Kc*liste_Yode_MCC[:,0]

# Graphe couple moteur

plt.figure(7)
plt.plot(t_MCC,Cm_CC,label='Odeint')
plt.title("Couple moteur") 
plt.xlabel('Temps (s)')
plt.ylabel('Cm(t) en N.m')
plt.grid()
plt.legend()
plt.show()

# Graphe vitesse angulaire

plt.figure(8)
plt.plot(t_MCC,liste_Yode_MCC[:,1],label='Odeint')
plt.title("Vitesse Angulaire") 
plt.xlabel('Temps (s)')
plt.ylabel('w(t) en rad/s')
plt.grid()
plt.legend()
plt.show()

#-------------------------------------------
#6. Mouvement d'une fusée

# données de l'énoncé

a_F = 0
b_F =160
N_F = 1000

# Résolution de l'équation intégrale avec Odeint

t_F = np.linspace(a_F, b_F, N_F)
Y0_F = [0,400,0]
liste_Yode_F = methode_odeint( fusee ,Y0_F, t_F)

#Graphe de la vitesse de la fusée

plt.figure(9)
plt.plot(t_F,liste_Yode_F[:,0])
plt.xlabel("Temps (s)")
plt.ylabel("Vitesse v (m/s)")
plt.title("Vitesse de la fusée")
plt.grid()
plt.show()

#Graphe de la trajectoire de la fusée

plt.figure(10)
plt.plot(t_F,liste_Yode_F[:,2])
plt.xlabel("Temps (s)")
plt.ylabel("Hauteur (m)")
plt.title("Trajectoire de la fusée")
plt.grid()
plt.show()

#-------------------------------------------
#7. Modèle proie-prédateur

# données de l'exercice

a_PP = 0
b_PP = 10
N_PP = 100
Y0_PP = [5,3]
h_PP = (b_PP - a_PP)/N_PP

# Evolution sans prédateurs

t_PP = np.linspace(a_PP, b_PP ,10)
Liste_Lievre = proies(10)
plt.figure(11)
plt.plot(t_PP,Liste_Lievre )
plt.xlabel("Nombre d'années")
plt.ylabel("Nombre d'animaux")
plt.title("Evolution du nombre de proies en fonction du temps sans prédateur")
plt.grid()
plt.show()

# Evolution sans proies

Liste_Lynx = predateurs(10)
plt.figure(12)
plt.plot(t_PP,Liste_Lynx )
plt.xlabel("Nombre d'années")
plt.ylabel("Nombre d'animaux")
plt.title("Evolution du nombre de proies en fonction du temps sans proie")
plt.grid()
plt.show()

# Résolution de l'équation avec Euler Explicite

t_PP2 = np.linspace(a_PP ,b_PP ,N_PP)
Liste_EE = Euler_Exp(proie_predateur, Y0_PP, h_PP,t_PP2)
plt.figure(13)
plt.plot(t_PP2,Liste_EE[:,0], label="proies")
plt.plot(t_PP2,Liste_EE[:,1], label="predateur")
plt.legend()
plt.xlabel("Nombre d'années")
plt.ylabel("Nombre d'animaux")
plt.ylim(-22,22)
plt.title("Evolution du nombre de proies et des prédateurs avec Euler")
plt.grid()
plt.show()

# Résolution de l'équation avec Odeint

Liste_O_PP = methode_odeint(proie_predateur, Y0_PP,t_PP2)
plt.figure(14)
plt.plot(t_PP2,Liste_O_PP[:,0], label="proies")
plt.plot(t_PP2,Liste_O_PP[:,1], label="predateur")
plt.legend()
plt.xlabel("Nombre d'années")
plt.ylabel("Nombre d'animaux")
plt.title("Evolution du nombre de proies et des prédateurs avec Odeint")
plt.grid()
plt.show()

# Portait de phase Euler

plt.figure(15)
plt.plot(Liste_EE[:,0],Liste_EE[:,1])
plt.xlabel("Proies")
plt.ylabel("Prédateurs")
plt.grid()
plt.legend()
plt.xlim(-20,15)
plt.title("Portrait de phase avec Euler")
plt.show()

# Portait de phase odeint

plt.figure(16)
plt.plot(Liste_O_PP[:,0],Liste_O_PP[:,1])
plt.xlabel("Proies")
plt.ylabel("Prédateurs")
plt.grid()
plt.legend()
plt.title("Portrait de phase avec Odeint")
plt.show()



