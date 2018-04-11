'''EP 2
   Aluno: Fabio Carvalho de Souza
   N° USP: 9425125
'''   

import numpy as np
import matplotlib.pyplot as plt
from random import random
import matplotlib.patches as patches
from random import randint
import matplotlib.path as mpath
import math
from random import randint
from scipy import stats
from scipy.stats import linregress

vet=[2**16,2**10,2**12,2**14]

def lin_regression(x, y):
    """ (y = m * x + b + error)"""
    m, b, R, p, SEm = linregress(x, y)
    n = len(x)
    SSx = np.var(x, ddof=1) * (n-1)
    SEb2 = SEm**2 * (SSx/n + np.mean(x)**2)
    SEb = SEb2**0.5
    return m, b, SEm, SEb, R, p

def l0(n):
    m = 1
    j = 1
    while m&(~n)==0:
        m *= 2
        j += 1
    return j

#função para gerar as sequencias de numeros aleatorios usando sequencias/ metodo sobol, com uso de um polinomio
# um valor de inicio para sequencia e uma precisão para o sistema

def sobolSeq(pol,inicia,precisao):

    if len(pol)!=len(inicia):
        raise ValueError('Valor Incosistente')

    for p in pol:
        if p!=0 and p!=1:
            raise ValueError('Valor Incosistente')

    for m in inicia:
        if m%2==0:
            raise ValueError('Valor Incosistente')

    if precisao<0:
        raise ValueError('Precisão negativa é invalida')

    q = len(inicia)
    M = inicia
    V = [ M[i]*2**(precisao-i-1) for i in range(q) ]
    A = pol

    for i in range(len(M),precisao):
        m = M[-q]^((2**q)*M[-q])
        for j in range(1,q):
            m ^= (2**j)*M[-j]*A[j]
        M.append( m )
        V.append( m*2**(precisao-i-1) )
    n = 1
    S = 0
    while True:
        S = S ^ V[l0(n)]
        n += 1
        yield float(S)/(2**precisao)

#função que junta as distribuições pseudo com as quasi aleatorias
def principal():
 #contadores para ambos os casos
    
 k=0
 conta=0
 q=[]*vet[0]
 c=[]*vet[0]
 t=[]*vet[0]
 s=[]*vet[0]


 l=[]*vet[0]
 w=[]*vet[0]
 a=[]*vet[0]
 z=[]*vet[0]
 
 s1 = sobolSeq([1,1],[1,1],32)
 s2 = sobolSeq([1,0,1],[1,3,7],32)

 while k<100:
  #inicializando variaveis para o metodo quasi aleatorio
  N = vet[0]   
  X =  [ next(s1) for _ in range(N) ]
  Y =  [ next(s2) for _ in range(N) ]

  #inicializando variaveis para o metodo pseudo aleatorio
  n=vet[0]
  conta2=0
  conta=0
  x = np.random.rand(n)
  y = np.random.rand(n)
  
  #adicionando num vetor as dimenções dos quadradinhos/ ladrilhos de verificação de ocorrencia de pontos
  i=randint(1,9)
  j=randint(1,9)
  L=[]
  L.append(i/10)
  L.append(((i+1)/10)+i/10)
  L.append((j)/10)
  L.append(((j+1)/10)+j/10)

  #plot do grafico para visualização dos pontos
  R=[]
  E=[]
  #fig1 = plt.figure()
  #ax1 = fig1.add_subplot(111, aspect='equal')
  #ax1.add_patch(patches.Rectangle((i/10, j/10),(i+1)/10,(j+1)/10,fill=False ,linewidth=1.5,edgecolor="blue"))

  #verificação no caso quasi aleatorio da ocorrencia de pontos dentro do quadrado interno e calcula razão entre o numero total de pontos e cria uma lista
  #com os valores para querem usados nas analises
  for r in range(n):
      if X[r]>=L[0] and X[r]<=L[1] and Y[r]>=L[2] and Y[r]<=L[3]:
          conta2=conta2+1
  E.append(conta2)
  l.append(E)
  w.append(conta2/N)
  #plt.axis([0,1, 0,1])
  #plt.scatter(x,y, c='r', s=2)
  #plt.title("Valores aleatórios no quadrado [0,1] x [0,1]")
  #plt.show()
  
 #verificação no caso pseudo aleatorio da ocorrencia de pontos dentro do quadrado interno e calcula razão entre o numero total de pontos e cria uma lista
 #com os valores para querem usados nas analises
  for p in range(n):
      if x[p]>=L[0] and x[p]<=L[1] and y[p]>=L[2] and y[p]<=L[3]:
          conta=conta+1        
  R.append(conta)
  q.append(R)
  c.append(conta/n)
  #plt.axis([0,1, 0,1])
  #plt.scatter(x,y, c='r', s=2)
  #plt.title("Valores aleatórios no quadrado [0,1] x [0,1]")
  #plt.show()
  k=k+1
 # uso das funções 1/sqrt(n) e ln(n)/n para plotar os graficos dessas funções e verificar se
 # é algo que as razões dos pontos se aproxima de alguma dessas curvas para o caso quasi aleatorio
 
 for h in range(1,len(c)):

      z.append( math.log(h)/h)
      a.append(1/math.sqrt(h))
      
 plt.plot(w,'o--', color='orange')
 plt.plot(a,'^--', color='red')
 plt.plot(z,'*--', color='blue')
 plt.title("Relação entre as funções e as razões dos valores quasi aleatorios")
 plt.show()
 print(np.std(w, ddof=1))
 print(np.std(l, ddof=1))
 print(np.std(a, ddof=1))
 print(np.std(z, ddof=1))
 
 # uso das funções 1/sqrt(n) e ln(n)/n para plotar os graficos dessas funções e verificar se
 # é algo que as razões dos pontos se aproxima de alguma dessas curvas para o caso pseudo aleatorio
 for h in range(1,len(c)+1):

      s.append( math.log(h)/h)
      t.append(1/math.sqrt(h))
      
 plt.plot(c,'o--', color='orange')
 plt.plot(t,'^--', color='red')
 plt.plot(s,'*--', color='blue')
 plt.title("Relação entre as funções e as razões dos valores pseudo aleatorios")
 plt.show()
 
 print(np.std(c, ddof=1))
 print(np.std(q, ddof=1))
 print(np.std(t, ddof=1))
 print(np.std(s, ddof=1))

 

# verificando comportamento das razões dos pontos entre os metodos pseudo e quasi aleatorio.
 plt.plot(c,'*-', color='green')
 plt.plot(w,'^-', color='red')
 plt.show()
 m, b, Sm, Sb, R, p = lin_regression(c, w)
 plt.plot(c,w,'o')
 plt.xlim(0,None)
 plt.ylim(0, None)

# desenho da recta, dados 2 pontos extremos
# escolhemos a origem e o max(x)
 x2 = np.array([0, max(c)])
 plt.plot(x2, m * x2 + b, '-')

# Anotação sobre o gráfico:
 plt.title("Relação de regressão entre os valores de variação dos pontos em cada modelo")
 plt.text(0.10,0.20,'m declive= {:>.4g} +- {:6.4f}'.format(m, Sm))
 plt.text(0.10,0.16,'b ordenada na origem= {:>.4g} +- {:6.4f}'.format(b, Sb))
 plt.text(0.10,0.12,'coeficiente de correlação (de Pearson) = {:7.5f}'.format(R**2))
 plt.text(0.10,0.10,'p-value do teste F : {:<8.6f}'.format(p))
 plt.text(0.10,0.08,'Erro de declive : {:<8.6f}'.format(Sm))
 plt.show()
principal()
