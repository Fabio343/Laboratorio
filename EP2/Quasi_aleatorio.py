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

#função principal que usa a sobolseq para gerar num vetor os valores quasi-aleatorios para um determinado N
# e gera uma imagem do quadrado [0,1]x[0,1] de forma a verificarmos as distribuição dos pontos na região
vet=[2**12,2**10,2**13,2**12]
def principal():
   
 s1 = sobolSeq([1,1],[1,1],32)
 s2 = sobolSeq([1,0,1],[1,3,7],32)

 N =vet[0]
 k=0
 conta=0
 q=[]*vet[0]
 c=[]*vet[0]
 t=[]*vet[0]
 s=[]*vet[0]
 while k<20:
  #inicializando variaveis
  n=vet[0]
  conta=0
  x =  [ next(s1) for _ in range(N) ]
  y =  [ next(s2) for _ in range(N) ]
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
  fig1 = plt.figure()
  ax1 = fig1.add_subplot(111, aspect='equal')
  ax1.add_patch(patches.Rectangle((i/10, j/10),(i+1)/10,(j+1)/10,fill=False ,linewidth=1.1,edgecolor="blue"))
  for p in range(n):
      if x[p]>=L[0] and x[p]<=L[1] and y[p]>=L[2] and y[p]<=L[3]:
          conta=conta+1
  R.append(conta)
  q.append(R)
  c.append(conta/n)
  plt.axis([0,1, 0,1])
  plt.scatter(x,y, c='r', s=1)
  plt.title("Valores aleatórios no quadrado [0,1] x [0,1]")
  plt.show()
  k=k+1
  print('Rotal de pontos',n)
  print(L, "dimenções do quadradinhos/ ladrilhos de verificação de ocorrencia de pontos" )
  print(R,'elementos quadrado interno')
  print(q,'numero de pontos do interior ')
  print(c,'razão entre pontos internos e externos')
 for h in range(1,len(c)+1):

      s.append( math.log(h)/h)
      t.append(1/math.sqrt(h))
 plt.title("Comparação entre as funções e a razão dos pontos caso quasi aleatório")     
 plt.plot(c,'o--', color='orange')
 plt.plot(t,'^--', color='red')
 plt.plot(s,'*--', color='blue')
 plt.show()
 
 m, b, Sm, Sb, R, p = lin_regression(c, s)
 plt.plot(c,s,'o')
 plt.xlim(0,None)
 plt.ylim(0, None)

# desenho da recta, dados 2 pontos extremos
# escolhemos a origem e o max(x)
 x2 = np.array([0, max(c)])
 plt.plot(x2, m * x2 + b, '-')

# Anotação sobre o gráfico:
 plt.title("Relação de regressão entre os valores de variação dos pontos e a função ln(n)/n")
 plt.text(0.10,0.35,'m declive= {:>.4g} +- {:6.4f}'.format(m, Sm))
 plt.text(0.10,0.30,'b ordenada na origem= {:>.4g} +- {:6.4f}'.format(b, Sb))
 plt.text(0.10,0.25,'coeficiente de correlação (de Pearson) = {:7.5f}'.format(R**2))
 plt.text(0.10,0.20,'p-value do teste F : {:<8.6f}'.format(p))
 plt.show()

 print(np.std(c, ddof=1), 'Desvio padrão dos pontos')
 print(np.std(t, ddof=1),'Desvio padrão dos pontos da função ')
 print(np.std(s, ddof=1),'Desvio padrão dos pontos da função')

 m, b, Sm, Sb, R, p = lin_regression(c, t)
 plt.plot(c,t,'o')
 plt.xlim(0,None)
 plt.ylim(0, None)

# desenho da recta, dados 2 pontos extremos
# escolhemos a origem e o max(x)
 x2 = np.array([0, max(c)])
 plt.plot(x2, m * x2 + b, '-')

# Anotação sobre o gráfico:
 plt.title("Relação de regressão entre os valores de variação dos pontos e a função 1/\sqrt(n)")
 plt.text(0.10,0.35,'m declive= {:>.4g} +- {:6.4f}'.format(m, Sm))
 plt.text(0.10,0.30,'b ordenada na origem= {:>.4g} +- {:6.4f}'.format(b, Sb))
 plt.text(0.10,0.25,'coeficiente de correlação (de Pearson) = {:7.5f}'.format(R**2))
 plt.text(0.10,0.20,'p-value do teste F : {:<8.6f}'.format(p))
 plt.show()

principal()


