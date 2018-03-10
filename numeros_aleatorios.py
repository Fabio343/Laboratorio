import numpy as np
import matplotlib.pyplot as plt
from random import random



def principal():
 n=int(input("entre com o valor de n: "))
 X=x = np.random.rand(n)
 Y=y = np.random.rand(n)
 print(X)
 print(Y)
 plt.axis([0,1, 0,1])
 plt.scatter(x,y)
 plt.title("Valores aleatórios no quadrado [0,1] x [0,1]")
 plt.show()

principal()
