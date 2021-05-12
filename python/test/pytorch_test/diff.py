from sympy import *

x=symbols("x")


y=5*pow(x,5)+2
z=4*y+2
dify=diff(y,x)
print(dify)

dify2=diff(z,x)
print(dify2)

