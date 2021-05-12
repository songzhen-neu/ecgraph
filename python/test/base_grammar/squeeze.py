import numpy as np

e=np.arange(10)
print(e)
e=e.reshape(1,10,1)
print(e)
# e=np.squeeze(e)
e=e.squeeze()
print(e)


