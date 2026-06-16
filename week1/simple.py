import numpy as np
a=[[1,2,3],[4,5,6]]
na=np.array(a)
na=na-1
print(na.reshape(-1,2))