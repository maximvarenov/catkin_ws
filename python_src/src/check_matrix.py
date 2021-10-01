import numpy as np

a = np.array([[-0.76686085, -0.59748648, -0.12047004],
 [ 0.63222172 ,-0.73562054, -0.31081137],
 [ 0.11054474, -0.31917445 , 0.94186252]])
row_sums = a.sum(axis=1)
a_n = a / row_sums[:, np.newaxis]
norm = np.linalg.norm(a) 
a_n =a/norm
b= np.linalg.inv(a_n)
c= np.transpose(a_n)  #transpose = inverse matrix
e= np.dot(a_n,c)
print(a,b,c,e)