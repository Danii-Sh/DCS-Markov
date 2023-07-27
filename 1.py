import itertools
import numpy as np
from scipy.linalg import solve

##a= transition_matrix
##
##solve a => stationary dist
##
##permutations of k
##
##calc all k routes with stats..- and trans_probs
##
##calc G with (Sum of -plogp)

print ('Enter number of Markov States')
dim = int(input ())
#print dim
print "****"

A = np.ndarray(shape=(dim,dim))
for count1 in range(0,dim):
    for count2 in range(0,dim):
        print ('Enter state '+str(count1)+' to state '+str(count2)+' transition probability')
        A[count1,count2]=input ()
print (type (A))
print A
print "****"

#second appraoch , delete a dependant row and replace it by sum-prob=1 
A2 = np.ndarray(shape=(dim,dim))
for count1 in range(0,dim):
        for count2 in range(0,dim):
            A2[count1,count2]=A[count2,count1]
for count1 in range(0,dim):
    A2[count1,count1]=A2[count1,count1]-1
    A2[dim-1,count1]=1 
print A2
print "****"

#b = np.random.random(3)
b = np.ndarray(shape=(dim))
for count1 in range(0,dim):
    b[count1]=0
b[dim-1]=1
print b
print "****"


#solution using LinAlg solver
x = solve(A2, b)
print x
print "****"



print ('Enter coding length k , to generate Gk')
k = int(input ())


states = []
for count1 in range (0,dim):
    states.append(count1)
print('*****Product*****')
CodedSymbols= list( itertools.product(states, repeat=k))

#print CodedSymbols

SymbolProb=[]
for count1 in range (0,len(CodedSymbols)):
    p=1
    for count2 in range (0,k-1):
        p=p*A[CodedSymbols[count1][count2]][CodedSymbols[count1][count2+1]]
    p=p*x[CodedSymbols[count1][0]]
    SymbolProb.append(p)

#print (SymbolProb,'length is ', len(SymbolProb))

Gk=0
for count1 in range(0,len(SymbolProb)):
    Gk+= (-SymbolProb[count1]*(np.log(SymbolProb[count1])/np.log(2)))
Gk=float(Gk)/float(k)
print Gk




