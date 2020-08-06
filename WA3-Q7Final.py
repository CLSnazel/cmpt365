## Chantal Snazel
## CMPT 365 - Summer 2020
## SN: 301247632
## Written Assignment 3: p278, Q7

import numpy as np
import math

def createDCTMatrix(n):

    matrixValues = np.array([])
    for i in range(n):
        row = np.array([])
        for j in range(n):
            
            aCoeff = 0

            if(i == 0):
                aCoeff = math.sqrt(1/n)
            else:
                aCoeff = math.sqrt(2/n)
            result = aCoeff * math.cos(((2*j+1)*i*math.pi)/(2*n))
            row = np.append(row, result)
            
        matrixValues = np.concatenate((matrixValues, row), axis=0)
    matrixValues = matrixValues.reshape(8,8)
    return matrixValues

def createDCTTranspose(n):
    matrixValues = np.array([])
    for j in range(n):
        row = np.array([])
        for i in range(n):
            
            aCoeff = 0

            if(i == 0):
                aCoeff = math.sqrt(1/n)
            else:
                aCoeff = math.sqrt(2/n)

            result = aCoeff * math.cos(((2*j+1)*i*math.pi)/(2*n))

            row = np.append(row, result)
        matrixValues = np.concatenate((matrixValues, row), axis=0)
    matrixValues = matrixValues.reshape(8,8)
    return matrixValues

T = (createDCTMatrix(8)) #create DCT matrix T
TT = (createDCTTranspose(8)) ## create DCT Matrix Transpose

#printing DCT Matrices Result
print("DCT Matrix T8:")
for i in T: 
    print(i)
print("DCT Matrix T8(transpose):")
for i in TT:
    print(i)

#proving that T*TT=I (Identity Matrix)
I = np.dot(TT, T)
I = np.round(I, 4)
print("Result of T*T(Transpose):")
print(I)


## Proving that each row and column of T are orthonormal vectors
## The inner product of a Trow(i) and TTcolumn(j)
## and inner product of a Tcolumn(j) and TTrow(i)
## will be 1 when i=j and 0 when i!=j

## Setting flags for these rules. If all column
allColumnOthronormal = False
allRowOrthonormal = False

#checking T-rows TT-cols
for i in range(len(TT)):
    for j in range(len(T)):
        tCol = T[::1,j]
        result = round(np.inner(tCol, TT[i]))
        #print(result, i , j)
        if(i == j and result==1):
            allColumnOrthonormal = True
        elif(i !=j and result==0):
            allColumnOrthonormal = True
        else:
            allColumnOrthonormal = False
            break
 
    if(allColumnOrthonormal == False):
        break

#checking T-cols TT-rows
for i in range(len(T)):
    for j in range(len(TT)):
        ttCol = TT[::1,j]
        result = round(np.inner(T[i],ttCol))
        #print(result, i , j)
        if(i == j and result==1):
            allRowOrthonormal = True
        elif(i !=j and result==0):
            allRowOrthonormal = True
        else:
            allRowOrthonormal = False
            break
        
    if(allRowOrthonormal == False):
        break
        
print("T Rows Orthonormal:", allRowOrthonormal)
print("T Columns Orthonormal:", allColumnOrthonormal)

input("Press ENTER to exit:")
