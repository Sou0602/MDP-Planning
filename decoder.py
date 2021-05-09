import numpy as np
from matplotlib import pyplot

def decoder_grid(input,pol,state_map,N,K,s0,st):
    s_i = s0
    s_f = st
    file = open( input,'r')
    grid = []
    lines = file.readlines()
    for line in lines:
        line = line.strip().split(' ')
        for l in line:
            grid = np.append(grid, float(l))
    grid = np.reshape(grid,(len(lines),-1))

    def index(i,j,N):
        ind = N*i + j
        return ind

    for i in range(0,np.shape(grid)[0]):
        strj = ''
        for j in range(0,np.shape(grid)[1]):
            if grid[i][j] == 1:
                strj = strj + '1' + ' '
            else:
                for k in range(0,len(state_map)):
                    if index(i,j,len(grid)) == state_map[k]:
                        actions = ['S' , 'E' , 'N' , 'W']
                        actions1 = ['S', 'N', 'E', 'W']
                        strj = strj + actions1[int(pol[0][int(k)])] + ' '
        print(strj)




