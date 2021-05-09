import numpy as np


def encoder_grid(input):

    file = open( input,'r')
    grid = []
    lines = file.readlines()
    for line in lines:
        line = line.strip().split(' ')
        for l in line:
            grid = np.append(grid, float(l))
    grid = np.reshape(grid,(len(lines),-1))
    print(grid)
    prob_mat = []
    states = []
    #Actions
    # 0 - down
    # 1 - right
    # 2 - up
    # 3 - left

    def index(i,j,N):
        ind = N*i + j
        return ind


    for i in range(0,np.shape(grid)[0]):
        for j in range(0,np.shape(grid)[1]):

            if grid[i][j] == 0 or grid[i][j] == 2:
                states = np.append(states,index(i,j,len(grid)))
                rew = 0
                for a in range(0,4):
                    if a == 0:
                        ni = i-1
                        nj = j
                    if a == 1:
                        ni = i
                        nj = j+1
                    if a == 2:
                        ni = i+1
                        nj = j
                    if a == 3:
                        ni = i
                        nj = j-1

                    if grid[ni][nj] == 3:
                        rew = 10000000
                        ns = index(ni, nj, len(grid))

                    elif grid[ni][nj] == 1 or grid[ni][nj] == 2:
                        rew = -10000000
                        ns = index(i,j,len(grid))
                    elif grid[ni][nj] == 0:
                        rew = -1
                        ns = index(ni, nj, len(grid))
                    pval = [index(i, j, len(grid)), a, ns, rew , 1]
                    prob_mat = np.append(prob_mat, pval)

            if grid[i][j] == 2:
                start_state = index(i,j,len(grid))

            if grid[i][j] == 3:
                end_state = index(i,j,len(grid))
                states = np.append(states, index(i, j, len(grid)))

    prob_mat = np.reshape(prob_mat,(-1,5))
    for n in range(0,len(states)):
        for r in range(0,len(prob_mat)):
            if prob_mat[r][0] == states[n]:
                prob_mat[r][0] = n
            if prob_mat[r][2] == states[n]:
                prob_mat[r][2] = n
        if states[n] == start_state:
            start_state = n

        if states[n] == end_state:
            end_state = n
    #print(prob_mat)

    N_s = int(len(prob_mat)/4 + 1)
    K = 4
    s0 = start_state
    st = end_state
    gamma = 0.9
    Transition_table = prob_mat
    mdptype = 'episodic'

    return states,N_s,K,s0,st,Transition_table,gamma,mdptype

state_map ,N,K,s0,st,Transtable, gamma , mdptype = encoder_grid('F:\csp-pa2\data\maze\grid10.txt')

f = open('encoder_grid.txt',"a")

f.write(str(N) + ' \n')
f.write(str(K) + ' \n')
f.write(str(s0)+ ' \n')
f.write(str(st)+ ' \n')
for i in range(0,len(Transtable)):
    f.write('transition' + ' ' + str(Transtable[i][0])+' '+str(Transtable[i][1])+' '+str(Transtable[i][2])+' '+str(Transtable[i][3])+' '+str(Transtable[i][4])+ ' \n')
f.close()


