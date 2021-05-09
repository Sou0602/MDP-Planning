import numpy as np
from pulp import LpMaximize, LpProblem , LpStatus, lpSum, LpVariable
from copy import deepcopy
from encoder import encoder_grid
from decoder import decoder_grid
from enc_w_ind import maze

def Value_Iteration(N,K,p_trans_mat,rew_mat,state_mat,gamma):
    eps = 0.00001
    v0 = np.ones((1,N))
    diff = 1
    v = np.ones((1, N))
    pol = np.zeros((1,N))
    v0 = get_value_from_policy(N,K,p_trans_mat,rew_mat,state_mat,pol,gamma)
    V_list = []

    while diff>eps :
        for s in range(0,N):

            Q = []
            for k in range(0,K):
                s_n = next_states(s,k,state_mat)
                qval = 0
                for i in s_n:
                    p = p_trans(s, k, i, p_trans_mat)
                    r = rew(s, k, i, rew_mat)
                    vval = v0[0][int(i)]
                    qval = qval + p * (r + gamma * vval)

                Q = np.append(Q,qval)
            qmax = np.max(Q)
            polmax = np.argmax(Q)
            v[0][s] = qmax
            pol[0][s] = polmax



        V_list = np.append(V_list,v)
        V_list = np.reshape(V_list,(-1,N))
        if len(V_list) == 1:
            diff = np.linalg.norm(V_list[-1] - v0)
        else:
            diff = np.linalg.norm(V_list[-1] - V_list[-2])
        v0[0] = V_list[-1]
    return v,pol

def get_value_from_policy(N,K,p_trans_mat,rew_mat,state_mat,pol,gamma):

    Coeffs = np.zeros((N,N))
    B = []
    for s in range(0,N):
        coeffs = np.zeros((1, N ))
        b = 0
        s_n = next_states(s,pol[0][s],state_mat)
        coeffs[0][s] = 1
        for sp in s_n:
            p = p_trans(s,pol[0][s],sp,p_trans_mat)
            r = rew(s,pol[0][s],sp,rew_mat)
            sp = int(sp)
            coeffs[0][sp] = -p * gamma
            b = b + p*r
        Coeffs[s][:] = coeffs[0]
        B = np.append(B,b)
    #Equations denote the linear equations where coeffs end is a constant vector
    # Slice last column of coefficients as a vector
    # solve for Ax = b
    B = np.reshape(B,(-1,1))
    vpi = np.linalg.solve(Coeffs,B)
    vpi = np.reshape(vpi,(1,N))
    vpi = np.round(vpi,6)

    return vpi

def value_eval(N,K,p_trans_mat,rew_mat, state_mat,pol,gamma):
    v0 = np.ones((1,N))
    v1 = np.zeros((1,N))
    while np.linalg.norm(v1-v0) > 1.01*1e-6:
        v0 = deepcopy(v1)
        for s in range(0,N):
            k = pol[0][s]
            val = 0
            for sn in next_states(s,k,state_mat):
                val += p_trans(s,k,sn,p_trans_mat) * (rew(s,k,sn,rew_mat)+gamma * v0[0][int(sn)])
            v1[0][s] = val

    return np.round(v1,6)


def Policy_Iteration(N,K,p_trans_mat,rew_mat,state_mat,gamma):
    np.random.seed(40)
    pol = np.zeros((1,N))
    sum_imps = 5
    vpi = get_value_from_policy(N, K, p_trans_mat, rew_mat, state_mat, pol, gamma)
    while sum_imps > 0:
        imp_s = np.zeros((1, N))
        vpi1 = get_value_from_policy(N,K,p_trans_mat,rew_mat,state_mat,pol,gamma)
        vpi = value_eval(N,K,p_trans_mat,rew_mat,state_mat,pol,gamma)

        for s in range(0,N):
            Q = []
            Qvals = []
            Kmat = []
            for k in range(0,K):
                s_n = next_states(s,k,state_mat)
                qval = 0
                for i in s_n:
                    p = p_trans(s,k,i,p_trans_mat)
                    r = rew(s,k,i,rew_mat)
                    v = vpi[0][int(i)]
                    qval = qval + p * (r +gamma*v)
                    qval = np.round(qval,6)
                    Qvals = np.append(Qvals,qval)
                    qbin = Qvals > vpi[0][s]
                    if (qval - vpi[0][s]) > 1.5*1e-6:
                        Q = np.append(Q, qval)
                        Kmat = np.append(Kmat, k)
                        imp_s[0][s] = imp_s[0][s] + 1
                #    imp_a[k][s] = k

            if len(Q) > 0:

                #qsel = np.random.choice(Q)
                qmax = np.argmax(Q)
                #qpol = np.where(Q == qsel)
                #qpol = qpol[0][0]
                qpol = qmax
                pol[0][s] = Kmat[qpol]
            else:
                imp_s[0][s] = 0

        sum_imps = np.sum(imp_s)
    vpi = value_eval(N,K,p_trans_mat,rew_mat,state_mat,pol,gamma)

    return vpi,pol

def Linear_program(N,K,p_trans_mat, rew_mat ,state_mat,gamma):
    v_opt = []
    pol_opt = np.zeros((1,N))
    model = LpProblem(name= "Maxv" , sense=LpMaximize)
    for s in range(0,N):
        varnames = [str(i) for i in range(0,N)]
    Lp_vars = LpVariable.matrix('v' , varnames  , cat="float")
    nlp_vars = np.array(Lp_vars)
    count = 1
    for s in range(0,N):

        for k in range(0,K):
            cons = 0
            s_n = next_states(s, k, state_mat)
            for sp in s_n:
                int(sp)
                cons = cons + p_trans(s,k,sp,p_trans_mat) * (rew(s,k,sp,rew_mat) + gamma * nlp_vars[int(sp)])

            model += (nlp_vars[s] >= cons , 'C'+str(count))
            count = count + 1

    model += -lpSum(nlp_vars)
    status = model.solve()

    for var in model.variables():
        v_opt = np.append(v_opt,var.value())
    v_opt = np.reshape(v_opt,(1,-1))
    for s in range(0,N):
        Q = []
        for k in range(0,K):
            s_n = next_states(s,k,state_mat)
            qval = 0
            for sp in s_n:
                qval = qval + p_trans(s, k, sp, p_trans_mat) * (rew(s, k, sp, rew_mat) + gamma * v_opt[0][int(sp)])
            Q = np.append(Q,qval)
        pol_opt[0][s] = np.argmax(Q)


    return v_opt , pol_opt


#input = F:\csp-pa2\data\mdp\continuing-mdp-2-2.txt
def mdp(input):
    Transition_table0 = []
    file = open(input,'r',encoding="utf-8")

    for line in file:
        line1 = line.strip().split(' ')
        trans = []
        if line1[0] == 'numStates':
            N0 = int(line1[1])
        if line1[0] == 'numActions':
            K0 = int(line1[1])
        if line1[0] == 'start':
            s00 = int(line1[1])
        if line1[0] == 'end':
            stt = int(line1[1])
        if line1[0] == 'mdptype':
            mdptype0 = line1[1]
        if line1[0] == 'discount':
            gamma0 = float(line1[2])
        if line1[0] == 'transition':
            for k in range(1,len(line1)):
                trans = np.append(trans,float(line1[k]))
            Transition_table0 = np.append(Transition_table0,trans,axis=0)

    Transition_table0 = np.reshape(Transition_table0,(-1,5))
    return N0,K0,s00,stt,Transition_table0,gamma0,mdptype0


def get_p_trans(Trans_table):

    p_trans = Trans_table[:,0:3]
    vals = np.reshape(Trans_table[:,4],(np.shape(p_trans)[0],-1))
    rew = np.append(p_trans,vals,axis = 1)
    return rew
def get_rew(Trans_table):
    rew = Trans_table[:,0:4]
    return rew
def state_matrix(Trans_table):
    matrix = Trans_table[:,0:3]
    return matrix

def next_states(s,k,state_mat):
    sp = []
    for i in range(len(state_mat)):
        if state_mat[i][0] == s and state_mat[i][1] == k:
            sp = np.append(sp,int(state_mat[i][2]))

    return sp

def p_trans(s,k,snext,p_trans_mat):
    for i in range(len(p_trans_mat)):
        if p_trans_mat[i][0] == s and p_trans_mat[i][1] == k and p_trans_mat[i][2] == snext:
            pval = p_trans_mat[i][3]
    return pval

def rew(s,k,snext,rew_mat):
    for i in range(len(rew_mat)):
        if rew_mat[i][0] == s and rew_mat[i][1] == k and rew_mat[i][2] == snext:
            r = rew_mat[i][3]
    return r



#for i in range(0,len(prob_mat)):
#    print('transition'+' '+str(prob_mat[i][0])+' '+str(prob_mat[i][1])+' '+str(prob_mat[i][2])+' '+str(prob_mat[i][3])+' '+str(prob_mat[i][4]))


N,K,s0,st,Transition_table,gamma,mdptype = mdp('F:\csp-pa2\data\mdp\continuing-mdp-50-20.txt')
#state_map,N,K,s0,st,Transition_table,gamma,mdptype = encoder_grid('F:\csp-pa2\data\maze\grid10.txt')
#N , K , s0,st,Transition_table,mdptype,gamma = maze('F:\csp-pa2\data\maze\grid10.txt')
#Transition_table = np.array(Transition_table)
p_trans_mat = get_p_trans(Transition_table)
rew_mat = get_rew(Transition_table)
state_mat = state_matrix(Transition_table)



v_opt , pol_opt = Value_Iteration(N,K,p_trans_mat,rew_mat,state_mat,gamma)
#v_opt , pol_opt = Linear_program(N,K,p_trans_mat,rew_mat,state_mat,gamma)
print(v_opt)
print(pol_opt)
#Policy_Iteration(N,K,p_trans_mat,rew_mat,state_mat,gamma)

#decoder_grid('F:\csp-pa2\data\maze\grid10.txt',pol_opt,state_map,N,K,s0,st)


