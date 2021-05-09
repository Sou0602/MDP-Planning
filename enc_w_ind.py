

def maze(gridfile):
    with open(gridfile) as f:
        grid = f.readlines()
    grid = [[int(j) for j in grid[i][:-1].split()] for i in range(len(grid))]

    def manhattan(point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    states = []
    ends = []
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] != 1:
                states.append((i, j))
            if grid[i][j] == 2:
                start = len(states) - 1
            if grid[i][j] == 3:
                end = len(states) - 1
                ends.append(end)
    end = ends[0]
    for check in ends[1:]:
        if manhattan(states[check], states[start]) < manhattan(states[end], states[start]):
            end = check

    transitions = []
    endstate = states[end]
    endReward = 1000000
    stateReward = -1
    invalidReward = -1000000
    for i in range(len(states)):
        state = states[i]
        neighbors = [(state[0] - 1, state[1]), (state[0] + 1, state[1]), (state[0], state[1] + 1),
                     (state[0], state[1] - 1)]

        if i == end:
            continue

        for action in range(4):
            neighbor = neighbors[action]
            if neighbor == endstate:
                transitions.append((i, action, end, endReward, 1.0))
            elif neighbor in states:
                transitions.append((i, action, states.index(neighbor), stateReward, 1.0))
            else:
                transitions.append((i, action, i, invalidReward, 1.0))
    mode = 'episodic'
    discount = 0.9
    numStates = len(states)
    return [numStates, 4, start, end, transitions, mode, discount]


'''''
numStates, numActions, start, end, transitions, mode, discount = maze('F:\csp-pa2\data\maze\grid10.txt')

f = open("enc_w_ind.txt","a")

f.write('numStates'+ ' '+str(numStates)+' \n')
f.write('numActions'+ ' ' + str( numActions)+' \n')
f.write('start' + ' '+ str (start)+' \n')
f.write('end' + str(end)+' \n')
for transition in transitions:
    f.write('transition'+' ' +str(transition[0])+ ' ' + str(transition[1]) + ' '+str(transition[2])+' '+str(transition[3])+' '+str(transition[4])+' \n')
f.close()
'''''