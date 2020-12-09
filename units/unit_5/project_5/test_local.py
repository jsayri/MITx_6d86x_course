# My test sandbox, evaluate function from the project 5: HomeWorld Game
import numpy as np
import agent_tabular_ql as atql

# test epsilon greedy algorithm (agent tabular ql)
def test_epsilon_greedy(q_size=(4,4,3,4)):

    print('Test epsilon greedy algorithm')

    # define a random q-function table
    q_func = np.random.random(q_size)

    # select one state 1
    state_1 = np.random.randint(q_func.shape[0])
    # select one state 2
    state_2 = np.random.randint(q_func.shape[1])

    # call for exploring action
    action_index, object_index = atql.epsilon_greedy(state_1, state_2, q_func, .9)
    # check random selection
    # to check random selection, a seed must be define so random numbers are expected...

    # call for greedy action
    action_index, object_index = atql.epsilon_greedy(state_1, state_2, q_func, .1)
    # check with expected greedy action & object selection
    ga_idx, go_idx = np.unravel_index(q_func[state_1, state_2].argmax(), q_func[state_1, state_2].shape)
    if ((ga_idx - action_index) + (go_idx - object_index)) == 0:
        print('took optimal')
    else:
        print('greedy method is not working')

# Call test functions
if __name__ == '__main__':

    # Q-learning tabular case
    # Test epsilon greedy algorithm
    test_epsilon_greedy((3, 3, 10, 10))