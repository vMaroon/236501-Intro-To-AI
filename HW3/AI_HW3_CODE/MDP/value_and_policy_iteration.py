from copy import deepcopy
import numpy as np


def get_local_utility(mdp, u, state, action):
    local_utility = 0.0
    for idx, probabilistic_action in enumerate(mdp.actions):
        s_prime = mdp.step(state, probabilistic_action)
        local_utility += mdp.transition_function[action][idx] * u[s_prime[0]][s_prime[1]]

    return local_utility


def get_max_future_utility(mdp, u, state):
    best_action = ''
    max_future_utility = float('-inf')  # calculate discounted max future util
    for action in mdp.actions:
        local_utility = get_local_utility(mdp, u, state, action)
        if local_utility > max_future_utility:
            max_future_utility = local_utility
            best_action = action

    return best_action, max_future_utility


def utility_get_matching_policies(mdp, u, state):
    possible_actions = []
    best_action = ''
    max_future_utility = float('-inf')  # calculate discounted max future util
    for action in mdp.actions:
        local_utility = get_local_utility(mdp, u, state, action)
        if float(mdp.board[state[0]][state[1]]) + local_utility == u[state[0]][state[1]]:
            possible_actions.append(action)
        if local_utility > max_future_utility:  # for safety
            max_future_utility = local_utility
            best_action = action

    return possible_actions, best_action


def value_iteration(mdp, u_init, epsilon=10 ** (-3)):
    u, u_prime = deepcopy(u_init), deepcopy(u_init)
    delta, once = 0, True

    for state in mdp.terminal_states:
        u_prime[state[0]][state[1]] = float(mdp.board[state[0]][state[1]])

    while once or (mdp.gamma == 1.0 and delta > 0) or \
            (mdp.gamma < 1.0 and delta >= float(epsilon) * float(1.0 - mdp.gamma) / mdp.gamma):
        u, delta, once = deepcopy(u_prime), 0, False

        for r, c in np.ndindex(mdp.num_row, mdp.num_col):
            if mdp.board[r][c] == 'WALL' or (r, c) in mdp.terminal_states:
                continue

            u_prime[r][c] = float(mdp.board[r][c]) + mdp.gamma * get_max_future_utility(mdp, u, (r, c))[1]

            delta = max([abs(u_prime[r][c] - u[r][c]), delta])

    return u


def get_policy(mdp, u):
    policy = np.empty((mdp.num_row, mdp.num_col), dtype='O')
    for r, c in np.ndindex(mdp.num_row, mdp.num_col):
        if mdp.board[r][c] == 'WALL' or (r, c) in mdp.terminal_states:
            continue

        possible_actions, best_action = utility_get_matching_policies(mdp, u, (r, c))
        if len(possible_actions) == 0:
            policy[r, c] = best_action
        else:
            policy[r, c] = possible_actions[0]

    return policy.tolist()


def policy_evaluation(mdp, policy):
    u = np.zeros((mdp.num_row, mdp.num_col), dtype=float)
    for state in mdp.terminal_states:
        u[state[0]][state[1]] = float(mdp.board[state[0]][state[1]])

    delta, once = 0, True
    while once or delta > 0:
        delta, once = 0, False

        for r, c in np.ndindex(mdp.num_row, mdp.num_col):
            if mdp.board[r][c] == 'WALL' or (r, c) in mdp.terminal_states:
                continue

            u_ = u[r][c]

            local_utility = get_local_utility(mdp, u, (r, c), policy[r][c])
            u[r][c] = float(mdp.board[r][c]) + mdp.gamma * local_utility

            delta = max([delta, abs(u[r][c] - u_)])

    return u.tolist()


def policy_iteration(mdp, policy_init):
    unchanged = False
    policy = deepcopy(policy_init)

    while not unchanged:
        u = policy_evaluation(mdp, policy)
        unchanged = True

        for r, c in np.ndindex(mdp.num_row, mdp.num_col):
            if mdp.board[r][c] == 'WALL' or (r, c) in mdp.terminal_states:
                continue

            best_action, max_future_utility = get_max_future_utility(mdp, u, (r, c))

            local_utility = get_local_utility(mdp, u, (r, c), policy[r][c])
            if max_future_utility > local_utility:
                policy[r][c] = best_action
                unchanged = False

    return policy
