""" You can import what ever you want """
import numpy as np
from value_and_policy_iteration import utility_get_matching_policies


def get_all_policies(mdp, u):  # You can add more input parameters as needed
    actions_to_arrow_map = {
        "UP": "UP",
        "DOWN": "DOWN",
        "LEFT": "LEFT",
        "RIGHT": "RIGHT",
        "".join(sorted(["UP", "DOWN"])): '│',
        "".join(sorted(["UP", "LEFT"])): '┘',
        "".join(sorted(["UP", "RIGHT"])): '└',
        "".join(sorted(["DOWN", "RIGHT"])): '┌',
        "".join(sorted(["DOWN", "LEFT"])): '┐',
        "".join(sorted(["RIGHT", "LEFT"])): '─',
        "".join(sorted(["UP", "DOWN", "LEFT"])): '┤',
        "".join(sorted(["UP", "DOWN", "RIGHT"])): '├',
        "".join(sorted(["UP", "LEFT", "RIGHT"])): '┴',
        "".join(sorted(["DOWN", "LEFT", "RIGHT"])): '┬',
        "".join(sorted(["UP", "DOWN", "LEFT", "RIGHT"])): '┼',
    }

    number_of_policies = 1
    policy = np.empty((mdp.num_row, mdp.num_col), dtype='O')
    for r, c in np.ndindex(mdp.num_row, mdp.num_col):
        if mdp.board[r][c] == 'WALL' or (r, c) in mdp.terminal_states:
            policy[r][c] = mdp.board[r][c]
            continue

        possible_actions, _ = utility_get_matching_policies(mdp, u, (r, c))
        number_of_policies *= len(possible_actions)

        policy[r][c] = actions_to_arrow_map["".join(sorted(possible_actions))]

    mdp.print_policy(policy)

    return number_of_policies



def get_policy_for_different_rewards(mdp):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp
    # print / displas the optimal policy as a function of r
    # (reward values for any non-finite state)
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
