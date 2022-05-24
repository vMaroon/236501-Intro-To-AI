import random
import time

from TaxiEnv import TaxiEnv
import argparse
import submission
import Agent


def run_agents():
    parser = argparse.ArgumentParser(description='Test your submission by pitting agents against each other.')
    parser.add_argument('agent0', type=str,
                        help='First agent')
    parser.add_argument('agent1', type=str,
                        help='Second agent')
    parser.add_argument('-t', '--time_limit', type=float, nargs='?', help='Time limit for each turn in seconds', default=1)
    parser.add_argument('-s', '--seed', nargs='?', type=int, help='Seed to be used for generating the game',
                        default=random.randint(0, 255))
    parser.add_argument('-c', '--count_steps', nargs='?', type=int, help='Number of steps each taxi gets before game is over',
                        default=4761)
    parser.add_argument('--print_game', action='store_true')

    args = parser.parse_args()

    agents = {
        "random": Agent.AgentRandom(),
        "greedy": Agent.AgentGreedy(),
        "improved_greedy": submission.AgentGreedyImproved(),
        "minimax": submission.AgentMinimax(),
        "alphabeta": submission.AgentAlphaBeta(),
        "expectimax": submission.AgentExpectimax()
    }

    # agent_names = sys.argv
    agent_names = [args.agent0, args.agent1]
    env = TaxiEnv()

    env.generate(args.seed, 2*args.count_steps)

    if args.print_game:
        print('initial board:')
        env.print()

    for _ in range(args.count_steps):
        for i, agent_name in enumerate(agent_names):
            agent = agents[agent_name]
            start = time.time()
            op = agent.run_step(env, i, args.time_limit)
            end = time.time()
            if end - start > args.time_limit:
                raise RuntimeError("Agent used too much time!")
            env.apply_operator(i, op)
            if args.print_game:
                print('taxi ' + str(i) + ' chose ' + op)
                env.print()
        if env.done():
            break
    balances = env.get_balances()
    print(balances)
    if balances[0] == balances[1]:
        print('draw')
    else:
        print('taxi', balances.index(max(balances)), 'wins!')


if __name__ == "__main__":
    run_agents()
