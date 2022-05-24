from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance, Taxi, Passenger
import random
import time

def normalize_lst(lst):
    sum_ = sum(lst)
    return [x / sum for x in lst]


class AgentGreedyImproved(AgentGreedy):
    # TODO: section a : 3
    MAX_MD = 14

    def run_step(self, env: TaxiEnv, taxi_id, time_limit):
        operators = env.get_legal_operators(taxi_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(taxi_id, op)

        children_heuristics = [self.heuristic(child, taxi_id) for child in children]
        min_heuristic = min(children_heuristics)
        index_selected = children_heuristics.index(min_heuristic)
        return operators[index_selected]

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        taxi = env.get_taxi(taxi_id)
        other_taxi = env.get_taxi((taxi_id + 1) % 2)
        cash_diff = taxi.cash - other_taxi.cash

        def msh(agent: Taxi, passenger: Passenger):
            return manhattan_distance(agent.position, passenger.position) + \
                   manhattan_distance(passenger.position, passenger.destination)

        distance_to_passengers = [msh(taxi, p) for p in env.passengers]

        if taxi.passenger is not None:  # IF TAXI HAS A PASSENGER THEN DELIVER
            return manhattan_distance(taxi.position, taxi.passenger.destination) - self.MAX_MD * taxi.cash \
                   - cash_diff

        if len(distance_to_passengers) == 1:  # other taxi already has a passenger
            return distance_to_passengers[0] - self.MAX_MD * taxi.cash \
                   - cash_diff

        other_taxi_distance_to_passengers = [msh(other_taxi, p) for p in env.passengers]

        # assuming that the other taxi will move towards its closer passenger
        #   from the passengers we can guarantee accessing,
        #   we choose the closest
        if distance_to_passengers[0] <= distance_to_passengers[1]:
            # if we know that it is safe to choose 0 (other_taxi wont get there first) OR
            # we know that other_taxi will choose 1
            if distance_to_passengers[0] <= other_taxi_distance_to_passengers[0] or \
                    other_taxi_distance_to_passengers[1] <= other_taxi_distance_to_passengers[0]:
                # for case of ==, it depends on the scheduling in the joint actions.
                # we chose to have the agent "compete" (50-50 chance) since that is at least as good
                # as two "greedy-for-closest-reward" agents
                return distance_to_passengers[0] - self.MAX_MD * taxi.cash \
                       - cash_diff
            return distance_to_passengers[1] - self.MAX_MD * taxi.cash \
                   - cash_diff

        if distance_to_passengers[1] <= other_taxi_distance_to_passengers[0] or \
                other_taxi_distance_to_passengers[0] <= other_taxi_distance_to_passengers[1]:
            # for case of ==, it depends on the scheduling in the joint actions.
            # we chose to have the agent "compete" (50-50 chance) since that is at least as good
            # as two "greedy-for-closest-reward" agents
            return distance_to_passengers[1] - self.MAX_MD * taxi.cash \
                   - cash_diff
        return distance_to_passengers[0] - self.MAX_MD * taxi.cash \
               - cash_diff


class AgentMinimax(Agent):
    # TODO: section b : 1
    def __init__(self):
        self.MAX_MD = 14
        self.taxi_in_turn_index = None
        self.agent_id = None
        self.kill_time = None

    def successors(self, env: TaxiEnv, taxi_id: int):
        operators = env.get_legal_operators(taxi_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(taxi_id, op)
        return operators, children

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        taxi = env.get_taxi(taxi_id)
        other_taxi = env.get_taxi((taxi_id + 1) % 2)
        cash_diff = taxi.cash - other_taxi.cash

        def msh(agent: Taxi, passenger: Passenger):
            return manhattan_distance(agent.position, passenger.position) + \
                   manhattan_distance(passenger.position, passenger.destination)

        distance_to_passengers = [msh(taxi, p) for p in env.passengers]

        if taxi.passenger is not None:  # IF TAXI HAS A PASSENGER THEN DELIVER
            return manhattan_distance(taxi.position, taxi.passenger.destination) - self.MAX_MD * taxi.cash \
                   - cash_diff

        if len(distance_to_passengers) == 1:  # other taxi already has a passenger
            return distance_to_passengers[0] - self.MAX_MD * taxi.cash \
                   - cash_diff

        other_taxi_distance_to_passengers = [msh(other_taxi, p) for p in env.passengers]

        # assuming that the other taxi will move towards its closer passenger
        #   from the passengers we can guarantee accessing,
        #   we choose the closest
        if distance_to_passengers[0] <= distance_to_passengers[1]:
            # if we know that it is safe to choose 0 (other_taxi wont get there first) OR
            # we know that other_taxi will choose 1
            if distance_to_passengers[0] <= other_taxi_distance_to_passengers[0] or \
                    other_taxi_distance_to_passengers[1] <= other_taxi_distance_to_passengers[0]:
                # for case of ==, it depends on the scheduling in the joint actions.
                # we chose to have the agent "compete" (50-50 chance) since that is at least as good
                # as two "greedy-for-closest-reward" agents
                return distance_to_passengers[0] - self.MAX_MD * taxi.cash \
                       - cash_diff
            return distance_to_passengers[1] - self.MAX_MD * taxi.cash \
                   - cash_diff

        if distance_to_passengers[1] <= other_taxi_distance_to_passengers[0] or \
                other_taxi_distance_to_passengers[0] <= other_taxi_distance_to_passengers[1]:
            return distance_to_passengers[1] - self.MAX_MD * taxi.cash \
                   - cash_diff
        return distance_to_passengers[0] - self.MAX_MD * taxi.cash \
               - cash_diff

    def rb_minimax(self, env: TaxiEnv, agent_id, depth):
        if time.time() > self.kill_time or env.done() or depth == 0:
            return self.heuristic(env, self.agent_id)

        operators, children = self.successors(env, agent_id)
        if self.agent_id == agent_id:
            curr_min = float('inf')
            for op in operators:
                if time.time() > self.kill_time:
                    return self.heuristic(env, self.agent_id)

                cloned_env = env.clone()
                cloned_env.apply_operator(agent_id, op)

                v = self.rb_minimax(cloned_env, 1 - agent_id, depth-1)

                if v < curr_min:
                    curr_min = v

            return curr_min
        else:
            curr_max = float('-inf')
            for op in operators:
                if time.time() > self.kill_time:
                    return self.heuristic(env, self.agent_id)

                cloned_env = env.clone()
                cloned_env.apply_operator(agent_id, op)

                v = self.rb_minimax(cloned_env, 1 - agent_id, depth-1)

                if v > curr_max:
                    curr_max = v

            return curr_max

    def anytime_minimax(self, env: TaxiEnv, agent_id, time_limit):
        operator = 'park'
        curr_min = float('inf')
        depth = 1
        self.kill_time = time.time() + 0.9 * time_limit

        while time.time() <= self.kill_time:
            local_op = 'park'
            local_min = float('inf')

            operators = env.get_legal_operators(agent_id)

            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                if time.time() > self.kill_time:
                    return operator

                self.taxi_in_turn_index = agent_id
                child.apply_operator(agent_id, op)

                v = self.rb_minimax(child, agent_id, depth)
                if v < local_min:
                    local_min = v
                    local_op = op

            if local_min < curr_min:
                curr_min = local_min
                operator = local_op

            depth += 1

        return operator


    def run_step(self,env:TaxiEnv, agent_id,time_limit):
        self.agent_id = agent_id

        return self.anytime_minimax(env, agent_id,time_limit)

class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def __init__(self):
        self.MAX_MD = 14
        self.taxi_in_turn_index = None
        self.agent_id = None
        self.kill_time = None


    def successors(self, env: TaxiEnv, taxi_id: int):
        operators = env.get_legal_operators(taxi_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(taxi_id, op)
        return operators, children

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        taxi = env.get_taxi(taxi_id)
        other_taxi = env.get_taxi((taxi_id + 1) % 2)
        cash_diff = taxi.cash - other_taxi.cash

        def msh(agent: Taxi, passenger: Passenger):
            return manhattan_distance(agent.position, passenger.position) + \
                   manhattan_distance(passenger.position, passenger.destination)

        distance_to_passengers = [msh(taxi, p) for p in env.passengers]

        if taxi.passenger is not None:  # IF TAXI HAS A PASSENGER THEN DELIVER
            return manhattan_distance(taxi.position, taxi.passenger.destination) - self.MAX_MD * taxi.cash \
                   - cash_diff

        if len(distance_to_passengers) == 1:  # other taxi already has a passenger
            return distance_to_passengers[0] - self.MAX_MD * taxi.cash \
                   - cash_diff

        other_taxi_distance_to_passengers = [msh(other_taxi, p) for p in env.passengers]

        # assuming that the other taxi will move towards its closer passenger
        #   from the passengers we can guarantee accessing,
        #   we choose the closest
        if distance_to_passengers[0] <= distance_to_passengers[1]:
            # if we know that it is safe to choose 0 (other_taxi wont get there first) OR
            # we know that other_taxi will choose 1
            if distance_to_passengers[0] <= other_taxi_distance_to_passengers[0] or \
                    other_taxi_distance_to_passengers[1] <= other_taxi_distance_to_passengers[0]:
                # for case of ==, it depends on the scheduling in the joint actions.
                # we chose to have the agent "compete" (50-50 chance) since that is at least as good
                # as two "greedy-for-closest-reward" agents
                return distance_to_passengers[0] - self.MAX_MD * taxi.cash \
                       - cash_diff
            return distance_to_passengers[1] - self.MAX_MD * taxi.cash \
                   - cash_diff

        if distance_to_passengers[1] <= other_taxi_distance_to_passengers[0] or \
                other_taxi_distance_to_passengers[0] <= other_taxi_distance_to_passengers[1]:
            return distance_to_passengers[1] - self.MAX_MD * taxi.cash \
                   - cash_diff
        return distance_to_passengers[0] - self.MAX_MD * taxi.cash \
               - cash_diff

    def rb_alphabeta(self, env: TaxiEnv, agent_id, depth, alpha, beta):
        if time.time() > self.kill_time or env.done() or depth == 0:
            return self.heuristic(env, self.agent_id)

        operators, children = self.successors(env, agent_id)
        if self.agent_id == agent_id:
            curr_min = float('inf')
            for op in operators:
                if time.time() > self.kill_time:
                    return self.heuristic(env, self.agent_id)

                cloned_env = env.clone()
                cloned_env.apply_operator(agent_id, op)

                v = self.rb_alphabeta(cloned_env, 1 - agent_id, depth-1, alpha, beta)

                if v < curr_min:
                    curr_min = v

                if curr_min < beta:
                    beta = curr_min

                if curr_min <= alpha:
                    return float('-inf')
            return curr_min
        else:
            curr_max = float('-inf')
            for op in operators:
                if time.time() > self.kill_time:
                    return self.heuristic(env, self.agent_id)

                cloned_env = env.clone()
                cloned_env.apply_operator(agent_id, op)

                v = self.rb_alphabeta(cloned_env, 1 - agent_id, depth-1, alpha, beta)

                if v > curr_max:
                    curr_max = v

                if curr_max > alpha:
                    alpha = curr_max

                if curr_max >= beta:
                    return float('inf')
            return curr_max

    def anytime_alphabeta(self, env: TaxiEnv, agent_id, time_limit):
        operator = 'park'
        curr_min = float('inf')
        depth = 1
        alpha = float('-inf')
        beta = float('inf')
        self.kill_time = time.time() + 0.9 * time_limit

        while time.time() <= self.kill_time:
            local_op = 'park'
            local_min = float('inf')

            operators = env.get_legal_operators(agent_id)

            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                if time.time() > self.kill_time:
                    return operator

                self.taxi_in_turn_index = agent_id
                child.apply_operator(agent_id, op)

                v = self.rb_alphabeta(child, agent_id, depth,alpha,beta)
                if v < local_min:
                    local_min = v
                    local_op = op

            if local_min < curr_min:
                curr_min = local_min
                operator = local_op

            depth += 1

        return operator

    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        self.agent_id = agent_id

        return self.anytime_alphabeta(env, agent_id, time_limit)


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()
