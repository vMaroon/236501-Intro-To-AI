from TaxiEnv import TaxiEnv, manhattan_distance
import random


class Agent:
    # returns the next operator to be applied - i.e. takes one turn
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()

    # returns list of legal operators and matching list of states reached by applying them
    def successors(self, env: TaxiEnv, taxi_id: int):
        operators = env.get_legal_operators(taxi_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(taxi_id, op)
        return operators, children

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        taxi = env.get_taxi(taxi_id)
        other_taxi = env.get_taxi((taxi_id+1) % 2)
        return taxi.cash - other_taxi.cash


# picks random operators from the legal ones
class AgentRandom(Agent):
    def run_step(self, env: TaxiEnv, taxi_id, time_limit):
        operators, _ = self.successors(env, taxi_id)
        return random.choice(operators)


class AgentGreedy(Agent):
    def run_step(self, env: TaxiEnv, taxi_id, time_limit):
        operators = env.get_legal_operators(taxi_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(taxi_id, op)
        children_heuristics = [self.heuristic(child, taxi_id) for child in children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]