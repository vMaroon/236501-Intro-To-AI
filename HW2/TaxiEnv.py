import random
from copy import copy


def manhattan_distance(p0, p1):
    return abs(p0[0] - p1[0]) + abs(p0[1] - p1[1])


class Taxi(object):
    def __init__(self, position, fuel, cash):
        self.position = position
        self.fuel = fuel
        self.cash = cash
        self.passenger = None

    def __repr__(self):
        return 'position:' + str(self.position) + ' fuel: ' + str(self.fuel) + \
               ' cash: ' + str(self.cash) + ' passenger: [' + str(self.passenger) + ']'


class Passenger(object):
    def __init__(self, position, destination):
        self.position = position
        self.destination = destination

    def __repr__(self):
        return 'position:' + str(self.position) + ' destination: ' + str(self.destination)


class GasStation(object):
    def __init__(self, position):
        self.position = position

    def __repr__(self):
        return 'position:' + str(self.position)


class TaxiEnv(object):
    def __init__(self):
        self.gas_stations = None
        self.passengers = None
        self.taxis = None
        self.seed = None
        self.num_steps = None

    def generate(self, seed, num_steps):
        self.num_steps = num_steps
        self.seed = seed
        self.taxis = [Taxi(p, 16, 0) for p in self.random_cells(2)]
        self.passengers = [Passenger(p, d) for _ in range(2) for p in self.random_cells(1) for d in
                           self.random_cells(1)]
        self.gas_stations = [GasStation(p) for p in self.random_cells(2)]

    def clone(self):
        cloned = TaxiEnv()
        cloned.num_steps = self.num_steps
        cloned.seed = self.seed
        cloned.taxis = [copy(t) for t in self.taxis]
        cloned.passengers = [copy(p) for p in self.passengers]
        cloned.gas_stations = [copy(g) for g in self.gas_stations]
        return cloned

    def random_cells(self, count: int):
        random.seed(self.seed)
        self.seed = random.randint(0, 255)
        return random.sample([(x, y) for x in range(4) for y in range(4)], count)

    def get_taxi(self, taxi_id):
        return self.taxis[taxi_id]

    def get_taxi_in(self, position):
        taxis = [taxi for taxi in self.taxis if taxi.position == position]
        if len(taxis) == 0:
            return None
        else:
            return taxis[0]

    def get_gas_station_in(self, position):
        gas_stations = [gas_station for gas_station in self.gas_stations if gas_station.position == position]
        if len(gas_stations) == 0:
            return None
        return gas_stations[0]

    def get_passenger_in(self, position):
        passengers = [passenger for passenger in self.passengers if passenger.position == position]
        if len(passengers) == 0:
            return None
        return passengers[0]

    def get_legal_operators(self, taxi_index: int):
        ops = []
        taxi = self.taxis[taxi_index]
        taxi_pos = taxi.position
        if taxi.fuel > 0:
            for op_move, op_disp in [('move north', (0, -1)), ('move south', (0, 1)),
                                     ('move west', (-1, 0)), ('move east', (1, 0))]:
                new_pos = (taxi_pos[0] + op_disp[0], taxi_pos[1] + op_disp[1])
                if 4 > new_pos[0] >= 0 and 4 > new_pos[1] >= 0 \
                        and self.get_taxi_in(new_pos) is None:
                    ops.append(op_move)
        else:
            ops.append('park')
        if self.get_gas_station_in(taxi_pos) and taxi.cash > 0:
            ops.append("refuel")
        if taxi.passenger is not None and taxi.passenger.destination == taxi_pos:
            ops.append("drop off passenger")
        if taxi.passenger is None and self.get_passenger_in(taxi.position) is not None:
            ops.append("pick up passenger")
        return ops

    def move_taxi(self, taxi_index: int, offset):
        p = self.taxis[taxi_index].position
        self.taxis[taxi_index].position = p[0] + offset[0], p[1] + offset[1]
        self.taxis[taxi_index].fuel -= 1

    def spawn_passenger(self):
        ps = self.random_cells(2)
        self.passengers.append(Passenger(ps[0], ps[1]))

    def apply_operator(self, taxi_index: int, operator: str):
        self.num_steps -= 1
        taxi = self.taxis[taxi_index]
        assert operator in self.get_legal_operators(taxi_index)
        assert not self.num_steps < 0
        if operator == 'park':
            pass
        elif operator == 'move north':
            self.move_taxi(taxi_index, (0, -1))
        elif operator == 'move south':
            self.move_taxi(taxi_index, (0, 1))
        elif operator == 'move east':
            self.move_taxi(taxi_index, (1, 0))
        elif operator == 'move west':
            self.move_taxi(taxi_index, (-1, 0))
        elif operator == 'pick up passenger':
            passenger = self.get_passenger_in(taxi.position)
            self.taxis[taxi_index].passenger = passenger
            self.passengers.remove(passenger)
        elif operator == 'refuel':
            taxi.fuel += taxi.cash
            taxi.cash = 0
        elif operator == 'drop off passenger':
            taxi.cash += manhattan_distance(taxi.passenger.position, taxi.passenger.destination) * 2
            self.spawn_passenger()
            taxi.passenger = None
        else:
            assert False

    def done(self):
        return len([taxi for taxi in self.taxis if taxi.fuel > 0]) == 0 or self.num_steps <= 0

    def get_balances(self):
        return [t.cash for t in self.taxis]

    def taxi_is_occupied(self, taxi_index):
        return self.taxis[taxi_index].passenger is not None

    def print(self):
        for y in range(4):
            for x in range(4):
                p = (x, y)
                taxi = self.get_taxi_in(p)
                passenger = self.get_passenger_in(p)
                gas_station = self.get_gas_station_in(p)
                passenger_destination = [passenger for passenger in self.passengers if passenger.destination == p]
                taxi_passenger_destination = [i for i, taxi in enumerate(self.taxis) if taxi.passenger is not None
                                              and taxi.passenger.destination == p]
                if taxi:
                    print('[T' + str(self.taxis.index(taxi)) + ']', end='')
                elif passenger:
                    print('[P' + str(self.passengers.index(passenger)) + ']', end='')
                elif gas_station:
                    print('[G' + str(self.gas_stations.index(gas_station)) + ']', end='')
                elif len(passenger_destination) > 0:
                    print('[D' + str(self.passengers.index(passenger_destination[0])) + ']', end='')
                elif len(taxi_passenger_destination) > 0:
                    print('[X' + str(taxi_passenger_destination[0]) + ']', end='')
                else:
                    print('[  ]', end='')
            print('')
        print('taxis: ', self.taxis)
        print('passengers on street: ', self.passengers)
        print('gas stations: ', self.gas_stations)
