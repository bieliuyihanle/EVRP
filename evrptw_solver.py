from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from targets import Customer, CharingStation, Target


@dataclass
class PeriodData:
    """Customer data that belongs to a single planning period."""

    name: str
    customers: List[Customer]


@dataclass
class MultiPeriodRoutingProblem:
    """Container with all information required to solve consecutive periods."""

    config: "RoutingProblemConfiguration"
    depot: Target
    charging_stations: List[CharingStation]
    periods: List[PeriodData]

    def is_multi_period(self) -> bool:
        return len(self.periods) > 1


@dataclass
class PeriodSolution:
    """Solution information for a single period."""

    name: str
    cost: float
    routes: List["Route"]
    remaining_energy: List[float]


@dataclass
class MultiPeriodSolution:
    """Aggregated solutions of all consecutive periods."""

    periods: List[PeriodSolution]

    @property
    def total_cost(self) -> float:
        return sum(period.cost for period in self.periods)



class RoutingProblemConfiguration:
    def __init__(self, tank_capacity, now_energy, payload_capacity, fuel_consumption_rate, charging_rate, velocity):
        self.tank_capacity = tank_capacity
        if isinstance(now_energy, (list, tuple, np.ndarray)):
            self.vehicle_initial_energies = [float(energy) for energy in now_energy]
            self.now_energy = float(self.vehicle_initial_energies[0]) if self.vehicle_initial_energies else 0.0
        else:
            self.vehicle_initial_energies = None
            self.now_energy = now_energy
        self.payload_capacity = payload_capacity
        self.fuel_consumption_rate = fuel_consumption_rate
        self.charging_rate = charging_rate
        self.velocity = velocity

    def get_initial_energy(self, vehicle_index=None):
        if self.vehicle_initial_energies is None:
            return self.now_energy
        if vehicle_index is None:
            return self.vehicle_initial_energies[0] if self.vehicle_initial_energies else 0.0
        if vehicle_index < len(self.vehicle_initial_energies):
            return self.vehicle_initial_energies[vehicle_index]
        return 1500


    def get_vehicle_count(self) -> int:
        if self.vehicle_initial_energies is None:
            return 1
        return len(self.vehicle_initial_energies)

    def clone_with_initial_energy(self, initial_energy):
        """Create a copy of the configuration using different initial energies."""

        if isinstance(initial_energy, list):
            energy_copy = [float(value) for value in initial_energy]
        elif isinstance(initial_energy, tuple):
            energy_copy = [float(value) for value in initial_energy]
        else:
            energy_copy = float(initial_energy)

        return RoutingProblemConfiguration(
            self.tank_capacity,
            energy_copy,
            self.payload_capacity,
            self.fuel_consumption_rate,
            self.charging_rate,
            self.velocity,
        )


class RoutingProblemInstance:
    def __init__(self, config, depot, customers, charging_stations):
        self.config = config
        self.depot = depot
        self.customers = customers
        self.charging_stations = charging_stations

        # distance matrices
        self.cust_cust_dist = np.zeros((len(self.customers), len(self.customers)))
        self.cust_cs_dist = np.zeros((len(self.customers), len(self.charging_stations)))

        # vertex lookup dict
        self.vertices = dict()

        # initialization of distance matrices
        for i in range(0, len(self.customers)):
            for j in range(0, len(self.customers)):

                if i == 0:
                    from_v = self.depot
                else:
                    from_v = self.customers[i-1]

                if j == 0:
                    to_v = self.depot
                else:
                    to_v = self.customers[j-1]

                self.cust_cust_dist[i, j] = from_v.distance_to(to_v)

        for i in range(1, len(self.customers)):
            for j in range(0, len(self.charging_stations)-1):
                if i == 0:
                    from_v = self.depot
                else:
                    from_v = self.customers[i-1]

                self.cust_cs_dist[i, j] = from_v.distance_to(self.charging_stations[j])

        # initialization of the lookup dict
        self.vertices[self.depot.id] = self.depot
        for c in self.customers:
            self.vertices[c.id] = c
        for cs in self.charging_stations:
            self.vertices[cs.id] = cs

    def get_initial_energy(self, vehicle_index=None):
        return self.config.get_initial_energy(vehicle_index)


class Route:
    def __init__(self, config, depot, vehicle_index=None, initial_energy=None):
        self.config = config
        self.route = [depot]
        self.depot = depot
        self.vehicle_index = vehicle_index
        if initial_energy is None:
            self.initial_energy = self.config.get_initial_energy(vehicle_index)
        else:
            self.initial_energy = initial_energy

    def __getitem__(self, key):
        # 如果key是切片对象，返回一个新的Route实例对象
        if isinstance(key, slice):
            # 创建一个新的Route实例对象
            sliced_route = Route(self.config, self.depot, vehicle_index=self.vehicle_index,
                                 initial_energy=self.initial_energy)
            # 将route列表进行切片，更新新实例的route属性
            sliced_route.route = self.route[key]
            return sliced_route
        # 否则返回对应索引的元素
        return self.route[key]

    @classmethod
    def make_list_to_route(cls, problem_instance, node_list, vehicle_index=None, initial_energy=None):
        route_instance = cls(problem_instance.config, problem_instance.depot, vehicle_index=vehicle_index,
                             initial_energy=initial_energy)
        route_instance.route = node_list  # 直接将传入的列表设置为路线，不再包含 depot
        return route_instance

    def generate_basic_route(self, target):
        self.route += [target]
        return self.route

    def remove_target(self, target):
        if target in self.route:
            self.route.remove(target)

    def is_nc_feasible(self):
        if self.tw_constraint_violated():
            return False
        elif self.payload_capacity_constraint_violated():
            return False
        else:
            return True

    def need_charge(self):
        return self.initial_energy - self.calculate_total_distance() * self.config.fuel_consumption_rate < 0.2 * self.config.tank_capacity

    def is_feasible(self):
        if self.tw_constraint_violated():
            return False
        elif self.tank_capacity_constraint_violated():
            return False
        elif self.payload_capacity_constraint_violated():
            return False
        else:
            return True

    def is_complete(self):
        return self.route[0] == self.depot and self.route[-1] == self.depot and self.depot not in self.route[1:-1]

    # CONSTRAINT VALIDATION METHODS
    def tw_constraint_violated(self):
        elapsed_time = self.route[0].call_time + self.route[0].service_time

        for i in range(1, len(self.route)):
            elapsed_time = elapsed_time + self.route[i - 1].distance_to(self.route[i]) / self.config.velocity
            if elapsed_time > self.route[i].due_date:
                return True

            if type(self.route[i]) is CharingStation:
                missing_energy = self.config.tank_capacity - self.calculate_remaining_tank_capacity(self.route[i])
                self.route[i].service_time = missing_energy * self.config.charging_rate

            elapsed_time += self.route[i].service_time

        return False

    def tank_capacity_constraint_violated(self):
        last = None
        tank_capacity = self.config.tank_capacity
        now_energy = self.initial_energy

        for t in self.route:
            if last is not None:
                distance = last.distance_to(t)
                consumption = distance * self.config.fuel_consumption_rate

                now_energy -= consumption

                if tank_capacity < 0:
                    return True

                if type(t) is CharingStation:
                    now_energy = self.config.tank_capacity
            last = t

        return False

    def payload_capacity_constraint_violated(self):
        total_demand = 0
        demand = self.calculate_demand()
        for t in range(1, len(self.route)):
            total_demand += demand[t-1]
        return total_demand > self.config.payload_capacity

    # STATUS CALCULATION METHODS
    def calculate_total_distance(self):
        last = None
        dist = 0

        for t in self.route:
            if last is not None:
                dist += last.distance_to(t)
            last = t

        return dist

    def calculate_total_cost(self):
        dist_cost = self.calculate_total_distance()
        arrival_times = self.calculate_arrival_times()
        time_cost = 0
        for i in range(1, len(self.route)):
            if type(self.route[i]) is CharingStation:
                time_cost += 0
            if self.route[i] is self.depot:
                time_cost += 0
            if type(self.route[i]) is Customer:
                time_cost += self.route[i].due_date-arrival_times[i-1]
        total_cost = dist_cost + 0.1 * time_cost + 1000

        return total_cost

    def calculate_remaining_tank_capacity(self, until=None):
        last = None
        now_energy = self.initial_energy
        total_consumption = 0
        for t in self.route:
            if last is not None:
                distance = last.distance_to(t)
                consumption = distance * self.config.fuel_consumption_rate
                # total_consumption += consumption
                now_energy -= consumption

                if until == t:
                    return now_energy

                if type(t) is CharingStation:
                    now_energy = self.config.tank_capacity
            last = t
            # print(total_consumption)
        return now_energy

    def calculate_remaining_energy(self):
        return self.calculate_remaining_tank_capacity()

    def calculate_arrival_times(self):
        # print(type(self.route))
        # print(self.route)
        elapsed_time = self.route[0].call_time + self.route[0].service_time
        arrival_times = []

        for i in range(1, len(self.route)):
            travel_time = self.route[i - 1].distance_to(self.route[i]) / self.config.velocity
            elapsed_time += travel_time

            # 记录当前节点的实际到达时间
            arrival_times.append(elapsed_time)

            if type(self.route[i]) is CharingStation:
                missing_energy = self.config.tank_capacity - self.calculate_remaining_tank_capacity(self.route[i])
                self.route[i].service_time = missing_energy * self.config.charging_rate

            # 在服务前计算等待时间
            elapsed_time += self.route[i].service_time

        return arrival_times

    def calculate_demand(self):
        arrival_times = self.calculate_arrival_times()
        demand = []
        for i in range(1, len(self.route)):
            if type(self.route[i]) is CharingStation:
                self.route[i].demand = 0
            if type(self.route[i]) is Customer:
                self.route[i].demand = (48 - self.route[i].stock_at_call_time + (arrival_times[i-1] -
                                                                                 self.route[i].call_time) / 30)*0.75
            if self.route[i] is self.depot:
                self.route[i].demand = 0
            demand.append(self.route[i].demand)
        return demand

    def calculate_total_duration(self):
        elapsed_time = self.route[0].call_time + self.route[0].service_time

        for i in range(1, len(self.route)):
            elapsed_time = elapsed_time + self.route[i - 1].distance_to(self.route[i]) / self.config.velocity

            if type(self.route[i]) is CharingStation:
                missing_energy = self.config.tank_capacity - self.calculate_remaining_tank_capacity(self.route[i])
                self.route[i] = missing_energy * self.config.charging_rate

            elapsed_time += self.route[i].service_time

        return elapsed_time

    def get_reachable_charging_stations(self, cust: Customer, capacity: float, problem_instance) -> list:
        max_dist = capacity / problem_instance.config.fuel_consumption_rate
        reachable_stations = []

        for cs in problem_instance.charging_stations:
            if cs.distance_to(cust) <= max_dist and cust.id != cs.id:
                reachable_stations.append(cs)

        return reachable_stations

    # def k_nearest_neighbor_min_due_date(self, customers, problem_instance, k=3):
    #
    #     giant_route = []
    #     serviced_customers = set()
    #
    #     while customers:
    #         route = Route(self.config, self.depot)
    #         last_position = self.depot
    #
    #         while customers:
    #             possible_successors = [customer for customer in customers if customer not in serviced_customers]
    #             possible_successors.sort(key=lambda n: n.distance_to(last_position))
    #             possible_successors = possible_successors[:k]
    #             if not possible_successors:
    #                 break
    #             successor = min(possible_successors, key=lambda n: n.due_date)
    #             route.generate_basic_route(successor)
    #
    #             if not route.is_nc_feasible():
    #                 route.remove_target(successor)
    #                 break
    #             serviced_customers.add(successor)
    #             last_position = successor
    #         route.generate_basic_route(self.depot)
    #         if len(route.route) > 1:
    #             giant_route.append(route)
    #             customers = [customer for customer in customers if customer not in serviced_customers]
    #
    #     return giant_route
    #
    # def process_route(self, solution, problem_instance):
    #     # print("111111111111111111111111111")
    #     # 初始化未访问的客户列表
    #     unvisited_customers = []
    #
    #     for idx, value in enumerate(solution):
    #
    #         # 判断当前路径是否需要充电
    #         while solution[idx].route.need_charge():
    #
    #             k = solution[idx].route.make_route_feasible_and_best(problem_instance)
    #             if k is not None:
    #                 # 找到可行的插入点，更新路径
    #                 solution[idx].route = k
    #                 break  # 找到充电位置，跳出while循环，继续下一条路径
    #             else:
    #                 # 找不到插入点，移除最小到达时间的客户点
    #                 earliest_customer = self.find_earliest_customer()
    #                 solution[idx].route.remove(earliest_customer)    # 从当前路径中移除
    #                 unvisited_customers.append(earliest_customer)  # 将该客户点标记为未访问
    #                 # 路径调整后，继续判断是否需要充电
    #
    #     # 循环直到所有未访问客户都被处理
    #     while unvisited_customers:
    #         # print("Current unvisited_customers:", unvisited_customers)
    #
    #         # 使用k-最近邻域启发式为未访问的客户生成新的路径
    #         depot = self.depot
    #         new_routes = self.k_nearest_neighbor_min_due_date(unvisited_customers, problem_instance)
    #         # print(new_routes)
    #         # 对新的路径进行充电站插入操作
    #         for idx, value in enumerate(new_routes):
    #             # print(value)
    #             # print(self.need_charge(value))
    #             while value.need_charge():
    #                 k = value.make_route_feasible_and_best(problem_instance)
    #                 if k is not None:
    #                     new_routes[idx] = k
    #                     break
    #                 else:
    #                     earliest_customer = new_routes[idx].find_earliest_customer()
    #                     new_routes[idx].remove_target(earliest_customer)
    #                     unvisited_customers.append(earliest_customer)
    #
    #         # 展开 new_routes，将其转化为单个客户的列表
    #         flattened_new_routes = [customer for route in new_routes for customer in route]
    #         # print(flattened_new_routes)
    #         # 更新 unvisited_customers 列表，确保未服务的客户仍然保留
    #         unvisited_customers = [customer for customer in unvisited_customers if customer not in flattened_new_routes]
    #
    #         # print("11111111Current unvisited_customers:", unvisited_customers)
    #         solution.extend(new_routes)
    #         # print(state.state)
    #     # print("22222222222222")
    #     return solution
    #
    # def find_earliest_customer(self):
    #     # 假设每个客户点包含属性 'due_date'
    #     # 返回路径中 due_date 最小的客户点
    #     earliest_customer = None
    #     min_due_date = float('inf')
    #
    #     for v in self.route:
    #         if isinstance(v, Customer):
    #             if v.due_date < min_due_date:
    #                 min_due_date = v.due_date
    #                 earliest_customer = v
    #
    #     return earliest_customer

    def find_optimal_charging_station_insertion(self, problem_instance):
        best_insertion_point = None
        best_station = None
        min_total_cost = float('inf')

        # 遍历路径中的每个顾客之间的位置作为插入点
        for i in range(1, len(self.route)):
            if not isinstance(self.route[i-1], CharingStation):

                reachable_stations = self.get_reachable_charging_stations(self.route[i - 1],
                                                                          self[:i].calculate_remaining_tank_capacity(),
                                                                          problem_instance)

                for j in reachable_stations:
                    temp_route = self[:i].generate_basic_route(j) + self[i:].route
                    temp_route = Route.make_list_to_route(problem_instance, temp_route,
                                                         vehicle_index=self.vehicle_index,
                                                         initial_energy=self.initial_energy)
                    # 判断插入后路径是否可行
                    if temp_route.is_feasible():
                        # 计算插入后的总成本
                        total_cost = temp_route.calculate_total_cost()
                        # 更新最优插入点和总成本
                        if total_cost < min_total_cost:
                            min_total_cost = total_cost
                            best_insertion_point = i
                            best_station = j

        return best_insertion_point, best_station, min_total_cost

    def make_route_feasible_and_best(self, problem_instance):
        best_insertion_point, best_station, min_total_cost = self.find_optimal_charging_station_insertion(problem_instance)
        best_feasible_route = self[:best_insertion_point].generate_basic_route(best_station) + self[best_insertion_point:].route
        # print(best_feasible_route)
        best_feasible_route1 = Route.make_list_to_route(problem_instance, best_feasible_route,
                                                       vehicle_index=self.vehicle_index,
                                                       initial_energy=self.initial_energy)

        if best_station is None:
            return None
        else:
            return best_feasible_route

    def remove_charging_station(self):
        self.route[:] = [i for i in self.route if type(i) is not CharingStation]
        return self.route

    def calculate_dist_to_first_customer(self, reverse=False):
        dist = 0
        last = None

        if reverse:
            self.route.reverse()

        for t in self.route:
            if last is not None:
                dist += last.distance_to(t)
                if type(t) is Customer:
                    if reverse:
                        self.route.reverse()
                    return dist
            last = t

        return dist

    def get_first_customer(self, reverse=False):
        if reverse:
            self.route.reverse()

        for t in self.route:
            if type(t) is Customer:
                if reverse:
                    self.route.reverse()
                return t

    def append_route(self, new_route):
        if new_route.route[0] == self.depot:
            route_to_append = new_route[1:]

        if self.route[-1] == self.depot:
            self.route = self.route[0:-1]

        self.route = self.route + route_to_append

    def __str__(self):
        route_str = '['

        for t in self.route:
            route_str += t.id + ', '

        route_str += ']'
        return route_str

    def __repr__(self):
        route_str = '['

        for t in self.route:
            route_str += t.id + ', '

        route_str += ']'
        return route_str


class EVRPTWSolver:
    """
    A simple framework for solving the EVRPTW (Electronic Vehicle Routing Problem with Time Windows)
    """

    def __init__(self, construction_heuristic, meta_heuristic=None):
        """
        :param construction_heuristic: heuristic for constructing a initial solution
        :param meta_heuristic: meta heuristic, that improves the construction heuristic solution
        """
        self.construction_heuristic = construction_heuristic
        self.meta_heuristic = meta_heuristic
        self.last_remaining_energy = []

    def solve(self, problem_instance):
        solution = self.construction_heuristic.solve(problem_instance)
        # print("hallo")
        if self.meta_heuristic:
            solution = self.meta_heuristic.improve(problem_instance, solution)

        cost = 0
        # print(solution)
        # solution.reverse()
        # print(solution)
        self.last_remaining_energy = []
        for route in solution:
            # print(route)
            # print(type(route))
            # r = Route.make_list_to_route(problem_instance, route)
            # print(type(r))
            cost += route.calculate_total_cost()
            if hasattr(route, "calculate_remaining_energy"):
                remaining_energy = route.calculate_remaining_energy()
                self.last_remaining_energy.append(remaining_energy)
        # print(solution)
        # print(type(solution))
        return cost, solution

    @staticmethod
    def _normalize_remaining_energy(remaining, previous, tank_capacity, desired_count):
        """Ensure that the remaining energy list has entries for all vehicles."""

        previous = previous or []
        desired_count = max(desired_count, len(remaining))
        normalized = []

        for index in range(desired_count):
            if index < len(remaining):
                normalized.append(float(remaining[index]))
            elif index < len(previous):
                normalized.append(float(previous[index]))
            else:
                normalized.append(float(tank_capacity))

        return normalized

    def solve_multi_period(self, multi_period_problem: MultiPeriodRoutingProblem) -> MultiPeriodSolution:
        """Solve a consecutive multi-period problem instance."""

        period_solutions: List[PeriodSolution] = []
        uses_vehicle_list = multi_period_problem.config.vehicle_initial_energies is not None

        if uses_vehicle_list:
            current_energy: List[float] = list(multi_period_problem.config.vehicle_initial_energies)
        else:
            current_energy = [multi_period_problem.config.now_energy]

        vehicle_count = len(current_energy)

        for period in multi_period_problem.periods:
            initial_energy = current_energy if uses_vehicle_list else current_energy[0]
            period_config = multi_period_problem.config.clone_with_initial_energy(initial_energy)
            problem_instance = RoutingProblemInstance(
                period_config,
                multi_period_problem.depot,
                list(period.customers),
                multi_period_problem.charging_stations,
            )

            cost, routes = self.solve(problem_instance)
            remaining = list(self.last_remaining_energy)
            normalized_remaining = self._normalize_remaining_energy(
                remaining, current_energy, period_config.tank_capacity, vehicle_count
            )

            period_solutions.append(
                PeriodSolution(
                    name=period.name,
                    cost=cost,
                    routes=routes,
                    remaining_energy=list(normalized_remaining),
                )
            )

            current_energy = list(normalized_remaining)
            vehicle_count = max(vehicle_count, len(current_energy))

        if period_solutions:
            self.last_remaining_energy = list(period_solutions[-1].remaining_energy)

        return MultiPeriodSolution(period_solutions)

def evaluate_period_solution(
    problem_instance: RoutingProblemInstance,
    solution_routes: Sequence[Union[Route, Sequence[Union[Target, str]]]],
    initial_energies: Union[float, Sequence[float], None] = None,
    period_name: str = "manual",
) -> PeriodSolution:
    """Calculate the cost of a manually provided solution for a single period.

    Parameters
    ----------
    problem_instance:
        ``RoutingProblemInstance`` that describes the considered period.
    solution_routes:
        Sequence containing either :class:`Route` objects or iterables with the
        identifiers of the visited vertices. When identifiers are provided the
        function resolves them using ``problem_instance.vertices``.
    initial_energies:
        Optional override of the vehicles' initial energies. A single value is
        broadcast to all routes while a sequence must provide at least as many
        entries as routes in ``solution_routes``. When omitted the energies from
        ``problem_instance.config`` are used.
    period_name:
        Optional label used for the returned :class:`PeriodSolution`.

    Returns
    -------
    PeriodSolution
        Object containing the evaluated routes, their total cost and the
        remaining energy after completing each route.
    """

    if not isinstance(solution_routes, Sequence) or isinstance(solution_routes, (str, bytes)):
        raise TypeError("solution_routes must be a sequence of routes.")

    route_count = len(solution_routes)
    if route_count == 0:
        raise ValueError("At least one route must be provided for evaluation.")

    if initial_energies is None:
        if problem_instance.config.vehicle_initial_energies is not None:
            energy_values: List[float] = list(problem_instance.config.vehicle_initial_energies)
        else:
            energy_values = [float(problem_instance.config.now_energy)]
    elif isinstance(initial_energies, (int, float)):
        energy_values = [float(initial_energies)]
    else:
        energy_values = [float(value) for value in initial_energies]

    if len(energy_values) == 1 and route_count > 1:
        energy_values = energy_values * route_count
    elif len(energy_values) < route_count:
        raise ValueError(
            "initial_energies must provide at least as many entries as routes in solution_routes."
        )

    def _make_route(
        raw_route: Union[Route, Sequence[Union[Target, str]]],
        vehicle_index: int,
        initial_energy: float,
    ) -> Route:
        if isinstance(raw_route, Route):
            route_instance = Route(
                problem_instance.config,
                problem_instance.depot,
                vehicle_index=vehicle_index,
                initial_energy=initial_energy,
            )
            route_instance.route = list(raw_route.route)
            return route_instance

        if isinstance(raw_route, (str, bytes)) or not isinstance(raw_route, Iterable):
            raise TypeError(
                "Each entry in solution_routes must be a Route or an iterable of targets/identifiers."
            )

        resolved_nodes: List[Target] = []
        for node in raw_route:
            if isinstance(node, Target):
                resolved_nodes.append(node)
            else:
                node_id = str(node)
                if node_id not in problem_instance.vertices:
                    raise KeyError(f"Unknown target identifier '{node_id}' in route definition.")
                resolved_nodes.append(problem_instance.vertices[node_id])

        if not resolved_nodes:
            raise ValueError("Route definitions must not be empty.")

        if resolved_nodes[0] is not problem_instance.depot:
            resolved_nodes.insert(0, problem_instance.depot)
        if resolved_nodes[-1] is not problem_instance.depot:
            resolved_nodes.append(problem_instance.depot)

        route_instance = Route(
            problem_instance.config,
            problem_instance.depot,
            vehicle_index=vehicle_index,
            initial_energy=initial_energy,
        )
        route_instance.route = resolved_nodes
        return route_instance

    evaluated_routes: List[Route] = []
    remaining_energy: List[float] = []
    total_cost = 0.0

    for index, raw_route in enumerate(solution_routes):
        route_energy = energy_values[index]
        route = _make_route(raw_route, index, route_energy)
        total_cost += route.calculate_total_cost()
        remaining_energy.append(float(route.calculate_remaining_energy()))
        evaluated_routes.append(route)

    return PeriodSolution(
        name=period_name,
        cost=total_cost,
        routes=evaluated_routes,
        remaining_energy=remaining_energy,
    )


def _coerce_target(
    raw_target: Union[Target, Dict[str, Any]],
    target_cls: type,
    fallback_idx: int,
) -> Target:
    """Convert a dictionary or existing object into a :class:`Target` subclass."""

    if isinstance(raw_target, target_cls):
        return raw_target

    if isinstance(raw_target, Target):
        if target_cls is Target or isinstance(raw_target, target_cls):
            return raw_target
        raise TypeError(
            "Provided target instance does not match the expected target type."
        )

    if not isinstance(raw_target, dict):
        raise TypeError(
            "Targets must be provided either as dictionaries or existing Target instances."
        )

    target_id = str(raw_target.get("id", fallback_idx))
    idx_value = int(raw_target.get("idx", fallback_idx))
    x = float(raw_target.get("x", 0.0))
    y = float(raw_target.get("y", 0.0))
    stock = float(raw_target.get("stock_at_call_time", 0.0))
    call_time = float(raw_target.get("call_time", 0.0))
    due_date = float(raw_target.get("due_date", 0.0))
    service_time = float(raw_target.get("service_time", 0.0))

    return target_cls(target_id, idx_value, x, y, stock, call_time, due_date, service_time)


def evaluate_manual_solution(
    depot: Union[Target, Dict[str, Any]],
    customers: Sequence[Union[Customer, Dict[str, Any]]],
    charging_stations: Sequence[Union[CharingStation, Dict[str, Any]]],
    solution_routes: Sequence[Sequence[Union[Target, str]]],
    tank_capacity: float,
    initial_energies: Union[float, Sequence[float]],
    fuel_consumption_rate: float,
    velocity: float,
    payload_capacity: Optional[float] = None,
    charging_rate: float = 0.0,
    period_name: str = "manual",
) -> PeriodSolution:
    """Evaluate a manually specified solution without building solver objects.

    Parameters
    ----------
    depot, customers, charging_stations:
        Points of the problem. Each entry can either be an existing ``Target``
        instance (``Target``, ``Customer`` or ``CharingStation``) or a
        dictionary with the fields ``id``, ``idx``, ``x``, ``y``,
        ``stock_at_call_time``, ``call_time``, ``due_date`` and ``service_time``.
        Missing dictionary fields default to ``0`` and ``id``/``idx`` fall back
        to the insertion order.
    solution_routes:
        Sequence describing the planned vehicle tours. Each route is a sequence
        containing either target identifiers (strings/integers) or direct target
        objects. Depot visits are added automatically when missing.
    tank_capacity:
        Maximum energy capacity of each vehicle.
    initial_energies:
        Initial energy for the vehicles. Provide a single numeric value to reuse
        it for all routes or a sequence with an explicit value per route.
    fuel_consumption_rate:
        Energy consumed per unit of travelled distance.
    velocity:
        Vehicle speed used to transform distances into travel times.
    payload_capacity:
        Optional payload capacity. When omitted an effectively unlimited
        capacity is used.
    charging_rate:
        Time required to recharge one unit of energy at a charging station.
    period_name:
        Name of the evaluated period for reporting purposes.

    Returns
    -------
    PeriodSolution
        Object containing the evaluated routes, their total cost and the
        remaining energy after each route.
    """

    depot_target = _coerce_target(depot, Target, 0)

    customer_targets = [
        _coerce_target(customer, Customer, index + 1)
        for index, customer in enumerate(customers)
    ]

    station_targets = [
        _coerce_target(station, CharingStation, index + 1)
        for index, station in enumerate(charging_stations)
    ]

    effective_payload_capacity = (
        float(payload_capacity)
        if payload_capacity is not None
        else float("inf")
    )

    config = RoutingProblemConfiguration(
        tank_capacity=tank_capacity,
        now_energy=initial_energies,
        payload_capacity=effective_payload_capacity,
        fuel_consumption_rate=fuel_consumption_rate,
        charging_rate=charging_rate,
        velocity=velocity,
    )

    problem_instance = RoutingProblemInstance(
        config=config,
        depot=depot_target,
        customers=[depot_target] + customer_targets,
        charging_stations=station_targets + [depot_target],
    )

    return evaluate_period_solution(
        problem_instance=problem_instance,
        solution_routes=solution_routes,
        initial_energies=initial_energies,
        period_name=period_name,
    )