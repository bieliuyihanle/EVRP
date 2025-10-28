import numpy as np
import sys
from targets import Customer
from evrptw_solver import Route


# =======================================================
#                GIANT ROUTE HEURISTICS
# =======================================================

def k_nearest_neighbor_min_ready_time(depot, customers, k=3):
    last_position = depot
    giant_route = []

    while len(giant_route) != len(customers):
        possible_successors = [n for n in customers if n not in giant_route]
        possible_successors.sort(key=lambda n: n.distance_to(last_position))
        possible_successors = possible_successors[:k]

        successor = min(possible_successors, key=lambda n: n.ready_time)
        giant_route.append(successor)

        last_position = successor

    return giant_route

def k_nearest_neighbor_min_due_date(depot, customers,problem_instance, k=3, start_vehicle_index=0):

    giant_route = []
    serviced_customers = set()

    while customers:
        route_index = start_vehicle_index + len(giant_route)
        route = Route(problem_instance.config, problem_instance.depot, vehicle_index=route_index)
        last_position = depot

        while customers:
            possible_successors = [customer for customer in customers if customer not in serviced_customers]
            possible_successors.sort(key=lambda n: n.distance_to(last_position))
            possible_successors = possible_successors[:k]
            if not possible_successors:
                break
            successor = min(possible_successors, key=lambda n: n.due_date)
            route.generate_basic_route(successor)
            demand = route.calculate_demand()
            if not route.is_nc_feasible():
                route.remove_target(successor)
                break
            serviced_customers.add(successor)
            last_position = successor
        route.generate_basic_route(depot)
        if len(route.route) > 1:
            giant_route.append(route)
            customers = [customer for customer in customers if customer not in serviced_customers]

    return giant_route

# def k_nearest_neighbor_min_due_date(depot, customers, k=3):
#     last_position = depot
#     giant_route = []
#     print("Customers:", customers)
#     print("Type of customers:", type(customers))
#     while len(giant_route) != len(customers):
#         possible_successors = [n for n in customers if n not in giant_route]
#         possible_successors.sort(key=lambda n: n.distance_to(last_position))
#         possible_successors = possible_successors[:k]
#
#         successor = min(possible_successors, key=lambda n: n.due_date)
#         giant_route.append(successor)
#
#         last_position = successor
#
#     return giant_route

def nearest_neighbor_tolerance_min_due_date(problem_instance, depot, customers, tolerance=1.3):
    giant_route = []
    # number = sum(len(sublist) for sublist in giant_route)
    number = 0
    while len(customers) != number:
        route_index = len(giant_route)
        route = Route(problem_instance.config, problem_instance.depot, vehicle_index=route_index)
        last_position = depot

        while len(customers) != number:
            possible_successors = [n for n in customers if n not in giant_route]
            min_distance = min(possible_successors, key=lambda x: x.distance_to(last_position)).distance_to(
                last_position)
            possible_successors = [n for n in possible_successors if
                                   n.distance_to(last_position) <= min_distance * tolerance]

            possible_successors.sort(key=lambda n: n.distance_to(last_position))
            successor = min(possible_successors, key=lambda n: n.due_date)
            route.generate_basic_route(successor)
            if not route.is_nc_feasible():
                route.remove_target(successor)
                break
            last_position = successor
            number += 1

        customers = route[1:]
        giant_route.append(customers)

    return giant_route


def nearest_neighbor_tolerance_min_ready_time(depot, customers, tolerance=1.3):
    last_position = depot
    giant_route = []

    while len(giant_route) != len(customers):
        possible_successors = [n for n in customers if n not in giant_route]
        min_distance = min(possible_successors, key=lambda x: x.distance_to(last_position)).distance_to(
            last_position)
        possible_successors = [n for n in possible_successors if
                               n.distance_to(last_position) <= min_distance * tolerance]

        possible_successors.sort(key=lambda n: n.distance_to(last_position))

        successor = min(possible_successors, key=lambda n: n.ready_time)
        giant_route.append(successor)

        last_position = successor

    return giant_route

def insert_cost(customer, route, idx):
    """
    Computes the insertion cost for inserting customer in route at idx.
    """
    pred = 0 if idx == 0 else route[idx - 1]
    succ = 0 if idx == len(route) else route[idx]
    return pred.distance_to(customer) + customer.distance_to(succ) - pred.distance_to(succ)


def generate_basic_route(from_route, target):
    from_route.route += [target]
    return from_route


def process_route(depot, solution, problem_instance):
    # print("111111111111111111111111111")
    # 初始化未访问的客户列表
    unvisited_customers = []

    for idx, value in enumerate(solution):
        value.vehicle_index = idx
        value.initial_energy = value.config.get_initial_energy(idx)

        # 判断当前路径是否需要充电
        while solution[idx].need_charge():

            k = solution[idx].make_route_feasible_and_best(problem_instance)
            if k is not None:
                # 找到可行的插入点，更新路径
                solution[idx].route = k
                break  # 找到充电位置，跳出while循环，继续下一条路径
            else:
                # 找不到插入点，移除最小到达时间的客户点
                earliest_customer = find_earliest_customer(solution[idx].route)
                solution[idx].route.remove(earliest_customer)    # 从当前路径中移除
                unvisited_customers.append(earliest_customer)  # 将该客户点标记为未访问
                # 路径调整后，继续判断是否需要充电

    # 循环直到所有未访问客户都被处理
    while unvisited_customers:
        # print("Current unvisited_customers:", unvisited_customers)

        # 使用k-最近邻域启发式为未访问的客户生成新的路径
        depot = depot
        start_index = len(solution)
        new_routes = k_nearest_neighbor_min_due_date(
            depot, unvisited_customers, problem_instance, start_vehicle_index=start_index)
        # print(new_routes)
        # 对新的路径进行充电站插入操作
        for idx, value in enumerate(new_routes):
            vehicle_index = start_index + idx
            value.vehicle_index = vehicle_index
            value.initial_energy = value.config.get_initial_energy(vehicle_index)
            # print(value)
            # print(self.need_charge(value))
            while value.need_charge():
                k = value.make_route_feasible_and_best(problem_instance)
                if k is not None:
                    new_routes[idx].route = k
                    break
                else:
                    earliest_customer = find_earliest_customer(new_routes[idx].route)
                    print(earliest_customer)
                    new_routes[idx].remove_target(earliest_customer)
                    unvisited_customers.append(earliest_customer)

        # 展开 new_routes，将其转化为单个客户的列表
        flattened_new_routes = [customer for route in new_routes for customer in route]
        # print(flattened_new_routes)
        # 更新 unvisited_customers 列表，确保未服务的客户仍然保留
        unvisited_customers = [customer for customer in unvisited_customers if customer not in flattened_new_routes]
        for route in new_routes:
            print(type(route))
        # print("11111111Current unvisited_customers:", unvisited_customers)
        solution.extend(new_routes)
        # print(state.state)
    # print("22222222222222")
    return solution


def find_earliest_customer(route):
    # 假设每个客户点包含属性 'due_date'
    # 返回路径中 due_date 最小的客户点
    earliest_customer = None
    min_due_date = float('inf')

    for v in route:
        if isinstance(v, Customer):
            if v.due_date < min_due_date:
                min_due_date = v.due_date
                earliest_customer = v

    return earliest_customer

class BeasleyHeuristic:
    def __init__(self, process_route, generate_giant_route, giant_route_args, generate_feasible_route=generate_basic_route):
        self.process_route = process_route
        self.generate_giant_route = generate_giant_route
        self.generate_feasible_route = generate_feasible_route
        self.giant_route_args = giant_route_args
        self.giant_route = None
        # self.problem_instance = problem_instance

    def solve(self, problem_instance):
        solution = []
        self.giant_route = self.generate_giant_route(problem_instance.depot, problem_instance.customers,problem_instance,
                                                     *self.giant_route_args)
        print(self.giant_route)
        solution = self.process_route(problem_instance.depot, self.giant_route, problem_instance)
        # for route in self.giant_route:
        #     if route.need_charge():
        #         r = route.make_route_feasible_and_best(problem_instance)
        #         r = Route.make_list_to_route(problem_instance, r)
        #         solution += [r]
        #     else:
        #         # r = Route.make_list_to_route(problem_instance, route)
        #         solution += [route]
        # print(solution)
        # for route in solution:
        #     print(type(route))
        return solution
