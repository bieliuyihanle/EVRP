from copy import deepcopy
from queue import LifoQueue
from random import randint, shuffle, random, choice
from math import exp
from evrptw_solver import RoutingProblemInstance, Route
from targets import Customer, CharingStation
from itertools import combinations, product
import matplotlib.pyplot as plt
import sys
import time
import os
from alns.accept import SimulatedAnnealing
import numpy as np
import numpy.random as rnd
from alns import ALNS
from alns.select import RouletteWheel
from alns.stop import MaxIterations, MaxRuntime, NoImprovement
import pandas as pd


K_MAX = 4
NO_IMPROVEMENT_TOLERANCE = 1


def calculate_route_remaining_energy(problem_instance: RoutingProblemInstance, route, vehicle_index: int) -> float:
    """Return the remaining energy of a route using the vehicle-specific initial energy."""

    if not route:
        return problem_instance.config.get_initial_energy(vehicle_index)

    current_energy = problem_instance.config.get_initial_energy(vehicle_index)
    last_position = problem_instance.depot
    # print(current_energy)

    for vertex_id in route[1:]:
        target = problem_instance.vertices[vertex_id]
        distance = last_position.distance_to(target)
        current_energy -= distance * problem_instance.config.fuel_consumption_rate
        # print(current_energy)
        if isinstance(target, CharingStation):
            current_energy = problem_instance.config.tank_capacity

        last_position = target

    return current_energy
def calculate_route_remaining_energy1(problem_instance: RoutingProblemInstance, route, vehicle_index: int) -> float:
    """Return the remaining energy of a route using the vehicle-specific initial energy."""

    if not route:
        return problem_instance.config.get_initial_energy(vehicle_index)

    current_energy = problem_instance.config.get_initial_energy(vehicle_index)
    last_position = problem_instance.depot
    print(current_energy)

    for vertex_id in route[1:]:
        target = problem_instance.vertices[vertex_id]
        distance = last_position.distance_to(target)
        current_energy -= distance * problem_instance.config.fuel_consumption_rate
        print(current_energy)
        # print(current_energy)
        if isinstance(target, CharingStation):
            current_energy = problem_instance.config.tank_capacity

        last_position = target

    return current_energy

# neighbourhoods
def two_opt_star(state, cost, remove_operator, nc_feasible_operator, state_feasibility_operator, route_cost_operator, need_operator,
                 insert_station_operation,process_state, min_cost=sys.maxsize):
    pre_state = deepcopy(state)
    for route in state:
        route = remove_operator(route)

    route_combinations = list(combinations(range(0, len(state)), 2))
    shuffle(route_combinations)

    for c in route_combinations:
        route_1 = list(state[c[0]])
        route_2 = list(state[c[1]])

        swap_points = list(product(range(1, len(route_1) - 1), range(1, len(route_2) - 1)))
        shuffle(swap_points)

        for sp in swap_points:
            temp_route_cost = 0
            temp_state = deepcopy(state)
            temp_state.remove(route_1)
            temp_state.remove(route_2)

            split_index_1 = sp[0]
            split_index_2 = sp[1]

            new_route_1 = route_1[:split_index_1] + route_2[split_index_2:]
            new_route_2 = route_2[:split_index_2] + route_1[split_index_1:]

            temp_state.append(new_route_1)
            temp_state.append(new_route_2)

            if nc_feasible_operator(temp_state):
                process_state(temp_state)
                # for idx, value in enumerate(temp_state):
                #     if need_operator(temp_state[idx]):
                #         k = insert_station_operation(value)
                #         if k is None:
                #             break
                #         temp_state[idx] = insert_station_operation(value)

                if state_feasibility_operator(temp_state) is True:
                    for k in temp_state:
                        temp_route_cost += route_cost_operator(k)
                    if temp_route_cost < min_cost:
                        return temp_state, temp_route_cost
                else:
                    continue
    return pre_state, cost


def two_opt(state, cost, remove_operator, nc_feasible_operator, state_feasibility_operator, route_cost_operator, need_operator,
            insert_station_operation,process_state, min_cost=sys.maxsize):
    pre_state = deepcopy(state)
    for route in state:
        route = remove_operator(route)

    route_indices = list(range(0, len(state)))
    shuffle(route_indices)

    for i in route_indices:
        route = state[i]
        cut_points = list(product(range(1, len(route) - 1), range(0, len(route) - 1)))
        shuffle(cut_points)

        for cp in cut_points:
            if cp[0] < cp[1] and cp[1] - cp[0] > 1:
                part_1 = route[:cp[0]]
                part_2 = route[cp[0]:cp[1]]
                part_3 = route[cp[1]:]
                part_2.reverse()

                temp_route_cost = 0
                temp_state = state[:i] + [part_1 + part_2 + part_3] + state[i + 1:]

                if nc_feasible_operator(temp_state):
                    process_state(temp_state)
                    # for idx, value in enumerate(temp_state):
                    #     if need_operator(temp_state[idx]):
                    #         k = insert_station_operation(value)
                    #         if k is None:
                    #             break
                    #         temp_state[idx] = insert_station_operation(value)

                    if state_feasibility_operator(temp_state) is True:
                        for k in temp_state:
                            temp_route_cost += route_cost_operator(k)
                        if temp_route_cost < min_cost:
                            return temp_state, temp_route_cost
                    else:
                        continue
            return pre_state, cost


def or_opt(state, cost, remove_operator, nc_feasible_operator, state_feasibility_operator, route_cost_operator, need_operator,
           insert_station_operation,process_state, min_cost=sys.maxsize):
    pre_state = deepcopy(state)
    for route in state:
        route = remove_operator(route)

    route_indices = list(range(0, len(state)))
    shuffle(route_indices)

    for i in route_indices:
        route = state[i]
        cut_points = list(product(range(1, len(route) - 1), range(0, len(route) - 1)))
        shuffle(cut_points)

        for cp in cut_points:
            if cp[0] < cp[1] and cp[1] - cp[0] > 1:
                part_1 = route[:cp[0]]
                part_2 = route[cp[0]:cp[1]]
                part_3 = route[cp[1]:]

                offset = randint(0, len(part_3) - 1)

                new_route = part_1 + part_3[:offset] + part_2 + part_3[offset:]
                temp_route_cost = 0
                temp_state = deepcopy(state)
                temp_state.remove(route)
                temp_state.append(new_route)

                if nc_feasible_operator(temp_state):
                    process_state(temp_state)
                    # for idx, value in enumerate(temp_state):
                    #     if need_operator(temp_state[idx]):
                    #         k = insert_station_operation(value)
                    #         if k is None:
                    #             break
                    #         temp_state[idx] = insert_station_operation(value)

                    if state_feasibility_operator(temp_state) is True:
                        for k in temp_state:
                            temp_route_cost += route_cost_operator(k)
                        if temp_route_cost < min_cost:
                            return temp_state, temp_route_cost
                    else:
                        continue
            return pre_state, cost
    return pre_state, cost


def cross_exchange(state, cost, remove_operator, nc_feasible_operator, state_feasibility_operator, route_cost_operator, need_operator,
                insert_station_operation,process_state, min_cost=sys.maxsize):
    pre_state = deepcopy(state)
    for route in state:
        route = remove_operator(route)

    route_combinations = list(combinations(range(0, len(state)), 2))
    shuffle(route_combinations)

    for c in route_combinations:

        route_1 = list(state[c[0]])
        route_2 = list(state[c[1]])

        swap_points = list(product(range(1, len(route_1) - 1), range(1, len(route_1) - 1), range(1, len(route_2) - 1),
                                   range(1, len(route_2) - 1)))
        shuffle(swap_points)

        for sp in swap_points:

            if sp[0] < sp[1] - 1 and sp[2] < sp[3] - 1:
                sr_1_s = min(sp[0], sp[1])
                sr_1_e = max(sp[0], sp[1])
                sr_2_s = min(sp[2], sp[3])
                sr_2_e = max(sp[2], sp[3])

                new_route_1 = route_1[:sr_1_s] + route_2[sr_2_s:sr_2_e] + route_1[sr_1_e:]
                new_route_2 = route_2[:sr_2_s] + route_1[sr_1_s:sr_1_e] + route_2[sr_2_e:]
                temp_route_cost = 0
                temp_state = deepcopy(state)
                temp_state.remove(route_1)
                temp_state.remove(route_2)
                temp_state.append(new_route_1)
                temp_state.append(new_route_2)

                if nc_feasible_operator(temp_state):
                    process_state(temp_state)
                    # for idx, value in enumerate(temp_state):
                    #     if need_operator(temp_state[idx]):
                    #         k = insert_station_operation(value)
                    #         if k is None:
                    #             break
                    #         temp_state[idx] = insert_station_operation(value)

                    if state_feasibility_operator(temp_state) is True:
                        for k in temp_state:
                            temp_route_cost += route_cost_operator(k)
                        if temp_route_cost < min_cost:
                            return temp_state, temp_route_cost
                    else:
                        continue
            return pre_state, cost
    return pre_state, cost


def merge_route(state, cost, remove_operator, nc_feasible_operator, state_feasibility_operator, route_cost_operator, need_operator,
                insert_station_operation, min_cost=sys.maxsize):
    pre_state = deepcopy(state)
    #
    # for route in state:
    #     route = remove_operator(route)
    #
    # route_combinations = list(combinations(range(0, len(state)), 2))
    # shuffle(route_combinations)
    #
    # for c in route_combinations:
    #     temp_route_cost = 0
    #     temp_state = deepcopy(state)
    #     route_1 = list(state[c[0]])
    #     route_2 = list(state[c[1]])
    #     new_route = route_1[:-1] + route_2[1:]
    #     temp_state.remove(route_1)
    #     temp_state.remove(route_2)
    #     temp_state.append(new_route)
    #
    #     if nc_feasible_operator(temp_state):
    #         for idx, value in enumerate(temp_state):
    #             if need_operator(temp_state[idx]):
    #                 k = insert_station_operation(value)
    #                 if k is None:
    #                     break
    #                 temp_state[idx] = insert_station_operation(value)
    #
    #         if state_feasibility_operator(temp_state) is True:
    #             for k in temp_state:
    #                 temp_route_cost += route_cost_operator(k)
    #             if temp_route_cost < min_cost:
    #                 return temp_state, temp_route_cost
    return pre_state, cost


class Adaptive:
    def __init__(self, problem_instance: RoutingProblemInstance, state, cost, fig_name, unassigned=None):
        self.problem_instance = problem_instance
        self.state = state
        self.cost = 0
        for idx, r in enumerate(self.state):
            self.problem_instance.config.now_energy = self.problem_instance.config.get_initial_energy(idx)
            self.cost += self.calculate_route_cost(r)
        self.unassigned = unassigned if unassigned is not None else []
        self.customers_id = []
        for k in self.problem_instance.customers:
            self.customers_id.append(k.id)
        self.fig_name = fig_name

    def _set_vehicle_energy(self, vehicle_index: int) -> None:
        self.problem_instance.config.now_energy = self.problem_instance.config.get_initial_energy(vehicle_index)

    def init_way(self, problem_instance: RoutingProblemInstance, routes, cost, unassigned=None):
        return Adaptive(self.problem_instance, self.state, self.cost, self.unassigned)

    # # 正交试验
    #     def improve_solution(self):
    #         problem_instance1 = self.problem_instance
    #         state1 = self.state
    #         cost1 = self.cost
    #         unassigned1 = self.unassigned
    #
    #         alns = ALNS(rnd.RandomState())
    #
    #         alns.add_destroy_operator(self.random_removal)
    #         alns.add_destroy_operator(self.random_route_removal)
    #         alns.add_destroy_operator(self.shortest_route_removal)
    #         alns.add_destroy_operator(self.least_cus_route_removal)
    #         alns.add_destroy_operator(self.worst_dist_cust_removal)
    #         alns.add_destroy_operator(self.worst_time_cust_removal)
    #         alns.add_destroy_operator(self.shaw_destroy)
    #         alns.add_destroy_operator(self.proximity_based_removal)
    #         alns.add_destroy_operator(self.time_based_removal)
    #         alns.add_destroy_operator(self.zone_removal)
    #
    #         alns.add_repair_operator(self.greedy_repair)
    #         alns.add_repair_operator(self.regret_2_insertion)
    #         alns.add_repair_operator(self.regret_3_insertion)
    #         alns.add_repair_operator(self.time_based_repair)
    #         alns.add_repair_operator(self.random_repair)
    #
    #         init = self.init_way(problem_instance1, state1, cost1, unassigned1)
    #
    #         orthogonal_table = [
    #             [100, 0.8, 0.1, [0.45, 0.3, 0.15, 0]],
    #             [100, 0.9, 0.4, [33, 13, 9, 0]],
    #             [100, 0.999, 0.6, [45, 15, 3, 0]],
    #             [100, 0.99999, 0.9, [100, 10, 1, 0]],
    #             [200, 0.8, 0.4, [45, 15, 3, 0]],
    #             [200, 0.9, 0.1, [100, 10, 1, 0]],
    #             [200, 0.999, 0.9, [0.45, 0.3, 0.15, 0]],
    #             [200, 0.99999, 0.6, [33, 13, 9, 0]],
    #             [300, 0.8, 0.6, [100, 10, 1, 0]],
    #             [300, 0.9, 0.9, [45, 15, 3, 0]],
    #             [300, 0.999, 0.1, [33, 13, 9, 0]],
    #             [300, 0.99999, 0.4, [0.45, 0.3, 0.15, 0]],
    #             [400, 0.8, 0.9, [33, 13, 9, 0]],
    #             [400, 0.9, 0.6, [0.45, 0.3, 0.15, 0]],
    #             [400, 0.999, 0.4, [100, 10, 1, 0]],
    #             [400, 0.99999, 0.1, [45, 15, 3, 0]]
    #         ]
    #
    #         # 实验记录表
    #         experiment_results = []
    #         num_repeats = 20
    #
    #         for i, (start_temp,  step, decay, weights) in enumerate(orthogonal_table):
    #             print(
    #                 f"Running experiment {i + 1} with parameters: weights={weights}, decay={decay}, start_temp={start_temp}, step={step}")
    #
    #             objective_values = []  # 记录20次实验的目标函数值
    #
    #             # 进行20次实验
    #             for _ in range(num_repeats):
    #
    #                 # 配置RouletteWheel和SimulatedAnnealing的参数
    #                 select = RouletteWheel(weights, decay, 10, 5)
    #                 accept = SimulatedAnnealing(start_temperature=start_temp, end_temperature=1, step=step,
    #                                             method='exponential')
    #                 stop = MaxRuntime(10)
    #
    #                 result = alns.iterate(init, select, accept, stop)
    #
    #                 solution = result.best_state
    #                 objective = solution.objective()
    #                 objective_values.append(objective)
    #
    #             # 计算S/N比，公式：S/N = -10 * log10( (1/n) * sum(Y_i^2) )
    #             sn_ratio = -10 * np.log10(np.mean(np.square(objective_values)))
    #
    #             # 保存结果
    #             experiment_results.append({
    #                 'Experiment': i + 1,
    #                 'weights': weights,
    #                 'decay': decay,
    #                 'start_temp': start_temp,
    #                 'step': step,
    #                 'S/N Ratio': sn_ratio,
    #                 'Objective Values': objective_values
    #             })
    #
    #         # 将结果保存到Excel
    #         df = pd.DataFrame(experiment_results)
    #         df.to_excel('orthogonal_experiment_results.xlsx', index=False)
    #
    #         # 显示结果
    #         print(df)

    def improve_solution(self):
        problem_instance1 = self.problem_instance
        state1 = self.state
        cost1 = self.cost
        unassigned1 = self.unassigned

        alns = ALNS(rnd.RandomState())

        alns.add_destroy_operator(self.random_removal)
        alns.add_destroy_operator(self.random_route_removal)
        alns.add_destroy_operator(self.shortest_route_removal)
        alns.add_destroy_operator(self.least_cus_route_removal)
        alns.add_destroy_operator(self.worst_dist_cust_removal)
        alns.add_destroy_operator(self.worst_time_cust_removal)
        alns.add_destroy_operator(self.shaw_destroy)
        alns.add_destroy_operator(self.proximity_based_removal)
        alns.add_destroy_operator(self.time_based_removal)
        alns.add_destroy_operator(self.zone_removal)

        alns.add_repair_operator(self.greedy_repair)
        alns.add_repair_operator(self.regret_2_insertion)
        alns.add_repair_operator(self.regret_3_insertion)
        alns.add_repair_operator(self.time_based_repair)
        alns.add_repair_operator(self.random_repair)

        init = self.init_way(problem_instance1, state1, cost1, unassigned1)

        select = RouletteWheel([45, 15, 3, 0], 0.9, 10, 5)
        accept = SimulatedAnnealing(start_temperature=400, end_temperature=1, step=0.9, method='exponential')
        # accept = SimulatedAnnealing.autofit(init.objective(), 0.05, 0.5, 30000)
        # stop = MaxIterations(30000)
        stop = MaxRuntime(1)


        result = alns.iterate(init, select, accept, stop)

        solution = result.best_state
        objective = solution.objective()

        # remaining_energies = [
        #     calculate_route_remaining_energy(self.problem_instance, route, idx)
        #     for idx, route in enumerate(solution.state)
        # ]
        #
        # print(f"Adaptive best solution  remaining energy: {remaining_energies:.3f}")
        # print(remaining_energies)

        # 从 result.statistics 中提取收敛数据
        statistics = result.statistics
        objectives = statistics.objectives
        # print(f"statistics is",statistics)
        # print(f"objectives is", objectives)

        best_objectives = np.minimum.accumulate(objectives)  # 计算最佳目标值的累积序列
        runtimes = np.cumsum(statistics.runtimes)  # 累积运行时间
        runtimes = np.append(runtimes, result.statistics.total_runtime)
        # print(f"best values is",best_objectives)
        # print(f"runtimes is", runtimes)
        return objective, solution.state,best_objectives.tolist(), runtimes.tolist()
        # self.plot_convergence_curve(result, self.fig_name)
        #
        # _, ax = plt.subplots(figsize=(12, 6))
        # result.plot_objectives(ax=ax)
        # # 保存图像到文件夹
        # # 创建保存路径
        # save_dir = r'C:\Users\10133\Desktop\evrp\改\收敛曲线'
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)  # 如果路径不存在则创建
        #
        # # 保存图像，文件名格式为 "实例名_迭代次数.png"
        # file_name = '{0}.png'.format(self.fig_name)
        # save_path = os.path.join(save_dir, file_name)
        # plt.savefig(save_path, format='png')
        # plt.close()  # 关闭图像以释放内存
        #
        #
        # return objective, solution.state



    def plot_convergence_curve(self, result, instance_name):
        # 获取统计信息
        statistics = result.statistics

        # 获取目标值 (objective values) 和 迭代时间 (runtimes)
        objectives = statistics.objectives
        best_result = np.minimum.accumulate(objectives)  # 计算最佳目标值
        runtimes = np.cumsum(statistics.runtimes)  # 累加迭代时间作为总运行时间
        runtimes = np.append(runtimes, result.statistics.total_runtime)

        # 绘制收敛曲线
        plt.figure(figsize=(12, 6))

        # 绘制当前目标值的收敛曲线
        # plt.plot(runtimes, objectives, label='Convergence Curve')

        # 绘制最优目标值的收敛曲线
        plt.plot(runtimes, best_result, label='Best Objective', linestyle='--')

        plt.xlabel('Runtime (seconds)')
        plt.ylabel('Objective Value')
        plt.title(f'Convergence Curve for {instance_name}')
        plt.legend()

        # 保存图像
        save_dir = r'C:\Users\10133\Desktop\evrp\改\收敛曲线'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = '{0}.png'.format(instance_name)
        save_path = os.path.join(save_dir, file_name)

        plt.savefig(save_path, format='png')
        plt.close()

    def copy(self):
        return Adaptive(self.problem_instance, deepcopy(self.state), self.cost,  self.unassigned.copy())

    def objective(self):
        """
        Computes the total route costs.
        """
        return sum(self.calculate_route_cost(route) for route in self.state)

    def find_route(self, customer):
        """
        Return the route that contains the passed-in customer.
        """
        for route in self.state:
            if customer in route:
                return route

        raise ValueError(f"Solution does not contain customer {customer}.")

    # @property
    # def cost(self):
    #     """
    #     Alias for objective method. Used for plotting.
    #     """
    #     return self.objective()

    def random_removal(self, state, rnd_state):
        lower_degree_of_destruction = 0.2
        higher_degree_of_destruction = 0.4
        lower_num_to_destroy = int(len(self.customers_id) * lower_degree_of_destruction)
        higher_num_to_destroy = int(len(self.customers_id) * higher_degree_of_destruction)
        customers_to_remove = rnd_state.randint(lower_num_to_destroy,higher_num_to_destroy)
        """
        Removes a number of randomly selected customers from the passed-in solution.
        """
        destroyed = state.copy()
        for route in destroyed.state:
            route = self.remove_charging_station(route)

        for customer in rnd_state.choice(
                self.customers_id, customers_to_remove, replace=False
        ):
            destroyed.unassigned.append(customer)
            route = destroyed.find_route(customer)
            route.remove(customer)
        # print(destroyed.unassigned)
        return self.remove_empty_routes(destroyed)

    def worst_dist_cust_removal(self, state, rnd_state):
        lower_degree_of_destruction = 0.2
        higher_degree_of_destruction = 0.4
        lower_num_to_destroy = int(len(self.customers_id) * lower_degree_of_destruction)
        higher_num_to_destroy = int(len(self.customers_id) * higher_degree_of_destruction)
        customers_to_remove = rnd_state.randint(lower_num_to_destroy,higher_num_to_destroy)

        destroyed = state.copy()
        for route in destroyed.state:
            route = self.remove_charging_station(route)

        cust_cost = []

        for route in destroyed.state:
            for idx in range(1, len(route) - 1):
                cust_cost_value = self.cust_cost(route[idx], route, idx)
                cust_cost.append((cust_cost_value, route[idx], route))
        sorted_cust = sorted(cust_cost, key=lambda x: x[0], reverse=True)
        worst_dist_cust = sorted_cust[:customers_to_remove]

        for _, customer, route in worst_dist_cust:
            route.remove(customer)
            destroyed.unassigned.append(customer)

        return self.remove_empty_routes(destroyed)

    def worst_time_cust_removal(self, state, rnd_state):
        lower_degree_of_destruction = 0.2
        higher_degree_of_destruction = 0.4
        lower_num_to_destroy = int(len(self.customers_id) * lower_degree_of_destruction)
        higher_num_to_destroy = int(len(self.customers_id) * higher_degree_of_destruction)
        customers_to_remove = rnd_state.randint(lower_num_to_destroy,higher_num_to_destroy)

        destroyed = state.copy()
        for route in destroyed.state:
            route = self.remove_charging_station(route)

        time_difference = []
        for route in destroyed.state:
            arrival_times = self.calculate_arrival_times(route)
            for i in range(1,len(route)):
                target = self.problem_instance.vertices[route[i]]
                if type(target) is Customer:
                    time_difference_value = target.due_date - arrival_times[i - 1]
                    time_difference.append((time_difference_value, route[i], route))

        sorted_cust = sorted(time_difference, key=lambda x: x[0], reverse=True)
        worst_time_cust = sorted_cust[:customers_to_remove]
        for _, customer, route in worst_time_cust:
            route.remove(customer)
            destroyed.unassigned.append(customer)
        # print(destroyed.unassigned)
        return self.remove_empty_routes(destroyed)

    # def shaw_destroy(self, state, rnd_state, num_customers_to_remove=3):
    #
    #     destroyed = state.copy()
    #     for route in destroyed.state:
    #         route = self.remove_charging_station(route)
    #
    #     initial_customer = rnd_state.choice(self.customers_id)
    #     # 计算每个客户的相关性
    #     relatedness_scores = []
    #     for customer in self.customers_id:
    #         if customer == initial_customer:
    #             continue
    #         relatedness_score = self.calculate_relatedness(initial_customer, customer)
    #         relatedness_scores.append((relatedness_score, customer))
    #     # 根据相关性对顾客进行排序
    #     sorted_customers = sorted(relatedness_scores, key=lambda x: x[0], reverse=True)
    #
    #     # 选择要移除的顾客
    #     customers_to_remove = [initial_customer] + [customer for _, customer in sorted_customers[:num_customers_to_remove - 1]]
    #
    #     for customer in customers_to_remove:
    #         route = destroyed.find_route(customer)
    #         route.remove(customer)
    #         destroyed.unassigned.append(customer)
    #
    #     return self.remove_empty_routes(destroyed)

    # 这个算子删除的客户点会*2
    def shaw_destroy(self, state, rnd_state, num_customers_to_remove=3):
        destroyed = state.copy()
        for route in destroyed.state:
            self.remove_charging_station(route)

        removed_customers = set()

        for _ in range(num_customers_to_remove):
            available_customers = list(set(self.customers_id) - set(destroyed.unassigned) - removed_customers)

            if not available_customers:
                break

            initial_customer = rnd_state.choice(available_customers)

            relatedness_scores = []
            for customer in available_customers:
                if customer == initial_customer:
                    continue
                relatedness_score = self.calculate_relatedness(initial_customer, customer, destroyed.state)
                relatedness_scores.append((relatedness_score, customer))

            if not relatedness_scores:
                break

            # 选择 relatedness_scores 最小的顾客点
            most_related_customer = min(relatedness_scores, key=lambda x: x[0])[1]

            route = destroyed.find_route(most_related_customer)
            if route:
                route.remove(most_related_customer)
                destroyed.unassigned.append(most_related_customer)
                removed_customers.add(most_related_customer)
            else:
                continue
        # print(destroyed.unassigned)
        return self.remove_empty_routes(destroyed)

    def calculate_relatedness(self, customer1, customer2, state):
        real_customer1 = self.problem_instance.vertices[customer1]
        real_customer2 = self.problem_instance.vertices[customer2]

        # 计算两者之间的距离
        distance_score = real_customer1.distance_to(real_customer2)
        # 根据需求、时间窗等增加其他相关性计算
        demand_score = abs(real_customer1.demand - real_customer2.demand)
        # print(real_customer1.demand,real_customer2.demand)
        time_window_score = abs(real_customer1.due_date - real_customer2.due_date)

        route_similarity_score = self.calculate_lij(customer1, customer2, state)

        # 加权组合得到最终的相关性分数
        relatedness_score = (1.0 * distance_score) + (1.0 * demand_score) + (0.5 * time_window_score) + (
                    1.0 * route_similarity_score)

        return relatedness_score

    def calculate_lij(self, customer1, customer2, state):
        # 假设self.problem_instance.routes是当前解的路径列表，每个路径包含多个顾客点
        for route in state:
            if customer1 in route and customer2 in route:
                return -1  # 在同一路线中
        return 1  # 不在同一路线中

    def proximity_based_removal(self, state, rnd_state, num_customers_to_remove=3):
        destroyed = state.copy()
        for route in destroyed.state:
            self.remove_charging_station(route)  # 移除充电站，确保只处理顾客点

        removed_customers = set()  # 用于跟踪已移除的顾客
        available_customers = list(set(self.customers_id) - set(destroyed.unassigned))

        # 随机选择一个初始顾客
        initial_customer = rnd_state.choice(available_customers)
        route = destroyed.find_route(initial_customer)
        route.remove(initial_customer)
        destroyed.unassigned.append(initial_customer)
        removed_customers.add(initial_customer)

        last_selected_customer = initial_customer

        for _ in range(num_customers_to_remove - 1):
            # 更新可移除的顾客列表
            available_customers = list(set(self.customers_id) - set(destroyed.unassigned) - removed_customers)

            if not available_customers:
                break  # 如果没有可移除的顾客，提前终止

            # 选择距离上一个顾客点最近的顾客点
            nearest_customer = min(available_customers, key=lambda customer:
            self.problem_instance.vertices[last_selected_customer].distance_to(
                self.problem_instance.vertices[customer]))

            # 移除该顾客
            route = destroyed.find_route(nearest_customer)
            route.remove(nearest_customer)
            destroyed.unassigned.append(nearest_customer)
            removed_customers.add(nearest_customer)

            # 更新上一个选择的顾客点
            last_selected_customer = nearest_customer
        # print(destroyed.unassigned)
        return self.remove_empty_routes(destroyed)

    def time_based_removal(self, state, rnd_state, num_customers_to_remove=3):
        destroyed = state.copy()
        for route in destroyed.state:
            self.remove_charging_station(route)  # 移除充电站，确保只处理顾客点

        removed_customers = set()  # 用于跟踪已移除的顾客
        available_customers = list(set(self.customers_id) - set(destroyed.unassigned))

        # 随机选择一个初始顾客
        initial_customer = rnd_state.choice(available_customers)
        route = destroyed.find_route(initial_customer)
        route.remove(initial_customer)
        destroyed.unassigned.append(initial_customer)
        removed_customers.add(initial_customer)

        last_selected_customer = initial_customer

        for _ in range(num_customers_to_remove - 1):
            # 更新可移除的顾客列表
            available_customers = list(set(self.customers_id) - set(destroyed.unassigned) - removed_customers)

            if not available_customers:
                break  # 如果没有可移除的顾客，提前终止

            # 选择时间窗最接近的顾客点
            nearest_time_customer = min(available_customers, key=lambda customer:
            abs(self.problem_instance.vertices[last_selected_customer].call_time -
                self.problem_instance.vertices[customer].call_time))

            # 移除该顾客
            route = destroyed.find_route(nearest_time_customer)
            route.remove(nearest_time_customer)
            destroyed.unassigned.append(nearest_time_customer)
            removed_customers.add(nearest_time_customer)

            # 更新上一个选择的顾客点
            last_selected_customer = nearest_time_customer
        # print(destroyed.unassigned)
        return self.remove_empty_routes(destroyed)

    def zone_removal(self, state, rnd_state, zone_size=30):
        lower_degree_of_destruction = 0.2
        higher_degree_of_destruction = 0.4
        lower_num_to_destroy = int(len(self.customers_id) * lower_degree_of_destruction)
        higher_num_to_destroy = int(len(self.customers_id) * higher_degree_of_destruction)
        num_customers_to_remove = rnd_state.randint(lower_num_to_destroy,higher_num_to_destroy)

        destroyed = state.copy()

        for route in destroyed.state:
            self.remove_charging_station(route)  # 移除充电站，确保只处理顾客点

        removed_customers = set()
        total_customers_removed = 0

        while total_customers_removed < num_customers_to_remove:
            available_customers = list(set(self.customers_id) - set(destroyed.unassigned) - removed_customers)

            if not available_customers:
                break  # 如果没有可移除的顾客，提前终止

            # 随机选择区域的左下角
            min_x = min(self.problem_instance.vertices[customer].x for customer in available_customers)
            max_x = max(self.problem_instance.vertices[customer].x for customer in available_customers)
            min_y = min(self.problem_instance.vertices[customer].y for customer in available_customers)
            max_y = max(self.problem_instance.vertices[customer].y for customer in available_customers)

            x_lower_bound = rnd_state.uniform(min_x-15, max_x - zone_size+15)
            y_lower_bound = rnd_state.uniform(min_y-15, max_y - zone_size+15)

            x_upper_bound = x_lower_bound + zone_size
            y_upper_bound = y_lower_bound + zone_size
            # print(x_lower_bound,x_upper_bound, y_lower_bound,y_upper_bound)
            # for i in available_customers:
            # print(self.problem_instance.vertices[i],self.problem_instance.vertices[i].x, self.problem_instance.vertices[i].y)
            # 找到在该区域内的顾客
            customers_in_zone = [
                customer for customer in available_customers
                if x_lower_bound <= self.problem_instance.vertices[customer].x <= x_upper_bound
                   and y_lower_bound <= self.problem_instance.vertices[customer].y <= y_upper_bound
            ]

            # 随机移除这些顾客
            rnd_state.shuffle(customers_in_zone)
            customers_to_remove = customers_in_zone[
                                  :min(len(customers_in_zone), num_customers_to_remove - total_customers_removed)]

            for customer in customers_to_remove:
                route = destroyed.find_route(customer)
                route.remove(customer)
                destroyed.unassigned.append(customer)
                removed_customers.add(customer)

            total_customers_removed += len(customers_to_remove)

        return self.remove_empty_routes(destroyed)
        # if any(c in destroyed.unassigned for c in ['C1', 'C2', 'C6', 'C8', 'C9']):
        #     print(f"hallo11111111111",destroyed.unassigned)

        # print(destroyed.unassigned)


    def shortest_route_removal(self, state, rnd_state): #防止局部最优
        min_distance = float('inf')
        shortest_route = None

        destroyed = state.copy()
        for route in destroyed.state:
            route = self.remove_charging_station(route)
        for route in destroyed.state:
            dis = self.calculate_route_distance(route)
            if dis < min_distance:
                min_distance = dis
                shortest_route = route

        for customer in shortest_route[1: len(shortest_route)-1]:
            destroyed.unassigned.append(customer)
        destroyed.state.remove(shortest_route)
        # print(destroyed.unassigned)
        return self.remove_empty_routes(destroyed)

    def least_cus_route_removal(self, state, rnd_state): #减少AGV数量

        min_cus_num = float('inf')
        least_cus_route = None

        destroyed = state.copy()
        for route in destroyed.state:
            route = self.remove_charging_station(route)

        for route in destroyed.state:
            cus_num = len(route)
            if cus_num < min_cus_num:
                min_cus_num = cus_num
                least_cus_route = route

        for customer in least_cus_route[1: len(least_cus_route)-1]:
            destroyed.unassigned.append(customer)
        destroyed.state.remove(least_cus_route)
        return self.remove_empty_routes(destroyed)

    def random_route_removal(self, state, rnd_state):
        destroyed = state.copy()
        for route in destroyed.state:
            route = self.remove_charging_station(route)

        route_index = rnd.randint(0, len(destroyed.state))
        route_to_remove = destroyed.state[route_index]

        for customer in route_to_remove[1: len(route_to_remove)-1]:
            destroyed.unassigned.append(customer)
        destroyed.state.remove(route_to_remove)
        return self.remove_empty_routes(destroyed)

    def remove_empty_routes(self, state):
        """
        Remove empty routes after applying the destroy operator.
        """
        state.route = [route for route in state.state if len(route) != 2]
        return state

    def greedy_repair(self, state, rnd_state):
        """
        Inserts the unassigned customers in the best route. If there are no
        feasible insertions, then a new route is created.
        """
        rnd_state.shuffle(state.unassigned)

        while len(state.unassigned) != 0:
            customer = state.unassigned.pop()
            route, idx = self.best_insert(customer, state)

            if route is not None:
                route.insert(idx, customer)
            else:
                state.state.append(["D0", customer, "D0"])

        state = self.process_route(state)

        return state

    def best_insert(self, customer, state):
        """
        Finds the best feasible route and insertion idx for the customer.
        Return (None, None) if no feasible route insertions are found.
        """
        best_cost, best_route, best_idx = None, None, None

        for route in state.state:
            for idx in range(1, len(route)):
                temp_route = deepcopy(route)
                temp_route.insert(idx, customer)
                if self.is_nc_feasible(temp_route):
                    cost = self.insert_cost(customer, temp_route, idx)

                    if best_cost is None or cost < best_cost:
                        best_cost, best_route, best_idx = cost, route, idx

        return best_route, best_idx

    def cust_cost(self, customer, route, idx):
        pre = route[idx - 1]
        suc = route[idx + 1]
        pred = self.problem_instance.vertices[pre]
        succ = self.problem_instance.vertices[suc]
        cust = self.problem_instance.vertices[customer]
        # Increase in cost of adding customer, minus cost of removing old edge
        return pred.distance_to(cust) + cust.distance_to(succ)

    def insert_cost(self, customer, route, idx):
        """
        Computes the insertion cost for inserting customer in route at idx.
        """
        pre = "D0" if idx == 0 else route[idx - 1]
        suc = "D0" if idx == len(route) else route[idx]
        pred = self.problem_instance.vertices[pre]
        succ = self.problem_instance.vertices[suc]
        cust = self.problem_instance.vertices[customer]
        # Increase in cost of adding customer, minus cost of removing old edge
        return pred.distance_to(cust) + cust.distance_to(succ) - pred.distance_to(succ)

    def regret_2_insertion(self, state, rnd_state, regret_threshold=float('inf')):
        while len(state.unassigned) > 0:
            max_regret_value = -1
            best_customer = None
            best_route = None
            best_insert_position = None

            for customer in state.unassigned:
                best_cost, second_best_cost, best_position, best_route_candidate = self.find_best_two_insert_positions(
                    customer, state)

                if best_cost is not None and second_best_cost is not None:
                    regret_value = second_best_cost - best_cost
                else:
                    regret_value = float('inf')

                if regret_value > max_regret_value:
                    max_regret_value = regret_value
                    best_customer = customer
                    best_route = best_route_candidate
                    best_insert_position = best_position

                if max_regret_value >= regret_threshold:
                    break

            if best_customer is not None:
                if best_route is not None:
                    best_route.insert(best_insert_position, best_customer)
                else:
                    new_route = ["D0", best_customer, "D0"]
                    state.state.append(new_route)

                state.unassigned.remove(best_customer)
            else:
                # 如果没有可行的客户插入，则结束循环
                break

        state = self.process_route(state)

        return state

    def find_best_two_insert_positions(self, customer, state):
        insertion_costs = []

        for route in state.state:
            for i in range(1, len(route)):
                temp_route = route[:i] + [customer] + route[i:]
                if self.is_nc_feasible(temp_route):
                    cost = self.calculate_nc_route_cost(temp_route)
                    insertion_costs.append((cost, i, route))

        # 按成本排序
        insertion_costs.sort(key=lambda x: x[0])
        if len(insertion_costs) < 2:
            return None, None, None, None

        best_cost, best_position, best_route = insertion_costs[0]
        second_best_cost, _, _ = insertion_costs[1]

        return best_cost, second_best_cost, best_position, best_route


    def regret_3_insertion(self, state, rnd_state, regret_threshold=float('inf')):
        """
        Implements the Regret-3 Insertion heuristic.
        Inserts unassigned customers into the current solution based on the
        difference in insertion cost between the best, second-best, and third-best positions.
        Early stops if a sufficiently high regret value is found.
        """
        while len(state.unassigned) > 0:
            max_regret_value = -1
            best_customer = None
            best_route = None
            best_insert_position = None

            for customer in state.unassigned:
                # 找到三个最佳插入位置及其对应的成本
                best_cost, second_best_cost, third_best_cost, best_position, best_route_candidate = self.find_best_three_insert_positions(
                    customer, state)

                if best_cost is not None and second_best_cost is not None and third_best_cost is not None:
                    regret_value = third_best_cost - best_cost
                else:
                    regret_value = float('inf')

                if regret_value > max_regret_value:
                    max_regret_value = regret_value
                    best_customer = customer
                    best_route = best_route_candidate
                    best_insert_position = best_position

                # 提前终止条件
                if max_regret_value >= regret_threshold:
                    break

            if best_customer is not None:
                if best_route is not None:
                    best_route.insert(best_insert_position, best_customer)
                else:
                    new_route = ["D0", best_customer, "D0"]
                    state.state.append(new_route)

                state.unassigned.remove(best_customer)
            else:
                break

        state = self.process_route(state)

        return state

    def find_best_three_insert_positions(self, customer, state):
        """
        Finds the best, second-best, and third-best insertion positions for the customer.
        Returns the best cost, second-best cost, third-best cost, best insertion index, and the corresponding route.
        """
        best_cost = None
        second_best_cost = None
        third_best_cost = None
        best_position = None
        best_route = None

        for route in state.state:
            for i in range(1, len(route)):
                temp_route = route[:i] + [customer] + route[i:]
                if not self.is_nc_feasible(temp_route):  # Assumes a method to check feasibility
                    continue

                cost = self.calculate_nc_route_cost(temp_route)

                if best_cost is None or cost < best_cost:
                    third_best_cost = second_best_cost
                    second_best_cost = best_cost
                    best_cost = cost
                    best_position = i
                    best_route = route
                elif second_best_cost is None or cost < second_best_cost:
                    third_best_cost = second_best_cost
                    second_best_cost = cost
                elif third_best_cost is None or cost < third_best_cost:
                    third_best_cost = cost

        if best_cost is None:
            return None, None, None, None, None

        if second_best_cost is None:
            second_best_cost = float('inf')

        if third_best_cost is None:
            third_best_cost = float('inf')

        return best_cost, second_best_cost, third_best_cost, best_position, best_route

    def time_based_repair(self, state, rnd_state, max_attempts=5, early_stop_threshold=5):
        """
        Optimized Time-based Repair operator.
        Inserts unassigned customers into the current solution by considering
        the change in the finish time of the route as the insertion cost.
        Includes early stopping and limited search space.
        """
        while len(state.unassigned) > 0:
            best_customer = None
            best_route = None
            best_insert_position = None
            min_increase_in_finish_time = float('inf')

            # 尝试在有限的路径和插入位置中找到最佳插入点
            for customer in state.unassigned:
                for _ in range(max_attempts):  # 限制尝试次数
                    if len(state.state) > 0:
                        route_index = rnd_state.randint(0, len(state.state))  # 随机选择路径
                        route = state.state[route_index]

                        # 如果路径是空路径 (即 [D0, D0])，在起点后直接插入
                        if len(route) == 2:
                            insert_position = 1  # 插入在起点 D0 后
                        else:
                            insert_position = rnd_state.randint(1, len(route) - 1)  # 在路径中随机选择插入位置

                        temp_route = route[:insert_position] + [customer] + route[insert_position:]

                        if not self.is_nc_feasible(temp_route):
                            continue

                        finish_time_before = self.calculate_finish_time(route)
                        finish_time_after = self.calculate_finish_time(temp_route)
                        increase_in_finish_time = finish_time_after - finish_time_before

                        if increase_in_finish_time < min_increase_in_finish_time:
                            min_increase_in_finish_time = increase_in_finish_time
                            best_customer = customer
                            best_route = route
                            best_insert_position = insert_position

                            # 提前终止条件
                            if min_increase_in_finish_time <= early_stop_threshold:
                                break

            # 插入最佳找到的客户
            if best_customer is not None:
                best_route.insert(best_insert_position, best_customer)
                state.unassigned.remove(best_customer)
            else:
                # 如果没有找到合适的位置，则将剩余的客户插入新路径
                customer = state.unassigned.pop()
                new_route = ["D0", customer, "D0"]
                state.state.append(new_route)

        # 处理插入后的路径
        state = self.process_route(state)

        return state

    def calculate_finish_time(self, route):
        """
        Calculates the finish time of a route.
        The finish time is defined as the time when the last customer is served.
        """
        finish_time = 0
        current_time = 0
        for i in range(len(route) - 1):
            current_customer = self.problem_instance.vertices[route[i]]
            next_customer = self.problem_instance.vertices[route[i + 1]]
            travel_time = current_customer.distance_to(next_customer) / self.problem_instance.config.velocity
            current_time += travel_time + current_customer.service_time

        finish_time = current_time + self.problem_instance.vertices[route[-1]].service_time
        return finish_time

    def random_repair(self, state, rnd_state, max_attempts=10):
        """
        Simplified Random Repair operator.
        Randomly inserts unassigned customers into the current solution.
        If no feasible insertion exists after a few attempts, a new route is created.
        """
        while len(state.unassigned) > 0:
            customer = state.unassigned.pop()
            inserted = False

            # 尝试在现有路径中插入客户
            for _ in range(max_attempts):
                if not state.state:
                    break  # 如果没有现有路径，跳出循环

                # 随机选择一个路径索引
                route_index = rnd_state.randint(0, len(state.state))
                route = state.state[route_index]

                # 随机选择一个插入位置（从1到len(route)，包括在末尾插入）
                insert_position = rnd_state.randint(1, len(route))

                # 创建临时路径以检查可行性
                temp_route = route[:insert_position] + [customer] + route[insert_position:]

                if self.is_nc_feasible(temp_route):  # 检查可行性
                    state.state[route_index].insert(insert_position, customer)
                    inserted = True
                    break  # 成功插入后，退出尝试循环

            # 如果在所有尝试中都未能插入，则创建新路径
            if not inserted:
                new_route = ["D0", customer, "D0"]
                state.state.append(new_route)

        state = self.process_route(state)

        return state

    # ['D0', 'S2', 'C4', 'C9', 'C2', 'C7', 'C3', 'C5', 'C8', 'C10', 'D0']
    # ['D0', 'S2', 'C6', 'C1', 'D0']
    def process_route(self, state):
        # if state.state == [['D0', 'C10', 'C4', 'C1', 'D0'], ['D0',  'C9', 'C5', 'C2', 'C7', 'C8', 'C3', 'C6', 'D0']]:
        #     print("111111111111111111")
        # elif state.state == [['D0',  'C9', 'C5', 'C2', 'C7', 'C8', 'C3', 'C6', 'D0'], ['D0', 'C10', 'C4', 'C1', 'D0']]:
        #     print("222222222222222222")
        # state.state = [['D0', 'C5', 'S0', 'C4', 'C3', 'C8', 'D0'], ['D0', 'C2', 'C1', 'S2', 'C6', 'C7', 'C9', 'C10', 'D0']]
        # print(state.state)
        # for route in state.state:
        #     print(self.is_feasible(route))
        #     print(self.calculate_route_cost(route))

        # elif state.state == [['D0', 'C2', 'C3', 'C5', 'D0'], ['D0', 'C4', 'C6', 'C1', 'D0']]:
        #     print("find state2")

        # print("111111111111111111111111111")
        # 初始化未访问的客户列表


        unvisited_customers = []

        for idx, value in enumerate(state.state):
            self._set_vehicle_energy(idx)
            # 判断当前路径是否需要充电
            while self.need_charge(state.state[idx]):
                # print(state.state[idx])
                # print(self.problem_instance.config.now_energy)
                self._set_vehicle_energy(idx)

                k = self.make_route_feasible_and_best(value)
                # if k == ['D0','S2', 'C6', 'C1', 'D0']:
                #     print("real charge is right")
                if k is not None:
                    # 找到可行的插入点，更新路径
                    state.state[idx] = k
                    break  # 找到充电位置，跳出while循环，继续下一条路径
                else:
                    # 找不到插入点，移除最小到达时间的客户点
                    earliest_customer = self.find_earliest_customer(state.state[idx])
                    state.state[idx].remove(earliest_customer)    # 从当前路径中移除
                    unvisited_customers.append(self.problem_instance.vertices[earliest_customer])  # 将该客户点标记为未访问
                    # 路径调整后，继续判断是否需要充电
        # if state.state == [['D0', 'S0', 'C4', 'C6', 'C1', 'D0'], ['D0', 'S0', 'C2', 'C3', 'C5', 'D0']]:
            # now_energy = self.problem_instance.config.now_energy
            # tank_capacity = self.problem_instance.config.tank_capacity
            #
            # velocity = self.problem_instance.config.velocity
            # fuel_consumption_rate = self.problem_instance.config.fuel_consumption_rate
            # charging_rate = self.problem_instance.config.charging_rate
            #
            # last_position = self.problem_instance.depot
            # time = self.problem_instance.depot.call_time + self.problem_instance.depot.service_time
            # for route in state.state:
            #     arrival_times = []
            #
            #     for v in route[1:]:
            #         target = self.problem_instance.vertices[v]
            #         print(target)
            #         d = last_position.distance_to(target)
            #         print(d)
            #         time += d / velocity
            #         print(time)
            #         arrival_times.append(time)
            #
            #         now_energy -= d * fuel_consumption_rate
            #
            #         if type(target) is Customer:
            #             time += target.service_time
            #             print(time)
            #         elif type(target) is CharingStation:
            #             time += (tank_capacity - now_energy) * charging_rate
            #             print((tank_capacity - now_energy) * charging_rate)
            #         last_position = target
            #
            #     print(arrival_times)
            #     a = self.calculate_demand(route)
            #     print((a))
            #     print(sum(a))
            # print(state.state)
            # print(state.objective())
        # elif state.state == [['D0', 'S0', 'C2', 'C3', 'C5', 'D0'], ['D0', 'S0', 'C4', 'C6', 'C1', 'D0']]:
        #     print("find state22222222")
        #     print(state.state)
        #     print(state.objective())
        # 循环直到所有未访问客户都被处理
        while unvisited_customers:
            # print("Current unvisited_customers:", unvisited_customers)

            # 使用k-最近邻域启发式为未访问的客户生成新的路径
            depot = self.problem_instance.depot
            new_routes = self.innh(depot, unvisited_customers, self.problem_instance)
            # print(new_routes)
            # 对新的路径进行充电站插入操作
            start_index = len(state.state)
            for offset, value in enumerate(new_routes):
                vehicle_index = start_index + offset
                self._set_vehicle_energy(vehicle_index)
                # print(value)
                # print(self.need_charge(value))
                while self.need_charge(value):
                    self._set_vehicle_energy(vehicle_index)
                    k = self.make_route_feasible_and_best(value)
                    if k is not None:
                        new_routes[offset] = k
                        break
                    else:
                        earliest_customer = self.find_earliest_customer(new_routes[offset])
                        new_routes[offset].remove(earliest_customer)
                        unvisited_customers.append(self.problem_instance.vertices[earliest_customer])

            # 展开 new_routes，将其转化为单个客户的列表
            flattened_new_routes = [customer for route in new_routes for customer in route]
            # print(flattened_new_routes)
            # 更新 unvisited_customers 列表，确保未服务的客户仍然保留

            unvisited_customers = [customer for customer in unvisited_customers if customer.id not in flattened_new_routes]

            # print("11111111Current unvisited_customers:", unvisited_customers)
            state.state.extend(new_routes)
        # print("22222222222222")
        return state

    def innh(self, depot, customers, problem_instance, k=3):
        giant_route = []
        serviced_customers = set()

        while customers:
            route = ["D0"]
            last_position = depot

            while customers:
                possible_successors = [customer for customer in customers if customer not in serviced_customers]
                possible_successors.sort(key=lambda n: n.distance_to(last_position))
                possible_successors = possible_successors[:k]
                if not possible_successors:
                    break
                successor = min(possible_successors, key=lambda n: n.due_date)
                route.append(successor.id)
                # demand = route.calculate_demand()
                if not self.is_nc_feasible(route):
                    route.remove(successor.id)
                    break
                serviced_customers.add(successor)
                last_position = successor
            route.append("D0")
            if len(route) > 1:
                giant_route.append(route)
                customers = [customer for customer in customers if customer not in serviced_customers]

        return giant_route

    def find_earliest_customer(self, route):
        # 假设每个客户点包含属性 'due_date'
        # 返回路径中 due_date 最小的客户点
        earliest_customer = None
        min_due_date = float('inf')

        for v in route:
            customer = self.problem_instance.vertices[v]
            if isinstance(customer, Customer):
                if customer.due_date < min_due_date:
                    min_due_date = customer.due_date
                    earliest_customer = v

        return earliest_customer

    def is_feasible(self, route):
        """
        slow implementation for checking the feasibility of a route
        :param route: route to check
        :return: returns True if the route is feasible
        """
        self.calculate_demand(route)

        tank_capacity = self.problem_instance.config.tank_capacity
        now_energy = self.problem_instance.config.now_energy
        load_capacity = self.problem_instance.config.payload_capacity
        velocity = self.problem_instance.config.velocity
        fuel_consumption_rate = self.problem_instance.config.fuel_consumption_rate
        charging_rate = self.problem_instance.config.charging_rate

        last_position = self.problem_instance.depot
        time = self.problem_instance.depot.call_time + self.problem_instance.depot.service_time

        # if route == ['D0', 'S0', 'C9', 'C5', 'C2', 'C7', 'C8', 'C3', 'C6', 'D0']:
        #     for v in route[1:]:
        #         target = self.problem_instance.vertices[v]
        #         d = last_position.distance_to(target)
        #         time += d / velocity
        #         print(target, f"的到达时间是", time)
        #         print(v, f"的容量为", load_capacity)
        #         now_energy -= d * fuel_consumption_rate
        #
        #         if now_energy < 0:
        #             print("energy_violated")
        #             return False
        #
        #         if time > target.due_date:
        #             print("time_violated")
        #             return False
        #
        #         if type(target) is Customer:
        #             load_capacity -= target.demand
        #             time += target.service_time
        #         elif type(target) is CharingStation:
        #             time += (tank_capacity - now_energy) * charging_rate
        #             now_energy = tank_capacity
        #         print(v, f"过后的剩余容量为", load_capacity)
        #         if load_capacity < 0:
        #             print("load_violated")
        #             return False
        #
        #         last_position = target
        # else:
        for v in route[1:]:
            target = self.problem_instance.vertices[v]
            d = last_position.distance_to(target)
            time += d / velocity

            now_energy -= d * fuel_consumption_rate

            if now_energy < 0:

                return False

            if time > target.due_date:

                return False

            if type(target) is Customer:
                load_capacity -= target.demand
                time += target.service_time
            elif type(target) is CharingStation:
                time += (tank_capacity - now_energy) * charging_rate
                now_energy = tank_capacity

            if load_capacity < 0:

                return False

            last_position = target

        return True

    def remove_charging_station(self, route):
        route[:] = [i for i in route if type(self.problem_instance.vertices[i]) is not CharingStation]
        return route

    def need_charge(self, route):
        return (self.problem_instance.config.now_energy - self.calculate_route_distance(route) *
                self.problem_instance.config.fuel_consumption_rate < 0.2 * self.problem_instance.config.tank_capacity)

    def nnh_need_charge(self, route):
        return (self.problem_instance.config.now_energy - self.calculate_total_distance(route) *
                self.problem_instance.config.fuel_consumption_rate < 0.2 * self.problem_instance.config.tank_capacity)

    def calculate_remaining_tank_capacity(self, route):
        last_position = self.problem_instance.depot
        now_energy = self.problem_instance.config.now_energy

        for t in route[1:]:
            target = self.problem_instance.vertices[t]
            distance = last_position.distance_to(target)
            consumption = distance * self.problem_instance.config.fuel_consumption_rate
            now_energy -= consumption

            if type(t) is CharingStation:
                now_energy = self.problem_instance.config.tank_capacity

            last_position = target
        return now_energy

    def get_reachable_charging_stations(self, cust: Customer, capacity: float) -> list:
        max_dist = capacity / self.problem_instance.config.fuel_consumption_rate
        reachable_stations = []

        for cs in self.problem_instance.charging_stations:
            if cs.distance_to(cust) <= max_dist and cust.id != cs.id:
                reachable_stations.append(cs)

        return reachable_stations

    def calculate_demand(self, route):
        arrival_times = self.calculate_arrival_times(route)
        demand = []

        i = 0
        for v in route[1:]:
            r = self.problem_instance.vertices[v]
            if type(r) is CharingStation:
                r.demand = 0
            if type(r) is Customer:
                r.demand = (48 - r.stock_at_call_time + (arrival_times[i] - r.call_time) / 30)*0.75
            if r == self.problem_instance.depot:
                r.demand = 0
            i += 1

            demand.append(r.demand)
        return demand

    def calculate_arrival_times(self, route):
        now_energy = self.problem_instance.config.now_energy
        tank_capacity = self.problem_instance.config.tank_capacity

        velocity = self.problem_instance.config.velocity
        fuel_consumption_rate = self.problem_instance.config.fuel_consumption_rate
        charging_rate = self.problem_instance.config.charging_rate

        last_position = self.problem_instance.depot
        time = self.problem_instance.depot.call_time + self.problem_instance.depot.service_time

        arrival_times = []

        for v in route[1:]:
            target = self.problem_instance.vertices[v]
            d = last_position.distance_to(target)
            time += d / velocity

            arrival_times.append(time)

            now_energy -= d * fuel_consumption_rate

            if type(target) is Customer:
                time += target.service_time
            elif type(target) is CharingStation:
                time += (tank_capacity - now_energy) * charging_rate

            last_position = target

        return arrival_times

    def is_nc_feasible(self, route):
        """
        slow implementation for checking the feasibility of a route
        :param route: route to check
        :return: returns True if the route is feasible
        """
        self.calculate_demand(route)
        load_capacity = self.problem_instance.config.payload_capacity
        velocity = self.problem_instance.config.velocity

        last_position = self.problem_instance.depot
        time = self.problem_instance.depot.call_time + self.problem_instance.depot.service_time

        for v in route[1:]:
            target = self.problem_instance.vertices[v]
            d = last_position.distance_to(target)
            time += d / velocity

            if time > target.due_date:

                return False

            if type(target) is Customer:
                load_capacity -= target.demand
                time += target.service_time

            if load_capacity < 0:

                return False

            last_position = target

        return True

    def calculate_route_distance(self, route):

        last_pos = self.problem_instance.depot
        dist = 0

        for r in route:
            v = self.problem_instance.vertices[r]
            dist += v.distance_to(last_pos)
            last_pos = v

        return dist

    def calculate_total_distance(self, route):
        last = None
        dist = 0

        for t in route:
            if last is not None:
                dist += last.distance_to(t)
            last = t

        return dist

    def calculate_total_cost(self, state):
        total_cost = 0
        for idx, r in enumerate(state):
            self._set_vehicle_energy(idx)
            total_cost += self.calculate_route_cost(r)
        return total_cost

    def calculate_nc_route_cost(self, route):
        dist_cost = self.calculate_route_distance(route)
        # print(f"dist{dist_cost}")
        arrival_times = self.calculate_arrival_times(route)
        time_cost = 0
        for i in range(1, len(route)):
            # print(route[i])
            target = self.problem_instance.vertices[route[i]]
            if type(target) is CharingStation:
                time_cost += 0
            if type(target) is Customer:
                time_cost += target.due_date-arrival_times[i-1]
            # print(f"time{time_cost}")
        route_cost = dist_cost + 0.5 * time_cost
        # print(route_cost)
        return route_cost

    def calculate_route_cost(self, route):
        route_cost = 0
        if not self.is_feasible(route):
            route_cost = float('inf')
        else:
            dist_cost = self.calculate_route_distance(route)
            # print(dist_cost)
            arrival_times = self.calculate_arrival_times(route)
            time_cost = 0
            for i in range(1, len(route)):
                # print(route[i])
                target = self.problem_instance.vertices[route[i]]
                if type(target) is CharingStation:
                    time_cost += 0
                if type(target) is Customer:
                    time_cost += target.due_date-arrival_times[i-1]
                # print(time_cost)
            route_cost = dist_cost + 0.5 * time_cost + 800

        return route_cost

    def energy_violated(self,route):
        tank_capacity = self.problem_instance.config.tank_capacity
        now_energy = self.problem_instance.config.now_energy
        fuel_consumption_rate = self.problem_instance.config.fuel_consumption_rate
        last_position = self.problem_instance.depot

        for v in route[1:]:
            target = self.problem_instance.vertices[v]
            d = last_position.distance_to(target)
            now_energy -= d * fuel_consumption_rate

            if now_energy < 0:
                print("energy_violated")
                return True

            if type(target) is CharingStation:
                now_energy = tank_capacity

            last_position = target

        return False

    def find_operate(self, route):
        while self.energy_violated(route):
            break
        return None

    def find_optimal_charging_station_insertion(self, route):
        # print(route)
        best_insertion_point = None
        best_station = None
        min_route_cost = float('inf')
        # if route == ['D0', 'C4', 'C6', 'C1', 'D0']:
        #     print("real route is found")
        #     for i in range(1, len(route)):
        #         v = self.problem_instance.vertices[route[i - 1]]
        #         if not isinstance(v, CharingStation):
        #
        #             reachable_stations = self.get_reachable_charging_stations(
        #                 self.problem_instance.vertices[route[i - 1]],
        #                 self.calculate_remaining_tank_capacity(route[:i]))
        #
        #             if reachable_stations is None:
        #                 continue
        #
        #             else:
        #                 for j in reachable_stations:
        #                     temp_route = route[:i] + [j.id] + route[i:]
        #                     # print(temp_route, f"可行", self.is_feasible(temp_route))
        #                     # 判断插入后路径是否可行
        #                     if self.is_feasible(temp_route):
        #
        #                         # 计算插入后的总成本
        #                         route_cost = self.calculate_route_cost(temp_route)
        #                         # Tr = self.calculate_arrival_times(temp_route)
        #                         # print(Tr)
        #                         # print(temp_route)
        #                         # print(route_cost)
        #                         # 更新最优插入点和总成本
        #                         if route_cost < min_route_cost:
        #                             min_route_cost = route_cost
        #                             best_insertion_point = i
        #                             best_station = j
        # # 遍历路径中的每个顾客之间的位置作为插入点
        # else:
        for i in range(1, len(route)):
            v = self.problem_instance.vertices[route[i-1]]
            if not isinstance(v, CharingStation):

                reachable_stations = self.get_reachable_charging_stations(self.problem_instance.vertices[route[i - 1]],
                                                                          self.calculate_remaining_tank_capacity(route[:i]))
                if reachable_stations is None:
                    continue

                else:
                    for j in reachable_stations:
                        temp_route = route[:i] + [j.id] + route[i:]

                        # 判断插入后路径是否可行
                        if self.is_feasible(temp_route):

                            # 计算插入后的总成本
                            route_cost = self.calculate_route_cost(temp_route)
                            # Tr = self.calculate_arrival_times(temp_route)
                            # print(Tr)
                            # print(temp_route)
                            # print(route_cost)
                            # 更新最优插入点和总成本
                            if route_cost < min_route_cost:
                                min_route_cost = route_cost
                                best_insertion_point = i
                                best_station = j
        # print(best_insertion_point, best_station, min_route_cost)
        return best_insertion_point, best_station, min_route_cost

    def make_route_feasible_and_best(self, route):
        best_insertion_point, best_station, min_route_cost = self.find_optimal_charging_station_insertion(route)
        if best_station == None:
            return None
        best_feasible_route = route[:best_insertion_point]+[best_station.id]+route[best_insertion_point:]

        # print(best_feasible_route)
        if not self.is_feasible(best_feasible_route):
            # print("NO")
            return None
        else:
            # print("YES")
            # print(best_feasible_route)
            return best_feasible_route

    def nc_feasible(self, state):
        for route in state:
            if self.is_nc_feasible(route):
                continue
            else:
                return False
        return True

    def state_feasible(self, state):
        for idx, route in enumerate(state):
            self._set_vehicle_energy(idx)
            if not self.is_feasible(route):
                return False
        return True


class FCFS:
    def __init__(self,problem_instance: RoutingProblemInstance):
        self.problem_instance = problem_instance

    def _set_vehicle_energy(self, vehicle_index: int) -> None:
        self.problem_instance.config.now_energy = self.problem_instance.config.get_initial_energy(vehicle_index)

    def improve_solution(self):
        cost = 0
        temp_solution = self.fcfs()
        solution = self.process_route(temp_solution)
        for idx, route in enumerate(solution):
            self._set_vehicle_energy(idx)
            cost += self.calculate_route_cost(route)

        # remaining_energies = [
        #     calculate_route_remaining_energy(self.problem_instance, route, idx)
        #     for idx, route in enumerate(solution)
        # ]
        # for idx, energy in enumerate(remaining_energies):
        #     print(f"FCFS best solution vehicle {idx} remaining energy: {energy:.3f}")

        costs = cost
        times = cost

        return cost, solution, costs, times

    def fcfs(self):

        solution = []
        customers = deepcopy(self.problem_instance.customers)
        customers.sort(key=lambda c: c.call_time, reverse=True)
        while customers:
            route = ["D0"]
            while customers:
                customer = customers.pop()
                route.append(customer.id)
                if not self.is_nc_feasible(route):
                    route.remove(customer.id)
                    customers.append(customer)
                    break
            route.append("D0")
            if len(route) > 1:
                solution.append(route)

        return solution

    def process_route(self, state):
        # if state.state == [['D0', 'C10', 'C4', 'C1', 'D0'], ['D0',  'C9', 'C5', 'C2', 'C7', 'C8', 'C3', 'C6', 'D0']]:
        #     print("111111111111111111")
        # elif state.state == [['D0',  'C9', 'C5', 'C2', 'C7', 'C8', 'C3', 'C6', 'D0'], ['D0', 'C10', 'C4', 'C1', 'D0']]:
        #     print("222222222222222222")
        # state.state = [['D0', 'C5', 'S0', 'C4', 'C3', 'C8', 'D0'], ['D0', 'C2', 'C1', 'S2', 'C6', 'C7', 'C9', 'C10', 'D0']]
        # print(state.state)
        # for route in state.state:
        #     print(self.is_feasible(route))
        #     print(self.calculate_route_cost(route))

        # elif state.state == [['D0', 'C2', 'C3', 'C5', 'D0'], ['D0', 'C4', 'C6', 'C1', 'D0']]:
        #     print("find state2")

        # print("111111111111111111111111111")
        # 初始化未访问的客户列表
        unvisited_customers = []

        for idx, value in enumerate(state):

            self._set_vehicle_energy(idx)

            # 判断当前路径是否需要充电
            while self.need_charge(state[idx]):
                self._set_vehicle_energy(idx)

                k = self.make_route_feasible_and_best(value)
                # if k == ['D0','S2', 'C6', 'C1', 'D0']:
                #     print("real charge is right")
                if k is not None:
                    # 找到可行的插入点，更新路径
                    state[idx] = k
                    break  # 找到充电位置，跳出while循环，继续下一条路径
                else:
                    # 找不到插入点，移除最小到达时间的客户点
                    earliest_customer = self.find_earliest_customer(state[idx])
                    state[idx].remove(earliest_customer)    # 从当前路径中移除
                    unvisited_customers.append(self.problem_instance.vertices[earliest_customer])  # 将该客户点标记为未访问
                    # 路径调整后，继续判断是否需要充电
        # if state.state == [['D0', 'S0', 'C4', 'C6', 'C1', 'D0'], ['D0', 'S0', 'C2', 'C3', 'C5', 'D0']]:
            # now_energy = self.problem_instance.config.now_energy
            # tank_capacity = self.problem_instance.config.tank_capacity
            #
            # velocity = self.problem_instance.config.velocity
            # fuel_consumption_rate = self.problem_instance.config.fuel_consumption_rate
            # charging_rate = self.problem_instance.config.charging_rate
            #
            # last_position = self.problem_instance.depot
            # time = self.problem_instance.depot.call_time + self.problem_instance.depot.service_time
            # for route in state.state:
            #     arrival_times = []
            #
            #     for v in route[1:]:
            #         target = self.problem_instance.vertices[v]
            #         print(target)
            #         d = last_position.distance_to(target)
            #         print(d)
            #         time += d / velocity
            #         print(time)
            #         arrival_times.append(time)
            #
            #         now_energy -= d * fuel_consumption_rate
            #
            #         if type(target) is Customer:
            #             time += target.service_time
            #             print(time)
            #         elif type(target) is CharingStation:
            #             time += (tank_capacity - now_energy) * charging_rate
            #             print((tank_capacity - now_energy) * charging_rate)
            #         last_position = target
            #
            #     print(arrival_times)
            #     a = self.calculate_demand(route)
            #     print((a))
            #     print(sum(a))
            # print(state.state)
            # print(state.objective())
        # elif state.state == [['D0', 'S0', 'C2', 'C3', 'C5', 'D0'], ['D0', 'S0', 'C4', 'C6', 'C1', 'D0']]:
        #     print("find state22222222")
        #     print(state.state)
        #     print(state.objective())
        # 循环直到所有未访问客户都被处理
        while unvisited_customers:
            # print("Current unvisited_customers:", unvisited_customers)

            # 使用k-最近邻域启发式为未访问的客户生成新的路径
            depot = self.problem_instance.depot
            new_routes = self.innh(depot, unvisited_customers, self.problem_instance)
            # print(new_routes)
            # 对新的路径进行充电站插入操作
            start_index = len(state)
            for offset, value in enumerate(new_routes):
                vehicle_index = start_index + offset
                self._set_vehicle_energy(vehicle_index)
                # print(value)
                # print(self.need_charge(value))
                while self.need_charge(value):
                    self._set_vehicle_energy(vehicle_index)
                    k = self.make_route_feasible_and_best(value)
                    if k is not None:
                        new_routes[offset] = k
                        break
                    else:
                        earliest_customer = self.find_earliest_customer(new_routes[offset])
                        new_routes[offset].remove(earliest_customer)
                        unvisited_customers.append(self.problem_instance.vertices[earliest_customer])

            # 展开 new_routes，将其转化为单个客户的列表
            flattened_new_routes = [customer for route in new_routes for customer in route]
            # print(flattened_new_routes)
            # 更新 unvisited_customers 列表，确保未服务的客户仍然保留

            unvisited_customers = [customer for customer in unvisited_customers if customer.id not in flattened_new_routes]

            # print("11111111Current unvisited_customers:", unvisited_customers)
            state.extend(new_routes)
        # print("22222222222222")
        return state

    def innh(self, depot, customers, problem_instance, k=3):
        giant_route = []
        serviced_customers = set()

        while customers:
            route = ["D0"]
            last_position = depot

            while customers:
                possible_successors = [customer for customer in customers if customer not in serviced_customers]
                possible_successors.sort(key=lambda n: n.distance_to(last_position))
                possible_successors = possible_successors[:k]
                if not possible_successors:
                    break
                successor = min(possible_successors, key=lambda n: n.due_date)
                route.append(successor.id)
                # demand = route.calculate_demand()
                if not self.is_nc_feasible(route):
                    route.remove_target(successor)
                    break
                serviced_customers.add(successor)
                last_position = successor
            route.append("D0")
            if len(route) > 1:
                giant_route.append(route)
                customers = [customer for customer in customers if customer not in serviced_customers]

        return giant_route

    def find_earliest_customer(self, route):
        # 假设每个客户点包含属性 'due_date'
        # 返回路径中 due_date 最小的客户点
        earliest_customer = None
        min_due_date = float('inf')

        for v in route:
            customer = self.problem_instance.vertices[v]
            if isinstance(customer, Customer):
                if customer.due_date < min_due_date:
                    min_due_date = customer.due_date
                    earliest_customer = v

        return earliest_customer

    def is_feasible(self, route):
        """
        slow implementation for checking the feasibility of a route
        :param route: route to check
        :return: returns True if the route is feasible
        """
        self.calculate_demand(route)

        tank_capacity = self.problem_instance.config.tank_capacity
        now_energy = self.problem_instance.config.now_energy
        load_capacity = self.problem_instance.config.payload_capacity
        velocity = self.problem_instance.config.velocity
        fuel_consumption_rate = self.problem_instance.config.fuel_consumption_rate
        charging_rate = self.problem_instance.config.charging_rate

        last_position = self.problem_instance.depot
        time = self.problem_instance.depot.call_time + self.problem_instance.depot.service_time

        # if route == ['D0', 'S0', 'C9', 'C5', 'C2', 'C7', 'C8', 'C3', 'C6', 'D0']:
        #     for v in route[1:]:
        #         target = self.problem_instance.vertices[v]
        #         d = last_position.distance_to(target)
        #         time += d / velocity
        #         print(target, f"的到达时间是", time)
        #         print(v, f"的容量为", load_capacity)
        #         now_energy -= d * fuel_consumption_rate
        #
        #         if now_energy < 0:
        #             print("energy_violated")
        #             return False
        #
        #         if time > target.due_date:
        #             print("time_violated")
        #             return False
        #
        #         if type(target) is Customer:
        #             load_capacity -= target.demand
        #             time += target.service_time
        #         elif type(target) is CharingStation:
        #             time += (tank_capacity - now_energy) * charging_rate
        #             now_energy = tank_capacity
        #         print(v, f"过后的剩余容量为", load_capacity)
        #         if load_capacity < 0:
        #             print("load_violated")
        #             return False
        #
        #         last_position = target
        # else:
        for v in route[1:]:
            target = self.problem_instance.vertices[v]
            d = last_position.distance_to(target)
            time += d / velocity

            now_energy -= d * fuel_consumption_rate

            if now_energy < 0:

                return False

            if time > target.due_date:

                return False

            if type(target) is Customer:
                load_capacity -= target.demand
                time += target.service_time
            elif type(target) is CharingStation:
                time += (tank_capacity - now_energy) * charging_rate
                now_energy = tank_capacity

            if load_capacity < 0:

                return False

            last_position = target

        return True

    def remove_charging_station(self, route):
        route[:] = [i for i in route if type(self.problem_instance.vertices[i]) is not CharingStation]
        return route

    def need_charge(self, route):
        return (self.problem_instance.config.now_energy - self.calculate_route_distance(route) *
                self.problem_instance.config.fuel_consumption_rate < 0.2 * self.problem_instance.config.tank_capacity)

    def nnh_need_charge(self, route):
        return (self.problem_instance.config.now_energy - self.calculate_total_distance(route) *
                self.problem_instance.config.fuel_consumption_rate < 0.2 * self.problem_instance.config.tank_capacity)

    def calculate_remaining_tank_capacity(self, route):
        last_position = self.problem_instance.depot
        now_energy = self.problem_instance.config.now_energy

        for t in route[1:]:
            target = self.problem_instance.vertices[t]
            distance = last_position.distance_to(target)
            consumption = distance * self.problem_instance.config.fuel_consumption_rate
            now_energy -= consumption

            if type(t) is CharingStation:
                now_energy = self.problem_instance.config.tank_capacity

            last_position = target
        return now_energy

    def get_reachable_charging_stations(self, cust: Customer, capacity: float) -> list:
        max_dist = capacity / self.problem_instance.config.fuel_consumption_rate
        reachable_stations = []

        for cs in self.problem_instance.charging_stations:
            if cs.distance_to(cust) <= max_dist and cust.id != cs.id:
                reachable_stations.append(cs)

        return reachable_stations

    def calculate_demand(self, route):
        arrival_times = self.calculate_arrival_times(route)
        demand = []

        i = 0
        for v in route[1:]:
            r = self.problem_instance.vertices[v]
            if type(r) is CharingStation:
                r.demand = 0
            if type(r) is Customer:
                r.demand = (48 - r.stock_at_call_time + (arrival_times[i] - r.call_time) / 30)*0.75
            if r == self.problem_instance.depot:
                r.demand = 0
            i += 1

            demand.append(r.demand)
        return demand

    def calculate_arrival_times(self, route):
        now_energy = self.problem_instance.config.now_energy
        tank_capacity = self.problem_instance.config.tank_capacity

        velocity = self.problem_instance.config.velocity
        fuel_consumption_rate = self.problem_instance.config.fuel_consumption_rate
        charging_rate = self.problem_instance.config.charging_rate

        last_position = self.problem_instance.depot
        time = self.problem_instance.depot.call_time + self.problem_instance.depot.service_time

        arrival_times = []

        for v in route[1:]:
            target = self.problem_instance.vertices[v]
            d = last_position.distance_to(target)
            time += d / velocity

            arrival_times.append(time)

            now_energy -= d * fuel_consumption_rate

            if type(target) is Customer:
                time += target.service_time
            elif type(target) is CharingStation:
                time += (tank_capacity - now_energy) * charging_rate

            last_position = target

        return arrival_times

    def is_nc_feasible(self, route):
        """
        slow implementation for checking the feasibility of a route
        :param route: route to check
        :return: returns True if the route is feasible
        """
        self.calculate_demand(route)
        load_capacity = self.problem_instance.config.payload_capacity
        velocity = self.problem_instance.config.velocity

        last_position = self.problem_instance.depot
        time = self.problem_instance.depot.call_time + self.problem_instance.depot.service_time

        for v in route[1:]:
            target = self.problem_instance.vertices[v]
            d = last_position.distance_to(target)
            time += d / velocity

            if time > target.due_date:

                return False

            if type(target) is Customer:
                load_capacity -= target.demand
                time += target.service_time

            if load_capacity < 0:

                return False

            last_position = target

        return True

    def calculate_route_distance(self, route):

        last_pos = self.problem_instance.depot
        dist = 0

        for r in route:
            v = self.problem_instance.vertices[r]
            dist += v.distance_to(last_pos)
            last_pos = v

        return dist

    def calculate_total_distance(self, route):
        last = None
        dist = 0

        for t in route:
            if last is not None:
                dist += last.distance_to(t)
            last = t

        return dist

    def calculate_total_cost(self, state):
        total_cost = 0
        for idx, r in enumerate(state):
            self._set_vehicle_energy(idx)
            total_cost += self.calculate_route_cost(r)
        return total_cost

    def calculate_nc_route_cost(self, route):
        dist_cost = self.calculate_route_distance(route)
        # print(f"dist{dist_cost}")
        arrival_times = self.calculate_arrival_times(route)
        time_cost = 0
        for i in range(1, len(route)):
            # print(route[i])
            target = self.problem_instance.vertices[route[i]]
            if type(target) is CharingStation:
                time_cost += 0
            if type(target) is Customer:
                time_cost += target.due_date-arrival_times[i-1]
            # print(f"time{time_cost}")
        route_cost = dist_cost + 0.5 * time_cost
        # print(route_cost)
        return route_cost

    def calculate_route_cost(self, route):
        route_cost = 0
        if not self.is_feasible(route):
            route_cost = float('inf')
        else:
            dist_cost = self.calculate_route_distance(route)
            # print(dist_cost)
            arrival_times = self.calculate_arrival_times(route)
            time_cost = 0
            for i in range(1, len(route)):
                # print(route[i])
                target = self.problem_instance.vertices[route[i]]
                if type(target) is CharingStation:
                    time_cost += 0
                if type(target) is Customer:
                    time_cost += target.due_date-arrival_times[i-1]
                # print(time_cost)
            route_cost = dist_cost + 0.5 * time_cost + 800

        return route_cost

    def energy_violated(self,route):
        tank_capacity = self.problem_instance.config.tank_capacity
        now_energy = self.problem_instance.config.now_energy
        fuel_consumption_rate = self.problem_instance.config.fuel_consumption_rate
        last_position = self.problem_instance.depot

        for v in route[1:]:
            target = self.problem_instance.vertices[v]
            d = last_position.distance_to(target)
            now_energy -= d * fuel_consumption_rate

            if now_energy < 0:
                print("energy_violated")
                return True

            if type(target) is CharingStation:
                now_energy = tank_capacity

            last_position = target

        return False

    def find_operate(self, route):
        while self.energy_violated(route):
            break
        return None

    def find_optimal_charging_station_insertion(self, route):
        # print(route)
        best_insertion_point = None
        best_station = None
        min_route_cost = float('inf')
        # if route == ['D0', 'C4', 'C6', 'C1', 'D0']:
        #     print("real route is found")
        #     for i in range(1, len(route)):
        #         v = self.problem_instance.vertices[route[i - 1]]
        #         if not isinstance(v, CharingStation):
        #
        #             reachable_stations = self.get_reachable_charging_stations(
        #                 self.problem_instance.vertices[route[i - 1]],
        #                 self.calculate_remaining_tank_capacity(route[:i]))
        #
        #             if reachable_stations is None:
        #                 continue
        #
        #             else:
        #                 for j in reachable_stations:
        #                     temp_route = route[:i] + [j.id] + route[i:]
        #                     # print(temp_route, f"可行", self.is_feasible(temp_route))
        #                     # 判断插入后路径是否可行
        #                     if self.is_feasible(temp_route):
        #
        #                         # 计算插入后的总成本
        #                         route_cost = self.calculate_route_cost(temp_route)
        #                         # Tr = self.calculate_arrival_times(temp_route)
        #                         # print(Tr)
        #                         # print(temp_route)
        #                         # print(route_cost)
        #                         # 更新最优插入点和总成本
        #                         if route_cost < min_route_cost:
        #                             min_route_cost = route_cost
        #                             best_insertion_point = i
        #                             best_station = j
        # # 遍历路径中的每个顾客之间的位置作为插入点
        # else:
        for i in range(1, len(route)):
            v = self.problem_instance.vertices[route[i-1]]
            if not isinstance(v, CharingStation):

                reachable_stations = self.get_reachable_charging_stations(self.problem_instance.vertices[route[i - 1]],
                                                                          self.calculate_remaining_tank_capacity(route[:i]))
                if reachable_stations is None:
                    continue

                else:
                    for j in reachable_stations:
                        temp_route = route[:i] + [j.id] + route[i:]

                        # 判断插入后路径是否可行
                        if self.is_feasible(temp_route):

                            # 计算插入后的总成本
                            route_cost = self.calculate_route_cost(temp_route)
                            # Tr = self.calculate_arrival_times(temp_route)
                            # print(Tr)
                            # print(temp_route)
                            # print(route_cost)
                            # 更新最优插入点和总成本
                            if route_cost < min_route_cost:
                                min_route_cost = route_cost
                                best_insertion_point = i
                                best_station = j
        # print(best_insertion_point, best_station, min_route_cost)
        return best_insertion_point, best_station, min_route_cost

    def make_route_feasible_and_best(self, route):
        best_insertion_point, best_station, min_route_cost = self.find_optimal_charging_station_insertion(route)
        if best_station == None:
            return None
        best_feasible_route = route[:best_insertion_point]+[best_station.id]+route[best_insertion_point:]

        # print(best_feasible_route)
        if not self.is_feasible(best_feasible_route):
            # print("NO")
            return None
        else:
            # print("YES")
            # print(best_feasible_route)
            return best_feasible_route


class SimulatedAnnealing1:
    def __init__(self, problem_instance: RoutingProblemInstance, state, cost, t_0, cooling_factor, fig_name, rep=1):
        self.problem_instance = problem_instance
        self.state = state
        self.cost = 0
        for idx, r in enumerate(self.state):
            self.problem_instance.config.now_energy = self.problem_instance.config.get_initial_energy(idx)
            self.cost += self.calculate_route_cost(r)
        self.t_0 = t_0
        self.temp = t_0
        self.cooling_factor = cooling_factor
        self.rep = rep

        self.fig_name = fig_name
        self.neighbour_hoods = [merge_route, two_opt, two_opt_star, or_opt, cross_exchange]

    def _set_vehicle_energy(self, vehicle_index: int) -> None:
        self.problem_instance.config.now_energy = self.problem_instance.config.get_initial_energy(vehicle_index)

    def _sum_route_costs(self, routes):
        total = 0
        for idx, route in enumerate(routes):
            self._set_vehicle_energy(idx)
            total += self.calculate_route_cost(route)
        return total
    def calculate_example(self, state):
        if self.state_feasible(state):
            print("feasible")

    import time

    # # 终止条件包含时间和迭代次数
    # def improve_solution(self):
    #     state_approx = self.state
    #     cost_approx = self.cost
    #
    #     start_time = time.time()  # 记录开始时间
    #     iteration = 0  # 初始化迭代次数
    #
    #     for r in range(0, self.rep):
    #         self.temp = self.t_0
    #         statistic = []
    #         temps = []
    #         current_state = state_approx
    #         current_cost = cost_approx
    #
    #         # 使用运行时间和迭代次数的双重终止条件
    #         while time.time() - start_time < 10 and iteration < 3000:
    #             random_neighbour, random_cost = self.get_random_feasible_neighbour(current_state, current_cost,
    #                                                                                )
    #             delta = (random_cost - current_cost) / current_cost
    #
    #             if delta <= 0:
    #                 current_state = random_neighbour
    #                 current_cost = random_cost
    #
    #                 if cost_approx > current_cost:
    #                     state_approx = current_state
    #                     cost_approx = current_cost
    #             else:
    #                 random_number = random()
    #                 if random_number < exp(-1 * delta / self.temp):
    #                     current_state = random_neighbour
    #                     current_cost = random_cost
    #
    #             statistic.append(current_cost)
    #             temps.append(self.temp)
    #             self.temp *= self.cooling_factor
    #             iteration += 1
    #
    #     plt.clf()
    #     plt.title('Instance {0}'.format(self.fig_name))
    #     plt.xlabel('iterations')
    #     plt.ylabel('cost')
    #     plt.plot(statistic)
    #     plt.savefig('figures/{0}.png'.format(self.fig_name))
    #
    #     state_approx, cost_approx = self.local_search(state_approx, cost_approx)
    #
    #     return cost_approx, state_approx

    # 运行时间为终止条件
    def improve_solution(self):
        state_approx = self.state
        cost_approx = self.cost

        start_time = time.time()  # 记录开始时间

        statistic = []  # 用于记录每次迭代的成本
        iteration_times = []  # 用于记录每次迭代的累计时间

        for r in range(0, self.rep):
            self.temp = self.t_0
            temps = []
            current_state = state_approx
            current_cost = cost_approx

            while time.time() - start_time < 1:  # 运行时间小于10秒
                random_neighbour, random_cost = self.get_random_feasible_neighbour(current_state, current_cost,
                                                                                   )
                delta = (random_cost - current_cost) / current_cost

                if delta <= 0:
                    current_state = random_neighbour
                    current_cost = random_cost

                    if cost_approx > current_cost:
                        state_approx = current_state
                        cost_approx = current_cost
                else:
                    random_number = random()
                    if random_number < exp(-1 * delta / self.temp):
                        current_state = random_neighbour
                        current_cost = random_cost

                statistic.append(cost_approx)
                iteration_times.append(time.time() - start_time)
                temps.append(self.temp)
                self.temp *= self.cooling_factor

        plt.clf()
        plt.title('Instance {0}'.format(self.fig_name))
        plt.xlabel('running time')
        plt.ylabel('cost')
        plt.plot(statistic)
        plt.savefig('figures/{0}.png'.format(self.fig_name))

        state_approx, cost_approx = self.local_search(state_approx, cost_approx)

        # remaining_energies = [
        #     calculate_route_remaining_energy(self.problem_instance, route, idx)
        #     for idx, route in enumerate(state_approx)
        # ]
        # for idx, energy in enumerate(remaining_energies):
        #     print(f"SimulatedAnnealing1 best solution vehicle {idx} remaining energy: {energy:.3f}")

        return cost_approx, state_approx, statistic, iteration_times

    # 迭代次数为终止条件
    # def improve_solution(self):
    #     state_approx = self.state
    #     cost_approx = self.cost
    #     iteration = 0
    #
    #     for r in range(0, self.rep):
    #         self.temp = self.t_0
    #         statistic = []
    #         temps = []
    #         current_state = state_approx
    #         current_cost = cost_approx
    #         while iteration < 3000:  # 0.0000000000001:
    #             random_neighbour, random_cost = self.get_random_feasible_neighbour(current_state,
    #                                                                                current_cost,
    #                                                                                iteration) # 去掉iteration行不？？
    #             delta = (random_cost - current_cost) / current_cost
    #
    #             if delta <= 0:
    #                 current_state = random_neighbour
    #                 current_cost = random_cost
    #
    #                 if cost_approx > current_cost:
    #                     state_approx = current_state
    #                     cost_approx = current_cost
    #             else:
    #                 random_number = random()
    #                 # print(delta)
    #                 if random_number < exp(-1 * delta / self.temp):
    #                     current_state = random_neighbour
    #                     current_cost = random_cost
    #
    #             statistic.append(current_cost)
    #             temps.append(self.temp)
    #             self.temp *= self.cooling_factor
    #             iteration += 1
    #
    #     plt.clf()
    #     plt.title('Instance {0}'.format(self.fig_name))
    #     plt.xlabel('iterations')
    #     plt.ylabel('cost')
    #     plt.plot(statistic)
    #     plt.savefig('figures/{0}.png'.format(self.fig_name))
    #
    #     state_approx, cost_approx = self.local_search(state_approx, cost_approx)
    #
    #     return cost_approx, state_approx

    def nc_feasible(self, state):
        for route in state:
            if self.is_nc_feasible(route):
                continue
            else:
                return False
        return True

    def state_feasible(self, state):
        for idx, route in enumerate(state):
            self._set_vehicle_energy(idx)
            if not self.is_feasible(route):
                return False
        return True

    def get_random_feasible_neighbour(self, state, cost):
        next_state = state
        next_cost = cost

        for nh_op in self.neighbour_hoods:
            if choice([True]):
                next_state, next_cost = nh_op(next_state, next_cost, self.remove_charging_station, self.nc_feasible,
                                              self.state_feasible,
                                              self.calculate_route_cost,
                                              self.need_charge,
                                              self.make_route_feasible_and_best,
                                              self.process_route
                                              )

        return next_state, next_cost

    def local_search(self, state, cost):
        current_state = state
        current_cost = cost

        while True:
            best_neighbour, best_neighbour_cost = self.get_best_neighbour(current_state, current_cost)

            if best_neighbour_cost < current_cost:
                current_state = best_neighbour
                current_cost = best_neighbour_cost
                continue

            break

        return current_state, current_cost

    def get_best_neighbour(self, state, cost):
        best_state = state
        best_cost = cost

        for nh_op in self.neighbour_hoods:
            best_state, best_cost = nh_op(best_state, best_cost, self.remove_charging_station, self.nc_feasible,
                                          self.state_feasible,
                                          self.calculate_route_cost,
                                          self.need_charge,
                                          self.make_route_feasible_and_best,
                                          self.process_route
                                          )

        return best_state, best_cost

    def process_route(self, state):
        # if state.state == [['D0', 'C10', 'C4', 'C1', 'D0'], ['D0',  'C9', 'C5', 'C2', 'C7', 'C8', 'C3', 'C6', 'D0']]:
        #     print("111111111111111111")
        # elif state.state == [['D0',  'C9', 'C5', 'C2', 'C7', 'C8', 'C3', 'C6', 'D0'], ['D0', 'C10', 'C4', 'C1', 'D0']]:
        #     print("222222222222222222")
        # state.state = [['D0', 'C5', 'S0', 'C4', 'C3', 'C8', 'D0'], ['D0', 'C2', 'C1', 'S2', 'C6', 'C7', 'C9', 'C10', 'D0']]
        # print(state.state)
        # for route in state.state:
        #     print(self.is_feasible(route))
        #     print(self.calculate_route_cost(route))

        # elif state.state == [['D0', 'C2', 'C3', 'C5', 'D0'], ['D0', 'C4', 'C6', 'C1', 'D0']]:
        #     print("find state2")

        # print("111111111111111111111111111")
        # 初始化未访问的客户列表
        unvisited_customers = []

        for idx, value in enumerate(state):
            self._set_vehicle_energy(idx)

            # 判断当前路径是否需要充电
            while self.need_charge(state[idx]):
                self._set_vehicle_energy(idx)

                k = self.make_route_feasible_and_best(value)
                # if k == ['D0','S2', 'C6', 'C1', 'D0']:
                #     print("real charge is right")
                if k is not None:
                    # 找到可行的插入点，更新路径
                    state[idx] = k
                    break  # 找到充电位置，跳出while循环，继续下一条路径
                else:
                    # 找不到插入点，移除最小到达时间的客户点
                    earliest_customer = self.find_earliest_customer(state[idx])
                    print(earliest_customer)
                    print(state[idx])
                    state[idx].remove(earliest_customer)    # 从当前路径中移除
                    unvisited_customers.append(self.problem_instance.vertices[earliest_customer])  # 将该客户点标记为未访问
                    # 路径调整后，继续判断是否需要充电
        # if state.state == [['D0', 'S0', 'C4', 'C6', 'C1', 'D0'], ['D0', 'S0', 'C2', 'C3', 'C5', 'D0']]:
            # now_energy = self.problem_instance.config.now_energy
            # tank_capacity = self.problem_instance.config.tank_capacity
            #
            # velocity = self.problem_instance.config.velocity
            # fuel_consumption_rate = self.problem_instance.config.fuel_consumption_rate
            # charging_rate = self.problem_instance.config.charging_rate
            #
            # last_position = self.problem_instance.depot
            # time = self.problem_instance.depot.call_time + self.problem_instance.depot.service_time
            # for route in state.state:
            #     arrival_times = []
            #
            #     for v in route[1:]:
            #         target = self.problem_instance.vertices[v]
            #         print(target)
            #         d = last_position.distance_to(target)
            #         print(d)
            #         time += d / velocity
            #         print(time)
            #         arrival_times.append(time)
            #
            #         now_energy -= d * fuel_consumption_rate
            #
            #         if type(target) is Customer:
            #             time += target.service_time
            #             print(time)
            #         elif type(target) is CharingStation:
            #             time += (tank_capacity - now_energy) * charging_rate
            #             print((tank_capacity - now_energy) * charging_rate)
            #         last_position = target
            #
            #     print(arrival_times)
            #     a = self.calculate_demand(route)
            #     print((a))
            #     print(sum(a))
            # print(state.state)
            # print(state.objective())
        # elif state.state == [['D0', 'S0', 'C2', 'C3', 'C5', 'D0'], ['D0', 'S0', 'C4', 'C6', 'C1', 'D0']]:
        #     print("find state22222222")
        #     print(state.state)
        #     print(state.objective())
        # 循环直到所有未访问客户都被处理
        while unvisited_customers:
            # print("Current unvisited_customers:", unvisited_customers)

            # 使用k-最近邻域启发式为未访问的客户生成新的路径
            depot = self.problem_instance.depot
            new_routes = self.innh(depot, unvisited_customers, self.problem_instance)
            # print(new_routes)
            # 对新的路径进行充电站插入操作
            start_index = len(state)
            for offset, value in enumerate(new_routes):
                vehicle_index = start_index + offset
                self._set_vehicle_energy(vehicle_index)
                # print(value)
                # print(self.need_charge(value))
                while self.need_charge(value):
                    self._set_vehicle_energy(vehicle_index)

                    k = self.make_route_feasible_and_best(value)
                    if k is not None:
                        new_routes[offset] = k
                        break
                    else:
                        earliest_customer = self.find_earliest_customer(new_routes[offset])
                        new_routes[offset].remove(earliest_customer)
                        unvisited_customers.append(self.problem_instance.vertices[earliest_customer])

            # 展开 new_routes，将其转化为单个客户的列表
            flattened_new_routes = [customer for route in new_routes for customer in route]
            # print(flattened_new_routes)
            # 更新 unvisited_customers 列表，确保未服务的客户仍然保留

            unvisited_customers = [customer for customer in unvisited_customers if customer.id not in flattened_new_routes]

            # print("11111111Current unvisited_customers:", unvisited_customers)
            state.extend(new_routes)
        # print("22222222222222")
        return state

    def innh(self, depot, customers, problem_instance, k=3):
        giant_route = []
        serviced_customers = set()

        while customers:
            route = ["D0"]
            last_position = depot

            while customers:
                possible_successors = [customer for customer in customers if customer not in serviced_customers]
                possible_successors.sort(key=lambda n: n.distance_to(last_position))
                possible_successors = possible_successors[:k]
                if not possible_successors:
                    break
                successor = min(possible_successors, key=lambda n: n.due_date)
                route.append(successor.id)
                # demand = route.calculate_demand()
                if not self.is_nc_feasible(route):
                    route.remove_target(successor)
                    break
                serviced_customers.add(successor)
                last_position = successor
            route.append("D0")
            if len(route) > 1:
                giant_route.append(route)
                customers = [customer for customer in customers if customer not in serviced_customers]

        return giant_route

    def find_earliest_customer(self, route):
        # 假设每个客户点包含属性 'due_date'
        # 返回路径中 due_date 最小的客户点
        earliest_customer = None
        min_due_date = float('inf')

        for v in route:
            customer = self.problem_instance.vertices[v]
            if isinstance(customer, Customer):
                if customer.due_date < min_due_date:
                    min_due_date = customer.due_date
                    earliest_customer = v

        return earliest_customer

    def is_feasible(self, route):
        """
        slow implementation for checking the feasibility of a route
        :param route: route to check
        :return: returns True if the route is feasible
        """
        self.calculate_demand(route)

        tank_capacity = self.problem_instance.config.tank_capacity
        now_energy = self.problem_instance.config.now_energy
        load_capacity = self.problem_instance.config.payload_capacity
        velocity = self.problem_instance.config.velocity
        fuel_consumption_rate = self.problem_instance.config.fuel_consumption_rate
        charging_rate = self.problem_instance.config.charging_rate

        last_position = self.problem_instance.depot
        time = self.problem_instance.depot.call_time + self.problem_instance.depot.service_time

        # if route == ['D0', 'S0', 'C9', 'C5', 'C2', 'C7', 'C8', 'C3', 'C6', 'D0']:
        #     for v in route[1:]:
        #         target = self.problem_instance.vertices[v]
        #         d = last_position.distance_to(target)
        #         time += d / velocity
        #         print(target, f"的到达时间是", time)
        #         print(v, f"的容量为", load_capacity)
        #         now_energy -= d * fuel_consumption_rate
        #
        #         if now_energy < 0:
        #             print("energy_violated")
        #             return False
        #
        #         if time > target.due_date:
        #             print("time_violated")
        #             return False
        #
        #         if type(target) is Customer:
        #             load_capacity -= target.demand
        #             time += target.service_time
        #         elif type(target) is CharingStation:
        #             time += (tank_capacity - now_energy) * charging_rate
        #             now_energy = tank_capacity
        #         print(v, f"过后的剩余容量为", load_capacity)
        #         if load_capacity < 0:
        #             print("load_violated")
        #             return False
        #
        #         last_position = target
        # else:
        for v in route[1:]:
            target = self.problem_instance.vertices[v]
            d = last_position.distance_to(target)
            time += d / velocity

            now_energy -= d * fuel_consumption_rate

            if now_energy < 0:

                return False

            if time > target.due_date:

                return False

            if type(target) is Customer:
                load_capacity -= target.demand
                time += target.service_time
            elif type(target) is CharingStation:
                time += (tank_capacity - now_energy) * charging_rate
                now_energy = tank_capacity

            if load_capacity < 0:

                return False

            last_position = target

        return True

    def remove_charging_station(self, route):
        route[:] = [i for i in route if type(self.problem_instance.vertices[i]) is not CharingStation]
        return route

    def need_charge(self, route):
        return (self.problem_instance.config.now_energy - self.calculate_route_distance(route) *
                self.problem_instance.config.fuel_consumption_rate < 0.2 * self.problem_instance.config.tank_capacity)

    def nnh_need_charge(self, route):
        return (self.problem_instance.config.now_energy - self.calculate_total_distance(route) *
                self.problem_instance.config.fuel_consumption_rate < 0.2 * self.problem_instance.config.tank_capacity)

    def calculate_remaining_tank_capacity(self, route):
        last_position = self.problem_instance.depot
        now_energy = self.problem_instance.config.now_energy

        for t in route[1:]:
            target = self.problem_instance.vertices[t]
            distance = last_position.distance_to(target)
            consumption = distance * self.problem_instance.config.fuel_consumption_rate
            now_energy -= consumption

            if type(t) is CharingStation:
                now_energy = self.problem_instance.config.tank_capacity

            last_position = target
        return now_energy

    def get_reachable_charging_stations(self, cust: Customer, capacity: float) -> list:
        max_dist = capacity / self.problem_instance.config.fuel_consumption_rate
        reachable_stations = []

        for cs in self.problem_instance.charging_stations:
            if cs.distance_to(cust) <= max_dist and cust.id != cs.id:
                reachable_stations.append(cs)

        return reachable_stations

    def calculate_demand(self, route):
        arrival_times = self.calculate_arrival_times(route)
        demand = []

        i = 0
        for v in route[1:]:
            r = self.problem_instance.vertices[v]
            if type(r) is CharingStation:
                r.demand = 0
            if type(r) is Customer:
                r.demand = (48 - r.stock_at_call_time + (arrival_times[i] - r.call_time) / 30)*0.75
            if r == self.problem_instance.depot:
                r.demand = 0
            i += 1

            demand.append(r.demand)
        return demand

    def calculate_arrival_times(self, route):
        now_energy = self.problem_instance.config.now_energy
        tank_capacity = self.problem_instance.config.tank_capacity

        velocity = self.problem_instance.config.velocity
        fuel_consumption_rate = self.problem_instance.config.fuel_consumption_rate
        charging_rate = self.problem_instance.config.charging_rate

        last_position = self.problem_instance.depot
        time = self.problem_instance.depot.call_time + self.problem_instance.depot.service_time

        arrival_times = []

        for v in route[1:]:
            target = self.problem_instance.vertices[v]
            d = last_position.distance_to(target)
            time += d / velocity

            arrival_times.append(time)

            now_energy -= d * fuel_consumption_rate

            if type(target) is Customer:
                time += target.service_time
            elif type(target) is CharingStation:
                time += (tank_capacity - now_energy) * charging_rate

            last_position = target

        return arrival_times

    def is_nc_feasible(self, route):
        """
        slow implementation for checking the feasibility of a route
        :param route: route to check
        :return: returns True if the route is feasible
        """
        self.calculate_demand(route)
        load_capacity = self.problem_instance.config.payload_capacity
        velocity = self.problem_instance.config.velocity

        last_position = self.problem_instance.depot
        time = self.problem_instance.depot.call_time + self.problem_instance.depot.service_time

        for v in route[1:]:
            target = self.problem_instance.vertices[v]
            d = last_position.distance_to(target)
            time += d / velocity

            if time > target.due_date:

                return False

            if type(target) is Customer:
                load_capacity -= target.demand
                time += target.service_time

            if load_capacity < 0:

                return False

            last_position = target

        return True

    def calculate_route_distance(self, route):

        last_pos = self.problem_instance.depot
        dist = 0

        for r in route:
            v = self.problem_instance.vertices[r]
            dist += v.distance_to(last_pos)
            last_pos = v

        return dist

    def calculate_total_distance(self, route):
        last = None
        dist = 0

        for t in route:
            if last is not None:
                dist += last.distance_to(t)
            last = t

        return dist

    def calculate_total_cost(self, state):
        return self._sum_route_costs(state)

    def calculate_nc_route_cost(self, route):
        dist_cost = self.calculate_route_distance(route)
        # print(f"dist{dist_cost}")
        arrival_times = self.calculate_arrival_times(route)
        time_cost = 0
        for i in range(1, len(route)):
            # print(route[i])
            target = self.problem_instance.vertices[route[i]]
            if type(target) is CharingStation:
                time_cost += 0
            if type(target) is Customer:
                time_cost += target.due_date-arrival_times[i-1]
            # print(f"time{time_cost}")
        route_cost = dist_cost + 0.5 * time_cost
        # print(route_cost)
        return route_cost

    def calculate_route_cost(self, route):
        route_cost = 0
        if not self.is_feasible(route):
            route_cost = float('inf')
        else:
            dist_cost = self.calculate_route_distance(route)
            # print(dist_cost)
            arrival_times = self.calculate_arrival_times(route)
            time_cost = 0
            for i in range(1, len(route)):
                # print(route[i])
                target = self.problem_instance.vertices[route[i]]
                if type(target) is CharingStation:
                    time_cost += 0
                if type(target) is Customer:
                    time_cost += target.due_date-arrival_times[i-1]
                # print(time_cost)
            route_cost = dist_cost + 0.5 * time_cost + 800

        return route_cost

    def energy_violated(self,route):
        tank_capacity = self.problem_instance.config.tank_capacity
        now_energy = self.problem_instance.config.now_energy
        fuel_consumption_rate = self.problem_instance.config.fuel_consumption_rate
        last_position = self.problem_instance.depot

        for v in route[1:]:
            target = self.problem_instance.vertices[v]
            d = last_position.distance_to(target)
            now_energy -= d * fuel_consumption_rate

            if now_energy < 0:
                print("energy_violated")
                return True

            if type(target) is CharingStation:
                now_energy = tank_capacity

            last_position = target

        return False

    def find_operate(self, route):
        while self.energy_violated(route):
            break
        return None

    def find_optimal_charging_station_insertion(self, route):
        # print(route)
        best_insertion_point = None
        best_station = None
        min_route_cost = float('inf')
        # if route == ['D0', 'C4', 'C6', 'C1', 'D0']:
        #     print("real route is found")
        #     for i in range(1, len(route)):
        #         v = self.problem_instance.vertices[route[i - 1]]
        #         if not isinstance(v, CharingStation):
        #
        #             reachable_stations = self.get_reachable_charging_stations(
        #                 self.problem_instance.vertices[route[i - 1]],
        #                 self.calculate_remaining_tank_capacity(route[:i]))
        #
        #             if reachable_stations is None:
        #                 continue
        #
        #             else:
        #                 for j in reachable_stations:
        #                     temp_route = route[:i] + [j.id] + route[i:]
        #                     # print(temp_route, f"可行", self.is_feasible(temp_route))
        #                     # 判断插入后路径是否可行
        #                     if self.is_feasible(temp_route):
        #
        #                         # 计算插入后的总成本
        #                         route_cost = self.calculate_route_cost(temp_route)
        #                         # Tr = self.calculate_arrival_times(temp_route)
        #                         # print(Tr)
        #                         # print(temp_route)
        #                         # print(route_cost)
        #                         # 更新最优插入点和总成本
        #                         if route_cost < min_route_cost:
        #                             min_route_cost = route_cost
        #                             best_insertion_point = i
        #                             best_station = j
        # # 遍历路径中的每个顾客之间的位置作为插入点
        # else:
        for i in range(1, len(route)):
            v = self.problem_instance.vertices[route[i-1]]
            if not isinstance(v, CharingStation):

                reachable_stations = self.get_reachable_charging_stations(self.problem_instance.vertices[route[i - 1]],
                                                                          self.calculate_remaining_tank_capacity(route[:i]))
                if reachable_stations is None:
                    continue

                else:
                    for j in reachable_stations:
                        temp_route = route[:i] + [j.id] + route[i:]

                        # 判断插入后路径是否可行
                        if self.is_feasible(temp_route):

                            # 计算插入后的总成本
                            route_cost = self.calculate_route_cost(temp_route)
                            # Tr = self.calculate_arrival_times(temp_route)
                            # print(Tr)
                            # print(temp_route)
                            # print(route_cost)
                            # 更新最优插入点和总成本
                            if route_cost < min_route_cost:
                                min_route_cost = route_cost
                                best_insertion_point = i
                                best_station = j
        # print(best_insertion_point, best_station, min_route_cost)
        return best_insertion_point, best_station, min_route_cost

    def make_route_feasible_and_best(self, route):
        best_insertion_point, best_station, min_route_cost = self.find_optimal_charging_station_insertion(route)
        if best_station == None:
            return None
        best_feasible_route = route[:best_insertion_point]+[best_station.id]+route[best_insertion_point:]

        # print(best_feasible_route)
        if not self.is_feasible(best_feasible_route):
            # print("NO")
            return None
        else:
            # print("YES")
            # print(best_feasible_route)
            return best_feasible_route


class VariableNeighbourhoodSearch:
    """
        N_0 -> 2-exchange within routes (intra route)
        N_1 -> 2-opt (intra route)
        N_2 -> Moving a customer to another route (inter route)
        N_3 -> Merge two routes (inter route)
        N_4 -> 2opt*-operator (inter route)
    """

    def __init__(self, problem_instance: RoutingProblemInstance, solution, cost):
        self.problem_instance = problem_instance
        self.solution = solution
        self.cost = cost

    def _set_vehicle_energy(self, vehicle_index: int) -> None:
        self.problem_instance.config.now_energy = self.problem_instance.config.get_initial_energy(vehicle_index)

    def _sum_route_costs(self, routes):
        total = 0
        for idx, route in enumerate(routes):
            self._set_vehicle_energy(idx)
            total += self.calculate_route_cost(route)
        return total

    # 终止条件为最大运行时间：10s
    def improve_solution(self):
        best_cost = self.cost
        best_solution = deepcopy(self.solution)
        result_cache = LifoQueue()  # 保存最近的5个结果

        route_cost_cache = 0

        for idx, r in enumerate(self.solution):
            self._set_vehicle_energy(idx)
            route_cost_cache += self.calculate_route_cost(r)

        statistic = []
        iteration_times = []

        start_time = time.time()  # 记录开始时间

        n = 1
        while time.time() - start_time < 5:  # 限制运行时间不超过10秒
            k = 0
            n += 1

            while k <= K_MAX:

                # 在内部循环中也检查时间
                if time.time() - start_time >= 5:
                    break

                next_rand_sol, next_route_cost_cache = self.get_next_random_feasible_solution(k, best_solution,
                                                                                              route_cost_cache)

                ls_solution, ls_route_cost_cache = self.do_local_search(next_rand_sol, next_route_cost_cache)

                if ls_route_cost_cache < best_cost:
                    # 如果找到更好的解，重置邻域为0
                    best_solution = ls_solution
                    best_cost = ls_route_cost_cache
                    k = 0
                else:
                    # 没有找到更好的解，继续增加邻域
                    k += 1
                statistic.append(best_cost)
                iteration_times.append(time.time() - start_time)

            result_cache.put(best_cost)

            if result_cache.qsize() == NO_IMPROVEMENT_TOLERANCE:
                prev_cost = result_cache.get()

                # 检查最近的 NO_IMPROVEMENT_TOLERANCE 轮中是否有改进
                if result_cache.qsize() == NO_IMPROVEMENT_TOLERANCE:
                    prev_cost = result_cache.get()
                    if prev_cost == best_cost:
                        # 如果没有改进，检查时间是否达标
                        if time.time() - start_time < 5:
                            # 如果未达到10秒，则重置k并继续
                            k = 0
                        else:
                            break  # 如果达到10秒，则停止算法
        # remaining_energies = [
        #     calculate_route_remaining_energy(self.problem_instance, route, idx)
        #     for idx, route in enumerate(best_solution)
        # ]
        # for idx, energy in enumerate(remaining_energies):
        #     print(f"VariableNeighbourhoodSearch best solution vehicle {idx} remaining energy: {energy:.3f}")

        return best_cost, best_solution, statistic, iteration_times

    # # 终止条件为多次迭代解未没有变好
    # def improve_solution(self):
    #     best_cost = self.cost
    #     best_solution = deepcopy(self.solution)
    #
    #     result_cache = LifoQueue()  # saves the last 5 results
    #
    #     route_cost_cache = 0
    #
    #     for r in self.solution:
    #         route_cost_cache += self.calculate_route_cost(r)
    #
    #     while True:
    #         k = 0
    #         while k <= K_MAX:
    #
    #             next_rand_sol, next_route_cost_cache = self.get_next_random_feasible_solution(k, best_solution,
    #                                                                                           route_cost_cache)
    #
    #             ls_solution, ls_route_cost_cache = self.do_local_search(next_rand_sol, next_route_cost_cache)
    #
    #             if ls_route_cost_cache < best_cost:
    #                 # the solution was better than the current -> restart with neighbourhood 0
    #                 best_solution = ls_solution
    #                 best_cost = ls_route_cost_cache
    #                 k = 0
    #             else:
    #                 # solution was not better -> try to improve with next neighbourhood
    #                 k += 1
    #
    #         result_cache.put(best_cost)
    #
    #         if result_cache.qsize() == NO_IMPROVEMENT_TOLERANCE:
    #             prev_cost = result_cache.get()
    #
    #             # In case there was no improvement within the last [NO_IMPROVEMENT_TOLERANCE] rounds -> terminate!
    #             if prev_cost == best_cost:
    #                 break
    #             else:
    #                 print('hallo')
    #
    #     return best_cost, best_solution

    def get_next_random_feasible_solution(self, neighbourhood_index, current_solution, route_cost_cache):
        now_solution = current_solution
        now_cost = route_cost_cache
        # N_0 -> 2-exchange within routes
        if neighbourhood_index == 0:
            # print("0")
            pre_solution = deepcopy(current_solution)
            pre_cost = route_cost_cache
            for route in pre_solution:
                route = self.remove_charging_station(route)
            rand_route_idx = randint(0, len(pre_solution) - 1)

            for i in range(0, len(pre_solution)):
                route_idx = (rand_route_idx + i) % (len(pre_solution))
                route = list(pre_solution[route_idx])

                for j in range(0, len(route) * len(route)):
                    from_idx = randint(0, len(route) - 1)
                    to_idx = randint(0, len(route) - 1)

                    if from_idx == to_idx:
                        to_idx = (to_idx + 1) % (len(route))

                    from_v = self.problem_instance.vertices[route[from_idx]]
                    to_v = self.problem_instance.vertices[route[to_idx]]
                    # print("original")
                    # print(current_solution)
                    if type(from_v) is Customer and type(to_v) is Customer:
                        route[from_idx], route[to_idx] = route[to_idx], route[from_idx]
                        if self.is_nc_feasible(route):

                            temp_solution = deepcopy(pre_solution)
                            temp_solution.remove(pre_solution[route_idx])
                            temp_solution.append(route)
                            # print("changed")
                            # print(current_solution)
                            # print(route)

                            temp_solution = self.process_route(temp_solution)
                            #原版本
                            # for idx, value in enumerate(temp_solution):
                            #     if self.need_charge(temp_solution[idx]):
                            #         k = self.make_route_feasible_and_best(value)
                            #         if k is None:
                            #             route[from_idx], route[to_idx] = route[to_idx], route[from_idx]
                            #             break
                            #         temp_solution[idx] = k

                            if self.state_feasible(temp_solution):
                                new_route_cost_cache = self._sum_route_costs(temp_solution)
                                return temp_solution, new_route_cost_cache
                        else:
                            route[from_idx], route[to_idx] = route[to_idx], route[from_idx]

        # N_1 -> 2opt
        elif neighbourhood_index == 1:
            # print("1")
            pre_solution = deepcopy(current_solution)
            pre_cost = route_cost_cache
            for route in pre_solution:
                route = self.remove_charging_station(route)
            route_indices = list(range(0, len(pre_solution) - 1))
            shuffle(route_indices)

            for route_idx in route_indices:
                route = list(pre_solution[route_idx])

                cut_points = list(product(range(1, len(route) - 1), range(0, len(route) - 1)))
                shuffle(cut_points)

                for cp in cut_points:
                    if cp[0] < cp[1] and cp[1] - cp[0] > 1:
                        part_1 = route[:cp[0]]
                        part_2 = route[cp[0]:cp[1]]
                        part_3 = route[cp[1]:]
                        new_route = part_1 + part_2 + part_3
                        if self.is_nc_feasible(new_route):

                            new_solution = deepcopy(pre_solution)
                            new_solution.remove(route)
                            new_solution.append(new_route)

                            new_solution = self.process_route(new_solution)
                            # for idx, value in enumerate(new_solution):
                            #     if self.need_charge(new_solution[idx]):
                            #         k = self.make_route_feasible_and_best(value)
                            #         if k is None:
                            #             break
                            #         new_solution[idx] = k

                            if self.state_feasible(new_solution):
                                new_route_cost_cache = self._sum_route_costs(new_solution)
                                return new_solution, new_route_cost_cache

        # N_2 -> Moving a customer to another route
        elif neighbourhood_index == 2:
            # print("2")
            pre_solution = deepcopy(current_solution)
            pre_cost = route_cost_cache
            for route in pre_solution:
                route = self.remove_charging_station(route)
            route_combinations = list(combinations(range(0, len(pre_solution)), 2))
            shuffle(route_combinations)

            for combination in route_combinations:
                from_route = list(pre_solution[combination[0]])
                to_route = list(pre_solution[combination[1]])
                transfer_points = list(product(range(0, len(from_route)), range(0, len(to_route))))
                shuffle(transfer_points)

                for tp in transfer_points:
                    from_v = self.problem_instance.vertices[from_route[tp[0]]]
                    to_v = self.problem_instance.vertices[to_route[tp[1]]]

                    if type(from_v) is Customer and type(to_v) is Customer:
                        from_route[tp[0]], to_route[tp[1]] = to_route[tp[1]], from_route[tp[0]]
                        if self.is_nc_feasible(from_route) and self.is_nc_feasible(to_route):
                            if combination[0] < combination[1]:
                                new_route_cost_cache = 0
                                new_solution = pre_solution[:combination[0]]
                                new_solution += [to_route]
                                new_solution += pre_solution[combination[0] + 1:combination[1]]
                                new_solution += [from_route]
                                new_solution += pre_solution[combination[1] + 1:]
                                new_solution = self.process_route(new_solution)
                                # for idx, value in enumerate(new_solution):
                                #     if self.need_charge(new_solution[idx]):
                                #         k = self.make_route_feasible_and_best(value)
                                #         if k is None:
                                #             break
                                #         new_solution[idx] = k

                                if self.state_feasible(new_solution):
                                    new_route_cost_cache = self._sum_route_costs(new_solution)
                                    return new_solution, new_route_cost_cache
                            else:

                                new_solution = pre_solution[:combination[1]]
                                new_solution += [from_route]
                                new_solution += pre_solution[combination[1] + 1:combination[0]]
                                new_solution += [to_route]
                                new_solution += pre_solution[combination[0] + 1:]

                                new_solution = self.process_route(new_solution)
                                # for idx, value in enumerate(new_solution):
                                #     if self.need_charge(new_solution[idx]):
                                #         k = self.make_route_feasible_and_best(value)
                                #         if k is None:
                                #             break
                                #         new_solution[idx] = k
                                if self.state_feasible(new_solution):
                                    new_route_cost_cache = self._sum_route_costs(new_solution)
                                    return new_solution, new_route_cost_cache
                        else:
                            from_route[tp[0]], to_route[tp[1]] = to_route[tp[1]], from_route[tp[0]]

        # N_3 -> Merge two routes
        elif neighbourhood_index == 3:
            # print("3")
            pre_solution = deepcopy(current_solution)
            pre_cost = route_cost_cache
            for route in pre_solution:
                route = self.remove_charging_station(route)
            route_combinations = list(combinations(range(0, len(pre_solution)), 2))
            shuffle(route_combinations)

            for combination in route_combinations:
                from_route = list(pre_solution[combination[0]])
                to_route = list(pre_solution[combination[1]])

                new_route = None
                if self.is_nc_feasible(from_route[:-1] + to_route[1:]):
                    new_route = from_route[:-1] + to_route[1:]
                elif self.is_nc_feasible(to_route[:-1] + from_route[1:]):
                    new_route = to_route[:-1] + from_route[1:]

                if new_route is not None:

                    new_solution = pre_solution[:combination[0]]
                    new_solution += [new_route]
                    new_solution += pre_solution[combination[0] + 1:combination[1]]
                    new_solution += pre_solution[combination[1] + 1:]

                    new_solution = self.process_route(new_solution)
                    if self.state_feasible(new_solution):
                        new_route_cost_cache = self._sum_route_costs(new_solution)
                        return new_solution, new_route_cost_cache
                else:
                    continue

        # N_4 -> 2opt*-operator
        elif neighbourhood_index == 4:
            # print("4")
            pre_solution = deepcopy(current_solution)
            pre_cost = route_cost_cache
            for route in pre_solution:
                route = self.remove_charging_station(route)
            route_combinations = list(combinations(range(0, len(pre_solution)), 2))
            shuffle(route_combinations)

            for combination in route_combinations:
                route_1 = list(pre_solution[combination[0]])
                route_2 = list(pre_solution[combination[1]])

                swap_points = list(product(range(1, len(route_1)), range(1, len(route_2))))
                shuffle(swap_points)

                for sp in swap_points:
                    split_index_1 = sp[0]
                    split_index_2 = sp[1]
                    # sp_1 = self.problem_instance.vertices[route_1[split_index_1]]
                    # sp_2 = self.problem_instance.vertices[route_2[split_index_2]]

                    if split_index_1 >= 2:
                        sp_prev = self.problem_instance.vertices[route_1[split_index_1 - 1]]

                        if type(sp_prev) is CharingStation:
                            split_index_1 -= 1
                            # sp_1 = sp_prev

                    if split_index_2 >= 2:
                        sp_prev = self.problem_instance.vertices[route_2[split_index_2 - 1]]

                        if type(sp_prev) is CharingStation:
                            split_index_2 -= 1
                            # sp_2 = sp_prev

                    new_route_1 = route_1[:split_index_1] + route_2[split_index_2:]
                    new_route_2 = route_2[:split_index_2] + route_1[split_index_1:]

                    if self.is_nc_feasible(new_route_1) and self.is_nc_feasible(new_route_2):
                        temp_route_cost_cache = 0
                        temp_solution = deepcopy(pre_solution)
                        temp_solution[combination[0]] = new_route_1
                        temp_solution[combination[1]] = new_route_2

                        temp_solution = self.process_route(temp_solution)
                        if self.state_feasible(temp_solution):
                            temp_route_cost_cache = self._sum_route_costs(temp_solution)

                            return temp_solution, temp_route_cost_cache

        return current_solution, route_cost_cache

    def do_local_search(self, initial_solution, route_cost_cache):
        # print("local")
        best_solution = initial_solution
        best_route_cost_cache = route_cost_cache

        solution_improved = False
        # print("n0")
        next_best_solution, next_best_route_cost_cache = self.get_next_best_n0_neighbour(best_solution,
                                                                                         best_route_cost_cache)
        if best_route_cost_cache > next_best_route_cost_cache:
            best_route_cost_cache = next_best_route_cost_cache
            best_solution = next_best_solution
            solution_improved = True
        # print("n1")
        next_best_solution, next_best_route_cost_cache = self.get_next_best_n1_neighbour(best_solution,
                                                                                         best_route_cost_cache)
        if best_route_cost_cache > next_best_route_cost_cache:
            best_route_cost_cache = next_best_route_cost_cache
            best_solution = next_best_solution
            solution_improved = True
        # print("n2")
        next_best_solution, next_best_route_cost_cache = self.get_next_best_n2_neighbour(best_solution,
                                                                                         best_route_cost_cache)

        if best_route_cost_cache > next_best_route_cost_cache:
            best_route_cost_cache = next_best_route_cost_cache
            best_solution = next_best_solution
            solution_improved = True
        # print("n3")
        next_best_solution, next_best_route_cost_cache = self.get_next_best_n3_neighbour(best_solution,
                                                                                         best_route_cost_cache)
        if best_route_cost_cache > next_best_route_cost_cache:
            best_route_cost_cache = next_best_route_cost_cache
            best_solution = next_best_solution
            solution_improved = True
        # print("n4")
        next_best_solution, next_best_route_cost_cache = self.get_next_best_n4_neighbour(best_solution,
                                                                                         best_route_cost_cache)
        while solution_improved:
            solution_improved = False
            best_solution = next_best_solution
            best_route_cost_cache = next_best_route_cost_cache

            next_best_solution, next_best_route_cost_cache = self.get_next_best_n0_neighbour(best_solution,
                                                                                             best_route_cost_cache)

            if best_route_cost_cache > next_best_route_cost_cache:
                best_route_cost_cache = next_best_route_cost_cache
                best_solution = next_best_solution
                solution_improved = True

            next_best_solution, next_best_route_cost_cache = self.get_next_best_n1_neighbour(best_solution,
                                                                                             best_route_cost_cache)

            if best_route_cost_cache > next_best_route_cost_cache:
                best_route_cost_cache = next_best_route_cost_cache
                best_solution = next_best_solution
                solution_improved = True

            next_best_solution, next_best_route_cost_cache = self.get_next_best_n2_neighbour(best_solution,
                                                                                             best_route_cost_cache)

            if best_route_cost_cache > next_best_route_cost_cache:
                best_route_cost_cache = next_best_route_cost_cache
                best_solution = next_best_solution
                solution_improved = True

            next_best_solution, next_best_route_cost_cache = self.get_next_best_n3_neighbour(best_solution,
                                                                                             best_route_cost_cache)

            if best_route_cost_cache > next_best_route_cost_cache:
                best_route_cost_cache = next_best_route_cost_cache
                best_solution = next_best_solution
                solution_improved = True

            next_best_solution, next_best_route_cost_cache = self.get_next_best_n4_neighbour(best_solution,
                                                                                             best_route_cost_cache)

            if best_route_cost_cache > next_best_route_cost_cache:
                best_route_cost_cache = next_best_route_cost_cache
                best_solution = next_best_solution
                solution_improved = True

        return best_solution, best_route_cost_cache

    def get_next_best_n0_neighbour(self, solution, route_cost_cache):
        best_solution = deepcopy(solution)
        # print("hallo")
        for route in best_solution:
            route = self.remove_charging_station(route)
        for route_idx in range(0, len(best_solution)):
            route = list(best_solution[route_idx])
            for from_idx in range(0, len(route) - 1):
                for to_idx in range(from_idx + 1, len(route)):
                    from_v = self.problem_instance.vertices[route[from_idx]]
                    to_v = self.problem_instance.vertices[route[to_idx]]

                    if type(from_v) is Customer and type(to_v) is Customer:
                        route[from_idx], route[to_idx] = route[to_idx], route[from_idx]

                        # print(best_solution)
                        # print(route)
                        if self.is_nc_feasible(route):

                            temp_solution = deepcopy(best_solution)
                            temp_solution[route_idx] = route

                            for idx, value in enumerate(temp_solution):
                                if self.need_charge(temp_solution[idx]):
                                    k = self.make_route_feasible_and_best(value)
                                    if k is None:
                                        route[from_idx], route[to_idx] = route[to_idx], route[from_idx]
                                        break
                                    temp_solution[idx] = k

                            # print(best_solution)
                            if self.state_feasible(temp_solution):
                                new_route_cost_cache = self._sum_route_costs(temp_solution)
                                if new_route_cost_cache < route_cost_cache:
                                    return temp_solution, new_route_cost_cache
                        else:
                            route[from_idx], route[to_idx] = route[to_idx], route[from_idx]

                        return solution, route_cost_cache

    def get_next_best_n1_neighbour(self, solution, route_cost_cache):
        best_solution = deepcopy(solution)
        for route in best_solution:
            route = self.remove_charging_station(route)
        route_indices = list(range(0, len(best_solution) - 1))
        shuffle(route_indices)

        for route_idx in route_indices:
            route = list(best_solution[route_idx])

            cut_points = list(product(range(1, len(route) - 1), range(0, len(route) - 1)))
            shuffle(cut_points)

            for cp in cut_points:
                if cp[0] < cp[1] and cp[1] - cp[0] > 1:
                    part_1 = route[:cp[0]]
                    part_2 = route[cp[0]:cp[1]]
                    part_3 = route[cp[1]:]
                    new_route = part_1 + part_2 + part_3
                    "hallo"
                    # print(new_route)
                    if self.is_nc_feasible(part_1 + part_2 + part_3):

                        new_solution = deepcopy(best_solution)
                        new_solution.remove(route)
                        new_solution.append(new_route)
                        new_solution = self.process_route(new_solution)
                        # for idx, value in enumerate(new_solution):
                        #     if self.need_charge(new_solution[idx]):
                        #         k = self.make_route_feasible_and_best(value)
                        #         if k is None:
                        #             break
                        #         new_solution[idx] = k

                        if self.state_feasible(new_solution):
                            new_route_cost_cache = self._sum_route_costs(new_solution)
                            if new_route_cost_cache < route_cost_cache:
                                return new_solution, new_route_cost_cache

        return solution, route_cost_cache

    def get_next_best_n2_neighbour(self, solution, route_cost_cache):
        best_solution = deepcopy(solution)
        best_cost = route_cost_cache

        for route in best_solution:
            route = self.remove_charging_station(route)

        route_combinations = list(combinations(range(0, len(best_solution)), 2))
        shuffle(route_combinations)

        for combination in route_combinations:
            from_route = list(best_solution[combination[0]])
            to_route = list(best_solution[combination[1]])

            transfer_points = list(product(range(0, len(from_route)), range(0, len(to_route))))
            shuffle(transfer_points)

            for tp in transfer_points:
                from_v = self.problem_instance.vertices[from_route[tp[0]]]
                to_v = self.problem_instance.vertices[to_route[tp[1]]]

                if type(from_v) is Customer and type(to_v) is Customer:
                    from_route[tp[0]], to_route[tp[1]] = to_route[tp[1]], from_route[tp[0]]
                    if self.is_nc_feasible(from_route) and self.is_nc_feasible(to_route):
                        if combination[0] < combination[1]:

                            new_solution = deepcopy(best_solution)
                            new_solution.remove(best_solution[combination[0]])
                            new_solution.remove(best_solution[combination[1]])
                            new_solution.append(from_route)
                            new_solution.append(to_route)

                            for idx, value in enumerate(new_solution):
                                if self.need_charge(new_solution[idx]):
                                    k = self.make_route_feasible_and_best(value)
                                    if k is None:
                                        from_route[tp[0]], to_route[tp[1]] = to_route[tp[1]], from_route[tp[0]]
                                        break
                                    new_solution[idx] = k

                                if self.state_feasible(new_solution):
                                    new_route_cost_cache = self._sum_route_costs(new_solution)
                                    if new_route_cost_cache < route_cost_cache:
                                        return new_solution, new_route_cost_cache

                        from_route[tp[0]], to_route[tp[1]] = to_route[tp[1]], from_route[tp[0]]

        return solution, route_cost_cache

    def get_next_best_n3_neighbour(self, solution, route_cost_cache):
        pre_solution = deepcopy(solution)
        pre_cost = route_cost_cache
        for route in pre_solution:
            route = self.remove_charging_station(route)
        route_combinations = list(combinations(range(0, len(pre_solution)), 2))
        shuffle(route_combinations)

        for combination in route_combinations:
            from_route = list(pre_solution[combination[0]])
            to_route = list(pre_solution[combination[1]])

            new_route = None
            if self.is_nc_feasible(from_route[:-1] + to_route[1:]):
                new_route = from_route[:-1] + to_route[1:]
            elif self.is_nc_feasible(to_route[:-1] + from_route[1:]):
                new_route = to_route[:-1] + from_route[1:]

            if new_route:

                new_solution = deepcopy(pre_solution)
                new_solution.remove(from_route)
                new_solution.remove(to_route)
                new_solution.append(new_route)

                new_solution = self.process_route(new_solution)

                if self.state_feasible(new_solution):
                    new_route_cost_cache = self._sum_route_costs(new_solution)
                    if new_route_cost_cache < route_cost_cache:
                        return new_solution, new_route_cost_cache

        return solution, route_cost_cache

    def get_next_best_n4_neighbour(self, solution, route_cost_cache):
        pre_solution = deepcopy(solution)
        pre_cost = route_cost_cache
        for route in pre_solution:
            route = self.remove_charging_station(route)
        route_combinations = list(combinations(range(0, len(pre_solution)), 2))
        shuffle(route_combinations)

        for combination in route_combinations:
            route_1 = list(pre_solution[combination[0]])
            route_2 = list(pre_solution[combination[1]])

            swap_points = list(product(range(1, len(route_1)), range(1, len(route_2))))
            shuffle(swap_points)

            for sp in swap_points:
                split_index_1 = sp[0]
                split_index_2 = sp[1]

                if split_index_1 >= 2:
                    sp_prev = self.problem_instance.vertices[route_1[split_index_1 - 1]]

                    if type(sp_prev) is CharingStation:
                        split_index_1 -= 1
                        # sp_1 = sp_prev

                if split_index_2 >= 2:
                    sp_prev = self.problem_instance.vertices[route_2[split_index_2 - 1]]

                    if type(sp_prev) is CharingStation:
                        split_index_2 -= 1
                        # sp_2 = sp_prev

                new_route_1 = route_1[:split_index_1] + route_2[split_index_2:]
                new_route_2 = route_2[:split_index_2] + route_1[split_index_1:]

                if self.is_nc_feasible(new_route_1) and self.is_nc_feasible(new_route_2):

                    new_solution = deepcopy(pre_solution)
                    new_solution.remove(pre_solution[combination[0]])
                    new_solution.remove(pre_solution[combination[1]])
                    new_solution.append(new_route_1)
                    new_solution.append(new_route_2)
                    new_solution = self.process_route(new_solution)
                    if self.state_feasible(new_solution):
                        new_route_cost_cache = self._sum_route_costs(new_solution)
                        if new_route_cost_cache < route_cost_cache:
                            return new_solution, new_route_cost_cache
        return solution, route_cost_cache

    def process_route(self, state):
        # if state.state == [['D0', 'C10', 'C4', 'C1', 'D0'], ['D0',  'C9', 'C5', 'C2', 'C7', 'C8', 'C3', 'C6', 'D0']]:
        #     print("111111111111111111")
        # elif state.state == [['D0',  'C9', 'C5', 'C2', 'C7', 'C8', 'C3', 'C6', 'D0'], ['D0', 'C10', 'C4', 'C1', 'D0']]:
        #     print("222222222222222222")
        # state.state = [['D0', 'C5', 'S0', 'C4', 'C3', 'C8', 'D0'], ['D0', 'C2', 'C1', 'S2', 'C6', 'C7', 'C9', 'C10', 'D0']]
        # print(state.state)
        # for route in state.state:
        #     print(self.is_feasible(route))
        #     print(self.calculate_route_cost(route))

        # elif state.state == [['D0', 'C2', 'C3', 'C5', 'D0'], ['D0', 'C4', 'C6', 'C1', 'D0']]:
        #     print("find state2")

        # print("111111111111111111111111111")
        # 初始化未访问的客户列表
        unvisited_customers = []

        for idx, value in enumerate(state):
            self._set_vehicle_energy(idx)

            # 判断当前路径是否需要充电
            while self.need_charge(state[idx]):
                self._set_vehicle_energy(idx)

                k = self.make_route_feasible_and_best(value)
                # if k == ['D0','S2', 'C6', 'C1', 'D0']:
                #     print("real charge is right")
                if k is not None:
                    # 找到可行的插入点，更新路径
                    state[idx] = k
                    break  # 找到充电位置，跳出while循环，继续下一条路径
                else:
                    # 找不到插入点，移除最小到达时间的客户点
                    earliest_customer = self.find_earliest_customer(state[idx])
                    state[idx].remove(earliest_customer)    # 从当前路径中移除
                    unvisited_customers.append(self.problem_instance.vertices[earliest_customer])  # 将该客户点标记为未访问
                    # 路径调整后，继续判断是否需要充电
        # if state.state == [['D0', 'S0', 'C4', 'C6', 'C1', 'D0'], ['D0', 'S0', 'C2', 'C3', 'C5', 'D0']]:
            # now_energy = self.problem_instance.config.now_energy
            # tank_capacity = self.problem_instance.config.tank_capacity
            #
            # velocity = self.problem_instance.config.velocity
            # fuel_consumption_rate = self.problem_instance.config.fuel_consumption_rate
            # charging_rate = self.problem_instance.config.charging_rate
            #
            # last_position = self.problem_instance.depot
            # time = self.problem_instance.depot.call_time + self.problem_instance.depot.service_time
            # for route in state.state:
            #     arrival_times = []
            #
            #     for v in route[1:]:
            #         target = self.problem_instance.vertices[v]
            #         print(target)
            #         d = last_position.distance_to(target)
            #         print(d)
            #         time += d / velocity
            #         print(time)
            #         arrival_times.append(time)
            #
            #         now_energy -= d * fuel_consumption_rate
            #
            #         if type(target) is Customer:
            #             time += target.service_time
            #             print(time)
            #         elif type(target) is CharingStation:
            #             time += (tank_capacity - now_energy) * charging_rate
            #             print((tank_capacity - now_energy) * charging_rate)
            #         last_position = target
            #
            #     print(arrival_times)
            #     a = self.calculate_demand(route)
            #     print((a))
            #     print(sum(a))
            # print(state.state)
            # print(state.objective())
        # elif state.state == [['D0', 'S0', 'C2', 'C3', 'C5', 'D0'], ['D0', 'S0', 'C4', 'C6', 'C1', 'D0']]:
        #     print("find state22222222")
        #     print(state.state)
        #     print(state.objective())
        # 循环直到所有未访问客户都被处理
        while unvisited_customers:
            # print("Current unvisited_customers:", unvisited_customers)

            # 使用k-最近邻域启发式为未访问的客户生成新的路径
            depot = self.problem_instance.depot
            new_routes = self.innh(depot, unvisited_customers, self.problem_instance)
            # print(new_routes)
            # 对新的路径进行充电站插入操作
            start_index = len(state)
            for offset, value in enumerate(new_routes):
                vehicle_index = start_index + offset
                self._set_vehicle_energy(vehicle_index)
                # print(value)
                # print(self.need_charge(value))
                while self.need_charge(value):
                    self._set_vehicle_energy(vehicle_index)
                    k = self.make_route_feasible_and_best(value)
                    if k is not None:
                        new_routes[offset] = k
                        break
                    else:
                        earliest_customer = self.find_earliest_customer(new_routes[offset])
                        new_routes[offset].remove(earliest_customer)
                        unvisited_customers.append(self.problem_instance.vertices[earliest_customer])

            # 展开 new_routes，将其转化为单个客户的列表
            flattened_new_routes = [customer for route in new_routes for customer in route]
            # print(flattened_new_routes)
            # 更新 unvisited_customers 列表，确保未服务的客户仍然保留

            unvisited_customers = [customer for customer in unvisited_customers if customer.id not in flattened_new_routes]

            # print("11111111Current unvisited_customers:", unvisited_customers)
            state.extend(new_routes)
        # print("22222222222222")
        return state

    def innh(self, depot, customers, problem_instance, k=3):
        giant_route = []
        serviced_customers = set()

        while customers:
            route = ["D0"]
            last_position = depot

            while customers:
                possible_successors = [customer for customer in customers if customer not in serviced_customers]
                possible_successors.sort(key=lambda n: n.distance_to(last_position))
                possible_successors = possible_successors[:k]
                if not possible_successors:
                    break
                successor = min(possible_successors, key=lambda n: n.due_date)
                route.append(successor.id)
                # demand = route.calculate_demand()
                if not self.is_nc_feasible(route):
                    route.remove_target(successor)
                    break
                serviced_customers.add(successor)
                last_position = successor
            route.append("D0")
            if len(route) > 1:
                giant_route.append(route)
                customers = [customer for customer in customers if customer not in serviced_customers]

        return giant_route

    def find_earliest_customer(self, route):
        # 假设每个客户点包含属性 'due_date'
        # 返回路径中 due_date 最小的客户点
        earliest_customer = None
        min_due_date = float('inf')

        for v in route:
            customer = self.problem_instance.vertices[v]
            if isinstance(customer, Customer):
                if customer.due_date < min_due_date:
                    min_due_date = customer.due_date
                    earliest_customer = v

        return earliest_customer

    def is_feasible(self, route):
        """
        slow implementation for checking the feasibility of a route
        :param route: route to check
        :return: returns True if the route is feasible
        """
        self.calculate_demand(route)

        tank_capacity = self.problem_instance.config.tank_capacity
        now_energy = self.problem_instance.config.now_energy
        load_capacity = self.problem_instance.config.payload_capacity
        velocity = self.problem_instance.config.velocity
        fuel_consumption_rate = self.problem_instance.config.fuel_consumption_rate
        charging_rate = self.problem_instance.config.charging_rate

        last_position = self.problem_instance.depot
        time = self.problem_instance.depot.call_time + self.problem_instance.depot.service_time

        # if route == ['D0', 'S0', 'C9', 'C5', 'C2', 'C7', 'C8', 'C3', 'C6', 'D0']:
        #     for v in route[1:]:
        #         target = self.problem_instance.vertices[v]
        #         d = last_position.distance_to(target)
        #         time += d / velocity
        #         print(target, f"的到达时间是", time)
        #         print(v, f"的容量为", load_capacity)
        #         now_energy -= d * fuel_consumption_rate
        #
        #         if now_energy < 0:
        #             print("energy_violated")
        #             return False
        #
        #         if time > target.due_date:
        #             print("time_violated")
        #             return False
        #
        #         if type(target) is Customer:
        #             load_capacity -= target.demand
        #             time += target.service_time
        #         elif type(target) is CharingStation:
        #             time += (tank_capacity - now_energy) * charging_rate
        #             now_energy = tank_capacity
        #         print(v, f"过后的剩余容量为", load_capacity)
        #         if load_capacity < 0:
        #             print("load_violated")
        #             return False
        #
        #         last_position = target
        # else:
        for v in route[1:]:
            target = self.problem_instance.vertices[v]
            d = last_position.distance_to(target)
            time += d / velocity

            now_energy -= d * fuel_consumption_rate

            if now_energy < 0:

                return False

            if time > target.due_date:

                return False

            if type(target) is Customer:
                load_capacity -= target.demand
                time += target.service_time
            elif type(target) is CharingStation:
                time += (tank_capacity - now_energy) * charging_rate
                now_energy = tank_capacity

            if load_capacity < 0:

                return False

            last_position = target

        return True

    def remove_charging_station(self, route):
        route[:] = [i for i in route if type(self.problem_instance.vertices[i]) is not CharingStation]
        return route

    def need_charge(self, route):
        return (self.problem_instance.config.now_energy - self.calculate_route_distance(route) *
                self.problem_instance.config.fuel_consumption_rate < 0.2 * self.problem_instance.config.tank_capacity)

    def nnh_need_charge(self, route):
        return (self.problem_instance.config.now_energy - self.calculate_total_distance(route) *
                self.problem_instance.config.fuel_consumption_rate < 0.2 * self.problem_instance.config.tank_capacity)

    def calculate_remaining_tank_capacity(self, route):
        last_position = self.problem_instance.depot
        now_energy = self.problem_instance.config.now_energy

        for t in route[1:]:
            target = self.problem_instance.vertices[t]
            distance = last_position.distance_to(target)
            consumption = distance * self.problem_instance.config.fuel_consumption_rate
            now_energy -= consumption

            if type(t) is CharingStation:
                now_energy = self.problem_instance.config.tank_capacity

            last_position = target
        return now_energy

    def get_reachable_charging_stations(self, cust: Customer, capacity: float) -> list:
        max_dist = capacity / self.problem_instance.config.fuel_consumption_rate
        reachable_stations = []

        for cs in self.problem_instance.charging_stations:
            if cs.distance_to(cust) <= max_dist and cust.id != cs.id:
                reachable_stations.append(cs)

        return reachable_stations

    def calculate_demand(self, route):
        arrival_times = self.calculate_arrival_times(route)
        demand = []

        i = 0
        for v in route[1:]:
            r = self.problem_instance.vertices[v]
            if type(r) is CharingStation:
                r.demand = 0
            if type(r) is Customer:
                r.demand = (48 - r.stock_at_call_time + (arrival_times[i] - r.call_time) / 30)*0.75
            if r == self.problem_instance.depot:
                r.demand = 0
            i += 1

            demand.append(r.demand)
        return demand

    def calculate_arrival_times(self, route):
        now_energy = self.problem_instance.config.now_energy
        tank_capacity = self.problem_instance.config.tank_capacity

        velocity = self.problem_instance.config.velocity
        fuel_consumption_rate = self.problem_instance.config.fuel_consumption_rate
        charging_rate = self.problem_instance.config.charging_rate

        last_position = self.problem_instance.depot
        time = self.problem_instance.depot.call_time + self.problem_instance.depot.service_time

        arrival_times = []

        for v in route[1:]:
            target = self.problem_instance.vertices[v]
            d = last_position.distance_to(target)
            time += d / velocity

            arrival_times.append(time)

            now_energy -= d * fuel_consumption_rate

            if type(target) is Customer:
                time += target.service_time
            elif type(target) is CharingStation:
                time += (tank_capacity - now_energy) * charging_rate

            last_position = target

        return arrival_times

    def is_nc_feasible(self, route):
        """
        slow implementation for checking the feasibility of a route
        :param route: route to check
        :return: returns True if the route is feasible
        """
        self.calculate_demand(route)
        load_capacity = self.problem_instance.config.payload_capacity
        velocity = self.problem_instance.config.velocity

        last_position = self.problem_instance.depot
        time = self.problem_instance.depot.call_time + self.problem_instance.depot.service_time

        for v in route[1:]:
            target = self.problem_instance.vertices[v]
            d = last_position.distance_to(target)
            time += d / velocity

            if time > target.due_date:

                return False

            if type(target) is Customer:
                load_capacity -= target.demand
                time += target.service_time

            if load_capacity < 0:

                return False

            last_position = target

        return True

    def calculate_route_distance(self, route):

        last_pos = self.problem_instance.depot
        dist = 0

        for r in route:
            v = self.problem_instance.vertices[r]
            dist += v.distance_to(last_pos)
            last_pos = v

        return dist

    def calculate_total_distance(self, route):
        last = None
        dist = 0

        for t in route:
            if last is not None:
                dist += last.distance_to(t)
            last = t

        return dist

    def calculate_total_cost(self, state):
        total_cost = 0
        for r in state:
            total_cost += self.calculate_route_cost(r)
        return total_cost

    def calculate_nc_route_cost(self, route):
        dist_cost = self.calculate_route_distance(route)
        # print(f"dist{dist_cost}")
        arrival_times = self.calculate_arrival_times(route)
        time_cost = 0
        for i in range(1, len(route)):
            # print(route[i])
            target = self.problem_instance.vertices[route[i]]
            if type(target) is CharingStation:
                time_cost += 0
            if type(target) is Customer:
                time_cost += target.due_date-arrival_times[i-1]
            # print(f"time{time_cost}")
        route_cost = dist_cost + 0.5 * time_cost
        # print(route_cost)
        return route_cost

    def calculate_route_cost(self, route):
        route_cost = 0
        if not self.is_feasible(route):
            route_cost = float('inf')
        else:
            dist_cost = self.calculate_route_distance(route)
            # print(dist_cost)
            arrival_times = self.calculate_arrival_times(route)
            time_cost = 0
            for i in range(1, len(route)):
                # print(route[i])
                target = self.problem_instance.vertices[route[i]]
                if type(target) is CharingStation:
                    time_cost += 0
                if type(target) is Customer:
                    time_cost += target.due_date-arrival_times[i-1]
                # print(time_cost)
            route_cost = dist_cost + 0.5 * time_cost + 800

        return route_cost

    def energy_violated(self,route):
        tank_capacity = self.problem_instance.config.tank_capacity
        now_energy = self.problem_instance.config.now_energy
        fuel_consumption_rate = self.problem_instance.config.fuel_consumption_rate
        last_position = self.problem_instance.depot

        for v in route[1:]:
            target = self.problem_instance.vertices[v]
            d = last_position.distance_to(target)
            now_energy -= d * fuel_consumption_rate

            if now_energy < 0:
                print("energy_violated")
                return True

            if type(target) is CharingStation:
                now_energy = tank_capacity

            last_position = target

        return False

    def find_operate(self, route):
        while self.energy_violated(route):
            break
        return None

    def find_optimal_charging_station_insertion(self, route):
        # print(route)
        best_insertion_point = None
        best_station = None
        min_route_cost = float('inf')
        # if route == ['D0', 'C4', 'C6', 'C1', 'D0']:
        #     print("real route is found")
        #     for i in range(1, len(route)):
        #         v = self.problem_instance.vertices[route[i - 1]]
        #         if not isinstance(v, CharingStation):
        #
        #             reachable_stations = self.get_reachable_charging_stations(
        #                 self.problem_instance.vertices[route[i - 1]],
        #                 self.calculate_remaining_tank_capacity(route[:i]))
        #
        #             if reachable_stations is None:
        #                 continue
        #
        #             else:
        #                 for j in reachable_stations:
        #                     temp_route = route[:i] + [j.id] + route[i:]
        #                     # print(temp_route, f"可行", self.is_feasible(temp_route))
        #                     # 判断插入后路径是否可行
        #                     if self.is_feasible(temp_route):
        #
        #                         # 计算插入后的总成本
        #                         route_cost = self.calculate_route_cost(temp_route)
        #                         # Tr = self.calculate_arrival_times(temp_route)
        #                         # print(Tr)
        #                         # print(temp_route)
        #                         # print(route_cost)
        #                         # 更新最优插入点和总成本
        #                         if route_cost < min_route_cost:
        #                             min_route_cost = route_cost
        #                             best_insertion_point = i
        #                             best_station = j
        # # 遍历路径中的每个顾客之间的位置作为插入点
        # else:
        for i in range(1, len(route)):
            v = self.problem_instance.vertices[route[i-1]]
            if not isinstance(v, CharingStation):

                reachable_stations = self.get_reachable_charging_stations(self.problem_instance.vertices[route[i - 1]],
                                                                          self.calculate_remaining_tank_capacity(route[:i]))
                if reachable_stations is None:
                    continue

                else:
                    for j in reachable_stations:
                        temp_route = route[:i] + [j.id] + route[i:]

                        # 判断插入后路径是否可行
                        if self.is_feasible(temp_route):

                            # 计算插入后的总成本
                            route_cost = self.calculate_route_cost(temp_route)
                            # Tr = self.calculate_arrival_times(temp_route)
                            # print(Tr)
                            # print(temp_route)
                            # print(route_cost)
                            # 更新最优插入点和总成本
                            if route_cost < min_route_cost:
                                min_route_cost = route_cost
                                best_insertion_point = i
                                best_station = j
        # print(best_insertion_point, best_station, min_route_cost)
        return best_insertion_point, best_station, min_route_cost

    def make_route_feasible_and_best(self, route):
        best_insertion_point, best_station, min_route_cost = self.find_optimal_charging_station_insertion(route)
        if best_station == None:
            return None
        best_feasible_route = route[:best_insertion_point]+[best_station.id]+route[best_insertion_point:]

        # print(best_feasible_route)
        if not self.is_feasible(best_feasible_route):
            # print("NO")
            return None
        else:
            # print("YES")
            # print(best_feasible_route)
            return best_feasible_route

    def nc_feasible(self, state):
        for route in state:
            if self.is_nc_feasible(route):
                continue
            else:
                return False
        return True

    def state_feasible(self, state):
        for idx, route in enumerate(state):
            self._set_vehicle_energy(idx)
            if not self.is_feasible(route):
                return False
        return True
