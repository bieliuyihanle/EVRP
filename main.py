import functools
import timeit
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import pandas as pd


from evrptw_meta import VariableNeighbourhoodSearch, SimulatedAnnealing1, Adaptive, FCFS
from evrptw_solver import EVRPTWSolver
from evrptw_utilities import load_problem_instance, load_solution, write_solution_to_file, write_solution_stats_to_file, \
    write_meta_heuristic_result_statistic_to_file
from heuristics.construction.beasley_heuristic import BeasleyHeuristic, process_route,  k_nearest_neighbor_min_due_date, \
    k_nearest_neighbor_min_ready_time, nearest_neighbor_tolerance_min_due_date, \
    nearest_neighbor_tolerance_min_ready_time

RESULT_STATISTICS_FILENAME = 'ex1_result_1126205.csv'
RESULT_STATISTICS_LATEX_TABLE = 'ex1_result_1126205.tex'

MAX_ITERATIONS = 10


def main():
    # best_score, best_heuristic, best_param = find_best_heuristic_setting_experiment()

    # CACHING OF BEST RESULTS
    best_score = 0
    best_heuristic = k_nearest_neighbor_min_due_date
    best_param = 3

    meta_heuristics_params = {
        'FCFS': {
            'class': FCFS,
            'params': lambda problem_instance, solution, cost, file, i: {}
        },
        'SimulatedAnnealing1': {
            'class': SimulatedAnnealing1,
            'params': lambda problem_instance, solution, cost, file, i: (
                solution, cost, 400, 0.9, '{0}_{1}'.format(file, i)
            )
        },
        'VariableNeighbourhoodSearch': {
            'class': VariableNeighbourhoodSearch,
            'params': lambda problem_instance, solution, cost, file, i: {
                'solution': solution,
                'cost': cost
            }
        },
        'Adaptive': {
            'class': Adaptive,
            'params': lambda problem_instance, solution, cost, file, i: (
                solution, cost, '{0}_{1}'.format(file, i)
            )
        }
    }

    print("The best score of {0} was achieved with {1} and parameter {2}".format(best_score, best_heuristic,
                                                                                 round(best_param, 3)))

    print()
    print("============================================")
    print("Generate initial solutions...")
    print("============================================")
    print()

    construction_heuristic = BeasleyHeuristic(process_route, best_heuristic, [best_param])
    test_case_statistics = []
    solver = EVRPTWSolver(construction_heuristic)
    for file in listdir('_problem_instances/exercise_instances/'):
        if file.endswith('.txt'):
            print('process file {0}'.format(file))
            print('load problem instance...')
            problem_instance = load_problem_instance('_problem_instances/exercise_instances/' + file)
            print('generate routes...')
            duration = timeit.timeit(functools.partial(solver.solve, problem_instance), number=1) * 1000
            cost, solution = solver.solve(problem_instance)
            test_case_statistics.append((file, cost, duration))
            print('write results to file ...')
            write_solution_to_file("_problem_solutions/solution_{0}".format(file), cost, solution)
            print()

            test_case_statistics.sort(key=lambda x: x[0])
            write_solution_stats_to_file(RESULT_STATISTICS_FILENAME, test_case_statistics)
            write_solution_stats_to_file(RESULT_STATISTICS_LATEX_TABLE, test_case_statistics, style='latex')

    print()
    print("============================================")
    print("Apply meta-heuristic to improve solutions...")
    print("============================================")
    print()

    dist_statistic = dict()
    time_statistic = dict()
    convergence_data = dict()  # 新增：用于存储每个算法和文件的收敛数据
    # 记录结果
    all_results = []
    convergence_records = []  # 用于保存所有运行的详细收敛数据
    all_convergence_runs = []

    for file in listdir('_problem_instances/exercise_instances/'):
        if file.endswith('.txt'):
            print("process file {0}".format(file))
            problem_instance = load_problem_instance('_problem_instances/exercise_instances/' + file)
            cost, solution = load_solution('_problem_solutions/solution_{0}'.format(file))

            print('start to improve the routes...')

            customers_id = [k.id for k in problem_instance.customers]

            convergence_data[file] = {}

            for meta_name, meta_data in meta_heuristics_params.items():
                meta_heuristic_class = meta_data['class']
                params_func = meta_data['params']

                # 为每个文件和算法初始化收敛数据
                convergence_data[file][meta_name] = {'costs': [], 'times': []}

                dist_statistic[file] = []
                time_statistic[file] = []
                print(f"Meta-heuristic {meta_name} is used")
                for i in range(10):  # 进行20次
                    print(i+1)
                    params = params_func(problem_instance, solution, cost, file, i)
                    # 如果参数是元组，直接按顺序解包；否则使用关键字参数解包
                    if isinstance(params, tuple):
                        meta_heuristic = meta_heuristic_class(problem_instance, *params)
                    else:
                        meta_heuristic = meta_heuristic_class(problem_instance, **params)

                    new_cost, new_solution, costs, times = meta_heuristic.improve_solution()

                    # 记录运行时间
                    duration = timeit.timeit(functools.partial(meta_heuristic.improve_solution), number=1)

                    dist_statistic[file].append(new_cost)
                    time_statistic[file].append(duration)

                    # 保存结果到列表
                    all_results.append((file, meta_name, i+1, new_cost, duration))
                    # print(all_results)
                    convergence_data[file][meta_name]['costs'].append(costs)
                    convergence_data[file][meta_name]['times'].append(times)

                    all_convergence_runs.append({
                        "File": file,
                        "Algorithm": meta_name,
                        "Run": i + 1,
                        "Final_Cost": new_cost,
                        "Costs": costs,  # 这是一个列表 [c1, c2, ...]
                        "Times": times  # 这是一个列表 [t1, t2, ...]
                    })

                    convergence_records.append({
                        "File": file,
                        "Algorithm": meta_name,
                        "Run": i + 1,  # 运行次数 i，从1开始
                        "Costs": costs,
                        "Times": times
                    })

                print(f"Meta-heuristic {meta_name} applied 20 times for file {file}")

    convergence_df = pd.DataFrame(all_convergence_runs)
    convergence_df.to_csv("all_convergence_runs.csv", index=False)

    # 保存结果到 Excel 文件
    df = pd.DataFrame(all_results, columns=['File', 'MetaHeuristic', 'Iteration', 'Cost', 'Duration'])
    df.to_excel('meta_heuristics_results with 1s and 30threshold.xlsx', index=False)

    # 保存 convergence_data 的详细记录到 Excel 文件
    convergence_df = pd.DataFrame(convergence_records)
    convergence_df.to_excel("convergence_data.xlsx", index=False)

    write_meta_heuristic_result_statistic_to_file('meta_heuristic_results.csv with 1s and 30threshold', dist_statistic, time_statistic)
    print("done")

    # #收敛曲线绘制
    # # 定义每个算法的颜色
    # algorithm_colors = {
    #     "SimulatedAnnealing1": "#F28080",  # SA -> red
    #     "VariableNeighbourhoodSearch": "#5861AC",  # VNS -> green
    #     "Adaptive": "#E5AB02"  # ALNS -> blue
    # }
    #
    # # 使用缩写名称代替完整名称
    # label_mapping = {
    #     "SimulatedAnnealing1": "SA",
    #     "VariableNeighbourhoodSearch": "VNS",
    #     "Adaptive": "ALNS"
    # }
    #
    # # 定义实例名称映射和标签映射
    # instance_mapping = {
    #     "Instance_26.txt": ("(d) T20I6", "Instance_26.txt"),
    #     "Instance_46.txt": ("(a) T30I6", "Instance_46.txt"),
    #     "Instance_66.txt": ("(b) T40I6", "Instance_66.txt"),
    #     "Instance_86.txt": ("(c) T50I6", "Instance_86.txt")
    # }
    #
    # # 设置全局字体为 Times New Roman
    # plt.rcParams['font.family'] = 'Times New Roman'
    #
    # # 创建2x2的子图布局
    # fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 设置图像大小
    #
    # # 存储数据的字典
    # data_to_save = {
    #     "best_index": [],  # 用于存储所有best_times数据
    # }
    #
    # # 筛选最佳结果并绘制收敛曲线
    # for i, (file, (instance_name, _)) in enumerate(instance_mapping.items()):
    #     ax = axs[i // 2, i % 2]  # 获取2x2网格中的位置
    #     algorithms_data = convergence_data.get(file, {})
    #
    #     for meta_name, data in algorithms_data.items():
    #         # 找到每个算法效果最好的一组数据，即最终目标函数值最小的一次运行
    #         min_cost_index = None
    #         min_cost = float('inf')
    #
    #         for idx, costs in enumerate(data['costs']):
    #             final_cost = costs[-1] if costs else float('inf')
    #             if final_cost < min_cost:
    #                 min_cost = final_cost
    #                 min_cost_index = idx
    #
    #         # 获取最佳的 times 和 costs 数据
    #         if min_cost_index is not None:
    #             best_times = data['times'][min_cost_index]
    #             best_costs = data['costs'][min_cost_index]
    #             # 将数据保存到字典中
    #             data_to_save["best_index"].append(min_cost_index)
    #             # 获取对应颜色
    #             color = algorithm_colors.get(meta_name, "#000000")  # 默认为黑色
    #             label = label_mapping.get(meta_name, meta_name)
    #             ax.plot(best_times, best_costs, label=label, color=color)
    #
    #     # 将数据保存为CSV格式
    #     df = pd.DataFrame(data_to_save)
    #     df.to_csv("best_index.csv", index=False)
    #     print("Data saved to index.csv")
    #
    #
    #     # 设置坐标标签和实例名称
    #     ax.set_xlabel("The computing time (ms)", fontsize=12)
    #     ax.set_ylabel("Objective value", fontsize=12)
    #
    #     # 使用 `fig.text()` 在横坐标下方添加实例名称
    #     ax.text(0.5, -0.25, instance_name, ha='center', va='top', transform=ax.transAxes, fontsize=12)
    #
    #     # 获取当前的默认横坐标刻度
    #     xticks = ax.get_xticks()
    #
    #     # 将刻度乘以 1000 转换为毫秒单位
    #     new_xticks = xticks * 1000
    #
    #     # 设置新的横坐标标签（单位为毫秒）
    #     ax.set_xticklabels([str(int(tick)) for tick in new_xticks], fontsize=12)
    #
    #     # 横坐标标签旋转 90 度
    #     ax.tick_params(axis='x', rotation=270)
    #
    #     ax.legend(loc='upper right')  # 将图例放置在右上角
    #     ax.grid(False)  # 显示网格
    #
    # # 调整子图之间的间距
    # plt.subplots_adjust(hspace=0.4, wspace=0.3)
    #
    # # 保存并显示图像
    # plt.savefig("convergence_curves_overall.pdf", bbox_inches="tight")
    # plt.show()


def find_best_heuristic_setting_experiment():
    print('NN tolerance heuristic with min due date:')
    print('================================')
    print('tolerance; average score')
    best_param = -1
    best_heuristic = None
    best_score = 0

    nnt_deadline_results = []
    nnt_readytime_results = []
    knn_deadline_results = []
    knn_readytime_results = []

    for tolerance in np.arange(1, 3, 0.1):
        distances = []
        construction_heuristic = BeasleyHeuristic(nearest_neighbor_tolerance_min_due_date, [round(tolerance, 2)])
        test_case_statistics = []
        solver = EVRPTWSolver(construction_heuristic)

        for file in listdir('_problem_instances/exercise_instances/'):
            if file.endswith('.txt'):
                problem_instance = load_problem_instance('_problem_instances/exercise_instances/' + file)
                distance, solution = solver.solve(problem_instance)
                distances.append(distance)

        if best_score == 0 or best_score > np.mean(distances):
            best_score = np.mean(distances)
            best_heuristic = nearest_neighbor_tolerance_min_due_date
            best_param = round(tolerance, 2)

        nnt_deadline_results.append(np.mean(distances))
        print("{0:.2f}; {1:.2f}".format(tolerance, np.mean(distances)))

    print('NN tolerance heuristic with min ready time:')
    print('================================')
    print('tolerance; average score')

    for tolerance in np.arange(1, 3, 0.1):
        distances = []
        construction_heuristic = BeasleyHeuristic(nearest_neighbor_tolerance_min_ready_time, [round(tolerance, 2)])
        solver = EVRPTWSolver(construction_heuristic)

        for file in listdir('_problem_instances/exercise_instances/'):
            if file.endswith('.txt'):
                problem_instance = load_problem_instance('_problem_instances/exercise_instances/' + file)

                distance, solution = solver.solve(problem_instance)
                write_solution_to_file("_problem_solutions/solution_{0}".format(file), distance, solution)
                distances.append(distance)

        if best_score == 0 or best_score > np.mean(distances):
            best_score = np.mean(distances)
            best_heuristic = nearest_neighbor_tolerance_min_ready_time
            best_param = round(tolerance, 2)

        nnt_readytime_results.append(np.mean(distances))
        print("{0:.2f}; {1:.2f}".format(tolerance, np.mean(distances)))

    plt.title('NN heuristic with tolerance')
    deadline, = plt.plot(np.arange(1, 3, 0.1), nnt_deadline_results, label='deadline minimized')
    readytime, = plt.plot(np.arange(1, 3, 0.1), nnt_readytime_results, label='readytime minimized')
    plt.xlabel('tolerance')
    plt.ylabel('average score')
    plt.legend([deadline, readytime], ['deadline minimized', 'readytime minimized'])
    plt.show()

    print('kNN heuristic with min due date:')
    print('================================')
    print('k; average score')
    for k in range(1, 10):
        distances = []
        construction_heuristic = BeasleyHeuristic(k_nearest_neighbor_min_due_date, [round(k, 2)])
        solver = EVRPTWSolver(construction_heuristic)

        for file in listdir('_problem_instances/exercise_instances/'):
            if file.endswith('.txt'):
                problem_instance = load_problem_instance('_problem_instances/exercise_instances/' + file)

                distance, solution = solver.solve(problem_instance)
                write_solution_to_file("_problem_solutions/solution_{0}".format(file), distance, solution)
                distances.append(distance)

        if best_score == 0 or best_score > np.mean(distances):
            best_score = np.mean(distances)
            best_heuristic = k_nearest_neighbor_min_due_date
            best_param = k

        knn_deadline_results.append(np.mean(distances))
        print("{0:.2f}; {1:.2f}".format(k, np.mean(distances)))

    print('kNN heuristic with min ready time:')
    print('================================')
    print('k; average score')
    for k in range(1, 10):
        distances = []
        construction_heuristic = BeasleyHeuristic(k_nearest_neighbor_min_ready_time, [round(k, 2)])
        solver = EVRPTWSolver(construction_heuristic)

        for file in listdir('_problem_instances/exercise_instances/'):
            if file.endswith('.txt'):
                problem_instance = load_problem_instance('_problem_instances/exercise_instances/' + file)
                distance, solution = solver.solve(problem_instance)
                write_solution_to_file("_problem_solutions/solution_{0}".format(file), distance, solution)
                distances.append(distance)

        if best_score == 0 or best_score > np.mean(distances):
            best_score = np.mean(distances)
            best_heuristic = k_nearest_neighbor_min_ready_time
            best_param = k

        knn_readytime_results.append(np.mean(distances))
        print("{0:.2f}; {1:.2f}".format(k, np.mean(distances)))

    plt.title('kNN heuristic')
    deadline, = plt.plot(range(1, 10), knn_deadline_results, label='deadline minimized')
    readytime, = plt.plot(range(1, 10), knn_readytime_results, label='readytime minimized')
    plt.xlabel('k')
    plt.ylabel('average score')
    plt.legend([deadline, readytime], ['deadline minimized', 'readytime minimized'])
    plt.show()

    return best_score, best_heuristic, best_param


if __name__ == "__main__":
    main()
