from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from evrptw_solver import (
    MultiPeriodRoutingProblem,
    PeriodData,
    RoutingProblemConfiguration,
    RoutingProblemInstance,
    Route,
)
from targets import Target, CharingStation, Customer


def read_instance_period_data(file: str):
    """Return the depot, charging stations and per-period customer lists.

    Parameters
    ----------
    file:
        Path to the instance file that should be parsed.

    Returns
    -------
    dict
        Dictionary containing the depot (``depot``), the list of charging
        stations (``charging_stations``), all unique customers appearing in the
        instance (``customers``) and a list that groups the customers for each
        period (``period_customers``).
    """

    parsed_instance = load_multi_period_instance(file)

    unique_customers = []
    seen_ids = set()
    for period in parsed_instance.periods:
        for customer in period.customers:
            if customer.id not in seen_ids:
                unique_customers.append(customer)
                seen_ids.add(customer.id)

    period_customers = [list(period.customers) for period in parsed_instance.periods]

    return {
        "depot": parsed_instance.depot,
        "charging_stations": list(parsed_instance.fuel_stations),
        "customers": unique_customers,
        "period_customers": period_customers,
    }


def _parse_target_line(line: str) -> Tuple[str, Target]:
    tokens = line.split()
    if len(tokens) < 8:
        raise ValueError(f"Malformed target line: '{line}'")

    idx = int(tokens[0][1:])
    x_coord = float(tokens[2])
    y_coord = float(tokens[3])
    stock = int(float(tokens[4]))
    call_time = int(float(tokens[5]))
    due_date = int(float(tokens[6]))
    service_time = int(float(tokens[7]))

    if tokens[1] == 'd':
        return 'depot', Target(tokens[0], idx, x_coord, y_coord, stock, call_time, due_date, service_time)
    if tokens[1] == 'f':
        return 'station', CharingStation(tokens[0], idx, x_coord, y_coord, stock, call_time, due_date, service_time)
    if tokens[1] == 'c':
        return 'customer', Customer(tokens[0], idx, x_coord, y_coord, stock, call_time, due_date, service_time)

    raise ValueError(f"Unknown target type token '{tokens[1]}' in line: '{line}'")


@dataclass(frozen=True)
class VehicleConfig:
    """Vehicle configuration shared by all periods of an instance."""

    tank_capacity: float
    now_energy: Union[float, List[float]]
    load_capacity: float
    fuel_consumption_rate: float
    charging_rate: float
    velocity: float


@dataclass(frozen=True)
class ParsedPeriod:
    """Container for customers that belong to a single period."""

    name: str
    customers: List[Customer]


@dataclass(frozen=True)
class MultiPeriodInstance:
    """Representation of an instance that may contain multiple periods."""

    source: str
    vehicle: VehicleConfig
    depot: Target
    fuel_stations: List[CharingStation]
    periods: List[ParsedPeriod]


def _parse_energy_values(raw_value: str) -> Union[float, List[float]]:
    """Parse the energy configuration value which may contain multiple entries."""

    cleaned = raw_value.replace(';', ',')
    values = [float(value.strip()) for value in cleaned.split(',') if value.strip()]

    if not values:
        raise ValueError("No initial energy values provided in instance file.")

    if len(values) == 1:
        return values[0]

    return values


def _read_targets_block(file_obj) -> Tuple[Target, List[Customer], List[CharingStation]]:
    """Read the first block containing depot, stations and optional base customers."""

    customers: List[Customer] = []
    fuel_stations: List[CharingStation] = []
    depot: Optional[Target] = None

    while True:
        pos = file_obj.tell()
        line = file_obj.readline()
        if not line:
            break

        stripped = line.strip()
        if not stripped:
            break

        if stripped.lower().startswith('period'):
            file_obj.seek(pos)
            break

        label, target = _parse_target_line(stripped)
        if label == 'depot':
            depot = target
        elif label == 'station':
            fuel_stations.append(target)
        elif label == 'customer':
            customers.append(target)

    if depot is None:
        raise ValueError("Instance file is missing a depot definition.")

    return depot, customers, fuel_stations


def _read_configuration_line(file_obj, description: str) -> str:
    """Return the raw configuration value for ``description``."""

    while True:
        line = file_obj.readline()
        if not line:
            raise ValueError(f"Unexpected end of file while reading {description}.")

        stripped = line.strip()
        if not stripped:
            continue

        if '/' not in stripped:
            raise ValueError(f"Malformed configuration line for {description}: '{stripped}'")

        value = stripped.split('/', 1)[1].strip()
        # Some instance files keep a trailing slash after the numeric value (e.g. "1500/").
        # Trim it alongside optional inline comments so downstream float conversions succeed.
        if '/' in value:
            value = value.split('/', 1)[0].strip()
        if '#' in value:
            value = value.split('#', 1)[0].strip()

        return value


def _read_vehicle_configuration(file_obj) -> VehicleConfig:
    """Parse the configuration section that contains vehicle parameters."""

    tank_capacity = float(_read_configuration_line(file_obj, 'vehicle tank capacity'))
    now_energy_raw = _read_configuration_line(file_obj, 'vehicle initial energy')
    now_energy = _parse_energy_values(now_energy_raw)
    load_capacity = float(_read_configuration_line(file_obj, 'vehicle load capacity'))
    fuel_consumption_rate = float(_read_configuration_line(file_obj, 'fuel consumption rate'))
    charging_rate = float(_read_configuration_line(file_obj, 'charging rate'))
    velocity = float(_read_configuration_line(file_obj, 'vehicle velocity'))

    return VehicleConfig(
        tank_capacity=tank_capacity,
        now_energy=now_energy,
        load_capacity=load_capacity,
        fuel_consumption_rate=fuel_consumption_rate,
        charging_rate=charging_rate,
        velocity=velocity,
    )


def _parse_period_sections(raw_text: str) -> List[Tuple[Optional[str], List[Customer]]]:
    """Parse additional period blocks from the remaining part of the instance file."""

    if not raw_text:
        return []

    periods: List[Tuple[Optional[str], List[Customer]]] = []
    current_customers: List[Customer] = []
    current_name: Optional[str] = None

    def flush_period() -> None:
        nonlocal current_customers, current_name
        if current_customers:
            periods.append((current_name, list(current_customers)))
        current_customers = []
        current_name = None

    for raw_line in raw_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        normalized = stripped.strip('[]')
        normalized = normalized.lstrip('#').strip()
        if normalized.lower().startswith('period'):
            flush_period()
            current_name = normalized
            continue

        label, target = _parse_target_line(stripped)
        if label != 'customer':
            raise ValueError(
                "Only customer (type 'c') entries are allowed inside period-specific blocks."
            )

        current_customers.append(target)

    flush_period()
    return periods


def _parse_instance(file: str) -> MultiPeriodInstance:
    """Read ``file`` and return a structured multi-period instance description."""
    with open(file) as f:
        f.readline()  # ignore header
        depot, base_customers, fuel_stations = _read_targets_block(f)
        vehicle_config = _read_vehicle_configuration(f)
        remaining = f.read()

    periods: List[ParsedPeriod] = []

    if base_customers:
        periods.append(ParsedPeriod(name='Period 1', customers=list(base_customers)))

    parsed_additional_periods = _parse_period_sections(remaining)
    next_index = len(periods) + 1
    for maybe_name, customers in parsed_additional_periods:
        period_name = maybe_name or f'Period {next_index}'
        periods.append(ParsedPeriod(name=period_name, customers=list(customers)))
        next_index += 1

    if not periods:
        periods.append(ParsedPeriod(name='Period 1', customers=[]))

    return MultiPeriodInstance(
        source=str(file),
        vehicle=vehicle_config,
        depot=depot,
        fuel_stations=fuel_stations,
        periods=periods,
    )


def load_multi_period_instance(file: str) -> MultiPeriodInstance:
    """Public helper returning the parsed multi-period instance description."""
    return _parse_instance(file)


def _build_problem_configuration(vehicle: VehicleConfig) -> RoutingProblemConfiguration:
    return RoutingProblemConfiguration(
        vehicle.tank_capacity,
        vehicle.now_energy,
        vehicle.load_capacity,
        vehicle.fuel_consumption_rate,
        vehicle.charging_rate,
        vehicle.velocity,
    )

def load_problem_instance(file, period_index: int = 0):
    parsed = _parse_instance(file)
    if period_index < 0 or period_index >= len(parsed.periods):
        raise IndexError(
            f"Requested period_index {period_index} but only {len(parsed.periods)} period(s) available."
        )
    config = _build_problem_configuration(parsed.vehicle)
    selected_customers = parsed.periods[period_index].customers

    return RoutingProblemInstance(
        config,
        parsed.depot,
        list(selected_customers),
        list(parsed.fuel_stations),
    )

def load_multi_period_problem_instance(file):
    parsed = _parse_instance(file)

    config = _build_problem_configuration(parsed.vehicle)
    periods = [
        PeriodData(name=period.name, customers=list(period.customers))
        for period in parsed.periods
    ]

    return MultiPeriodRoutingProblem(
        config,
        parsed.depot,
        list(parsed.fuel_stations),
        periods,
    )


def load_solution(file):
    with open(file, 'r') as f:
        cost = float(f.readline())
        solution = []

        for line in f:
            # [:-1] is needed, otherwise the last element of the list would be an empty string
            route = line.split(', ')[:-1]

            solution.append(route)

        return cost, solution


def write_solution_to_file(file, distance, routes):
    with open(file, 'w') as f:
        f.write('{0}\n'.format(round(distance, 3)))

        for r in routes:
            if type(r) is Route:
                if len(r.route) > 2:
                    for t in r.route:
                        f.write('{0}, '.format(t.id))
                    f.write('\n')
            elif type(r) is list:
                if len(r) > 2:
                    for v in r:
                        f.write('{0}, '.format(v))
                    f.write('\n')


def write_solution_stats_to_file(file, stat, style='csv'):
    with open(file, 'w') as result_file:
        if style == 'csv':
            result_file.writelines('testcase;distance;runtime (in ms)\n')
            for r in stat:
                result_file.write('{0} ; {1} ; {2}\n'.format(r[0], round(r[1], 3), round(r[2], 3)))
        elif style == 'latex':
            result_file.write("\\begin{table}[t]\n")
            result_file.write("\\label{tab:result}\n")
            result_file.write("\\begin{tabular}{lrr}\n")
            result_file.write("\\toprule\n")
            result_file.write("instance & distance & runtime (in ms) \\\\ \n")
            result_file.write("\\midrule")

            for r in stat:
                result_file.write('{0} & {1} & {2} \\\\ \n'.format(r[0], round(r[1], 3), round(r[2], 3)))

            result_file.write("\\bottomrule \n")
            result_file.write("\\end{tabular} \n")
            result_file.write("\\end{table} \n")


def write_meta_heuristic_result_statistic_to_file(file, dist_stat, time_stat):
    with open(file, 'w') as f:
        f.writelines('file;distance(avg);distance(std);distance(min);time(avg);time(std);time(max)\n')

        instances = list(dist_stat.keys())
        instances.sort()

        for i in instances:
            f.writelines(
                '{0};{1};{2};{3};{4};{5};{6}\n'.format(i, np.average(dist_stat[i]), np.std(dist_stat[i]),
                                                     np.min(dist_stat[i]),
                                                     np.average(time_stat[i]), np.std(time_stat[i]),
                                                     np.max(time_stat[i])))
