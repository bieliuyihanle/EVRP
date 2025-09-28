import math as m

class Target:
    def __init__(self, id, idx, x, y, stock_at_call_time, call_time, due_date, service_time):
        self.id = id
        self.idx = idx
        self.x = x
        self.y = y
        self.stock_at_call_time = stock_at_call_time
        self.call_time = call_time
        self.due_date = due_date
        self.service_time = service_time

    def distance_to(self, compared_target):
        # print(f"dist between", self, "and", compared_target, "is", abs(self.x - compared_target.x) + abs(self.y - compared_target.y))
        return abs(self.x - compared_target.x) + abs(self.y - compared_target.y)

    def get_coordinates(self):
        return self.x, self.y

    def __str__(self):
        return "type: {0}, id: {1}, x: {2}, y: {3}".format(type(self), self.id, self.x, self.y)

    def __repr__(self):
        return "type: {0}, id: {1}, x: {2}, y: {3}".format(type(self), self.id, self.x, self.y)


class Customer(Target):
    def __init__(self, id, idx, x, y, stock_at_call_time, call_time, due_date, service_time):
        super(Customer, self).__init__(id, idx, x, y, stock_at_call_time, call_time, due_date, service_time)



class CharingStation(Target):
    def __init__(self, id, idx, x, y, stock_at_call_time, call_time, due_date, service_time):
        super(CharingStation, self).__init__(id, idx, x, y, stock_at_call_time, call_time, due_date, service_time)
