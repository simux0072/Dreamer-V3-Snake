import random
import numpy

DIRECTIONS = {
    0: [0, -1],
    1: [1, 0],
    2: [0, 1],
    3: [-1, 0]
}

MAP_SIZE = [22, 22]

class Snake:
    def __init__(self):
        self.head = Body_Head()
        self.body_parts: list[Body_Part] = []
        self.food = Food()
        self.food.update_food_location([self.head])

    def update(self):
        if len(self.body_parts) != 0:
            ...
        
    def update_map(self):
        current_map = numpy.zeros(MAP_SIZE, dtype=int)
        current_map[self.head.coords[1]][self.head.coords[0]] = 1
        for body_part in self.body_parts:
            current_map[body_part.coords[1]][body_part.coords[0]] = 2
        current_map[self.food.coords[1]][self.food.coords[0]] = 3
        return current_map
        
        
class Body_Part:
    def __init__(self, prev_part=None):
        self.coords: list[int, int] = []
        self.prev_part:Body_Part = prev_part
    
    def update_part(self):
        self.coords = [self.prev_part.coords[0], self.prev_part.coords[1]]

class Body_Head:
    def __init__(self):
        self.coords = [int(MAP_SIZE[0]/2), int(MAP_SIZE[1]/2)]
        self.direction = 1
    
    def update_head(self, input: int):
        self.direction = (self.direction + input) % 4
        self.coords = [self.coords[0] + DIRECTIONS[self.direction][0], self.coords + DIRECTIONS[self.direction][1]]
        
class Food:
    def __init__(self):
        self.coords = []
    
    def update_food_location(self, body_parts: list[Body_Part]):
        self.coords = [random.randint(1, MAP_SIZE[0] - 1), random.randint(1, MAP_SIZE[1] - 1)]
        for part in body_parts:
            if part.coords == self.coords:
                self.update_food_location(body_parts)
                break