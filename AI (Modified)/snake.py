import random
import pygame
from pygame import Vector2
from pygame import time as pygame_time

WIDTH, HEIGHT = 1400, 900
FPS = 10

class player():
    # Body data
    body_list = []
    body_num = 0

    # Head data
    points = 0
    head_coordinates = Vector2(0, 0)
    head_move_queue = []
    direction = 3
    head_speed = Vector2(1, 0)
    frame_count = Vector2(0, 0)

    # Head/Body "to draw" data
    head_color = (0, 146, 15)
    head_border_Radius = 3
    body_color = (0, 77, 8)
    body_border_Radius =  0
    length = Vector2(50, 50)
    
    # Updates coordinates for head/body
    def update(self, frame_count, speed, coordinates):
        frame_count.x += speed.x
        frame_count.y += speed.y
        coordinates.x = frame_count.x * self.length.x
        coordinates.y = frame_count.y * self.length.y

    # Handels head/body logic
    def player_logic(self, Food):
        self.update(self.frame_count, self.head_speed, self.head_coordinates)
        for i in self.body_list:
            self.update(i[2], i[1], i[0])

            # Handels logic for eating the food and creating a new body
        if self.head_coordinates == Food.coordinates:
            self.points += 1
            self.body_num += 1
            if self.body_num == 1:
                self.body_list.append([Vector2(self.head_coordinates.x - 50 * self.head_speed.x, self.head_coordinates.y - 50 * self.head_speed.y), 
                                        self.head_speed, 
                                        Vector2(self.frame_count.x - 6 * self.head_speed.x, self.frame_count.y - 6 * self.head_speed.y)])
            else:
                self.body_list.append([Vector2(self.body_list[-1][0].x - 50 * self.body_list[-1][1].x, self.body_list[-1][0].y - 50 * self.body_list[-1][1].y), 
                                        self.body_list[-1][1], 
                                        Vector2(self.body_list[-1][2].x - 6 * self.body_list[-1][1].x, self.body_list[-1][2].y - 6 * self.body_list[-1][1].y)])

        # Updates the body speed values
        if self.body_num > 0:
            for i in range(self.body_num - 1, -1, -1):
                if i == 0:
                    self.body_list[0][1] = self.head_speed
                else:
                    self.body_list[i][1] = self.body_list[i - 1][1]
        # Handels direction change from player input
        if len(self.head_move_queue) > 0:
            self.direction = self.head_move_queue.pop(0)
        if self.direction == 0:
            self.head_speed = Vector2(0, -1)
        elif self.direction == 1:
            self.head_speed = Vector2(0, 1)
        elif self.direction == 2:
            self.head_speed = Vector2(-1, 0)
        elif self.direction == 3:
            self.head_speed = Vector2(1, 0)
    
        # Checks and handles collisions between body and head
        for i in self.body_list:
            if self.head_coordinates == i[0]:
                return self.points, True
        
        # Checks if head is out of bounds
        if self.head_coordinates.x >= WIDTH or self.head_coordinates.x < 0 or self.head_coordinates.y >= HEIGHT or self.head_coordinates.y < 0:
            return self.points, True

        # Draws the objects to screen
        Food.generate(self)
        return self.points, False

class food():
    coordinates = Vector2()
    color = (205, 0, 0)
    length = Vector2(50, 50)
    border_Radius = 3
    # Function to get random coordinates for food
    def random(self):
        self.coordinates.x = random.randrange(0, WIDTH - 50, 50)
        self.coordinates.y = random.randrange(0, HEIGHT - 50, 50)

    # Function to call the random function and update the food location
    def generate(self, Player):
        while True:
            m = 0
            if self.coordinates == Player.head_coordinates:
                self.random()
            else:
                for i in Player.body_list:
                    if self.coordinates == i[0]:
                        m += 1
                        self.random()
                if m == 0:
                    break

Player = player()
Food = food()

Food.random()
Food.generate(Player)

# Main game loop
def main(direction):
    clock = pygame_time.Clock()
    run = True
    while run:           
        if direction == 3 and Player.direction != 3 and Player.direction != 2:
            Player.head_move_queue.append(3)
        elif direction == 2 and Player.direction != 2 and Player.direction != 3:
            Player.head_move_queue.append(2)
        elif direction == 0 and Player.direction != 0 and Player.direction != 1:
            Player.head_move_queue.append(0)
        elif direction == 1 and Player.direction != 1 and Player.direction != 0:
            Player.head_move_queue.append(1)
        clock.tick(FPS)
        point, game_over = Player.player_logic()
        print(point)
        if game_over:
            run = False
    pygame.quit()

if __name__ == "__main__":
    main()