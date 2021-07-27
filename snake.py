import pygame
import random
from pygame import Vector2, display, font
from pygame import time as pygame_time
from pygame import key
from pygame import draw
from pygame import event
from pygame.constants import K_DOWN, K_LEFT, K_RIGHT, K_SPACE, K_UP, K_d

pygame.init()


#  Global variables
WIDTH, HEIGHT = 1400, 900
FPS = 60
SCREEN = display.set_mode((WIDTH, HEIGHT))
display.set_caption("Testing game")
SCREEN_COLOR = (0, 0, 0)
Font = font.SysFont('aria', 50)

# Function to draw the Body/Head/Food to screen
def update(screen, coordinates, length, color, border_Radius):
    rect = pygame.Rect(coordinates.x, coordinates.y, length.x, length.y)
    draw.rect(screen, color, rect, border_radius=border_Radius)

#  Function to draw the food
def update_circle(color, coordinates, circle_radius):
    draw.circle(SCREEN, color, coordinates, circle_radius)

class player():
    # Body data
    body_list = []
    body_num = 0

    # Head data
    points = 0
    head_coordinates = Vector2(0, 0)
    head_move_queue = []
    direction = 3
    speed_per = 50/0.1/FPS
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
        coordinates.x = round(frame_count.x * self.speed_per, 7)
        coordinates.y = round(frame_count.y * self.speed_per, 7)

    # Handels head/body logic
    def player_logic(self, screen, Food):
        self.update(self.frame_count, self.head_speed, self.head_coordinates)
        for i in self.body_list:
            self.update(i[2], i[1], i[0])
        
        if self.head_coordinates.x % 50 == 0 and self.head_coordinates.y % 50 == 0:

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
                    pygame.quit()
        
        # Checks if head is out of bounds
        if self.head_coordinates.x >= WIDTH or self.head_coordinates.x < 0 or self.head_coordinates.y >= HEIGHT or self.head_coordinates.y < 0:
            pygame.quit()

        # Draws the objects to screen
        for i in self.body_list:
            update(screen, i[0], self.length, self.body_color, self.body_border_Radius)
        Food.generate(self)
        update(screen, self.head_coordinates, self.length, self.head_color, self.head_border_Radius)

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
        new_coordinates = Vector2(self.coordinates.x + self.length.x/2, self.coordinates.y + self.length.y/2)
        update_circle(self.color, new_coordinates, self.length.x/2)

Player = player()
Food = food()

# Function to draw to screen and call other functions
def draw_window(player, food, screen, screen_color):
    SCREEN.fill(screen_color)
    player.player_logic(screen, food)
    text = Font.render("Score: " + str(player.points), True, (255, 255, 255))
    screen.blit(text, Vector2(0, 0)) # Renders text to screen (aka score)
    display.update()

SCREEN.fill(SCREEN_COLOR)
Food.random()
Food.generate(Player)

# Main game loop
def main():
    clock = pygame_time.Clock()
    run = True
    while run:
        for Event in event.get():
            if Event.type == pygame.QUIT:
                run = False
            elif key.get_focused and Event.type == pygame.KEYDOWN:
                if Event.key == K_RIGHT:
                    try:
                        if Player.head_move_queue[-1] != 3 or Player.head_move_queue[-1] != 2:
                            Player.head_move_queue.append(3)
                    except IndexError:
                        Player.head_move_queue.append(3)
                elif Event.key == K_LEFT:
                    try:
                        if Player.head_move_queue[-1] != 2 or Player.head_move_queue[-1] != 3:
                            Player.head_move_queue.append(2)
                    except IndexError:
                        Player.head_move_queue.append(2)
                elif Event.key == K_UP:
                    try:
                        if Player.head_move_queue[-1] != 0 or Player.head_move_queue[-1] != 1:
                            Player.head_move_queue.append(0)
                    except IndexError:
                        Player.head_move_queue.append(0)
                elif Event.key == K_DOWN:
                    try:
                        if Player.head_move_queue[-1] != 1 or Player.head_move_queue[-1] != 0:
                            Player.head_move_queue.append(1)
                    except IndexError:
                        Player.head_move_queue.append(1)

        clock.tick(FPS)
        draw_window(Player, Food, SCREEN, SCREEN_COLOR)
    pygame.quit()

if __name__ == "__main__":
    main()