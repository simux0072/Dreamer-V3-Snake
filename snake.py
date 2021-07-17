import fractions
import pygame
import random
from pygame import Vector2, display
from pygame import time as pygame_time
from pygame import key
from pygame import event
from pygame import draw
from pygame.constants import K_DOWN, K_LEFT, K_RIGHT, K_SPACE, K_UP, K_d

WIDTH, HEIGHT = 1400, 900
FPS = 60
SCREEN = display.set_mode((WIDTH, HEIGHT))
display.set_caption("Testing game")
SCREEN_COLOR = (0, 0, 0)
PIX_ARR = pygame.PixelArray(SCREEN)

class player():
    points = 0
    move_queue = []
    coordinates = Vector2(0, 0)
    direction = 3
    speed_per = 25/0.1/FPS
    speed = Vector2(1, 0)
    frame_count = Vector2(0, 0)
    color = (0, 146, 15)
    length = Vector2(25, 25)
    border_Radius = 3

    def draw(self, screen, coordinates, length, color, border_Radius):
        rect = pygame.Rect(coordinates.x, coordinates.y, length.x, length.y)
        draw.rect(screen, color, rect, border_radius=border_Radius)
    
    def update(self, screen, Food):
        self.frame_count.x += self.speed.x
        self.frame_count.y += self.speed.y
        self.coordinates.x = round(self.frame_count.x * self.speed_per, 7)
        self.coordinates.y = round(self.frame_count.y * self.speed_per, 7)
        # print(self.coordinates)
        if self.coordinates.x % 25 == 0 and self.coordinates.y % 25 == 0 and len(self.move_queue) > 0:
            self.direction = self.move_queue.pop(0)
            if self.direction == 0:
                self.speed = Vector2(0, -1)
            elif self.direction == 1:
                self.speed = Vector2(0, 1)
            elif self.direction == 2:
                self.speed = Vector2(-1, 0)
            elif self.direction == 3:
                self.speed = Vector2(1, 0)
        
        if self.coordinates.x >= WIDTH or self.coordinates.x < 0 or self.coordinates.y >= HEIGHT or self.coordinates.y < 0:
            pygame.quit()

        if self.coordinates.x == Food.coordinates.x and self.coordinates.y == Food.coordinates.y:
            self.points += 1
            print(self.points)
        Food.generate(Player=self, screen=screen)
        self.draw(screen, self.coordinates, self.length, self.color, self.border_Radius)
class food():
    coordinates = Vector2()
    color = (205, 0, 0)
    length = Vector2(25, 25)
    border_Radius = 3

    def random(self):
        self.coordinates.x = random.randrange(0, WIDTH - 25, 25)
        self.coordinates.y = random.randrange(0, HEIGHT - 25, 25)

    def draw(self, screen):
        rect = pygame.Rect(self.coordinates.x, self.coordinates.y, self.length.x, self.length.y)
        draw.rect(screen, self.color, rect, border_radius=self.border_Radius)

    def generate(self, Player, screen):
        while self.coordinates.x == Player.coordinates.x and self.coordinates.y == Player.coordinates.y:
            self.random()

        self.draw(screen)

Player = player()
Food = food()

def draw_window(player, food, screen, screen_color):
    SCREEN.fill(screen_color)
    player.update(screen, food)
    display.update()

SCREEN.fill(SCREEN_COLOR)
Food.random()
Food.generate(Player, SCREEN)

def main():
    clock = pygame_time.Clock()
    ONE_AT_A_TIME = False
    run = True
    while run:
        for Event in event.get():
            if Event.type == pygame.QUIT:
                run = False
            elif key.get_focused and Event.type == pygame.KEYDOWN:
                if Event.key == K_SPACE and ONE_AT_A_TIME == False:
                    ONE_AT_A_TIME = True
                elif Event.key == K_SPACE and ONE_AT_A_TIME == True:
                    ONE_AT_A_TIME = False
                elif Event.key == K_d and ONE_AT_A_TIME:
                    draw_window(Player, Food, SCREEN, SCREEN_COLOR)
                elif Event.key == K_RIGHT and Player.direction != 3 and Player.direction != 2:
                    Player.move_queue.append(3)
                elif Event.key == K_LEFT and Player.direction != 2 and Player.direction != 3:
                    Player.move_queue.append(2)
                elif Event.key == K_UP and Player.direction != 0 and Player.direction != 1:
                    Player.move_queue.append(0)
                elif Event.key == K_DOWN and Player.direction != 1 and Player.direction != 0:
                    Player.move_queue.append(1)

        if ONE_AT_A_TIME == False:
            clock.tick(FPS)
            draw_window(Player, Food, SCREEN, SCREEN_COLOR)
    pygame.quit()

if __name__ == "__main__":
    main()