import pygame
import random
from pygame import Vector2, display
from pygame import time
from pygame import key
from pygame import event
from pygame import draw
from pygame.constants import K_DOWN, K_LEFT, K_RIGHT, K_SPACE, K_UP, K_d

WIDTH, HEIGHT = 1400, 900
FPS = 60
DT = 1/FPS
SCREEN = display.set_mode((WIDTH, HEIGHT))
display.set_caption("Testing game")

SCREEN_COLOR = (0, 0, 0)
PIX_ARR = pygame.PixelArray(SCREEN)

class player():
    updated = True
    points = 0
    move_queue = []
    coordinates = Vector2(0, 0)
    direction = 3
    # Need to solve this problem
    speed_per = 25/(FPS/8)
    speed = Vector2(25/(FPS/10), 0)
    color = (0, 146, 15)
    length = Vector2(25, 25)
    border_Radius = 3

    def draw(self, screen, coordinates, length, color, border_Radius):
        rect = pygame.Rect(coordinates.x, coordinates.y, length.x, length.y)
        draw.rect(screen, color, rect, border_radius=border_Radius)
    
    def update(self, screen, Food):
        print(self.coordinates)
        if self.coordinates.x % 25 == 0 and self.coordinates.y % 25 == 0 and len(self.move_queue) != 0:
            direction = self.move_queue.pop(0)
            if direction == 0:
                self.speed = Vector2(0, -1 * self.speed_per)
            elif direction == 1:
                self.speed = Vector2(0, self.speed_per)
            elif direction == 2:
                self.speed = Vector2(-1 * self.speed_per, 0)
            elif direction == 3:
                self.speed = Vector2(self.speed_per, 0)
            
            self.updated = True
        
        self.coordinates.x += self.speed.x
        self.coordinates.y += self.speed.y
        if self.coordinates.x >= WIDTH:
            self.coordinates.x = 0
        elif self.coordinates.x < 0:
            self.coordinates.x = WIDTH
        elif self.coordinates.y >= HEIGHT:
            self.coordinates.y = 0
        elif self.coordinates.y < 0:
            self.coordinates.y = HEIGHT - 25

        if self.coordinates.x == Food.coordinates.x and self.coordinates.y == Food.coordinates.y:
            self.points += 1
            print(self.points)
        
        self.draw(screen, self.coordinates, self.length, self.color, self.border_Radius)
        Food.generate(Player=self, screen=screen)
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
    clock = time.Clock()
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