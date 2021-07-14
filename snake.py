import pygame
from pygame import Vector2, display
from pygame import time
from pygame import key
from pygame import event
from pygame import draw

WIDTH, HEIGHT = 1400, 900
FPS = 60
SCREEN = display.set_mode((WIDTH, HEIGHT))
display.set_caption("Testing game")

SCREEN_COLOR = (0, 0, 0)
PIX_ARR = pygame.PixelArray(SCREEN)

class player():
    coordinates = Vector2(0, 0)
    color = (0, 146, 15)
    length = Vector2(25, 25)
    border_radius = 3
    rect = pygame.Rect(coordinates.x, coordinates.y, length.x, length.y)

    def draw(self, screen):
        draw.rect(screen, self.color, self.rect, border_radius=self.border_radius)

Player = player()

def draw_window(player, screen):
    SCREEN.fill(SCREEN_COLOR)
    player.draw(screen)
    display.update()

def main():
    clock = time.Clock()
    run = True
    while run:
        for Event in event.get():
            if Event.type == pygame.QUIT:
                run = False

            # elif key.get_focused and Event.type == pygame.KEYUP:
            #     if Event.key == 
        clock.tick(FPS)
        draw_window(Player, SCREEN)
    pygame.quit()

if __name__ == "__main__":
    main()