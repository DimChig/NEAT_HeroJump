import math
import random
import time

import pygame

pygame.init()

screen = pygame.display.set_mode((1500, 1000))
screen_width = screen.get_width()
screen_height = screen.get_height()

OBSTACLES_GAP = int(screen_width / 3)


pygame.display.set_caption("Jump!")

def ccw(A, B, C):
    return (C[1]- A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

# Return true if line segments AB and CD intersect
def doLinesOveralp(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def getNextPosByAngle(x, y, angle, speed):
    x += speed * math.cos(math.radians(angle))
    y += speed * math.sin(math.radians(angle))
    return x, y

trail_x = 0
trail_y = screen_height * 2

start_angle = -80
angle = start_angle
force = 1.5
gravity = 0.001
velocity_y = 0
speed_x = 0
trail_prev_x = trail_x
trail_prev_y = trail_y

screen.fill((255, 255, 255))

color = (0, 0, 0)

obs_x = 800
obs_y = screen_height - 200
obs_width = 100
obs_height = 70

def reset(a):
    global trail_x, trail_y, start_angle, color, angle, speed_x, velocity_y, trail_prev_x, trail_prev_y, gravity
    trail_x = 50
    trail_y = screen_height - 200
    trail_prev_x = trail_x
    trail_prev_y = trail_y
    #start_angle = random.randint(-85, -10)
    start_angle = a
    #gravity = random.randint(1, 10) / 1000
    angle = a

    speed_bust = math.pow(0.7 - math.cos(math.radians(start_angle)), 2)
    speed_x = math.pow(math.cos(math.radians(start_angle)), 2)
    #speed_x = math.cos(math.radians(start_angle))
    print(start_angle, speed_bust)
    velocity_y = abs(math.sin(math.radians(start_angle)))
    #screen.fill((255, 255, 255))

gravity = 0.005
angle = -85

run = True
while run:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False


    #move
    '''angle += gravity
    if angle > -start_angle: angle = -start_angle
    trail_x, trail_y = getNextPosByAngle(trail_x, trail_y, angle, force)'''

    if trail_y > screen_height or trail_y < 0:
        angle += 1
        if angle > 0:
            angle = -90
            gravity -= 0.001
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        reset(angle)

    k = 10
    #print(angle," =>",velocity_y)
    velocity_y -= gravity * k

    trail_prev_x = trail_x
    trail_prev_y = trail_y
    trail_x += speed_x * k
    trail_y -= velocity_y * k

    #draw
    pygame.draw.circle(screen, color, (trail_x, trail_y), 5)



    pygame.display.flip()
    #screen.fill((0, 0, 0))




pygame.quit()