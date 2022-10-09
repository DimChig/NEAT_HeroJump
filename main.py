import time

import numpy as np
import pygame
import math
import random
import time

import os.path
import neat
import matplotlib.pyplot as plt
import pickle as pkl
from TelegramNotifier import notifyDimChig

SCORE_MAX = 15000
EPOCHS = 10000
PLAY_TYPE = 1
model_path = "winner2.pkl"
pygame.init()

gravity = 0.0005
CALC_PERCISION = 100

screen = pygame.display.set_mode((1920, 1080))
screen_width = screen.get_width()
screen_height = screen.get_height()

player_height = int(screen_height/15)
player_width = player_height/2

OBSTACLES_GAP_MIN = int(screen_width / 5)
OBSTACLES_GAP_MAX = int(screen_width / 3)
#OBSTACLES_WIDTH_MIN = int(player_width * 2)
OBSTACLES_WIDTH_MIN = int(player_width * 2)
OBSTACLES_WIDTH_MAX = int(player_width * 2)
OBSTACLES_HEIGHT_MIN = int(screen_height * 0.3)
OBSTACLES_HEIGHT_MAX = int(screen_height * 0.97)
OBSTACLES_TRESHOLD_MAX = int((OBSTACLES_HEIGHT_MAX - OBSTACLES_HEIGHT_MIN) / 4)


pygame.display.set_caption("Jump!")
pygame.font.init()
font_score = pygame.font.Font("fonts/main.TTF", 50)
font_stats = pygame.font.Font("fonts/main.TTF", 30)



cam_offset_x = screen_width/3
cam_x = -cam_offset_x


generation = 0
FPS = 30
current_FPS = 0
fps_last_time = 0
t_last_fast_update = 0
current_Total = []

cam_speed = FPS

run = False

instances = []
obstacles = []
nets = []
ge = []
gradient_colors = []

current_max_score = 0
max_score = 0
epoch_scores = []

nearestPile = None

clock = pygame.time.Clock()
color_idx = 0

def ccw(A, B, C):
    return (C[1]- A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
def doLinesOveralp(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def getNextPosByAngle(x, y, angle, speed):
    x += speed * math.cos(math.radians(angle))
    y += speed * math.sin(math.radians(angle))
    return x, y

class Player:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y

        self.color = color
        self.isJumping = False

        self.angle = 0
        self.trail_x = 0
        self.trail_y = 0
        self.trail_prev_x = 0
        self.trail_prev_y = 0
        self.speed_x = 0
        self.velocity_y = 0

        self.jump_start_x = 0
        self.jump_start_y = 0

    def jump(self, angle, force):
        self.angle = angle
        self.force = force

class Obs:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.isPassed = False


class Instance:
    def __init__(self, player, genom):
        self.player = player
        self.score = 0
        self.genom = genom
        self.isDead = False


def generate_gradient_rgbs():
    arr = []
    for i in range(0,255):
        arr.append((255,i,0))
    for i in range(0, 255):
        arr.append((255-i, 255, 0))
    for i in range(0, 255):
        arr.append((0, 255, i))
    for i in range(0, 255):
        arr.append((0, 255-i, 255))
    for i in range(0, 255):
        arr.append((i, 0, 255))
    for i in range(0, 255):
        arr.append((255, 0, 255-i))
    return arr


def restart():
    global instances, obstacles, run, gradient_colors, nets, ge, epoch_scores, current_max_score, max_score, current_max_score
    run = True

    if current_max_score > max_score: max_score = current_max_score
    epoch_scores.append(current_max_score)

    if PLAY_TYPE == 0:
        f = open("log.txt", "w")
        f.write(str(epoch_scores))
        f.close()

    current_max_score = 0
    instances = []
    obstacles = []
    nets = []
    ge = []
    gradient_colors = generate_gradient_rgbs()


def drawScore():
    global current_max_score, font_score, font_stats, max_score, generation, current_FPS, current_Total, PLAY_TYPE
    text = font_score.render("Score: " + str(int(current_max_score)), True, (0, 0, 0))
    screen.blit(text, (screen_width/2 - text.get_width()/2, 10))

    if PLAY_TYPE == 1 or PLAY_TYPE == 2:
        return
    offX = 100
    pygame.draw.rect(screen, (255, 255, 255), (5 + offX, 5, 300, 200))
    pygame.draw.rect(screen, (100, 100, 100), (5 + offX, 5, 300, 200), 3)

    s = "Generation: " + str(generation)
    text1 = font_stats.render(s, True, (0, 0, 0))
    screen.blit(text1, (offX + 310 / 2 - text1.get_width() / 2, 10))

    s = "Max score: " + str(max_score)
    text2 = font_stats.render(s, True, (0, 0, 0))
    screen.blit(text2, (offX + 310 / 2 - text2.get_width() / 2, 10 + text1.get_height()))

    s = "Alive: " + str(len(instances))
    text3 = font_stats.render(s, True, (0, 0, 0))
    screen.blit(text3, (offX + 310 / 2 - text3.get_width() / 2, 10 + text1.get_height() * 2))

    if len(current_Total) != 0:
        sum = 0
        for f in current_Total: sum += f
        cur_fps = int(sum / len(current_Total))
        s = "FPS: " + str(cur_fps)
        text4 = font_stats.render(s, True, (0, 0, 0))
        screen.blit(text4, (offX + 310 / 2 - text4.get_width() / 2, 10 + text1.get_height() * 3))

def getDarkenColor(color, darkness):
    return (max(0, int(color[0] / darkness)), max(0, int(color[1] / darkness)), max(0, int(color[2] / darkness)))

def mymap(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)

def drawInstances(isPlayerHidden):
    global instances, cam_x, color_idx

    p_width = player_width
    p_height = player_height
    for i, instance in enumerate(instances):
        p = instance.player
        if isPlayerHidden:
            if instance.isDead: continue
        if PLAY_TYPE == 1:
            color_idx = (color_idx + 1) % len(gradient_colors)
            p.color = gradient_colors[color_idx]
        pygame.draw.rect(screen, p.color, (p.x - p_width / 2 - cam_x, p.y - p_height / 2, p_width, p_height))
        #eyes
        pygame.draw.rect(screen, (255, 255, 255), (p.x - cam_x, p.y - p_height / 4, p_width / 2, p_height / 6))
        pygame.draw.rect(screen, (0, 0, 0), (p.x + p_width / 4 - cam_x, p.y - p_height / 4 + 3, p_width / 4, p_height / 6 - 6))

        pygame.draw.rect(screen, getDarkenColor(p.color, 2),(p.x - p_width / 2 - cam_x, p.y - p_height / 2, p_width, p_height), 3)

        cluv_size = p_height / 8
        pygame.draw.polygon(screen, color=(255, 157, 0), points=[(p.x + p_width / 2 - cam_x, p.y - cluv_size), (p.x + p_width / 2 - cam_x, p.y + cluv_size), (p.x + p_width / 2 + cluv_size*1.4 - cam_x, p.y)])
        cluv_size = p_height / 10
        pygame.draw.polygon(screen, color=(255, 213, 0), points=[(p.x + p_width / 2 - cam_x, p.y - cluv_size), (p.x + p_width / 2 - cam_x, p.y + cluv_size), (p.x + p_width / 2 + cluv_size*1.4 - cam_x, p.y)])
        #draw player angle
        pygame.draw.line(screen, p.color, (p.jump_start_x - cam_x, p.jump_start_y), getNextPosByAngle(p.jump_start_x - cam_x, p.jump_start_y, p.angle, player_height), 5)

def drawObstacles():
    for obs in obstacles:
        pygame.draw.rect(screen, (50, 50, 50), (obs.x - obs.width / 2 - cam_x, obs.y, obs.width, obs.height))
        pygame.draw.rect(screen, (0, 0, 0), (obs.x - obs.width / 2 - cam_x, obs.y, obs.width, obs.height), 5)

def getPlayerSpectate():
    spectate_player = instances[0].player
    max = 0
    for i in instances:
        if i.isDead == False and i.player.x > max:
            max = i.player.x
            spectate_player = i.player
    return spectate_player

def drawAll():
    global cam_x, instances, cam_offset_x, cam_speed
    spectate_player = getPlayerSpectate()

    target_cam_x = spectate_player.x - cam_offset_x
    if target_cam_x != cam_x:
        if abs(target_cam_x - cam_x) <= cam_speed:
            cam_x = target_cam_x
        elif target_cam_x > cam_x:
            cam_x = target_cam_x
        elif target_cam_x < cam_x:
            cam_x -= cam_speed

    screen.fill((255, 255, 255))
    drawObstacles()
    drawInstances(True)
    drawScore()

prev_obstacle_y = -1
def spawnObstacle():
    global obstacles, prev_obstacle_y, OBSTACLES_GAP_MIN, OBSTACLES_GAP_MAX, player_height, OBSTACLES_WIDTH_MIN, OBSTACLES_WIDTH_MAX, OBSTACLES_HEIGHT_MIN, OBSTACLES_HEIGHT_MAX
    x = 0
    if len(obstacles) > 0:
        x = obstacles[len(obstacles) - 1].x + random.randint(int(OBSTACLES_GAP_MIN), int(OBSTACLES_GAP_MAX))

    y = prev_obstacle_y
    if y == -1:
        y = random.randint(OBSTACLES_HEIGHT_MIN, OBSTACLES_HEIGHT_MAX)
    else:
        y += random.randint(-OBSTACLES_TRESHOLD_MAX, OBSTACLES_TRESHOLD_MAX)
        y = max(y, OBSTACLES_HEIGHT_MIN)
        y = min(y, OBSTACLES_HEIGHT_MAX)
    prev_obstacle_y = y

    width = random.randint(OBSTACLES_WIDTH_MIN, OBSTACLES_WIDTH_MAX)
    height = screen_height - y
    obstacles.append(Obs(x, y, width, height))

def spawnObstacles():
    global obstacles
    for i in range(0, 10):
        spawnObstacle()


def main(genomes, config):
    global instances, ge, nets, t_last_fast_update, cam_x, cam_offset_x, run, nearestPile, obstacles, generation, FPS, current_FPS, fps_last_time, t_last_fast_update, EPOCHS, SCORE_MAX, current_Total, PLAY_TYPE, current_max_score, max_score
    restart()
    generation += 1
    cam_x = -cam_offset_x

    spawnObstacles()
    nearestPile = obstacles[0]

    t_last_fast_update = 0

    index = -1
    color_step = max(1, math.floor(len(gradient_colors) / len(genomes)))
    for i, g in genomes:
        index += 1
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)

        color = gradient_colors[(color_step * i) % len(gradient_colors)]

        #player = Player(int(mymap(index, 0, len(genomes), -obstacles[0].width/2 + player_width/2, obstacles[0].width/2 - player_width/2)), obstacles[0].y - player_height/2, color)
        player = Player(0, obstacles[0].y - player_height/2, color)
        instances.append(Instance(player, g))

        g.fitness = 0
        ge.append(g)

    run = True
    fps_last_time = time.time()
    while run:
        if FPS != 2000: clock.tick(FPS)
        t = time.time()
        if t != fps_last_time:
            current_FPS = int(1/(t - fps_last_time))
            if len(current_Total) > 50:
                current_Total.pop(0)
            current_Total.append(current_FPS)
        fps_last_time = time.time()


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        if FPS != 2000:
            drawAll()
        else:
            if t - t_last_fast_update > 2:
                t_last_fast_update = t
                screen.fill((0, 0, 0))
                text = font_score.render("Gen: " + str(generation) + ", Max score: " + str(max_score), True, (255, 255, 255))
                screen.blit(text,(screen_width / 2 - text.get_width() / 2, text.get_height()))

                # draw graph
                offset = screen_width / 15
                graph_width = screen_width - offset * 2
                graph_height = int(screen_height / 1.2 - offset * 2)
                pygame.draw.line(screen, (255, 255, 255), (offset, screen_height - offset),
                                 (offset + graph_width, screen_height - offset), 5)
                pygame.draw.line(screen, (255, 255, 255), (offset + graph_width, screen_height - offset + 7),
                                 (offset + graph_width, screen_height - offset - 7), 4)

                pygame.draw.line(screen, (255, 255, 255), (offset, screen_height - offset),
                                 (offset, screen_height - offset - graph_height), 5)
                pygame.draw.line(screen, (255, 255, 255), (offset + 7, screen_height - offset - graph_height),
                                 (offset - 7, screen_height - offset - graph_height), 4)

                text = font_stats.render(str(max_score), True, (255, 255, 255))
                screen.blit(text,
                            (offset - text.get_width() / 2, screen_height - offset - graph_height - text.get_height()))

                text = font_stats.render("Epochs: " + str(generation), True, (255, 255, 255))
                screen.blit(text,
                            (offset + graph_width - text.get_width(), screen_height - offset + text.get_height() / 2))

                mean_points = 30
                mean_step = len(epoch_scores) / mean_points
                mean_point = (0, 0)
                mean_cnt = 0
                mean_point_prev = (offset, screen_height - offset)

                epoch = generation
                #arr = epoch_scores[:]
                if not (len(epoch_scores) == 0 or epoch == 0 or max_score == 0):
                    prev_dot = (offset, screen_height - offset)
                    for x, y in enumerate(epoch_scores):
                        nx = int(mymap(x, 0, epoch, offset, offset + graph_width))
                        ny = int(mymap(y, 0, max_score, screen_height - offset, screen_height - offset - graph_height))
                        pygame.draw.line(screen, (255, 0, 0), prev_dot, (nx, ny), 3)
                        prev_dot = (nx, ny)

                        if len(epoch_scores) < mean_points: continue
                        mean_point = (mean_point[0] + nx, mean_point[1] + ny)
                        mean_cnt += 1
                        if mean_cnt >= mean_step or x == len(epoch_scores) - 1:
                            px, py = int(mean_point[0] / mean_cnt), int(mean_point[1] / mean_cnt)
                            mean_cnt = 0
                            pygame.draw.line(screen, (0, 180, 255), mean_point_prev, (px, py), 5)
                            mean_point_prev = (px, py)
                            mean_point = (0, 0)

        #calculate distances
        # get nearest pile

        isEveryBodyFinished = True
        cnt_alive = 0
        for x, instance in enumerate(instances):
            if x >= len(instances): continue
            if instance.isDead == True: continue
            cnt_alive += 1
            #ge[x].fitness += 0.05

            player = instance.player
            if player.isJumping == True:
                isEveryBodyFinished = False

                #move player
                player.velocity_y -= gravity * CALC_PERCISION

                player.trail_prev_x = player.x + player_width / 2
                player.trail_prev_y = player.y + player_height / 2
                player.trail_x += player.speed_x * CALC_PERCISION
                player.trail_y -= player.velocity_y * CALC_PERCISION

                player.x = player.trail_x
                player.y = player.trail_y

                if FPS != 2000:
                    pygame.draw.line(screen, player.color,
                                     (player.x - cam_x + player_width / 2, player.y),
                                     (player.x - cam_x + nearestPile.x - player.x, player.y), 4)
                    pygame.draw.line(screen, player.color,
                                     (player.x + nearestPile.x - player.x - cam_x, player.y),
                                     (player.x - cam_x + nearestPile.x - player.x, player.y + nearestPile.y - player.y),
                                     4)

                if nearestPile.x - nearestPile.width / 2 - player_width/2 < player.trail_x < nearestPile.x + nearestPile.width / 2 + player_width /2 and nearestPile.y - player_height/2 < player.trail_y < nearestPile.y + nearestPile.height + player_height/2:
                    # inside

                    if doLinesOveralp((player.trail_prev_x, player.trail_prev_y), (player.x + player_width / 2, player.y + player_height / 2), (nearestPile.x - nearestPile.width / 2, nearestPile.y), (nearestPile.x + nearestPile.width / 2 + player_width, nearestPile.y)):
                        player.y = nearestPile.y - player_height/2
                        #player.x = nearestPile.x - player_width/2
                        max_fitness = 10
                        max_bonus = int(10)
                        bonus = max(0, min(mymap(abs(player.x - nearestPile.x), nearestPile.width / 2, 0, 0, max_bonus), max_bonus))
                        ge[x].fitness += max_fitness + bonus
                        instance.score += 1
                        if instance.score > current_max_score:
                            current_max_score = instance.score
                            if current_max_score > max_score:
                                max_score = current_max_score
                    else:
                        instance.isDead = True
                        ge[x].fitness -= 10

                    #pygame.display.flip()

                    player.isJumping = False
                    player.trail_x = player.x
                    player.trail_y = player.y
                    player.trail_prev_x = player.trail_x
                    player.trail_prev_y = player.trail_y

                #ceiling  or player.trail_y - player_height/2 < 0
                elif player.trail_y + player_height/2 >= screen_height or (player.trail_y > nearestPile.y and player.velocity_y < 0) or player.x + player_width > nearestPile.x + nearestPile.width:
                    instance.isDead = True

                # if instance.isDead == True:
                #     ge[x].fitness -= 1
                #     #instances.pop(x)
                #     #ge.pop(x)
                #     #nets.pop(x)
                #     continue

        #CALCULATE ANGLES
        if isEveryBodyFinished == True:
            nearestPile.isPassed = True
            spawnObstacle()
            for obs in obstacles:
                if obs.isPassed == False and obs.x > nearestPile.x:
                    nearestPile = obs
                    break
            if nearestPile == None:
                continue

            for x, instance in enumerate(instances):
                if x >= len(instances): continue
                if instance.isDead == True: continue

                if PLAY_TYPE == 0 or PLAY_TYPE == 1:

                    player = instance.player

                    distanceX = nearestPile.x - player.x
                    distanceX = mymap(distanceX, OBSTACLES_GAP_MIN - OBSTACLES_WIDTH_MAX, OBSTACLES_GAP_MAX + OBSTACLES_WIDTH_MAX, 0, 1)
                    #print(distanceX)
                    #distanceY = nearestPile.y - player.y + player_height/2
                    pile_pos_y = mymap(nearestPile.y, 0, screen_height, 0, 1)
                    pos_y = mymap(player.y, 0, screen_height, 0, 1)
                    #print((pos_y, distanceX, pile_pos_y))

                    #output = nets[x].activate((pos_y, distanceX, pile_pos_y))
                    output = nets[x].activate((pile_pos_y, pos_y, distanceX))
                    #if x == 0: print((pile_pos_y, pos_y, distanceX))
                    #angle = mymap(output[0], 0, 1, -80, -10)
                    angle = mymap(output[0], -1, 1, -90, -1)
                    #print(output[0]," ->",angle)
                    #angle = random.randint(-80, -10)

                    player.angle = angle
                    player.speed_x = math.cos(math.radians(angle))
                    player.velocity_y = abs(math.sin(math.radians(angle)))

                    player.isJumping = True
                    player.trail_x = player.x
                    player.trail_y = player.y
                    player.trail_prev_x = player.trail_x
                    player.trail_prev_y = player.trail_y
                    player.jump_start_x = player.x
                    player.jump_start_y = player.y + player_height/2




        if len(instances) <= 0 or cnt_alive == 0 or (PLAY_TYPE == 0 and current_max_score > SCORE_MAX):
            #drawAll()
            #drawInstances(False)
            #pygame.display.flip()
            if FPS != 2000:
                #time.sleep(2)
                pass
            run = False
            break

        pygame.display.flip()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            pygame.quit()
            break
        if keys[pygame.K_f]:
            FPS = 1000
            cam_x = getPlayerSpectate().x - cam_offset_x
        elif keys[pygame.K_s]:
            FPS = 30
            cam_x = getPlayerSpectate().x - cam_offset_x
        elif keys[pygame.K_g]:
            FPS = 2000
        elif keys[pygame.K_t]:
            FPS = 1
        elif keys[pygame.K_p]:
            plt.plot(epoch_scores)
            plt.ylabel('Epochs: ' + str(generation))
            plt.show()

def runProgram(config_pass):
    global t_last_fast_update
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_pass)

    if PLAY_TYPE == 0:

        #generate population
        p = neat.Population(config)
        #stats reporter
        #p.add_reporter(neat.StdOutReporter(True))
        #p.add_reporter(neat.StatisticsReporter())

        winner = p.run(main, EPOCHS)

        print('\nBest genome:\n{!s}'.format(winner))
        print("Gen " + str(generation) + " score =", max_score)
        #save
        with open(model_path, "wb") as f:
            pkl.dump(winner, f)
            f.close()

        notifyDimChig("Finished Training! Reached max score *" + str(max_score) + "* at gen *" + str(generation) + "*! Model saved to *" + str(model_path) + "*")
    else:
        with open(model_path, "rb") as f:
            genome = pkl.load(f)
        genomes = [(1, genome)]
        main(genomes, config)
        print("Reached score:",max_score)



if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_neat.txt")
    runProgram(config_path)

