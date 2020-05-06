import pygame
import random
import os
import neat
import time
import pickle
import numpy as np
import math

pygame.init()
pygame.font.init()
STAT_FONT = pygame.font.SysFont("comicsans", 50)
WIDTH = 1000
HEIGHT = 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.Surface.fill(screen, (255, 255, 255))
pygame.display.set_caption("Snake")

mindist = 20
size = 13
dist = size * mindist + 100
fields = []
gen = 0
snakes = []
stolbK = int((WIDTH + 100) / dist)
print(stolbK)


class Field:
    def __init__(self, ind, screen):
        self.ind = ind
        self.strok = ind // stolbK
        self.stolb = ind % stolbK
        self.sc = screen
        self.dist = dist
        self.mindist = mindist

    def draw(self):
        for x in range(size + 1):
            pygame.draw.line(self.sc, (0, 0, 0), (self.stolb * self.dist + x * self.mindist, self.strok * self.dist),
                             (self.stolb * self.dist + x * self.mindist, self.strok * self.dist + self.mindist * size))
        for y in range(size + 1):
            pygame.draw.line(self.sc, (0, 0, 0), (self.stolb * self.dist, self.strok * self.dist + y * self.mindist),
                             (self.stolb * self.dist + self.mindist * size, self.strok * self.dist + y * self.mindist))


class Snake:

    def __init__(self, ind, screen):
        self.body = [(3, 2), (2, 2), (1, 2)]
        self.direction = 1  # Вверх, вправно, вниз, влево : 0, 1, 2, 3
        self.ind = ind
        self.sc = screen
        self.food = createfood(self)
        self.plan = 1
        self.eatedFood = False
        self.predgolova = (3, 2)
        self.dist = round(math.sqrt((self.food[0] - self.body[0][0]) ** 2 + (self.food[1] - self.body[0][1]) ** 2), 2)
        self.newdist = round(math.sqrt((self.food[0] - self.body[0][0]) ** 2 + (self.food[1] - self.body[0][1]) ** 2),
                             2)
        self.timewithnoeat = 50

    def move(self):
        self.direction = self.plan
        if self.body[0] == self.food:
            self.body.append(self.food)
            self.food = createfood(self)
            self.eatedFood = True
        for part in range(len(self.body) - 1, 0, -1):
            self.body[part] = self.body[part - 1]
        if self.direction == 0:
            self.body[0] = (self.body[0][0], self.body[0][1] - 1)
        elif self.direction == 1:
            self.body[0] = (self.body[0][0] + 1, self.body[0][1])
        elif self.direction == 2:
            self.body[0] = (self.body[0][0], self.body[0][1] + 1)
        else:
            self.body[0] = (self.body[0][0] - 1, self.body[0][1])

        self.dist = self.newdist
        self.newdist = round(math.sqrt((self.food[0] - self.body[0][0]) ** 2 + (self.food[1] - self.body[0][1]) ** 2),
                             2)
        self.timewithnoeat -= 1

    def change_dir(self, x):

        if abs(self.direction - x) != 2:
            self.plan = x

    def draw(self):
        strok = self.ind // stolbK
        stolb = self.ind % stolbK
        pygame.draw.rect(self.sc, (255, 0, 0), (
            stolb * dist + mindist * self.food[0] + 1, strok * dist + mindist * self.food[1] + 1, mindist - 1,
            mindist - 1))
        for part in self.body:
            x0, y0 = stolb * dist + mindist * part[0], strok * dist + mindist * part[1]
            color = (0, 100, 0)
            if part == self.body[0]:
                color = (0, 150, 0)
            pygame.draw.rect(self.sc, color, (x0 + 1, y0 + 1, mindist - 1, mindist - 1))


def window_draw(fields, snakes):
    global gen
    pygame.Surface.fill(screen, (255, 255, 255))
    for field in fields:
        field.draw()
    for snake in snakes:
        snake.draw()


def createfood(snake):
    global size
    body = snake.body
    while True:
        food = (random.randrange(size), random.randrange(size))
        if not food in body:
            return food


def eval_genomes(genomes, config):
    global screen, gen, size, dist, mindist, stolbK

    gen += 1

    nets = []
    ge = []
    fields = []
    snakes = []

    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        snakes.append(Snake(len(snakes), screen))
        fields.append(Field(len(fields), screen))
        ge.append(genome)

    run = True
    clock = pygame.time.Clock()
    while run and len(snakes) > 0:
        clock.tick(20)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break
        for x, snake in enumerate(snakes):
            # ge[x].fitness -= 0.01
            if snake.eatedFood:
                ge[x].fitness += 15
                snake.eatedFood = False
                snake.timewithnoeat = 50

            foodx, foody = snake.food[0], snake.food[1]
            headx, heady = snake.body[0][0], snake.body[0][1]
            for distance in range(1, size + 1):
                if (headx + distance, heady) in snake.body or headx + distance >= size:
                    dis1 = round(1 / distance, 3)
                if (headx, heady + distance) in snake.body or heady + distance >= size:
                    dis2 = round(1 / distance, 3)
                if (headx, heady - distance) in snake.body or heady - distance < 0:
                    dis0 = round(1 / distance, 3)
                if (headx - distance, heady) in snake.body or heady - distance < 0:
                    dis3 = round(1 / distance, 3)
            if foodx > headx:
                xb = 1
            else:
                xb = 0
            if foodx < headx:
                xm = 1
            else:
                xm = 0
            if foody > heady:
                ym = 1
            else:
                ym = 0
            if foody < heady:
                yb = 1
            else:
                yb = 0

            output = nets[x].activate((dis0, dis1, dis2, dis3, ym, xb, yb, xm))
            maxim = [0, 0]
            for info in output:
                if info > maxim[0]:
                    maxim.append(info)
                    maxim.sort()
                    maxim = maxim[1:]
            directions = [output.index(maxim[0]), output.index(maxim[1])]
            now = snake.direction
            if abs(directions[1] - now) != 2:
                newdir = directions[1]
            else:
                newdir = directions[0]
            snake.change_dir(newdir)
            snake.move()

        rem = []
        for snake in snakes:
            ind = snakes.index(snake)
            body = snake.body
            if body[0] in body[1:]:
                rem.append(snake)
            if body[0][0] < 0 or body[0][0] >= size or body[0][1] >= size or body[0][1] < 0:
                rem.append(snake)
            if snake.timewithnoeat <= 0:
                rem.append(snake)
        for snake in rem:
            ind = snakes.index(snake)
            nets.pop(ind)
            ge.pop(ind)
            snakes.pop(ind)
        window_draw(fields, snakes)
        pygame.display.update()


def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 10000)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run('config-feedforward.txt')
