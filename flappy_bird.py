"""
flappy_bird.py
==============
NEAT training script for Flappy Bird, extended with:
  1. Real-time neural-network visualization of the current "leader" bird
     (the bird with the highest fitness in the population right now),
     using the NeuralNetworkVisualizer class from nn_visualizer.py.
  2. Per-generation CSV logging (best fitness, average fitness, species
     count, max score reached)

Uses the NeuralNetworkVisualizer class exactly as implemented in
nn_visualizer.py:

    visualizer = NeuralNetworkVisualizer(genome, config, panel_x, panel_y, panel_w, panel_h)
    visualizer.update((bird_y, dist_top, dist_bottom))
    visualizer.set_genome(new_genome)   # called when the leader bird changes
    visualizer.draw(window)             # paints its own panel bg/border/title

Note: NeuralNetworkVisualizer.draw() already fills and borders its own
panel and renders a "Neural Network" title, so flappy_bird.py does not
duplicate that background/label - it only draws a fallback panel when no
bird is alive to visualize.
"""

import os
import csv
import pickle
import random
from datetime import datetime

import neat
import pygame

from bird import Bird
from pipe import Pipe
from base import Base
from nn_visualizer import NeuralNetworkVisualizer

pygame.font.init()  # initialize pygame's font module before creating any fonts

# --------------------------------------------------------------------------
# WINDOW / LAYOUT CONSTANTS
# --------------------------------------------------------------------------
# The window is now split into two regions side by side:
#   [ game area (left) ][ neural-network visualization panel (right) ]
GAME_WIDTH   = 500                       # width of the actual playable game area
PANEL_WIDTH  = 800                       # width reserved for the NN visualizer
WIN_HEIGHT   = 800
WIN_WIDTH    = GAME_WIDTH + PANEL_WIDTH  # total window width


GAME_H= 730
GAME_W=500


PANEL_X = GAME_WIDTH   # NN panel starts right after the game area
PANEL_Y = 0
PANEL_H = WIN_HEIGHT

DRAW_LINES = True   # draw debug lines from bird -> pipe gap (game area only)
GEN = 0              # global generation counter, incremented once per eval_genomes() call

# Colors used for the NN panel background / separators
PANEL_BG_COLOR   = (25, 25, 35)
PANEL_LINE_COLOR = (60, 60, 75)

BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))
STAT_FONT  = pygame.font.SysFont("comicsans", 50)
PANEL_FONT = pygame.font.SysFont("comicsans", 22)

# --------------------------------------------------------------------------
# CSV LOGGING SETUP
# --------------------------------------------------------------------------
# One row is appended per generation. Columns are kept generic on purpose so
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "results", "neat_training_log.csv")
CSV_HEADERS = [
    "timestamp",
    "algorithm",
    "generation",
    "best_fitness",
    "avg_fitness",
    "species_count",
    "max_score",
]


def _init_csv_log():
    """Create the CSV file with headers if it doesn't already exist."""
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)


def _log_generation(gen, genome_list, species_count, max_score):
    """
    Append one row of stats for the generation that just finished.

    genome_list: the list of `neat.DefaultGenome` objects that were alive
                 at some point during this generation (their .fitness has
                 already been set by eval_genomes()).
    """
    fitnesses = [g.fitness for g in genome_list if g.fitness is not None]
    best_fitness = max(fitnesses) if fitnesses else 0
    avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            "NEAT",
            gen,
            round(best_fitness, 3),
            round(avg_fitness, 3),
            species_count,
            max_score,
        ])


# --------------------------------------------------------------------------
# DRAWING
# --------------------------------------------------------------------------
def draw_window(win, birds, pipes, base, score, gen, pipe_ind, visualizer):
    """Draw the game area (left) and the NN visualization panel (right)."""

    # ---- Game area -------------------------------------------------
    win.blit(BG_IMG, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)

    for bird in birds:
        # draw debug lines from the bird to the gap it is aiming for
        if DRAW_LINES:
            try:
                pygame.draw.line(
                    win, (255, 0, 0),
                    (bird.x + bird.img.get_width() / 2, bird.y + bird.img.get_height() / 2),
                    (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width() / 2, pipes[pipe_ind].height),
                    5,
                )
                pygame.draw.line(
                    win, (255, 0, 0),
                    (bird.x + bird.img.get_width() / 2, bird.y + bird.img.get_height() / 2),
                    (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.get_width() / 2, pipes[pipe_ind].bottom),
                    5,
                )
            except IndexError:
                # pipe_ind can momentarily point past the end of `pipes`
                # right after a pipe is removed - just skip the debug lines.
                pass
        bird.draw(win)

    # SCORE (top right of the game area)
    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (GAME_WIDTH - 10 - text.get_width(), 10))

    # GENERATION (top left)
    text = STAT_FONT.render("Gen: " + str(gen), 1, (255, 255, 255))
    win.blit(text, (10, 10))

    # ALIVE COUNT
    text = STAT_FONT.render("Alive: " + str(len(birds)), 1, (255, 255, 255))
    win.blit(text, (10, 50))

    # ---- NN visualization panel -------------------------------------
    # NeuralNetworkVisualizer.draw() paints its own panel background,
    # border and "Neural Network" title inside (self.x, self.y, self.width,
    # self.height), so we don't need to draw a separate panel rect here -
    # doing so would just draw an extra rect underneath/behind it.
    if visualizer is not None:
        visualizer.draw(win)
    else:
        pygame.draw.rect(win, PANEL_BG_COLOR, pygame.Rect(PANEL_X, PANEL_Y, PANEL_WIDTH, PANEL_H))
        pygame.draw.line(win, PANEL_LINE_COLOR, (PANEL_X, 0), (PANEL_X, WIN_HEIGHT), 2)
        msg = PANEL_FONT.render("No birds alive", 1, (150, 150, 150))
        win.blit(msg, (PANEL_X + 15, 50))

    pygame.display.update()


# --------------------------------------------------------------------------
# NEAT FITNESS FUNCTION (called once per generation by neat.Population.run)
# --------------------------------------------------------------------------
def eval_genomes(genomes, config):
    """
    Runs a single generation: simulates every genome's bird simultaneously,
    assigns fitness based on how long it survives / how many pipes it
    passes, and visualizes whichever bird currently has the best fitness.
    """
    global GEN
    GEN += 1

    nets = []   # neural network for each genome, same index as `birds`
    ge = []     # genome objects themselves, same index as `birds`
    birds = []  # Bird instances, same index as `nets` / `ge`

    # genomes is a list of (genome_id, genome) tuples handed to us by NEAT
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        ge.append(g)

    base = Base(730)
    pipes = [Pipe(600)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("NEAT Flappy Bird - Live NN Visualization")
    clock = pygame.time.Clock()

    run = True
    score = 0

    # NN-VIS: state used to (re)build the visualizer only when the leader
    # bird actually changes, instead of rebuilding it every single frame.
    visualizer = None
    leader_idx = None
    # per-frame inputs fed to each bird's network, kept so we can hand the
    # leader's exact inputs to the visualizer after the decision loop below.
    bird_inputs = {}

    while run:
        clock.tick(3000)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # decide which of the (at most 2) on-screen pipes is the "next" one
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            # every bird in this generation has died - end the generation
            run = False
            break

        # ---- move every bird and let its network decide whether to jump ----
        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1  # small reward per frame survived

            inputs = (
                bird.y/GAME_H,  # normalize to [0,1] range
                abs(bird.y - pipes[pipe_ind].height)/GAME_H ,
                abs(bird.y - pipes[pipe_ind].bottom)/GAME_H ,
                abs(bird.x - pipes[pipe_ind].x)/GAME_W,
            )
            bird_inputs[x] = inputs

            output = nets[x].activate(inputs)
            if output[0] > 0.5:  
                bird.jump()

        # ---- pipe collisions / passing ----
        rem = [] # pipes to be removed after the loop (we can't remove while iterating)
        add_pipe = False
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)
                    bird_inputs.pop(x, None)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)
            pipe.move()

        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5      #add fitness for passing a pipe
            pipes.append(Pipe(600)) #new pipe added to the right of the screen

        for r in rem:
            pipes.remove(r)

        # ---- floor / ceiling collisions ----
        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                ge[x].fitness -= 1
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)
                bird_inputs.pop(x, None)

        # bird_inputs keys can go stale after pops above (indices shift),
        # so rebuild it fresh each frame from the surviving birds instead
        # of trying to patch it incrementally.
        bird_inputs = {
            x: (
                bird.y/GAME_H,
                abs(bird.y - pipes[pipe_ind].height)/GAME_H if pipe_ind < len(pipes) else 0,
                abs(bird.y - pipes[pipe_ind].bottom)/GAME_H if pipe_ind < len(pipes) else 0,
                abs(bird.x - pipes[pipe_ind].x)/GAME_W if pipe_ind < len(pipes) else 0,
            )
            for x, bird in enumerate(birds)
        }

        # ---- pick the current leader (highest fitness) and update the NN panel ----
        if birds:
            new_leader_idx = max(range(len(ge)), key=lambda i: ge[i].fitness)

            if visualizer is None:
                # first bird of the generation - construct the visualizer once
                leader_idx = new_leader_idx
                visualizer = NeuralNetworkVisualizer(
                    ge[leader_idx], config, PANEL_X, PANEL_Y, PANEL_WIDTH, PANEL_H
                )
            elif new_leader_idx != leader_idx:
                # NN-VIS: leadership changed hands - swap the genome being
                # shown instead of rebuilding the whole visualizer object
                # (set_genome() re-derives the network/node layout for us).
                leader_idx = new_leader_idx
                visualizer.set_genome(ge[leader_idx])

            visualizer.update(bird_inputs[leader_idx])
        else:
            visualizer = None
            leader_idx = None

        base.move()
        draw_window(win, birds, pipes, base, score, GEN, pipe_ind, visualizer)

        # ---- early-stop conditions for this generation ----
        if score > 100:
            run = False
            break

        if score > 50:
            best_genome = max(ge, key=lambda g: g.fitness)
            best_net = neat.nn.FeedForwardNetwork.create(best_genome, config)
            with open("models/4.pickle", "wb") as f:
                pickle.dump({"genome": best_genome, "network": best_net}, f)
            run = False
            break

    # ---- log this generation's results to CSV ----
    # `genomes` (the raw list passed in by NEAT) still holds every genome
    # that took part this generation, including ones that died early, and
    # by now every genome's .fitness has been finalized by NEAT/us.
    all_genomes_this_gen = [g for _, g in genomes]
    # NOTE: real species count isn't available inside eval_genomes() - NEAT
    # only tracks it on the StatisticsReporter/Population objects in run().
    # This is a placeholder; wire up the real value if compare_algorithms.py
    # needs it (see the note in run() below).
    species_count = 1
    _log_generation(GEN, all_genomes_this_gen, species_count, score)


# --------------------------------------------------------------------------
# TRAINING ENTRY POINT
# --------------------------------------------------------------------------
def run(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    _init_csv_log()

    winner = p.run(eval_genomes, 50)  # run for at most 50 generations

    print("\nBest genome:\n{!s}".format(winner))

    # also persist the overall best genome/network found across the whole run
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open("models/best_neat4.pickle", "wb") as f:
        pickle.dump({"genome": winner, "network": winner_net}, f)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)