"""
Human vs. AI Mode (NEAT)
=======================
A split-screen mode where a human player competes against a trained NEAT AI model.

Usage:
    python human_vs_ai.py

Controls:
    SPACE - Jump (human bird)
    Q     - Quit / Forfeit current round (triggers AI victory & score screen)
    1     - Start game vs NEAT AI (on selection screen)
    R     - Restart game (on game over screen)
    ESC   - Exit game entirely
"""
# besy_neat2.pickle for direct raw vals of distances
# best_neat4.pickle for normalized vals of distances 
# so if using refactor accordingly
import os
import pickle
import random
import pygame
from bird import Bird
from pipe import Pipe
from base import Base

pygame.font.init()

WIN_WIDTH = 1000  # Two panels of 500px each
PANEL_WIDTH = 500
WIN_HEIGHT = 800
GROUND_Y = 730
GAME_W = 500
GAME_H = 730
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

STAT_FONT = pygame.font.SysFont("comicsans", 40)
MED_FONT = pygame.font.SysFont("comicsans", 28)
SMALL_FONT = pygame.font.SysFont("comicsans", 20)
TITLE_FONT = pygame.font.SysFont("comicsans", 50)


# ============================================================
# NEAT Model Loader
# ============================================================

def load_neat_model():
    """Load the trained NEAT model from pickle file."""
    model_path = "models/best_neat4.pickle"
    if not os.path.exists(model_path):
        model_path = "best.pickle"
    if not os.path.exists(model_path):
        return None, None
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data.get('network') or data.get('genome'), 'neat'


# ============================================================
# NEAT Controller
# ============================================================

class NEATAIController:
    """Controls a bird using a trained NEAT network."""

    def __init__(self, network):
        self.network = network

    def should_jump(self, bird, pipe):
        output = self.network.activate((
            bird.y/GAME_H,
            abs(bird.y - pipe.height)/GAME_H,
            abs(bird.y - pipe.bottom)/GAME_H,
            abs(bird.x - pipe.x)/GAME_W
        ))
        return output[0] > 0.5


# ============================================================
# Selection Screen
# ============================================================

def draw_selection_screen(win):
    """Draw the AI opponent selection screen."""
    win.blit(BG_IMG, (0, 0))

    # Dark overlay
    overlay = pygame.Surface((WIN_WIDTH, WIN_HEIGHT))
    overlay.set_alpha(180)
    overlay.fill((0, 0, 0))
    win.blit(overlay, (0, 0))

    # Title & Subtitle
    title = TITLE_FONT.render("HUMAN vs NEAT AI", True, (255, 255, 255))
    win.blit(title, (WIN_WIDTH // 2 - title.get_width() // 2, 150))

    subtitle = MED_FONT.render("Play against the trained NEAT agent", True, (200, 200, 200))
    win.blit(subtitle, (WIN_WIDTH // 2 - subtitle.get_width() // 2, 230))

    # Option status check
    neat_path = "models/best_neat4.pickle"
    alt_path = "best.pickle"
    available = os.path.exists(neat_path) or os.path.exists(alt_path)

    color = (100, 255, 100) if available else (150, 150, 150)
    text = MED_FONT.render("[1] NEAT (NeuroEvolution)", True, color)
    win.blit(text, (WIN_WIDTH // 2 - text.get_width() // 2, 330))

    status = "Ready" if available else "Model not found"
    status_color = (100, 255, 100) if available else (255, 100, 100)
    status_text = SMALL_FONT.render(status, True, status_color)
    win.blit(status_text, (WIN_WIDTH // 2 - status_text.get_width() // 2, 365))

    # Instructions
    instr = SMALL_FONT.render("Press 1 to start  |  ESC to quit", True, (180, 180, 200))
    win.blit(instr, (WIN_WIDTH // 2 - instr.get_width() // 2, WIN_HEIGHT - 80))

    pygame.display.update()


def show_selection_screen():
    """Show the selection screen and return the NEAT controller or None."""
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    while True:
        clock.tick(30)
        draw_selection_screen(win)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return None
                elif event.key == pygame.K_1:
                    network, _ = load_neat_model()
                    if network is not None:
                        return NEATAIController(network)
                    else:
                        show_message(
                            win,
                            "NEAT model not found!",
                            "Run your NEAT trainer script first to generate best.pickle."
                        )


def show_message(win, title, message):
    """Show a temporary message on screen."""
    win.blit(BG_IMG, (0, 0))
    overlay = pygame.Surface((WIN_WIDTH, WIN_HEIGHT))
    overlay.set_alpha(180)
    overlay.fill((0, 0, 0))
    win.blit(overlay, (0, 0))

    title_surf = TITLE_FONT.render(title, True, (255, 100, 100))
    win.blit(title_surf, (WIN_WIDTH // 2 - title_surf.get_width() // 2, WIN_HEIGHT // 2 - 50))

    msg_surf = MED_FONT.render(message, True, (200, 200, 200))
    win.blit(msg_surf, (WIN_WIDTH // 2 - msg_surf.get_width() // 2, WIN_HEIGHT // 2 + 20))

    hint = SMALL_FONT.render("Press any key to go back...", True, (180, 180, 200))
    win.blit(hint, (WIN_WIDTH // 2 - hint.get_width() // 2, WIN_HEIGHT // 2 + 80))

    pygame.display.update()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN):
                waiting = False


# ============================================================
# Game Mode
# ============================================================

def draw_game(win, ai_bird, human_bird, pipes, base, ai_score, human_score,
              ai_name, ai_alive, human_alive, game_over=False, winner_text=""):
    """Draw the split-screen game."""

    # Full-window background render
    win.blit(BG_IMG, (0, 0))
    win.blit(BG_IMG, (PANEL_WIDTH, 0))

    # --- Left panel: AI ---
    win.set_clip(pygame.Rect(0, 0, PANEL_WIDTH, WIN_HEIGHT))

    for pipe in pipes:
        pipe.draw_offset(win, 0)
    base.draw(win)

    if ai_bird and ai_alive:
        ai_bird.draw(win)

    label = MED_FONT.render(f"AI ({ai_name})", True, (255,255,255))
    win.blit(label, (10, 10))
    score_text = STAT_FONT.render(str(ai_score), True, (255, 255, 255))
    win.blit(score_text, (PANEL_WIDTH - score_text.get_width() - 10, 10))

    win.set_clip(None)

    # --- Right panel: Human ---
    win.set_clip(pygame.Rect(PANEL_WIDTH, 0, PANEL_WIDTH, WIN_HEIGHT))

    for pipe in pipes:
        pipe.draw_offset(win, PANEL_WIDTH)
    base.draw_offset(win, PANEL_WIDTH)

    if human_bird and human_alive:
        human_bird.draw_offset(win, PANEL_WIDTH)

    label = MED_FONT.render("YOU", True, (255, 255, 255))
    win.blit(label, (PANEL_WIDTH + 10, 10))
    score_text = STAT_FONT.render(str(human_score), True, (255, 255, 255))
    win.blit(score_text, (WIN_WIDTH - score_text.get_width() - 10, 10))

    win.set_clip(None)

    # Divider line
    pygame.draw.line(win, (200, 200, 200), (PANEL_WIDTH, 0), (PANEL_WIDTH, WIN_HEIGHT), 3)

    if not game_over:
        hint = SMALL_FONT.render("SPACE = Jump  |  Q = Forfeit Round", True, (0, 0,0))
        win.blit(hint, (WIN_WIDTH // 2 - hint.get_width() // 2, WIN_HEIGHT - 30))

    if game_over:
        overlay = pygame.Surface((WIN_WIDTH, WIN_HEIGHT))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        win.blit(overlay, (0, 0))

        winner_surf = TITLE_FONT.render(winner_text, True, (255, 255, 100))
        win.blit(winner_surf, (WIN_WIDTH // 2 - winner_surf.get_width() // 2, WIN_HEIGHT // 2 - 80))

        score_line = MED_FONT.render(
            f"AI: {ai_score}  vs  You: {human_score}", True, (255, 255, 255)
        )
        win.blit(score_line, (WIN_WIDTH // 2 - score_line.get_width() // 2, WIN_HEIGHT // 2))

        restart = SMALL_FONT.render("Press R to play again  |  ESC to quit app", True, (200, 200, 200))
        win.blit(restart, (WIN_WIDTH // 2 - restart.get_width() // 2, WIN_HEIGHT // 2 + 60))

    pygame.display.update()


def run_human_vs_ai():
    """Main game loop for human vs AI mode."""
    ai_controller = show_selection_screen()
    if ai_controller is None:
        return

    ai_name = "NEAT"
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption(f"Human vs AI ({ai_name})")
    clock = pygame.time.Clock()

    while True:
        _play_round(win, clock, ai_controller, ai_name)

        # Check restart/exit options after game over
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                    elif event.key == pygame.K_r:
                        waiting = False


def _play_round(win, clock, ai_controller, ai_name):
    """Play a single round of human vs AI."""
    ai_bird = Bird(230, 350)
    human_bird = Bird(230, 350)

    random.seed(42)
    pipes = [Pipe(600)]
    base = Base(GROUND_Y)

    ai_score = 0
    human_score = 0
    ai_alive = True
    human_alive = True
    game_over = False
    winner_text = ""

    run = True
    while run:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and human_alive:
                    human_bird.jump()
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    # Player forfeit / Quit mid-game
                    human_alive = False
                    ai_alive = False
                    game_over = True
                    winner_text = f"AI ({ai_name}) WINS! (FORFEIT)"

        # Find closest pipe ahead of the birds (bird.x is 230)
        pipe_ind = 0
        if len(pipes) > 1 and 230 > pipes[0].x + pipes[0].PIPE_TOP.get_width():
            pipe_ind = 1

        if not game_over:
            # --- AI bird logic ---
            if ai_alive:
                ai_bird.move()
                if ai_controller.should_jump(ai_bird, pipes[pipe_ind]):
                    ai_bird.jump()

                if pipes[pipe_ind].collide(ai_bird) or \
                   ai_bird.y + ai_bird.img.get_height() >= GROUND_Y or \
                   ai_bird.y < 0:
                    ai_alive = False

            # --- Human bird logic ---
            if human_alive:
                human_bird.move()
                if pipes[pipe_ind].collide(human_bird) or \
                   human_bird.y + human_bird.img.get_height() >= GROUND_Y or \
                   human_bird.y < 0:
                    human_alive = False

            # --- Pipe score tracking & Pipe creation ---
            add_pipe = False
            current_pipe = pipes[pipe_ind]

            # Score update for AI
            if ai_alive and not hasattr(current_pipe, 'ai_passed'):
                if current_pipe.x < ai_bird.x:
                    current_pipe.ai_passed = True
                    ai_score += 1

            # Score update for Human
            if human_alive and not hasattr(current_pipe, 'human_passed'):
                if current_pipe.x < human_bird.x:
                    current_pipe.human_passed = True
                    human_score += 1

            # Trigger new pipe generation
            if not current_pipe.passed and current_pipe.x < 230:
                current_pipe.passed = True
                add_pipe = True

            # Move pipes & clear offscreen
            rem = []
            for pipe in pipes:
                if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                    rem.append(pipe)
                pipe.move()

            if add_pipe:
                pipes.append(Pipe(600))

            for r in rem:
                pipes.remove(r)

            base.move()

            # Check normal end condition (both birds died naturally)
            if not ai_alive and not human_alive:
                game_over = True
                if ai_score > human_score:
                    winner_text = f"AI ({ai_name}) WINS!"
                elif human_score > ai_score:
                    winner_text = "YOU WIN!"
                else:
                    winner_text = "IT'S A TIE!"

        # Draw frame / Game over screen
        draw_game(
            win, ai_bird, human_bird, pipes, base,
            ai_score, human_score, ai_name, ai_alive, human_alive,
            game_over, winner_text
        )

        if game_over:
            run = False


if __name__ == "__main__":
    run_human_vs_ai()