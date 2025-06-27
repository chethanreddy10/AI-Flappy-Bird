import pygame
import random
import os


PIPE_IMG= pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))

class Pipe:
    GAP = 200 #gap between the top and bottom pipe
    VEL = 5 #speed of the pipe moving left to right. as flappy bird doesnt move horizonatally it gives the effect of moving left.
    
    def __init__(self,x):
        self.x = x  # y because we do that randomly calc
        self.height = 0
        self.gap = 100
        
        self.top = 0 #?
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)#flip the pipe image for top pipe.
        self.PIPE_BOTTOM = PIPE_IMG
        
        self.passed = False #if the bird has passed the pipe or not. for collision detection, Ai
        self.set_height()
        
    def set_height(self):
        self.height = random.randrange(50, 450) #random height of the pipe
        self.top = self.height - self.PIPE_TOP.get_height() #bottom of the top pipe !!!doubt
        self.bottom = self.height + self.GAP # top of the bottom pipe
    
    def move(self):
        self.x -= self.VEL
        #move the pipe left by the velocity.
        
    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))    
    
    
    #!IMPORT
    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)    
        
        #get the mask of the bird and the top and bottom pipe.
        #round of bord y as it is a float value.
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))
        
        
        #check for collision
        b_point = bird_mask.overlap(top_mask, top_offset) #get the point of collision
        t_point = bird_mask.overlap(bottom_mask, bottom_offset) #get the point of collision
        
        if b_point or t_point: #if there is a point of collision
            return True #collision detected
        return False
