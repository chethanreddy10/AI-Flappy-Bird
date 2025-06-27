import pygame
import os

BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))

class Base:
    #as it is moving left to right. so it should be a class.
    VEL = 5 #speed of the base moving left to right. same as pipe.
    WIDTH = BASE_IMG.get_width() #width of the base image
    IMG= BASE_IMG
    
    
    # 2 images of the base are needed to create the effect of moving left to right.
    def __init__(self, y):
        self.y = y
        self.x1 = 0 #x position of the first base image
        self.x2 = self.WIDTH #x position of the second base image
        
    def move(self):
        #move the base left by the velocity.
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            # we move the first iamge to the right of the second image.
            self.x1 = self.x2 + self.WIDTH
            
            
        if self.x2 + self.WIDTH < 0:
            # we move the first iamge to the right of the second image.
            self.x2 = self.x1 + self.WIDTH
            
            # like a cycle of 2 images. it keeps the screen full of base.
    
    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))
