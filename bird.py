
import pygame
import os

BIRDS_IMGS = [
    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))
]

class Bird:
    IMGS = BIRDS_IMGS
    MAX_ROTATION = 25 #looks like aiming towards sky.
    ROT_VEL=20  # arbitary based on choice each frame. every time we move the bird 20 deg per frame
    ANIMATION_TIME= 5 #how long bird animation should be.
    
    def __init__(self,x,y):
        self.x=x
        self.y=y #intit position of bird
        self.tilt =0 # look flat
        self.tick_count =0
        self.vel=0
        self.height =0
        self.img_count=0 # which img showed currently
        self.img=self.IMGS[0]
        #init done
    def jump(self):
        self.vel = -10.5    #(0,0) is the top left. 
        self.tick_count=0   #as soon as we jump reset so that freshly equn (displacement) can be applied.
        self.height=self.y  #from where jumping
    
    def move(self):
        self.tick_count+=1 # a tick happend a frame went by
        
        displacement = self.vel*self.tick_count +1.5*self.tick_count**2
        # -10.5*1 +1.5*1**2=-9 => 9 pixels upward..
        #upwards is - disp downword is pos disp
        #there 3 is the arbitary g constant instead of real for smooth exp
        
        
        if displacement>=16 :
            #terminal vel check
            #if moving down more than 16 pixels down...make sure it not fall infintly
            displacement = 16
        
        if displacement<0:
            displacement-=2 # Makes upward jumps feel more responsive and floaty. your choice!
        
        self.y=self.y + displacement
        
        if displacement<0 or self.y <self.height +50:
            if self.tilt < self.MAX_ROTATION:
                #in that direction tilts upwar to MAX_ROT.
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL
                       
    def _animate(self):
        """Update the bird animation frame."""
        self.img_count += 1
        
        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count >= self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2

    def draw(self, win):
        self._animate()
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        # rotate the image about the center of the bird's rectangle, and then get the new rectangle that bounds the rotated image. This ensures that the bird is drawn at the correct position after rotation.
        win.blit(rotated_image, new_rect.topleft)
        # blit -> blit is a method in Pygame that draws one image onto another. In this case, it draws the rotated bird image onto the game window.
        # win -> the game window (destination)
        # rotated_image   -> bird image after rotation (source)
        # new_rect.topleft -> where to place it
        # Draw the rotated bird image on the window at this position.

    def draw_offset(self, win, x_offset):
        """Draw the bird at an x-offset (for split-screen mode)."""
        self._animate()
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x + x_offset, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)


    def get_mask(self):
        return pygame.mask.from_surface(self.img)
    #get the mask of the image for collision detection.

