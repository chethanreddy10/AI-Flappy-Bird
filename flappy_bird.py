import pygame
import neat
import os
import random
import pickle
from bird import Bird
from pipe import Pipe
from base import Base

pygame.font.init()  #initialize the font module of pygame

WIN_WIDTH = 500
WIN_HEIGHT = 800
DRAW_LINES = True
GEN=0
#global variable to keep track of the generation number.

BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))


STAT_FONT = pygame.font.SysFont("comicsans", 50)

def draw_window(win, birds,pipes, base,score,GEN,pipe_ind):
    win.blit(BG_IMG, (0, 0))
    for pipe in pipes:
        pipe.draw(win)
    base.draw(win)  #draw the base on the window.
   
    
    for bird in birds:
        #draw lines from bird to pipe
        if DRAW_LINES:
            try:
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width()/2, pipes[pipe_ind].height), 5)
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.get_width()/2, pipes[pipe_ind].bottom), 5)
            except:
                pass
        bird.draw(win)  #draw the bird on the window.
        
     
    #SCORE
    text= STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10)) #aaccomodate the text in the top right corner.
    #render the score text in the top right corner.
    
    #GENERATION
    text= STAT_FONT.render("Gen: " + str(GEN), 1, (255, 255, 255))
    win.blit(text, (10, 10)) #aaccomodate the text in the top right corner.

     # alive
    text = STAT_FONT.render("Alive: " + str(len(birds)),1,(255,255,255))
    win.blit(text, (10, 50))
        
    pygame.display.update()





def main(genomes,config):
    global GEN 
    GEN+=1 #increment the generation number for each call of main.
    #bird=Bird(220,340) #init bird at 220,340 pos.
    #we will use this to create a population of birds.
    #genomes are neural networks that will control the birds.
    nets=[]
    ge=[]
    birds=[]
    #each position of lists corresponds to a bird prop
    #genomes is a list of tuples (genome_id, genome) where genome is a neat genome object so _,g for destructuring.
    for _,g in genomes:
        net =neat.nn.FeedForwardNetwork.create(g,config)
        #create a neural network for each genome.
        nets.append(net)  #add the neural network to the list of networks.
        birds.append(Bird(230, 350))  #add a bird to the list of birds.
        g.fitness =0 #set the fitness of the genome to 0.
        ge.append(g)    #add the genome to the list of genomes.
        
    
    
    base = Base(730) #init base at 730 y pos
    pipes=[Pipe(600)] #init pipes at 600 x pos
    win=pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()  #for controlling the frame rate
    run =True
    score=0
    
    
    
    
    
    
    while run:
        clock.tick(30) #atmost 30 ticks frames per second.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        #bird.move()
        
        # at max 2 pipes are there at a time.
        pipe_ind=0
        if len(birds)>0:
            if len(pipes)>1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                #if there are more than 1 pipe and the bird is past the first pipe.
                pipe_ind=1
        #as input of network -> bird  y pos, pipe top y, pipe bottom y. so which pipe to use.
        
        else:
            run = False
            break
            #if there are no birds left then quit generation. 
        
        
        
        
        for x,bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1 #increment the fitness of the genome by 0.1 for each frame.
            #ass the bird is alive it gets a fitness score.
            #why low as it 30 frames per second so 3 sec = 3 fitness score.
            #encourage the bird to stay alive.

            output=nets[x].activate((bird.y, abs(bird.y- pipes[pipe_ind].height),abs(bird.y - pipes[pipe_ind].bottom)))
            #output is a list of values from the neural network. but in ome other cases other than flappy bird it can be a tuple.


            #how
            if output[0]>0.5:
                #output is -1 to 1 as activation function.
                #if the output of the neural network is greater than 0.5 then jump.
                bird.jump()

            # pass to the nn and get the output.
        
        
        
        
        #list of rem
        rem= []
        add_pipe=False
        for pipe in pipes:

            for x,bird in enumerate(birds):
                #for each bird in the population.
                if pipe.collide(bird):
                  #then i want to remove that bird also from the list of birds.  
                  #x is the index of the bird in the list of birds.
                  
                  ge[x].fitness -= 1 #decrease the fitness of the genome by 1.
                  #ex: two birds at same x dist one collide other not then the one that colliding will have low fitness.
                  #so that it will not be selected for the next generation.
                  
                  birds.pop(x) #remove the bird from the list of birds
                  nets.pop(x) #remove the net from the list of nets.
                  ge.pop(x)   #remove the genome from the list of genomes.
                  
                  
                #as soon as a pipe is passed by the bird we set the passed to true.
                #dont confuse pipe.passed in init with false.
                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True #we need to add a new pipe.
                    #if the pipe is not passed and the x pos of the pipe is less than the x pos of the bird.
                    #then we set the passed to true.
                    #this is used for AI to know if it has passed the pipe or not.
           
            #one a pipe reaches a certaion x pos then we add a new pipe.
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)
                #completely of the screen so remove it.
            pipe.move()
        
        if add_pipe:
           score += 1 #increment the score if a pipe is passed.
           for g in ge:
               g.fitness +=5 #reward the genome for passing a pipe.
           pipes.append(Pipe(600)) #add a new pipe at 600 x pos respawn

        for r in rem:
            pipes.remove(r)
            #now remove.
        
        for x,bird in enumerate(birds):
            if(bird.y + bird.img.get_height() >= 730 or bird.y < 0):
                #if the bird is out of the screen then quit the game.
                #hit the base or the top of the screen.
                #in og version only bottom is checked.
                
                #we will remove only . we will not decrease the fitness of the genome.'
                ge[x].fitness -= 1 #decrease the fitness of the genome by 1.
                birds.pop(x) #remove the bird from the list of birds    
                nets.pop(x) #remove the net from the list of nets.
                ge.pop(x)   #remove the genome from the list of genomes.
        
        if(score > 100):
            #if the score is greater than 100 then quit the game.
            run = False
            break
            #as it is a simple game so we can quit after 100 score.
        
        
        
        
        base.move()
        #move the bird according to the physics and every frame.
        #if applied directly then quickly down.. so tick rate for while loop is needed.
        #so that it moves at a constant speed.        
        draw_window(win, birds,pipes,base,score,GEN-1,pipe_ind)
        
        # break if score gets large enough
        if score > 50:
           # Find the genome with highest fitness
           best_genome = max(ge, key=lambda g: g.fitness)
           
           # Get its corresponding network
           best_net = neat.nn.FeedForwardNetwork.create(best_genome, config)
           
           # Save both the genome and network
           pickle.dump({
               'genome': best_genome,
               'network': best_net
           }, open("best.pickle", "wb"))
           break

   #main()


def run(config_path):
   config= neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path) # create the config object
   
   p=neat.Population(config)  #create a population of genomes   
   
   
   
   p.add_reporter(neat.StdOutReporter(True))  #population to print the output to the console
   stats=neat.StatisticsReporter()  #create a statistics reporter to keep track of the population
   p.add_reporter(stats)  #add the statistics reporter to the population
   
   
   winner=p.run(main,50) #run for max 50 until generations.
   #the run method takes a function as an argument which is called for each genome in the population.
   #the function is responsible for evaluating the genome and returning a fitness score.
   
   # show final stats
   print('\nBest genome:\n{!s}'.format(winner))

if __name__ == "__main__":
    
    local_dir = os.path.dirname(__file__)# path of the current file
    config_path = os.path.join(local_dir, "config-feedforward.txt")  # path to the config file
    #neat config file for the neural network.
    
    run(config_path)  # run the neat algorithm with the config file.
    
    
    
    
    
    
    #note: if pop size so high then it essentially cereates many variations of neural networks
    #we got the best neural net in that way luckily....
    #as this is a simple game so it works well.
    #for complex games like chess or go it takes a lot of time to train the neural network.