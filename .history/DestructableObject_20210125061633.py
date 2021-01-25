import turtle as tl 
import pygame
import os
 

class DestructableObject(object):      
    
        nextId= 0     
        bounds = [0.0,0.0]
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GREEN = (0, 255, 0)
        RED =  (0, 0, 0)
    
        def __init__(self,screen,image, attackDmg=1, damage=0, health=1,startPosition=(0.0,0.0),movespeed=20,name = "Destructable-"):                    
                self.isDisposed = False
                self.font = pygame.font.SysFont('Arial', 12, True, False)
                self.name = name 
                self.title = self.font.render(self.name,True, DestructableObject.WHITE) 
                self.damage = damage
                self.health = health
                self.currentPos = startPosition
                if isinstance(image, str):  
                    self.image = pygame.image.load(image)
                else: self.image =image
                self.image_size = self.image.get_rect().size
                self.movespeed = movespeed; 
                self.attackDmg = attackDmg
                DestructableObject.bounds = pygame.display.get_window_size()
                DestructableObject.screen = screen

        def dispose(self):
            self.isDisposed = True
        
        def checkHealth(self):
            if self.health <= 0:
               self.dispose() 

        def takeDamage(self,damage = 1):
            if self.health - damage < 0: 
               self.health = 0
            else:
               self.health-=damage 
            self.checkHealth();

        def CheckOutOfBounds(self,axis):
            switch = {
             "xmax": lambda:(self.currentPos[0] > DestructableObject.bounds[0] - self.image_size[0]),
             "xmin": lambda:(self.currentPos[0] < 1),
             "ymax": lambda:(self.currentPos[1] > DestructableObject.bounds[1] - self.image_size[1]),
             "ymin": lambda:(self.currentPos[1] < 1)
            }
            res =switch.get(axis, "Only xmin..max,ymax.. allowed")
            return res() 

                     
        def draw(self):
                # DestructableObject.screen.blit(self.title,(self.currentPos[0]-13,self.currentPos[1]-15))  #print name & id  -35px behind object
                DestructableObject.screen.blit(self.image,self.currentPos)
                


 