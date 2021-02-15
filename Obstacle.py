from DestructableObject import DestructableObject
 
import pygame
import random 
import copy

class Obstacle(DestructableObject): 

       def __init__(self,screen,image,attackDmg=1,damage=0,health=1,movespeed=1,startPosition=(0.0,0.0),name="Obstacle-"):            
               super().__init__(screen,image,attackDmg,damage,health,startPosition,movespeed)  
               self.name = name      
               self.title = self.font.render(self.name,True, self.WHITE) 
               self.count=0
               self.pos=(300,300)
               self.obs_ID = 0
 
       def dispose(self):
            self.isDisposed = True
            del self

       def updatePosition(self):
               self.currentPos=(self.currentPos[0],self.currentPos[1]+self.movespeed) 
               self.boundsCheck()
               if (self.count < 1000):
                   #self.printState()
                   self.pos=(self.pos[0]-.10,self.pos[1]-.07)

       def boundsCheck(self):
           if(self.CheckOutOfBounds("ymax")): 
              self.currentPos  =  (self.currentPos[0],self.bounds[1]+self.image_size[1])
              self.dispose()
              Obstacle.fails+=1; 
             
       