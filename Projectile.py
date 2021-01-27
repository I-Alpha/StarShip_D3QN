from Obstacle import Obstacle
import pygame

class Projectile(Obstacle): 
        out_of_bounds = False
        def __init__(self,screen,image,attackDmg=1,damage=0,health=1,movespeed=-1,startPosition=(0.0,0.0),name="Projectile-"):            
                super().__init__(screen,image,attackDmg,damage,health,movespeed,startPosition)        
                self.name = name    
                self.title = self.font.render(self.name,True, self.WHITE)

        def dispose(self):             
            self.isDisposed = True
            del self
        
        def draw(self):                
                # Projectile.screen.blit(self.title,(self.currentPos[0]-13,self.currentPos[1]+5))
                Projectile.screen.blit(self.image,self.currentPos) 
                self.updatePosition()
             
        def updatePosition(self):
               self.currentPos=(self.currentPos[0],self.currentPos[1]+self.movespeed) 
               self.boundsCheck()

        def boundsCheck(self):
               if(self.CheckOutOfBounds("ymin")): 
                    self.currentPos  =  (self.currentPos[0],0-self.image_size[1])
                    self.dispose()
                    Projectile.out_of_bounds=True  
        
        def reset(): 
                Projectile.out_of_bounds=0