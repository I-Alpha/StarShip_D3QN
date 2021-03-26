from DestructableObject import DestructableObject
from Projectile import Projectile
import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_LEFT, K_RIGHT, K_SPACE, K_a, K_d,K_UP,K_DOWN


class SpaceShipSprite(DestructableObject):

      deadProjectiles =0;
      liveProjectiles =[]
      
      

      def __init__(self,screen,image,attackDmg= 1,damage=0,health=4,startPosition=(0.0,0.0),ammoMax=5, name="SpaceShip-"):
               super().__init__(screen,image,attackDmg,damage,health,startPosition)               
               self.rect = (image)
               self.name = name 
               self.title = self.font.render(self.name,True, DestructableObject.WHITE) 
               self.currentAmmo = ammoMax
               self.ammoMax = ammoMax
               self.screen = screen
               self.lives =1 
               self.maxhp=health
               self.ammoCounter =.3
               self.firedAt=0      
               self.maxProjectiles_on_screen =15
               self.action_status = 0
               SpaceShipSprite.liveProjectiles =[]
      def move(self,keys):
           self.action_status = 0
           event = "none"         
           if  isinstance(keys,type(pygame.key.get_pressed())):
                if len(keys)>0:
                        if (keys[K_SPACE]):
                              self.fireprojectile() 
                        elif (keys[K_RIGHT]):
                              event ="right"
                        elif (keys[K_LEFT]):
                              event ="left"
                        elif (keys[K_UP]):
                              event ="up"
                        elif (keys[K_DOWN]):
                              event ="down"
           elif isinstance(keys,str):
                  event = keys
                  if keys=="space" and len(SpaceShipSprite.liveProjectiles) < self.maxProjectiles_on_screen:
                        self.fireprojectile() 

           self.currentPos = self.updatePosition(event)
           self.boundsCheck()
           return self.action_status
            
 
      def updatePosition(self,argument):
            switcher = {
                "left": lambda: (self.currentPos[0]-self.movespeed,self.currentPos[1]),
                "right": lambda: (self.currentPos[0]+self.movespeed,self.currentPos[1]),
                "up": lambda: (self.currentPos[0],self.currentPos[1]-self.movespeed),
                "down": lambda: (self.currentPos[0],self.currentPos[1]+self.movespeed),
                }
            res = switcher.get(argument,"Invalid Movement") 
            return self.currentPos if res == "Invalid Movement" else res() 
               

      def boundsCheck(self):
            if(self.CheckOutOfBounds("ymin")):
                   self.currentPos  =  (self.currentPos[0],5)
                   self.action_status = - 1
            elif(self.CheckOutOfBounds("ymax")):
                   self.currentPos  =  (self.currentPos[0],DestructableObject.bounds[1]-self.image_size[1]-5)         
                   self.action_status = - 1  
            elif(self.CheckOutOfBounds("xmin")): 
                  self.currentPos  =  (5,self.currentPos[1])       
                  self.action_status = - 1 
            elif(self.CheckOutOfBounds("xmax")):
                   self.currentPos  =  (DestructableObject.bounds[0]-self.image_size[0]-5,self.currentPos[1])  
                   self.action_status = - 1
      def fireprojectile(self):      
            newprojectile = Projectile(self.screen,r'Assets\imgs\bullet1.png',movespeed=-8,startPosition=(self.currentPos[0],self.currentPos[1]-14))
            mps = pygame.time.get_ticks()/1000 - self.firedAt/1000
            if self.currentAmmo > 0 :
                   if mps > 1 and self.currentAmmo < self.ammoMax:
                         self.currentAmmo += 1
                   if mps > .4:
                         self.firedAt = pygame.time.get_ticks()
                         SpaceShipSprite.liveProjectiles.append(newprojectile) 
                         self.currentAmmo-=1
                   else:
                        self.action_status = -1
            else:                   
                  if mps >= self.ammoCounter:
                        self.currentAmmo = self.ammoMax
                        self.fireprojectile()
                  else:
                        self.action_status = -1
         
         

      def takeDamage(self,obs):
            if self.health > 0:  
                  self.health -= obs.attackDmg
            else:
                  self.lives -= 1
                  self.health= self.maxhp
                  if self.lives ==0:
                        return False
            return True  