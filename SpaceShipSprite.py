from DestructableObject import DestructableObject
from Projectile import Projectile
import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_LEFT, K_RIGHT, K_SPACE, K_a, K_d,K_UP,K_DOWN


class SpaceShipSprite(DestructableObject):

      deadProjectiles =0;
      liveProjectiles =[]
      
      

      def __init__(self,screen,image,attackDmg= 1,damage=0,health=4,startPosition=(0.0,0.0),ammoMax=4, name="SpaceShip-"):
               super().__init__(screen,image,attackDmg,damage,health,startPosition)               
               self.rect = (image)
               self.name = name 
               self.title = self.font.render(self.name,True, DestructableObject.WHITE) 
               self.currentAmmo = ammoMax
               self.ammoMax = ammoMax
               self.screen = screen
               self.lives =1 
               self.maxhp=health
               self.ammoCounter =5500
               self.firedAt=0      
               self.maxProjectiles_on_screen =20
               SpaceShipSprite.liveProjectiles =[]
      def move(self,keys):
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
            if(self.CheckOutOfBounds("ymin")): self.currentPos  =  (self.currentPos[0],5)
            if(self.CheckOutOfBounds("ymax")): self.currentPos  =  (self.currentPos[0],DestructableObject.bounds[1]-self.image_size[1]-5)         
            if(self.CheckOutOfBounds("xmin")): self.currentPos  =  (5,self.currentPos[1])        
            if(self.CheckOutOfBounds("xmax")): self.currentPos  =  (DestructableObject.bounds[0]-self.image_size[0]-5,self.currentPos[1])  
               
      def fireprojectile(self):      
            newprojectile = Projectile(self.screen,r'Assets\imgs\bullet1.png',movespeed=-8,startPosition=(self.currentPos[0]+5,self.currentPos[1]-14))
        
            if self.currentAmmo > 0 :
                   self.firedAt = pygame.time.get_ticks()
                   SpaceShipSprite.liveProjectiles.append(newprojectile) 
                   self.currentAmmo-=1
            else:            
                  reloadTime = pygame.time.get_ticks() - self.firedAt
                  if reloadTime >= self.ammoCounter:
                        self.currentAmmo = self.ammoMax
                        self.fireprojectile()
         
         

      def takeDamage(self,obs):
            if self.health > 0:  
                  self.health -= obs.attackDmg
            else:
                  self.lives -= 1
                  self.health= self.maxhp
                  if self.lives ==0:
                        return False
            return True  