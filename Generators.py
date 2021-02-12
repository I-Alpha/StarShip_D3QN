import turtle as tl 
import pygame
import random 
import copy
import os
from Obstacle import Obstacle
from Projectile import Projectile
class ObstacleGenerator():       
    liveProjectiles =[] 
    deadProjectiles = 0
    hits = 0;
    fails = 0;
    liveObstacles = []
    deadObstacles = 0
    r_num = 0
    prevnum = 0
    prevPos = 0
    p_out_of_bounds=False
    def __init__(self,screen, image = r'Assets\imgs\brick.png'):     
       ObstacleGenerator.screen = screen
       ObstacleGenerator.bounds = screen.get_size()
       ObstacleGenerator.image = image 
       ObstacleGenerator.prevT = pygame.time.get_ticks()
       ObstacleGenerator.nextId=0         
       self.levelHeight = 70 
       self.obsimg = Obstacle(screen,image).image_size
       self.initilaizeRandom()
       self.reset()
      
    def drawAll(self): 
           for i in ObstacleGenerator.liveObstacles: 
               i.draw()
           for x in ObstacleGenerator.liveProjectiles:
               x.draw()
         
    def initilaizeRandom(self,snum=5):     
        
          if ObstacleGenerator.r_num == 20000 :
             ObstacleGenerator.r_num=0
          else:
            ObstacleGenerator.r_num+=1
   # get step within range of game window X bounds -50 each side 
          step = ((ObstacleGenerator.bounds[0])-10)/5
           # generate a list of   
          random.seed(ObstacleGenerator.r_num)
          ObstacleGenerator.prevnum = pygame.time.get_ticks()+1/(pygame.time.get_ticks()*random.randint(0,1)+1)
          ObstacleGenerator.listPos = random.sample(range(20,ObstacleGenerator.bounds[0]- self.obsimg[0],int(step+1)),snum)
          if snum == 1 : 
                while ObstacleGenerator.listPos ==  ObstacleGenerator.prevPos  :
                    ObstacleGenerator.listPos = random.sample(range(20,ObstacleGenerator.bounds[0]-self.obsimg[0],int(step+1)),snum)
                ObstacleGenerator.prevPos = ObstacleGenerator.listPo
          random.shuffle(ObstacleGenerator.listPos)
 
          
             

             

    def updateAll(self):
           for i in ObstacleGenerator.liveObstacles:
               i.updatePosition()               
               ObstacleGenerator.fails += Obstacle.fails;
           self.updateList() 

    def updateList(self):           
           
           for i in ObstacleGenerator.liveObstacles: 
               if i.isDisposed == True:
                   ObstacleGenerator.liveObstacles.remove(i)
                   ObstacleGenerator.deadObstacles+=1
                   del i
           for x in ObstacleGenerator.liveProjectiles:
               if x.isDisposed == True:
                   ObstacleGenerator.liveProjectiles.remove(x)
                   ObstacleGenerator.deadProjectiles+=1
                   del x
           ObstacleGenerator.p_out_of_bounds = Projectile.out_of_bounds
           Projectile.out_of_bounds = False

    def generate(self,delay=50000,num=5,seed=5, limit = 5):
           diff = pygame.time.get_ticks() - ObstacleGenerator.prevT
           if delay== 0:
               self.generateLevels(num,seed)
           elif (diff>delay) and (len(ObstacleGenerator.liveObstacles)<=limit):
                self.generateLevels(num,seed)
                ObstacleGenerator.prevT +=diff 
          

    def generateLevels(self,num=5,seed=5,delay=0):             
               for i in self.generateObstacles(num,seed):
                   ObstacleGenerator.liveObstacles.append(i)

    def generateObstacles(self,num=5,seed=5,snum=5):                       
   
            self.initilaizeRandom(snum)
            
            if num <5 : 
                ObstacleGenerator.listPos = random.sample(ObstacleGenerator.listPos,num)           
            tempList = []
            for i in ObstacleGenerator.listPos:  
                self.temp = Obstacle(ObstacleGenerator.screen, ObstacleGenerator.image, name ="G-O-",movespeed=4)
                self.temp.currentPos = (float(i),0.0)             
                tempList.append(self.temp)
            return tempList

 
          

    def checkHits(self):
           #check if any projectile and obstacles have made contact 
           for i in self.liveObstacles:
                obYcoord = i.currentPos[1]
                obXcoord = i.currentPos[0]
                obHeight = i.image_size[1]
                obWidth = i.image_size[0] 
                if any(((x.currentPos[0] + x.image_size[0]>=obXcoord) and (x.currentPos[0] <= obXcoord + obWidth) and (x.currentPos[1] + x.image_size[1]>=obYcoord) and (x.currentPos[1] <= obYcoord + obHeight)) for x in ObstacleGenerator.liveProjectiles):
                    xlist =[x for x in ObstacleGenerator.liveProjectiles if ((x.currentPos[0] + x.image_size[0]>=obXcoord) and (x.currentPos[0] <= obXcoord + obWidth) and (x.currentPos[1] + x.image_size[1]>=obYcoord) and (x.currentPos[1] <= obYcoord + obHeight))]
                    for x in xlist:
                        if len(xlist)>2:
                            print("hey")
                        i.health -= x.attackDmg
                        if i.health<1:
                            i.dispose()                  
                        x.dispose() 
                        del x
                        self.updateList()                
                        self.generate(num=1, delay =1000,seed=pygame.time.get_ticks())  
                        ObstacleGenerator.hits+=1                                        
                    return True
           return False

           
    def reset(self):
        ObstacleGenerator.deadProjectiles=0
        ObstacleGenerator.liveProjectiles=[]
        ObstacleGenerator.liveObstacles = []
        ObstacleGenerator.deadObstacles = 0
        ObstacleGenerator.hits = 0; 
        ObstacleGenerator.prevT = pygame.time.get_ticks()
        ObstacleGenerator.nextId=0 ;
        ObstacleGenerator.fails=0;
        Obstacle.fails =0
        ObstacleGenerator.p_out_of_bounds=0
        Projectile.reset() 
        
