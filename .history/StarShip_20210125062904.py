import pygame
from SpaceShipSprite import SpaceShipSprite
import sys
from Generators import ObstacleGenerator
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_UP,K_DOWN, K_LEFT, K_RIGHT, K_SPACE, K_a, K_d,K_s
from Obstacle import Obstacle
import numpy as np
import copy 
import pandas as pd
import cv2   

class StarShipGame:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    liveObstacles = []
    deadObstacles = 0
    liveProjectiles =[]

    def __init__(self,graphics,screen_size =(400, 400)):
        pygame.font.init()
        pygame.init()
        self.action=0
        StarShipGame.screen = pygame.display.set_mode(screen_size)
        self.done = False
        self.playing = False
        self.reward = 0
        self.graphics = graphics 
        self.score =0
        self.key =0
        #background  
        self.font = pygame.font.SysFont('Arial', 25, True, False)     
        self.spaceShipSprite = SpaceShipSprite(StarShipGame.screen,r'Assets\imgs\triangleShip.png',startPosition=((screen_size[0]/2)-60,screen_size[1]/2))   
        self.obstacleGenerator = ObstacleGenerator(StarShipGame.screen,r'Assets\imgs\brick.png')
        self.screen_size = screen_size
        self.FPS = 60
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("StarShip")
        self.obstacleGenerator.generate(0); 
        StarShipGame.liveObstacles = self.obstacleGenerator.liveObstacles
        self.save=False 
        self.REM_STEP = 4
        self.ROWS =  100
        self.COLS = 100
        self.image_memory = np.zeros((self.REM_STEP, self.ROWS, self.COLS))
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
        self.x =0 

    def vectorize_func(self,m):
        return m/255
    
    def render(self):            
        return (copy.deepcopy(pygame.surfarray.array3d(self.screen)))
    
    def render_wrap(self):
        temp = self.render()
        temp=np.reshape(temp,(-1,3))
        temp= pd.DataFrame(temp,columns=['r','g','b'])
        # conditions = [ 
        #     (temp == 255)
        # ]
        # temp2=pd.Series()
        # temp2['r'] = 0         
        # temp2['g'] = 0         
        # # temp2['b'] = 0 
        # temp.loc[(temp['r'] >1) & (temp['g']>1) & (temp['b']>1)] =  [0,0,0]

        # temp =    np.where(temp['r'] == 0 & temp['g']==3 & temp['b'] == 0 ),[22,33,44], temp).tolist()
        # temp =    np.where(temp == [255,255,255] ,[0,0,0], temp).tolist()
        return temp.values
    

    def getPixelsOnScreen(self):  
        temp =copy.deepcopy(pygame.surfarray.array3d(self.screen))
        temp=np.reshape(temp,(-1,3))
        temp= pd.DataFrame(temp,columns=('r','g','b'))
        conditions = [
            (temp['r'] == 190) & (temp['g'] == 0) & (temp['b'] == 0),
            (temp['r'] == 0) & (temp['g'] == 255) & (temp['b'] == 0),
            (temp['r'] == 0) & (temp['g'] == 0) & (temp['b'] == 255)
            (temp['r'] == 0) & (temp['g'] == 0) & (temp['b'] == 255)
        ]
        choices = [1, 2, 3]
        #np.select(conditions, choices, default=0)
      
        return self.vectorize_func(np.select(conditions, choices, default=0))
       
        # new =[]
        # for x,i in enumerate(temp):
        #     new.append(colors.get((tuple)(i),0))

        # return new
        # self.vectorize_func(temp)
      #    temp= [ p.array([0,0,0],dtype='uint8') if np.array_equal(i,np.array([255,255,255])) else i for i in j  for j in temp]
      #   # for y in temp:
        # #test for white pixels
        #     if np.array_equal(y,target):
        #         print("W found")
    def getPixelsOnScreenNew(self):            
        img= self.render_wrap()
 
        img=np.reshape(img,(400,400,3)) 
            
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   
        img = cv2.resize(img, (100, 100))  
        img = np.transpose(img)
        ret, img = cv2.threshold(img,1,255,cv2.THRESH_BINARY)   
        img = self.vectorize_func(img)
        self.image_memory = np.roll(self.image_memory, 1, axis = 0)
        self.image_memory[0,:,:] = img
           
        cv2.namedWindow("Input")
        cv2.imshow("Input",  np.reshape(self.image_memory,(200,200,1)))
        #np.select(conditions, choices, default=0)
        return np.expand_dims(self.image_memory, axis=0)

    def getEnvStateOnScreen(self):
        obs_coor=[]
        for obs in  self.obstacleGenerator.liveObstacles :
               x,y= self.Obs_pos(obs)
               obs_coor.append([x,y])
        agentX,agentY = self.agent_pos()    
        state = [
            #*self.getPixelsOnScreen()              
		    [agentX,agentY],
             *obs_coor,
            [0,self.obstacleGenerator.hits],          
            [0,len(self.obstacleGenerator.liveObstacles)],
            [0,self.obstacleGenerator.deadObstacles], 
            [0,self.obstacleGenerator.fails] ,
            [0,self.clock.get_time()] 
        ]
        state = np.reshape(state,(11,2))
        self.image_memory = np.roll(self.image_memory, 1, axis = 0)
        self.image_memory[0,:,:] = state
        #np.select(conditions, choices, default=0)
        return np.expand_dims(self.image_memory, axis=0)

    def game_loop(self):
            for event in pygame.event.get():
                if event.type == QUIT:           # terminates the game when game window is closed
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:    # terminates the game when esc is pressed
                        pygame.quit()
                        sys.exit()  
                    if event.key == K_s:  
                        self.save=True
                  
                          
            self.obstacleGenerator.updateAll()
            StarShipGame.liveProjectiles=self.spaceShipSprite.liveProjectiles             
            ObstacleGenerator.liveProjectiles=StarShipGame.liveProjectiles        
            hit = self.obstacleGenerator.checkHits()       
            if hit:
                self.reward+=.5
                hit=False            
            if ObstacleGenerator.p_out_of_bounds:
                self.reward-=.2
                ObstacleGenerator.p_out_of_bounds=0
            self.checkHits()
            if (ObstacleGenerator.fails) > 0 :
                   self.done=True

    def play(self):
        self.playing = True     
  
        while True: #begin gameloop 
            # handle mouse and keyboard events   
            self.clock.tick(self.FPS)
            self.game_loop()            
            keys = pygame.key.get_pressed()
            self.spaceShipSprite.move(keys)

            #termination condition 
            if self.check_shipalive() or ObstacleGenerator.fails>0 or Obstacle.fails  > 0: 
               ObstacleGenerator.reset()
               self.spaceShipSprite.liveProjectiles=[]
               self.done=True
               self.reward += int(self.clock.get_time()/1000)
               pygame.quit()
               sys.exit()
            
            if self.graphics:              
                self.draw_all()  
                pygame.display.update()
        pygame.quit()
    
    def reset(self):
        
        self.spaceShipSprite = SpaceShipSprite(StarShipGame.screen,r'Assets\imgs\triangleShip.png',startPosition=(self.screen_size[0]/2,self.screen_size[1]/2))   
        SpaceShipSprite.liveProjectiles=[]  
        ObstacleGenerator.reset()
        self.obstacleGenerator = ObstacleGenerator(StarShipGame.screen,r'Assets\imgs\brick.png')
        StarShipGame.liveObstacles = []
        StarShipGame.deadObstacles = 0
        StarShipGame.liveProjectiles =[]      
        ObstacleGenerator.fails=0
        Obstacle.fails=0
        self.score =0
        self.hit=0
        self.fails=0
        self.reward= 0
        StarShipGame.fails =0
        self.done=False
        self.clock = pygame.time.Clock()     
        self.obstacleGenerator.generate(0);
        StarShipGame.liveObstacles =  self.obstacleGenerator.liveObstacles 
        obs_coor=[]
        for obs in  self.obstacleGenerator.liveObstacles :
               x,y= self.Obs_pos(obs)
               obs_coor.append([x,y])
        agentX,agentY = self.agent_pos()    
        state = [
            #*self.getPixelsOnScreen()              
		    [agentX,agentY],
            *obs_coor, 
            [-1,self.obstacleGenerator.hits],          
            [-1,len(self.obstacleGenerator.liveObstacles)],
            [-1,(self.obstacleGenerator.deadObstacles)], 
            [-1,self.obstacleGenerator.fails] ,
            [-1,self.clock.get_time()] 
          ]
        
        pygame.Surface.unlock(self.screen)
        # for i in obs_coor:
        #     state.append(i) 
        return state
    
    def resetNew(self):        
        self.spaceShipSprite = SpaceShipSprite(StarShipGame.screen,r'Assets\imgs\triangleShip.png',startPosition=(self.screen_size[0]/2,self.screen_size[1]/2))   
        SpaceShipSprite.liveProjectiles=[]  
        ObstacleGenerator.reset()
        self.obstacleGenerator = ObstacleGenerator(StarShipGame.screen,r'Assets\imgs\brick.png')
        StarShipGame.liveObstacles = []
        StarShipGame.deadObstacles = 0
        StarShipGame.liveProjectiles =[]      
        ObstacleGenerator.fails=0
        Obstacle.fails=0
        self.score =0
        self.hit=0
        self.fails=0
        self.x =0
        self.score =0
        self.image_memory = np.zeros((self.REM_STEP, self.ROWS, self.COLS))
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
        self.reward= 0
        StarShipGame.fails =0
        self.done=False
        self.clock = pygame.time.Clock()     
        self.obstacleGenerator.generate(0);
        StarShipGame.liveObstacles =  self.obstacleGenerator.liveObstacles  
        for i in range(self.REM_STEP):
              state = np.reshape(self.getPixelsOnScreenNew(),(4,400,400,1))
        # for i in obs_coor:
        #     state.append(i) 
      
        return (state).flatten()
  
    def check_shipalive(self):
                if self.spaceShipSprite.lives <1 and self.spaceShipSprite.health==0:
                     print("You died !") 
                     self.done= True
                     return True
                if ObstacleGenerator.fails>0:
                    self.done =True
                return False
            
    def checkHits(self): 
                i = self.spaceShipSprite
                if any((( i.currentPos[0] + i.image_size[0] >= x.currentPos[0]) and (i.currentPos[1] >=  x.currentPos[1] - i.image_size[1]) and (i.currentPos[0] <= x.currentPos[0] + x.image_size[0]) and  (i.currentPos[1] <= x.currentPos[1] + x.image_size[1])) for x in StarShipGame.liveObstacles):
                   #if collision
                    y =next(x for x in StarShipGame.liveObstacles if (( i.currentPos[0] + i.image_size[0] >= x.currentPos[0]) and (i.currentPos[1] >=  x.currentPos[1] - i.image_size[1]) and (i.currentPos[0] <= x.currentPos[0] + x.image_size[0]) and  (i.currentPos[1] <= x.currentPos[1] + x.image_size[1])))
                    alive = (i.takeDamage(y))
                    y.dispose()
                    StarShipGame.liveObstacles.remove(y)
                    self.obstacleGenerator.liveObstacles =StarShipGame.liveObstacles
                    self.obstacleGenerator.updateList()
                    self.obstacleGenerator.generate(delay=0,num=1)
                    StarShipGame.liveObstacles =self.obstacleGenerator.liveObstacles                 
                
                    if alive :
                        self.reward-=.2
                    else: 
                        self.reward-=.5


    def draw_score(self,color ='white'):
            self.x += self.reward
            scoreboard = ")---Score-->  " + "{:.2f}".format(self.x) + "/-hits : " +  str(self.obstacleGenerator.hits) + "/fails : " +  str(ObstacleGenerator.fails) + "/liveObs : " +  str(len(self.obstacleGenerator.liveObstacles))  +"/Missiles live: "+ str(len(self.obstacleGenerator.liveProjectiles)) +"/key : " +  str((self.key)) +"/Action : " +  str((self.action))+ "/deadObs : " +  str(self.obstacleGenerator.deadObstacles) + "/Ship-lives : " +  str((self.spaceShipSprite.lives))+ "/Ship-health: " +  str((self.spaceShipSprite.health))   
            ls = scoreboard.split("/");
            xmov=0
            if color == "black":
               color = StarShipGame.BLACK
            else :
               color = StarShipGame.WHITE
            # for i in ls:
            #         scoreboardf = self.font.render(i,True, pygame.Color(color))
            #         StarShipGame.screen.blit(scoreboardf, [50,100+xmov]) 
            #         xmov +=25        
            
            # Create a black image
            img = np.zeros((512,512,3), np.uint8)

            # Write some Text

            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (50,30)
            fontScale              = 1
            fontColor              = (255,255,255)
            lineType               = 2
            for i in ls:
                cv2.putText(img,i, 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
                bottomLeftCornerOfText = (bottomLeftCornerOfText[0],bottomLeftCornerOfText[1] + 50)
            
            cv2.imshow('input',img)
                    

    def draw_all(self,xmov=.10):
         #clear/fill screen
            self.draw_score()
            StarShipGame.screen.fill(pygame.Color(StarShipGame.BLACK))                
            # display image on screen
            # screen.blit(image, [600, 100])
            self.spaceShipSprite.draw()            
            self.obstacleGenerator.drawAll() 


    


    def step(self,action):
        self.done=0
        self.reward =0
        key = "nothing"
        #key pressed
        
        #left
        if action == 0:
            self.reward -= .1
            key="left"
         
        #right
        elif action == 1: 
            self.reward -= .1
            key="right"
        #Up
        elif action == 2: 
            self.reward -= .1
            key="up"
        #Down
        elif action == 3: 
            self.reward -= .1
            key="down"

        #Space-Shoot
        elif action == 4: 
            self.reward -=.1      
            key="space"

        elif action == 5:
            self.reward -= .1

        self.key=key 
        self.action=action

        self.spaceShipSprite.move(key)
        self.game_loop() 
        obs_coor=[]
        n=len(self.obstacleGenerator.liveObstacles)
        if  n  < 5:
            x=5-n
            self.obstacleGenerator.generate(delay=0,num=x)
        StarShipGame.liveObstacles =self.obstacleGenerator.liveObstacles       
        obs_coor=[]          
        for obs in  self.obstacleGenerator.liveObstacles :
               x,y= self.Obs_pos(obs)
               obs_coor.append([x,y])
        agentX,agentY = self.agent_pos()    
        state = [
            #*self.getPixelsOnScreen()              
		    [agentX,agentY],
            *obs_coor, 
            [-1,self.obstacleGenerator.hits],          
            [-1,len(self.obstacleGenerator.liveObstacles)],
            [-1,(self.obstacleGenerator.deadObstacles)], 
            [-1,self.obstacleGenerator.fails] ,
            [-1,self.clock.get_time()] 
          ]
        
        pygame.Surface.unlock(self.screen) 
        # for i in obs_coor:
        #     state.append(i)
        # for i in obs_coor:
        #     state.append(i)
        if self.graphics:  
            self.clock.tick(self.FPS)
            self.draw_all()           # generates new frame
            pygame.display.update()            
      
      
        return self.reward, state,self.done

    def stepNew(self,action):
        self.done=0
        self.reward =0
        key = "nothing"
        #key pressed
        
        #left
        if action == 0:
            self.reward -= .1
            key="left"
         
        #right
        elif action == 1: 
            self.reward -= .1
            key="right"
        #Up
        elif action == 2: 
            self.reward -= .1
            key="up"
        #Down
        elif action == 3: 
            self.reward -= .1
            key="down"

        #Space-Shoot
        elif action == 4: 
            self.reward -=.2      
            key="space"

        elif action == 5:
            self.reward -= .15

        self.key=key 
        self.action=action

        self.spaceShipSprite.move(key)
        self.game_loop() 
        obs_coor=[]
        n=len(self.obstacleGenerator.liveObstacles)
        if  n  < 5:
            x=5-n
            self.obstacleGenerator.generate(delay=0,num=x)
        StarShipGame.liveObstacles =self.obstacleGenerator.liveObstacles    
        
        state = (self.getPixelsOnScreenNew())
        # if  np.any(state[:,0] == 1):
        #     print("Yes 1 detected")
        #     f=open("res.csv",'w')
        #     print(np.reshape(state,(160,160)).tolist())
        #     f.write(str(np.reshape(state,(160,160)).tolist()))
        #     np.savetxt('res2.csv', np.reshape(state,(160,160)))
        #     f.close()       
        # for i in obs_coor:
        #     state.append(i)  
        pygame.Surface.unlock(self.screen) 
        # for i in obs_coor:
        #     state.append(i)
        # for i in obs_coor:
        #     state.append(i)
        if self.graphics:  
            self.clock.tick(self.FPS)
            self.draw_all()           # generates new frame
            pygame.display.update()         
         
       
        
        return self.reward, state.flatten(),self.done
    
    def agent_pos(self):
        aCenterX = (self.spaceShipSprite.currentPos[0])+(self.spaceShipSprite.image_size[0]/2)
        aCenterY = (self.spaceShipSprite.currentPos[1])+(self.spaceShipSprite.image_size[1]/2)
        formatdCX = aCenterX*(1/self.screen_size[0])
        formatdCY =aCenterY*(1/self.screen_size[1])
        return formatdCX,formatdCY
    
    def Obs_pos(self,obs):
        aCenterX = (obs.currentPos[0]) + (obs.image_size[0]/2)
        aCenterY =(obs.currentPos[1]) + (obs.image_size[1]/2)
        formatdCX = aCenterX*(1/self.screen_size[0])
        formatdCY =aCenterY*(1/self.screen_size[1])
        return formatdCX,formatdCY
 


   
if __name__ == '__main__':
	env = StarShipGame(graphics=True)
	env.play()