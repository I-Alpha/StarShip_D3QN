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
import time

  
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
        self.ROWS =  1
        self.COLS = 54
        self.image_memory = np.zeros((self.REM_STEP, self.ROWS, self.COLS))
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
        self.x =0   
        self.time_multipliyer= 1 
        self.timer1 = 0
        self.counter = 0

    def vectorize_func(self,m):
        return m/255
    
    def imshow(self, image, rem_step=0):
        cv2.imshow("SpS", image[rem_step,...])
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return

    def render(self):            
        return (copy.deepcopy(pygame.surfarray.array3d(self.screen)))
    
    def render_wrap(self):
        temp = self.render()
        temp=np.reshape(temp,(-1,3))
        temp= pd.DataFrame(temp,columns=['r','g','b'])
        conditions = [ 
            (temp == 255)
        ]
        temp2=pd.Series()
        temp2['r'] = 0         
        temp2['g'] = 0         
        temp2['b'] = 0 
        temp.loc[(temp['r'] >1) & (temp['g']>1) & (temp['b']>1)] =  [0,0,0]
        temp =    np.where(temp['r'] == 0 & temp['g']==3 & temp['b'] == 0 ,[22,33,44], temp.tolist())
        temp =    np.where(temp == [255,255,255] ,[0,0,0], temp).tolist()
        return temp.values
    
    def getPixelsOnScreen(self):            
        img= self.render()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_rgb_resized = cv2.resize(img_rgb, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
        img_rgb_resized[img_rgb_resized > 5] = 255
        img_rgb_resized = np.transpose(img_rgb_resized / 255)
        self.image_memory = np.roll(self.image_memory, 1, axis = 0)
        self.image_memory[0,:,:] = img_rgb_resized    
        self.imshow(self.image_memory,0)
        return np.expand_dims(self.image_memory, axis=0)

    def getEnvStateOnScreen(self):
        obs_collection={}
        count = 0
        agentX,agentY,agent_X_offset,agent_Y_offset, agent_CX_offset,agent_CY_offset,agent_w,agent_h= self.Obj_state(self.spaceShipSprite)

        for obs in  self.obstacleGenerator.liveObstacles :
               x,y,x_offset,y_offset, Cx_offset,Cy_offset,w,h= self.Obj_state(obs)
               label = "obs_" + str(count) + "_"
               obs_collection[label+"ID"]= obs.obs_ID              
               obs_collection[label+"X"]=x              
               obs_collection[label +"Y"]=y
               obs_collection[label+"CX"]=Cx_offset              
               obs_collection[label +"CY"]=Cy_offset
               obs_collection[label+"width"]=w
               obs_collection[label+"height"]=h

               count+=1
       
        # count=0
        # livep ={}
        # for proj in self.obstacleGenerator.liveProjectiles:
        #        x,y,x_offset,y_offset, cx_offset,cy_offset,w,h= self.Obj_state(proj)
        #        label = "live_projectile_" + str(count) + "_"
        #        livep[label+"X"]=x               
        #        livep[label +"Y"]=y    
        #     #    livep[label+"X_offset"]=x_offset
        #     #    livep[label+"Y_offset"]=y_offset
        #     #    livep[label+"CX_offset"]=cx_offset
        #     #    livep[label+"CY_offset"]=cy_offset
        #        livep[label+"width"]=w
        #        livep[label+"height"]=h 
        #        count+=1
        
        count=0
        state = {                              
                    
                    "dead_obstacles":(self.obstacleGenerator.deadObstacles),
                    "live_projectiles_num" : (len(self.obstacleGenerator.liveProjectiles)),
                    "live_projectiles_last_fired_at": (self.spaceShipSprite.firedAt/1000),
                    "live_projectiles_miss": (self.obstacleGenerator.p_out_of_bounds), 
                    "hits":(self.obstacleGenerator.hits),   
                    "fails":(self.obstacleGenerator.fails),
                    "counter":  (self.counter),
                    "score":(self.score),     
                    # "agent_X_offset":agent_X_offset,
                    # "agent_Y_offset":agent_Y_offset,                    
                    "agent_CX_offset":agent_CX_offset,
                    "agent_CY_offset":agent_CY_offset, 
                    "agent_ammo_current":self.spaceShipSprite.currentAmmo,
                    "game_timer":(pygame.time.get_ticks()- self.timer1)/1000,
                    "agent_health" :(self.spaceShipSprite.health),
                    "agent_damage" : self.spaceShipSprite.damage,
                    "agent_reward":self.reward,            
                    # "live_obstacles_num":len(self.obstacleGenerator.liveObstacles),
                   "agent_X" : agentX,
                    "agent_Y":agentY,
                    "agent_width": agent_w,
                    "agent_height": agent_h,                     
                } 
        for m, (k, v) in enumerate(obs_collection.items()):
                state[k]=v 

        # livep_len = len(livep)
        # livep_len = (int)(livep_len/4)
        # if livep_len < self.spaceShipSprite.maxProjectiles_on_screen:        
        #     for l in range(livep_len,self.spaceShipSprite.maxProjectiles_on_screen):
        #             label = "live_projectile_placeHolder_" + str(l) + "_"
        #             livep[label+"X"]=0              
        #             livep[label +"Y"]=0
        #             # livep[label+"X_offset"]=0
        #             # livep[label+"Y_offset"]=0
        #             # livep[label+"CX_offset"]=0
        #             # livep[label+"CY_offset"]=0
        #             livep[label+"width"]=0
        #             livep[label+"height"]=0


        # for t, (k, v) in enumerate(livep.items()):
        #         state[k]=v 
        state =[ *state.values()]

        self.image_memory = np.roll(self.image_memory, 1, axis = 0)
        self.image_memory[0,:,:] = state
        #np.select(conditions, choices, default=0)
        return np.expand_dims(self.image_memory, axis=0)

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
                    ObstacleGenerator.deadObstacles+=1
                
                    if alive :
                        self.reward-=3
                    else: 
                        self.reward-=7
                        self.done = True


    def draw_score(self,color ='white'):
            self.x += self.reward
            scoreboard = "<<<Score>>>  " + "{:.2f}".format(self.x) + "/-hits : " +  str(self.obstacleGenerator.hits) + "/fails : " +  str(ObstacleGenerator.fails) + "/liveObs : " +  str(len(self.obstacleGenerator.liveObstacles))  +"/Missiles live: "+ str(len(self.obstacleGenerator.liveProjectiles)) +"/key : " +  str((self.key)) +"/Action : " +  str((self.action))+ "/deadObs : " +  str(self.obstacleGenerator.deadObstacles) + "/Ship-lives : " +  str((self.spaceShipSprite.lives))+ "/Ship-health: " +  str((self.spaceShipSprite.health))   
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

    def Obj_state(self,obs):
        aCenterX = (obs.currentPos[0]) + (obs.image_size[0]/2)
        aCenterY =(obs.currentPos[1]) + (obs.image_size[1]/2)
        formatdCX = aCenterX*(1/self.screen_size[0])
        formatdCY =aCenterY*(1/self.screen_size[1])
        offsetCX =   0.5 - formatdCX
        offsetCY =  0.5 - formatdCY               
        w= obs.image_size[0]*(1/self.screen_size[0])
        h= obs.image_size[1]*(1/self.screen_size[1])
        return formatdCX,formatdCY, *self.getOffsets_formatted(obs), offsetCX,offsetCY,w,h
 

    def getOffsets_formatted(self,obs):
        aCenterX = (obs.offset[0]) + (obs.image_size[0]/2)
        aCenterY =(obs.offset[1]) + (obs.image_size[1]/2)
        offsetX = aCenterX*(1/self.screen_size[0])
        offsetY =aCenterY*(1/self.screen_size[1])  
        return offsetX, offsetY
   

    def game_loop(self):
           
            self.obstacleGenerator.updateAll()                
            StarShipGame.liveProjectiles=self.spaceShipSprite.liveProjectiles             
            ObstacleGenerator.liveProjectiles=StarShipGame.liveProjectiles        
            hit = self.obstacleGenerator.checkHits()       
            if hit:
                self.reward+=1
                hit=False            
            if ObstacleGenerator.p_out_of_bounds:
                self.reward-=.3
                ObstacleGenerator.p_out_of_bounds=0
            self.checkHits()
            if (ObstacleGenerator.fails) > 0 :
                    self.reward -= (ObstacleGenerator.fails) - (self.time_multipliyer+1.4)
                    self.done=True           
                           
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

    def play(self):
        self.playing = True     
  
        while True: #begin gameloop 
            # handle mouse and keyboard events   
            self.clock.tick(self.FPS)
            print(self.reward)
            self.reward =0
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
        pygame.init()     
        self.spaceShipSprite = SpaceShipSprite(StarShipGame.screen,r'Assets\imgs\triangleShip.png',startPosition=(self.screen_size[0]/2,self.screen_size[1]/2))   
        SpaceShipSprite.liveProjectiles=[]
        ObstacleGenerator.obs_ID=0  
        self.obstacleGenerator = ObstacleGenerator(StarShipGame.screen,r'Assets\imgs\brick.png')
        StarShipGame.liveObstacles = []
        StarShipGame.deadObstacles = 0
        StarShipGame.liveProjectiles =[]      
        ObstacleGenerator.fails=0
        Obstacle.fails=0
        self.score =0
        self.counter =0
        self.hit=0
        self.fails=0
        self.x =0
        self.action=0
        self.key=0
        self.score =0
        self.image_memory = np.zeros((self.REM_STEP, self.ROWS, self.COLS))
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
        self.reward= 0
        self.time_multipliyer =1
        StarShipGame.fails =0
        self.done=False  
        self.timer1 = pygame.time.get_ticks()
        self.clock = pygame.time.Clock()     
        self.obstacleGenerator.generate(0);
        StarShipGame.liveObstacles =  self.obstacleGenerator.liveObstacles  
        for i in range(self.REM_STEP):
            state = self.getEnvStateOnScreen()
        # for i in obs_collection:
        #     state.append(i) 
      
        return (state).flatten()
  
     
    def step(self,action):
        
        self.counter += 1
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
            self.reward -= .1

        self.key=key 
        self.action=action

        self.spaceShipSprite.move(key)
        self.game_loop() 
        
        obs_collection=[]
        n=len(self.obstacleGenerator.liveObstacles)
        if  n  < 5:
            x=5-n
            self.obstacleGenerator.generate(delay=0,num=x)
        if self.done:        
            state = self.getEnvStateOnScreen()
            return self.reward, state.flatten(),self.done
        StarShipGame.liveObstacles =self.obstacleGenerator.liveObstacles    
        state = self.getEnvStateOnScreen()
        # if  np.any(state[:,0] == 1):
        #     print("Yes 1 detected")
        #     f=open("res.csv",'w')
        #     print(np.reshape(state,(160,160)).tolist())
        #     f.write(str(np.reshape(state,(160,160)).tolist()))
        #     np.savetxt('res2.csv', np.reshape(state,(160,160)))
        #     f.close()       
        # for i in obs_collection:
        #     state.append(i)  
        # for i in obs_collection:
        #     state.append(i)
        # for i in obs_collection:
        #     state.append(i)
        if self.graphics:  
            self.clock.tick(self.FPS)
            self.draw_all()           # generates new frame
            pygame.display.update()         
         
       
        
        return self.reward, state.flatten(),self.done
     
     
if __name__ == '__main__':
        env = StarShipGame(graphics=True)
        env.FPS = 20
        env.play()
