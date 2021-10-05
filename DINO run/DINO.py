import cv2
import os
import keyboard
import time
from grabscreen import grab_screen
import random



my_path = os.path.abspath(os.path.dirname("__file__"))
path = os.path.join(my_path, "img\\")

start =0 
end =0.01
var_list=[0]
upline = True
downline = False
i=0
while True :
    
    # 獲得遊戲畫面
 
    img = grab_screen(region=(700, 430, 1250, 500))

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    (thresh, img_gray) = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    
    # 儲存畫面
    
    # if count % 50 == 0:
        # cv2.imwrite(f"{path}BotView{pic}.jpg", img)
        # print(f"{path}BotView{pic}.jpg")
        # print(f"Saved Picture numder: {pic}")
        # pic += 1
    # count += 1
    
    # print(img.shape)
    
    
    
    
    
    limit_start = 550
    limit_end = 350
    
    #  界線圖 
    cv2.rectangle(img, (limit_start,0), (limit_start,50), (0,255,0), 3)
    cv2.rectangle(img, (limit_end,0), (limit_end,50), (0,0,255), 3)
  
    
    # 獲取通過時間
    
    if upline:
        if 0 in img_gray[:40,550]:
            start = time.time()
            upline = False
            downline=True
            
    if downline:
        if 0 in img_gray[:40,350]:
            end = time.time()
            var = 200/(end-start)/100
            var_list[0]= var
            
            upline = True
            downline=False
            
        

    
    dino_x, dino_y = 38,45
    small_jump = 22
    big_jump = 22. 
     

    distance = int(var_list[0]*small_jump)
    
    cv2.rectangle(img, (dino_x+distance,0), (dino_x+distance,50), (255,0,255), 3)
                                                                                             
    
    x =  dino_x+distance
    
    if img_gray[dino_y-10,x] == 0 or img_gray[dino_y-10,x-2]== 0 or img_gray[dino_y-10,x-3]== 0:
    
        
        keyboard.release("down")
        keyboard.press("space")
        # print("跳啊~幹~")
        # time.sleep(0.01)
        # keyboard.release("space")                                                                                                       
            
               
    elif img_gray[dino_y-38,x]==0 or img_gray[dino_y-36,x]==0 or img_gray[dino_y-35,x]==0 :
        keyboard.press("down")
                
 
    

                                            
    cv2.imshow('',img)
    
    if cv2.waitKey(1)==ord('q'):
        break
    
cv2.destroyAllWindows()
