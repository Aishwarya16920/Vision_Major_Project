from kivy.core.window import Window
from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen,ScreenManager
from kivy.lang import Builder
from kivy.uix.image import Image
from kivy.clock import Clock
import cv2
import numpy as np
import pytesseract 
import pyttsx3
from kivy.graphics.texture import Texture
pytesseract.pytesseract.tesseract_cmd= 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
from pytesseract import Output
import speech_recognition as sr

Window.size= (400,600)

#Builder String
KV_string = ''' 
ScreenManager:
    App:
<App>:
    name: 'app'
    BoxLayout:
        orientation:'vertical'
        
        MDToolbar:
            title: 'Vision App'
            md_bg_color: .2, .2, .2, 1
            specific_text_color: 1, 1, 1, 1
        
        MDBottomNavigation:
            panel_color: 0,0,0,0
            
            MDBottomNavigationItem:
                id: 'first'
                name: 'screen-1'
                text: 'Explore'
                on_tab_press: app.Explore()
                
                MDLabel:
                    text: 'Obstacle Detection'
                    halign: 'center'
                    
                    
            MDBottomNavigationItem:
                id: 'second'
                name: 'screen-2'
                text: 'Read'
                on_tab_press: app.Read()
               
                MDLabel:
                    text: 'Text Recognition'
                    halign: 'center'
            
            MDBottomNavigationItem:
                id: 'third'
                name: 'screen-3'
                text: 'Find'
                on_tab_press: app.Find()
                
                MDLabel:
                    text: 'Find Items'
                    halign: 'center'
'''

class App(Screen):
    pass

sm = ScreenManager()

sm.add_widget(App(name = 'app'))

class Vision(MDApp):
    def build(self):
        self.icon= 'logo.png'
        screen = Screen()
        self.theme_cls.theme_style = "Dark"
        self.kv_str = Builder.load_string(KV_string)
        screen.add_widget(self.kv_str)
        self.image= Image()
        screen.add_widget(self.image)
        engine = pyttsx3.init()
        engine.say('Welcome to Vision App!')
        engine.say('Click bottom left to explore, Click bottom middle to read, Click bottom right to find.')
        engine.runAndWait()
        self.capture= cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture= cv2.VideoCapture(0, cv2.CAP_DSHOW)
            engine = pyttsx3.init()
            engine.say('You are using front camera!')
            engine.runAndWait()
            if not self.capture.isOpened():
                engine = pyttsx3.init()
                engine.say('Please enable your camera!')
                engine.runAndWait()
                exit()
        return screen
    
    def on_start(self):
        Clock.schedule_interval(self.load_video, 0.5/30.0)

    
    def Explore(self):
        Clock.unschedule(self.TextRecognition)
        Clock.unschedule(self.FindObjects)
        engine = pyttsx3.init()
        engine.say('You are in explore mode.')
        engine.runAndWait()
        Clock.schedule_interval(self.ObstacleDetection, 0.5/30.0)
        
        
    def Read(self):
        Clock.unschedule(self.ObstacleDetection)
        Clock.unschedule(self.FindObjects)
        engine = pyttsx3.init()
        engine.say('You are in read mode.')
        engine.runAndWait()
        Clock.schedule_interval(self.TextRecognition, 100/30.0)
        
    def Find(self):
        Clock.unschedule(self.ObstacleDetection)
        Clock.unschedule(self.TextRecognition)
        engine = pyttsx3.init()
        engine.say('You are in find mode.')
        engine.runAndWait()
        Clock.schedule_interval(self.FindObjects, 0.5/30.0)
        
    
    def load_video(self,*args):
        ret,frame = self.capture.read()
        self.image_frame= frame
        imgH,imgW,_ = frame.shape
        buffer= cv2.flip(frame, 0).tobytes()
        texture= Texture.create(size= (frame.shape[1], frame.shape[0]), colorfmt= 'bgr')
        texture.blit_buffer(buffer, colorfmt= 'bgr', bufferfmt= 'ubyte')
        self.image.texture= texture
        
    def ObstacleDetection(self,*args):
        ret,frame = self.capture.read()
        self.image_frame= frame
        imgH,imgW,_ = frame.shape
        
        net= cv2.dnn.readNet('yolov4/yolov4.weights', 'yolov4/yolov4.cfg' )
        classes = []
        with open('yolov4/coco.names', 'r') as f:
            classes = f.read().splitlines()
        blob= cv2.dnn.blobFromImage(frame, 1/255, (416, 416),(0,0,0), swapRB= True, crop=False)
        net.setInput(blob)
        output_layers_names= net.getUnconnectedOutLayersNames()
        layerOutputs= net.forward(output_layers_names)
        boxes= []
        confidences= []
        class_ids= []
        for output in layerOutputs:
            for detection in output:
                scores= detection[5:]
                class_id= np.argmax(scores)
                confidence= scores[class_id]
                if confidence > 0.5:
                    center_x= int(detection[0]* imgW)
                    center_y= int(detection[1]* imgH)
                    w=int(detection[2]* imgW)
                    h= int(detection[3]* imgH)
                    
                    x= int(center_x- w/2)
                    y= int(center_y- h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
        indexes= cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font= cv2.FONT_HERSHEY_PLAIN
        colors= np.random.uniform(0, 255, size= (len(boxes), 3))
        
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h= boxes[i]
                label= str(classes[class_ids[i]])
                confidence= str(round(confidences[i],2))
                color= colors[i]
                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                cv2.putText(frame, label+ " "+ confidence, (x, y+20), font, 2, (255,255,255), 2) 
            engine = pyttsx3.init()
            engine.say(label)
            engine.runAndWait()
                
        buffer= cv2.flip(frame, 0).tobytes()
        texture= Texture.create(size= (frame.shape[1], frame.shape[0]), colorfmt= 'bgr')
        texture.blit_buffer(buffer, colorfmt= 'bgr', bufferfmt= 'ubyte')
        self.image.texture= texture
     
    def TextRecognition(self, *args):
        ret,scene = self.capture.read()
        self.image_frame= scene

        imgH,imgW,_ = scene.shape
        frame= cv2.cvtColor(scene,cv2.COLOR_BGR2GRAY)
        
        x1, y1, w1, h1= 0, 0, imgH, imgW
        results = pytesseract.image_to_data(frame, output_type=Output.DICT)
        for i in range(0, len(results["text"])):
            x = results["left"][i]
            y = results["top"][i]
            w = results["width"][i]
            h = results["height"][i]
            text = results["text"] 
            conf = int(float(results["conf"][i]))
            if conf > 70:
                s= ""
                for i in text:
                    s+=i+" "
                characters_to_remove = "!()@—*“>=+-\/,'|£#%$&^_~?"
                global phrase
                phrase= s
                for character in characters_to_remove:
                    phrase = phrase.replace(character, "")
                cv2.rectangle(scene, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        engine = pyttsx3.init()
        engine.say(phrase)
        engine.runAndWait()       
        buffer= cv2.flip(scene, 0).tobytes()
        texture= Texture.create(size= (scene.shape[1], scene.shape[0]), colorfmt= 'bgr')
        texture.blit_buffer(buffer, colorfmt= 'bgr', bufferfmt= 'ubyte')
        self.image.texture= texture
        
    def FindObjects(self,*args):
        ret,frame = self.capture.read()
        self.image_frame= frame
        imgH,imgW,_ = frame.shape
        
        net= cv2.dnn.readNet('yolov4/yolov4.weights', 'yolov4/yolov4.cfg' )
        classes = []
        with open('yolov4/coco.names', 'r') as f:
            classes = f.read().splitlines()
        blob= cv2.dnn.blobFromImage(frame, 1/255, (416, 416),(0,0,0), swapRB= True, crop=False)
        net.setInput(blob)
        output_layers_names= net.getUnconnectedOutLayersNames()
        layerOutputs= net.forward(output_layers_names)
        boxes= []
        confidences= []
        class_ids= []
        for output in layerOutputs:
            for detection in output:
                scores= detection[5:]
                class_id= np.argmax(scores)
                confidence= scores[class_id]
                if confidence > 0.5:
                    center_x= int(detection[0]* imgW)
                    center_y= int(detection[1]* imgH)
                    w=int(detection[2]* imgW)
                    h= int(detection[3]* imgH)
                    
                    x= int(center_x- w/2)
                    y= int(center_y- h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
        indexes= cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font= cv2.FONT_HERSHEY_PLAIN
        colors= np.random.uniform(0, 255, size= (len(boxes), 3))
        
        objdis={}
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h= boxes[i]
                label= str(classes[class_ids[i]])
                confidence= str(round(confidences[i],2))
                color= colors[i]
                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                distance= (2 * 3.14 * 180) / (w + h * 360) * 1000 + 3
                d=round((distance*2.54),2)
                objdis[label]=d
                cv2.putText(frame, label+ " "+ confidence, (x, y+20), font, 2, (255,255,255), 2)  
                cv2.putText(frame, "%.2fcm" % (distance*2.54),(x+250,y+20), font,2, (0, 255, 0), 2)  
        
        r = sr.Recognizer()						
        check= 1
        while(check):
            try:	
                engine = pyttsx3.init()
                engine.say('What is the object you want to find?')
                engine.runAndWait()
                with sr.Microphone() as source2:
                    r.adjust_for_ambient_noise(source2,duration=0.1)
                    audio2 = r.listen(source2)
                    MyText = r.recognize_google(audio2)
                    MyText = MyText.lower()

                    text = "Did you say "+MyText
                    engine = pyttsx3.init()
                    engine.say(text)
                    engine.runAndWait()
                    with sr.Microphone() as source3:
                        r.adjust_for_ambient_noise(source3, duration=0.1)
                        audio3 = r.listen(source3)
                        MyText1 = r.recognize_google(audio3)
                        MyText1 = MyText1.lower()

                        text2 = "Did you say "+MyText1
                        engine = pyttsx3.init()
                        engine.say(text2)
                        engine.runAndWait()
                    
                        flag=0
                        if(MyText1=='yes'):
                            for i in objdis:
                                if i == MyText:
                                    flag =1
                                    dist=objdis[i]
                                    break
                        elif(MyText1=='no'):
                            continue

                        if flag == 1:
                            engine.say('Object found at a distance of '+ str(dist)+'centimeter')
                            engine.runAndWait()
                        else:
                            engine.say('Object not found in frame')
                            engine.runAndWait()
            except:
                pass
            check = 0
                
        buffer= cv2.flip(frame, 0).tobytes()
        texture= Texture.create(size= (frame.shape[1], frame.shape[0]), colorfmt= 'bgr')
        texture.blit_buffer(buffer, colorfmt= 'bgr', bufferfmt= 'ubyte')
        self.image.texture= texture

if __name__ == '__main__':           
    Vision().run()