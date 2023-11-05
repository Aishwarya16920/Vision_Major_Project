import cv2 
import numpy as np
import pyttsx3

def ObstacleDetection():
    net= cv2.dnn.readNet('yolov4/yolov4.weights', 'yolov4/yolov4.cfg' )
    
    classes = []

    with open('yolov4/coco.names', 'r') as f:
        classes = f.read().splitlines()
        
    cam= cv2.VideoCapture(1)
    if not cam.isOpened():
            cam= cv2.VideoCapture(0)
            if not cam.isOpened():
                raise IOError("Cannot open Camera") 

    while True:
        ret,frame=cam.read()
        imgH,imgW,_ = frame.shape
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
        cv2.imshow("Object Detection",frame)
        engine = pyttsx3.init()
        engine.say(label)
        engine.runAndWait()
        if cv2.waitKey(2) & 0xFF==ord('q'):
            break
    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ObstacleDetection()