import speech_recognition as sr
import cv2 
import numpy as np
import pyttsx3        

net= cv2.dnn.readNet('YoloV4/yolov4.weights', 'YoloV4/yolov4.cfg' )
classes = []
with open('YoloV4/coco.names', 'r') as f:
	classes = f.read().splitlines()
cam= cv2.VideoCapture(1)
if not cam.isOpened():
    cam= cv2.VideoCapture(0)
    if not cam.isOpened():
        raise IOError("Cannot open Camera")
	 
while True:
	ret,frame = cam.read()
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
			print(objdis)
			cv2.putText(frame, label+ " "+ confidence, (x, y+20), font, 2, (255,255,255), 2)  
			cv2.putText(frame, "%.2fcm" % (distance*2.54),(x+250,y+20), font,2, (0, 255, 0), 2)  

	cv2.imshow("Object Detection",frame)
	#print(obj)
	
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
				#print(text)
				engine.runAndWait()
				with sr.Microphone() as source3:
					r.adjust_for_ambient_noise(source3, duration=0.1)
					audio3 = r.listen(source3)
					MyText1 = r.recognize_google(audio3)
					MyText1 = MyText1.lower()

					text2 = "Did you say "+MyText1
					engine = pyttsx3.init()
					engine.say(text2)
					#print(text2)
					engine.runAndWait()
				
					flag=0
					if(MyText1=='yes'):
						for i in objdis:
							print(i)
							if i == MyText:
								flag =1
								#dist=objdis.get(i)
								dist=objdis[i]
								print(dist)
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
	if cv2.waitKey(2) & 0xFF==ord('q'):
		break
cam.stop()
cv2.destroyAllWindows()