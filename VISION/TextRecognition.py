import cv2
import pytesseract 
import pyttsx3
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd= 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
     
def TextRecognition():
    cam= cv2.VideoCapture(1)
    if not cam.isOpened():
            cam= cv2.VideoCapture(0)
            if not cam.isOpened():
                raise IOError("Cannot open Camera")  
    while True:
        ret,scene=cam.read()
        imgH,imgW,_ = scene.shape
        
        frame= cv2.cvtColor(scene,cv2.COLOR_BGR2GRAY)
        
        x1, y1, w1, h1= 0, 0, imgH, imgW
        # text=pytesseract.image_to_string(frame,lang="eng")
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
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Text Recognition",scene) 
        engine = pyttsx3.init()
        engine.say(phrase)
        engine.runAndWait()
        if cv2.waitKey(2) & 0xFF==ord('q'):
            break
    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    TextRecognition()

