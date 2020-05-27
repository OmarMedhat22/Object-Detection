import numpy as np
import cv2


classes = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]




colors = np.random.rand(21,3)*255

model = "MobileNetSSD_deploy.caffemodel"
protxt = "MobileNetSSD_deploy.prototxt.txt"

mobile_net = cv2.dnn.readNetFromCaffe(protxt,model)

cap = cv2.VideoCapture("test_video.mp4")

x=0
while True:

        ret,frame = cap.read()
        frame = cv2.resize(frame, (400,300) )
        model_frame = cv2.resize(frame ,(300,300) )
        (h,w) = frame.shape[:2]

        blob_image = cv2.dnn.blobFromImage(model_frame,0.007843,(300,300),127.5)
        mobile_net.setInput(blob_image)
        detections = mobile_net.forward()

        for i in range(0,detections.shape[2]):

                probability = detections[0,0,i,2]

                if probability > 0.2 :

                        class_number = int(detections[0,0,i,1])
                        box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                        (x1,y1,x2,y2) = box.astype("int")
                        label = classes[class_number] + " " + str(int(probability *100)) + "%"
                        cv2.rectangle(frame , (x1,y1) , (x2,y2) , colors[class_number] )
                        cv2.putText(frame,label,(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.5,colors[class_number],2)
                        
        cv2.imwrite(str(x) + ".jpg", frame)
        x=x+1

        cv2.imshow("frame",frame)

        key = cv2.waitKey(1)& 0xFF
        if key == ord("q"):
                break
cv2.destroyAllWindows()
                        
                        

                        
        
