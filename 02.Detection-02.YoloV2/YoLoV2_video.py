from darkflow.net.build import TFNet
import cv2
import random

cap = cv2.VideoCapture("test.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter("test.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, size)
model_path = "your_network_cfg.cfg"
weights_path = "your_network_weights.weights"

options = {"model": "cfg/yolo.cfg", "load": "yolo.weights", "threshold": 0.1}
tfnet = TFNet(options)
color = [(255,0,0), (0,0,255), (0,255,0), (255,255,0)]

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        result = tfnet.return_predict(frame)

        for _, box in enumerate(result):
            idx = 0
            label = box['label']
            confidence = box['confidence']
            topleft = box['topleft']
            bottomright = box['bottomright']
            if confidence > 0.4 :
                cv2.rectangle(frame,(topleft['x'],topleft['y']),(bottomright['x'],bottomright['y']),color[idx],3)

                cv2.putText(frame, label, org=(topleft['x'],topleft['y']), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0,
                            color=color[idx])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        videoWriter.write(frame)
    else:
        break
cap.release()
videoWriter.release()
cv2.destroyAllWindows()
print('Done')