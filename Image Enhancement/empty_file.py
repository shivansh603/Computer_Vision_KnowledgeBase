import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import cv2
from image_enhancement import image_enhancement




new_model = tf.keras.models.load_model('/home/beelink/onm/object_inhand/cnn_model/my_model')



cam = cv2.VideoCapture(2)
    # get first video frame
    # ok, frame = cap.read()
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc("M","J","P","G"))


video_fps = cam.get(cv2.CAP_PROP_FPS) 
width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

size = (width,height)   

out = cv2.VideoWriter('dense.mp4',  
                        cv2.VideoWriter_fourcc(*'mp4v'), 
                        video_fps, (640, 480)) 


# out2 = cv2.VideoWriter('dense_con.mp4',  
#                         cv2.VideoWriter_fourcc(*'MJPG'), 
#                         video_fps, size) 



count = 0 
while True:
    
    
    ok, frame = cam.read()
    if not ok:
        print("[ERROR] reached end of file")
        print(count)
        
        break
        
    if count == 0:
        
        first_frame = frame
        hsv_canvas = np.zeros_like(first_frame)
# set saturation value (position 2 in HSV space) to 255
        hsv_canvas[..., 1] = 255

        frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # get next frame
        
    ie = image_enhancement.IE(frame, 'RGB')
    output = ie.BBHE()
    print(frame.shape)

    frame_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    
    
    # compare initial frame with current frame
    # flow = cv2.calcOpticalFlowFarneback(frame_gray_init, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow = cv2.calcOpticalFlowFarneback(frame_gray_init,frame_gray, None,
                                        pyr_scale=0.5,
                                        levels=3,
                                        winsize=15,
                                        iterations=3,
                                        poly_n=5,
                                        poly_sigma=1.2,
                                        # flags = 0)
                                        flags= cv2.OPTFLOW_FARNEBACK_GAUSSIAN )
    # get x and y coordinates
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # set hue of HSV canvas (position 1)
    hsv_canvas[..., 0] = angle*(180/(np.pi/2))
    # set pixel intensity value (position 3
    hsv_canvas[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    






    flow = cv2.normalize(magnitude,None, 0,255,cv2.NORM_MINMAX)
    




    # frame_rgb = cv2.cvtColor(hsv_canvas, cv2.COLOR_HSV2BGR)
    frame_rgb = cv2.cvtColor(hsv_canvas, cv2.COLOR_HSV2BGR)
    # out.write(frame_rgb)
    cv2.imwrite("image.png",frame_rgb)

    image_size = (256, 256)

    img = keras.utils.load_img("image.png", target_size=image_size)
# plt.imshow(img)
    

    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = new_model.predict(img_array)
    score = float(tf.sigmoid(predictions[0][0]))
    print(f"This image is {100 * (1 - score):.2f}% with_object and {100 * score:.2f}% without_obect.")


  

    cv2.imshow('Frame', frame_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

 
    frame_gray_init = frame_gray
    count += 1