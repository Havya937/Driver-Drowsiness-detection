import streamlit as st
import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
st.set_page_config(page_title="DRIVER DROWSINESS DETECTION SYSTEM",page_icon="https://33.media.tumblr.com/5c79953db232e69e2f07f58b0a25c70f/tumblr_ncq090P25b1tlnptjo1_1280.gif")
st.title("DRIVER DROWSINESS DETECTION SYSTEM")
choice=st.sidebar.selectbox("My Menu",("HOME","URL","CAMERA","Feedback"))
if(choice=="HOME"):

    st.image("https://th.bing.com/th/id/OIP.gC1o75jJN-xJLKr8B0hiQwHaD4?rs=1&pid=ImgDetMain")   
    st.write("This is Driver Drowsiness Detection system developed using opencv and dlib")
elif(choice=="URL"):
    url=st.text_input("Enter the url")
    btn=st.button("Start Detection")
    window=st.empty()
    if btn:
        i=1
        btn2=st.button('Stop detection')
        facemodel=cv2.CascadeClassifier("face.xml")
        eyemodel=load_model("eyes.h5",compile=False)
        vid=cv2.VideoCapture(url)
        if btn2:
            vid.release()
            st.experimental_rerun()
        while(vid.isOpened()):
            flag,frame=vid.read()
            if(flag):
                faces=facemodel.detectMultiScale(frame)
                for(x,y,w,h) in faces:
                    face_img=frame[y:y+h,x:x+w]
                    size = (224, 224)
                    face_img = ImageOps.fit(Image.fromarray(face_img), size, Image.LANCZOS)
                    face_img = (np.asarray(face_img).astype(np.float32) / 127.5) - 1
                    face_img=np.expand_dims(face_img,axis=0)
                    pred=eyemodel.predict(face_img)[0][0]
                    if(pred>0.9):
                        path="data/"+str(i)+".jpg"
                        cv2.imwrite(path,frame[y:y+h,x:x+w])
                        cv2.rectangle(frame,(x,y),(x+h,y+w),(0,0,255),3)
                    else:
                        cv2.rectangle(frame,(x,y),(x+h,y+w),(0,255,0),3)
                window.image(frame,channels="BGR")
elif(choice=="CAMERA"):
    cam=st.selectbox("Choose 0 for primary camera and 1 for secondary camera",("None",0,1))
    btn=st.button("Start Detection")
    window=st.empty()
    if btn:
        i=1
        btn2=st.button('Stop detection')
        facemodel=cv2.CascadeClassifier("face.xml")
        eyemodel=load_model("eyes.h5",compile=False)
        vid=cv2.VideoCapture(cam)
        if btn2:
            vid.release()
            st.experimental_rerun()
        while(vid.isOpened()):
            flag,frame=vid.read()
            if(flag):
                faces=facemodel.detectMultiScale(frame)
                for(x,y,w,h) in faces:
                    face_img=frame[y:y+h,x:x+w]
                    size = (224, 224)
                    face_img = ImageOps.fit(Image.fromarray(face_img), size, Image.LANCZOS)
                    face_img = (np.asarray(face_img).astype(np.float32) / 127.5) - 1
                    face_img=np.expand_dims(face_img,axis=0)
                    pred=eyemodel.predict(face_img)[0][0]
                    if(pred>0.9):
                        path="data/"+str(i)+".jpg"
                        cv2.imwrite(path,frame[y:y+h,x:x+w])
                        cv2.rectangle(frame,(x,y),(x+h,y+w),(0,0,255),3)
                    else:
                        cv2.rectangle(frame,(x,y),(x+h,y+w),(0,255,0),3)
                window.image(frame,channels="BGR")
elif(choice=="Feedback"):
    st.markdown('<iframe src="https://docs.google.com/forms/d/e/1FAIpQLSfKoxF-E03CCFog8ycsyNiaAkBqRh9Is-HbnhGQbfIl7_aUDw/viewform?embedded=true" width="640" height="1214" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>',unsafe_allow_html=True)
       

