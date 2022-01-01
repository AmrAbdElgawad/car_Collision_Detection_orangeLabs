from object_tracker import accidentDetection
import streamlit as st
import streamlit_authenticator as stauth
import smtplib, ssl
import moviepy.editor as moviepy
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf

####################--Load Model--#############
weights= "/content/project/checkpoints/yolov4-416"
@st.cache
def load_model():
	  return tf.saved_model.load(weights, tags=[tag_constants.SERVING])
##########################################



st.title("Car collision detection")
names = ['Mohamed Magdi','Amr Eid','Esraa Hazem','Mohamed Sbae']
Emails = ['muhammedmagdi411@gmail.com','amr.ead2015@gmail.com','mohamedsebaie1@gmail.com','eng.esraa.ghazy1520@alexu.edu.eg']
passwords = ['mm','ae','ms','eh']
hashed_passwords = stauth.hasher(passwords).generate()
authenticator = stauth.authenticate(names,Emails,hashed_passwords,
    'some_cookie_name','some_signature_key',cookie_expiry_days=30)
name, authentication_status = authenticator.login('Login','main')
if authentication_status:
    st.write('Welcome *%s*' % (name))
    st.title('Let\'s detect any collisions')
    Url = st.text_input("Add Url Here")
    result=""
    #Text box to recieve Url
    if st.button("Submit"):
      result = Url
      st.success(result)

      #########- Call our Function-##############
      
      Output=accidentDetection(result,"/content/infer1.mp4",load_model())
      Output
      ######-Print video result-##########

      #############-Convert Video to mp4-###########
      clip = moviepy.VideoFileClip("/content/infer1.mp4")
      clip.write_videofile("/content/infer2.mp4")

      #############-Display Video-#####
      video_file = open("/content/infer2.mp4", 'rb')
      video_bytes = video_file.read()
      st.video(video_bytes)

      ##############-Send Email-########
      sender = 'car.crash.orange2@gmail.com'
      password ='ITI123456' 
      port = 465  
      receiver=  Emails[names.index(name)]
      server = smtplib.SMTP('smtp.gmail.com', 587)
      server.starttls()
      server.login(sender,password)
      server.sendmail(sender, receiver,Output)
      server.quit()
            


elif authentication_status == False:
    st.error('Email/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your Email and password')
    