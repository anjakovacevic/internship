# https://www.youtube.com/watch?v=z_dbnYHAQYg
# with spoofing  https://www.youtube.com/watch?v=_KvtVk8Gk1A
# https://github.com/computervisioneng/face-attendance-system

import os.path
import datetime
import pickle

import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
import subprocess
# from test import test
import util

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("950x520+250+100")
        self.main_window.title("Face Attendance System")
        self.main_window.configure(bg='#eeeeee')

        self.button_frame = tk.Frame(self.main_window, bg='#eeeeee')
        self.button_frame.place(x=650, y=150, width=350, height=300)

        self.login_button_main_window = util.get_button(self.button_frame, 'Login', '#F282D0', self.login)
        self.login_button_main_window.pack(pady=10)

        # self.logout_button_main_window = util.get_button(self.main_window, 'logout', 'red', self.logout)
        # self.logout_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(self.button_frame, 'Register New User', '#42BCE5', 
                                                                    self.register_new_user, fg='black')
        self.register_new_user_button_main_window.pack(pady=10)

        # self.webcam_frame = tk.Frame(self.main_window, bg='#dddddd', bd=2, relief='solid')  # Create a frame with a border
        # self.webcam_frame.place(x=10, y=10, width=700, height=500)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=10, width=700, height=500)

        self.add_webcam(self.webcam_label)

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)
        
        self._label = label
        self.process_webcam()
    
    def process_webcam(self):
        ret, frame = self.cap.read()
        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)
        # repeat process every 20us, everything looks like a video but is a series of picures
        self._label.after(20, self.process_webcam)

    def login(self):
        
        unknown_img_path ='./.tmp.jpg'
        cv2.imwrite(unknown_img_path, self.most_recent_capture_arr)
        output = str(subprocess.check_output(['face_recognition', self.db_dir, unknown_img_path]))
        name = output.split(',')[1][:-5]
        # print(name), da vidis koji su svi moguci outputi ove situacije
        if name in ['unknown_person', 'no_persons_found']:
            util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
        else:
            util.msg_box('Welcome back!', 'Welcome {}.'.format(name))
            with open(self.log_path, 'a') as f:
                f.write('{}, {}\n'.format(name, datetime.datetime.now()))
                f.close()

        os.remove(unknown_img_path)
        
        # label = test(
        #         image=self.most_recent_capture_arr,
        #         model_dir=r'C:\Users\anja.kovacevic\kod\face\FR_softver\resources\anti_spoof_models',
        #         device_id=0
        #         )
        # # 1 is a real image and 0 is a fake
        # if label == 1:

        #     name = util.recognize(self.most_recent_capture_arr, self.db_dir)

        #     if name in ['unknown_person', 'no_persons_found']:
        #         util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
        #     else:
        #         util.msg_box('Welcome back !', 'Welcome, {}.'.format(name))
        #         with open(self.log_path, 'a') as f:
        #             f.write('{},{},in\n'.format(name, datetime.datetime.now()))
        #             f.close()

        # else:
        #     util.msg_box('You are a spoofer!', 'You are fake !')

    
    # def logout(self):
    #     label = test(
    #             image=self.most_recent_capture_arr,
    #             model_dir=r'C:\Users\anja.kovacevic\kod\face\FR_softver\resources\anti_spoof_models',
    #             device_id=0
    #             )

    #     if label == 1:

    #         name = util.recognize(self.most_recent_capture_arr, self.db_dir)

    #         if name in ['unknown_person', 'no_persons_found']:
    #             util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
    #         else:
    #             util.msg_box('Hasta la vista !', 'Goodbye, {}.'.format(name))
    #             with open(self.log_path, 'a') as f:
    #                 f.write('{},{},out\n'.format(name, datetime.datetime.now()))
    #                 f.close()

    #     else:
    #         util.msg_box('Hey, you are a spoofer!', 'You are fake !')
        

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1100x520+250+100")

        #buttons
        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', '#F282D0', self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=720, y=300)
        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try again', '#42BCE5', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=720, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=720, y=150)
        #self.entry_text_register_new_user.pack(pady=10)


        self.text_label_register_new_user = util.get_text_label(self.register_new_user_window, 'Please, enter username:')
        # self.text_label_register_new_user.pack(pady=10)
        self.text_label_register_new_user.place(x=720, y=70)


    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def accept_register_new_user(self):
        '''
        # getting the information that user put in the text field
        name = self.entry_text_register_new_user.get(1.0,  "end-1c")
        # saving the photo of a new registered user
        cv2.imwrite(os.path.join(self.db_dir, '{}.jpg'.format(name)), self.register_new_user_capture)

        # let the user know that registeration was succesful
        util.msg_box('Success!', 'User was registered successfully!')

        self.register_new_user_window.destroy()
            '''
        name = self.entry_text_register_new_user.get(1.0, "end-1c")

        embeddings = face_recognition.face_encodings(self.register_new_user_capture)[0]

        file = open(os.path.join(self.db_dir, '{}.pickle'.format(name)), 'wb')
        pickle.dump(embeddings, file)

        util.msg_box('Success!', 'User was registered successfully !')

        self.register_new_user_window.destroy()

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    
    def start(self):
        self.main_window.mainloop()   



if __name__ == "__main__":
    app = App()
    app.start()
