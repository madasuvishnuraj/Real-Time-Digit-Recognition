from tkinter import *
from tkinter import messagebox
from PIL import Image
import io
import numpy as np
import keras
import cv2



class ui_digit_recognition():

    def __init__(self,application,model):
        self.ui_application = application
        self.ui_application.title('Digit Recognition')
        self.ui_application.geometry("400x550")
        self.canvas = None
        self.pred_button = None
        self.fileName = 'digit_image'
        self.lasx = 0
        self.lasy = 0
        self.label = None
        self.model = model
        self.count = 0
       
    def get_x_and_y(self,event):
        self.lasx, self.lasy = event.x, event.y

    def draw_smth(self,event):
        self.canvas.create_line((self.lasx, self.lasy, event.x, event.y),fill="#023047",width=20)
        self.lasx, self.lasy = event.x, event.y

    def create_canvas(self):
        self.canvas = Canvas(self.ui_application, bg='#ffb703',height=300, width=300)
        self.canvas.pack()
        self.canvas.place(x=50, y=15)
        self.canvas.bind("<Button-1>", self.get_x_and_y)
        self.canvas.bind("<B1-Motion>", self.draw_smth)
        
    def predict_the_value(self):
        
        self.count=self.count+1
        ps = self.canvas.postscript(file = self.fileName + '.eps',colormode='gray')
        
        img = Image.open(self.fileName + '.eps') 
        img.save(self.fileName + '.jpg') 
        im = cv2.imread(self.fileName + '.jpg')
        gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        final_img = ~np.array(gray_image)         # changing black pixel to white and white pixel to black
        final_img[final_img >= 200 ] = 255       # fine tuning the image
        final_img = cv2.resize(final_img, (28, 28), interpolation = cv2.INTER_NEAREST)
        
        #im = Image.fromarray(final_img)                             # comment this  line
        #im.save('create_dataset/2/0_{}'.format(self.count)+'.jpg')   #comment this line
             
        #cv2.imshow('Grayscale', final_img)
        final_img = np.expand_dims(final_img, [0,-1])
        #print(final_img.shape)  1,28,28,1
        predected_label =  self.model.predict(final_img/255)
        print(predected_label)
        final_val = np.argmax(predected_label)
        print(final_val)
        self.label = Label(app,text="Predicted value: {}".format(final_val),font=('Helvetica bold', 18),fg="#023047")
        self.label.pack(pady=20)
        self.label.place(x=60, y=450)
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.label.after(250, self.label.destroy())
        self.label=None
        
    def predict_button(self):
        self.pred_button = Button(app, text ="Predict", command = self.predict_the_value,height= 3, width=12)
        self.pred_button.pack()
        self.pred_button.place(x=100, y=350)
        
    def clear_button(self):
        self.clr_button = Button(app, text ="Clear", command = self.clear_canvas,height= 3, width=12)
        self.clr_button.pack()
        self.clr_button.place(x=215, y=350)

    def create_ui(self):
        self.create_canvas()
        self.predict_button()
        self.clear_button()
    
        
if __name__=="__main__":

    app = Tk()
    model = keras.models.load_model('simple_model4.h5')
    x = ui_digit_recognition(app,model)
    x.create_ui()
    app.mainloop()


    

    