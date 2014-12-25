# -*- coding: utf-8 -*- 
# Filename:
#     MainWindow.py
# Description:
#     This file contains "class MainWindow" which deals with all
# GUI routines in this software.

from Tkinter import *
from tkMessageBox import *

class usrPixel:
    def __init__(self, col, obj):
        self.color = col
        self.object = obj

class MainWindow:
    def __init__(self):
        # Initialize the window
        self.root = Tk()
        self.root.title('Monkey Read Numbers')
        self.root.geometry('510x500')
        
        # Initialize Menus
        self.menu = Menu(self.root)
        self.root.config(menu=self.menu)
        self.Modemenu = Menu(self.menu)
        self.menu.add_cascade(label="Mode", menu=self.Modemenu)
        self.Modemenu.add_command(label="Recognize", command=self.switchmode0)
        self.Modemenu.add_command(label="Training", command=self.switchmode1)
        
        self.Funcmenu = Menu(self.menu)
        self.menu.add_cascade(label="Function", menu=self.Funcmenu)
        self.Funcmenu.add_command(label="Reset", command=self.reset)
        self.Funcmenu.add_command(label="About", command=self.about)

        # Initialize Labels
        self.title0 = Label(self.root, text="Now in Recogization mode", font=("Arial, 20"))
        self.title1 = Label(self.root, text="Now in Training mode", font=("Arial, 20"))
        self.label1 = Label(self.root, text="Please select the number you are drawing")

        # Initialize Buttons
        self.cv = Canvas(self.root, width=400, height=400, bg="black")
        self.buttonFin = Button(self.root, text="Finish")
        self.buttonClear = Button(self.root, text="Clear")
        self.buttonSample = Button(self.root, text="Sample")
        self.buttonTrain = Button(self.root, text="Train")
        
        # Initialize Events
        self.cv.bind('<B1-Motion>', self.usrDraw)
        self.buttonFin.bind('<Button-1>', self.usrFinish)
        self.buttonClear.bind('<Button-1>', self.usrClear)
        self.buttonSample.bind('<Button-1>', self.sample)
        self.buttonTrain.bind('<Button-1>', self.train)

        # Initialize Canvas for user to draw
        self.usrPixmap = list()
        for i in range(20):
            for j in range(20):
                self.usrPixmap.append(usrPixel(0, self.cv.create_rectangle(i*20, j*20, i*20+20, j*20+20, fill="black")))

        # Initialize Radiobutton for user when training
        self.Radio = IntVar()
        self.Radio.set(0)
        self.RadioButtons = list()
        for i in range(10):
            Rad = Radiobutton(self.root, variable=self.Radio, text='%d'%(i), value=i)
            Rad.grid(row=5, column=i)
            self.RadioButtons.append(Rad)
        
        # Grid layout management        
        self.cv.grid(row=1,column=0, rowspan=3, columnspan=9)
        self.buttonFin.grid(row=1,column=10)
        self.buttonClear.grid(row=2,column=10)
        
        # Switch to Mode0 (Recognization Mode)
        self.switchmode0()
        
    def show(self):
        self.root.mainloop()
        
    def switchmode0(self):
        # Switch to Recognization Mode, Modify the GUI
        self.Mode = 0
        self.buttonSample.grid(row=3, column=10)
        self.buttonTrain.grid_forget()
        self.title0.grid(row=0, column=0, columnspan=10)
        self.title1.grid_forget()
        self.label1.grid_forget()
        for radio in self.RadioButtons:
            radio.grid_forget()

    def switchmode1(self):
        # Switch to Training Mode, Modify the GUI
        self.Mode = 1
        self.buttonSample.grid_forget()
        self.buttonTrain.grid(row=3, column=10)
        self.title0.grid_forget()
        self.title1.grid(row=0, column=0, columnspan=10)
        self.label1.grid(row=4, column=0, columnspan=10)
        for i in range(10):
            self.RadioButtons[i].grid(row=5, column=i)

        # Load the existing training examples into memory
        import scipy.io as spio
        raw = spio.loadmat('sample.mat')
        self.X = list()
        for tmp in raw['X']:
            tmp1 = list()
            for element in tmp:
                tmp1.append(element)
            self.X.append(tmp1)
        self.y = list()
        for tmp in raw['y']:
            tmp1 = list()
            for element in tmp:
                tmp1.append(element)
            self.y.append(tmp1)
        
    def about(self):
        showinfo("Monkey Read Numbers", "I'm a Monkey that can read numbers.\nYou can also train me to do so.")

    def reset(self):
        # Replace 'sample.mat' by 'reset.mat' and train again
        showinfo("Reset", "Press 'OK' to start! \nThere'll be another window appear when finished!\nPLEASE WAIT\n")
        import os
        from shutil import copy
        try:
            os.remove('sample.mat')
            copy('reset.mat', 'sample.mat')
        except:
            print "File Error"
            showinfo("Reset", "Error while reset!\n(file missing or permission denied)")
            return None
        import learning as LN
        LN.train('sample.mat', 'Theta.mat')
        showinfo("Reset", "Reset successfully!")        
        

    def usrClear(self, event):
        # Clear the canvas for user
        for i in range(400):
            self.usrPixmap[i].color = 0
            self.cv.itemconfig(self.usrPixmap[i].object, fill="black")
    
    def usrDraw(self, event):
        # Deal with draw events
        x = event.x / 20
        y = event.y / 20
        if (x >= 0 and x < 20 and y >= 0 and y < 20):
            if (self.usrPixmap[y*20+x].color < 255):
                self.usrPixmap[y*20+x].color = 255
            self.cv.itemconfig(self.usrPixmap[x*20+y].object, fill='white')
            
    def usrFinish(self, event):
        # Deal with user press "Finish" button
        import predict as pd
        import numpy as np
        import scipy.io as spio
        import ShowResult as sr
        
        if (self.Mode == 0):
            # Recognization Mode
            theta = spio.loadmat('Theta.mat')
            Theta1 = np.mat(theta['Theta1'])
            Theta2 = np.mat(theta['Theta2'])
            tmp = np.zeros((1, 400))
            for p in range(400):
                tmp[0, p] = self.usrPixmap[p].color/255.0
            sr.ShowResult(pd.predict(Theta1, Theta2, tmp))
        else:
            # Training Mode
            tmp = list()
            for i in range(400):
                tmp.append(self.usrPixmap[i].color/255)
            self.X.append(tmp)
            self.y.append([self.Radio.get()])
        
    def sample(self, event):
        # Randomly generate an example in 'sample.mat' and show the image
        from random import random
        import ShowNum as sn
        import scipy.io as spio
        import numpy.core.fromnumeric as npfunc
        raw = spio.loadmat('sample.mat')
        X = raw['X']
        row = int(random()*npfunc.shape(X)[0])
        sn.createImg(X[row]).show()
        
    def train(self, event):
        # Save the new training examples and start trainging
        import learning as Ln
        import scipy.io as spio
        Total = {'X': self.X, 'y': self.y}
        spio.savemat('sample.mat', Total)
        showinfo("Training", "Press 'OK' to start! \nThere'll be another window appear when finished!\nPLEASE WAIT\n")
        Ln.train('sample.mat', 'Theta.mat')
        showinfo("Training", "Training Finished!")
