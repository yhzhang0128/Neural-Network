# -*- coding: utf-8 -*- 

from Tkinter import *
from tkMessageBox import *

class usrPixel:
    def __init__(self, col, obj):
        self.color = col
        self.object = obj

class MainWindow:
    def __init__(self):
        self.root = Tk()
        self.root.title('Monkey Read Numbers')
        self.root.geometry('500x450')
        
        self.menu = Menu(self.root)
        self.root.config(menu=self.menu)
        self.Modemenu = Menu(self.menu)
        self.menu.add_cascade(label="Mode", menu=self.Modemenu)
        self.Modemenu.add_command(label="Recognize", command=self.switchmode0)
        self.Modemenu.add_command(label="Training", command=self.switchmode1)
        
        self.Funcmenu = Menu(self.menu)
        self.menu.add_cascade(label="Function", menu=self.Funcmenu)
        self.Funcmenu.add_command(label="About", command=self.about)

        self.cv = Canvas(self.root, width=400, height=400, bg="black")
        self.buttonFin = Button(self.root, text="Finish")
        self.buttonClear = Button(self.root, text="Clear")
        self.buttonSample = Button(self.root, text="Sample")
        
        self.cv.bind('<B1-Motion>', self.usrDraw)
        self.buttonFin.bind('<Button-1>', self.usrFinish)
        self.buttonClear.bind('<Button-1>', self.usrClear)
        self.buttonSample.bind('<Button-1>', self.sample)

        self.usrPixmap = list()
        for i in range(20):
            for j in range(20):
                self.usrPixmap.append(usrPixel(0, self.cv.create_rectangle(i*20, j*20, i*20+20, j*20+20, fill="black")))

#        self.menu.grid(row=0,column=0)
        self.cv.grid(row=1,column=0, rowspan=10)
        self.buttonFin.grid(row=1,column=1)
        self.buttonClear.grid(row=2,column=1)
        self.buttonSample.grid(row=3,column=1)
        
#        self.cv.pack()
#        self.buttonFin.pack()
#        self.buttonClear.pack()
        
    def show(self):
        self.root.mainloop()


    def usrClear(self, event):
        for i in range(400):
            self.usrPixmap[i].color = 0
            self.cv.itemconfig(self.usrPixmap[i].object, fill="black")
    
    def usrDraw(self, event):
        x = event.x / 20
        y = event.y / 20
        if (x >= 0 and x < 20 and y >= 0 and y < 20):
            if (self.usrPixmap[y*20+x].color < 255):
                self.usrPixmap[y*20+x].color = 255
            self.cv.itemconfig(self.usrPixmap[x*20+y].object, fill='white')
            
    def usrFinish(self, event):
        #print "Finish"
        import predict as pd
        import numpy as np
        import scipy.io as spio
        import ShowResult as sr
        
        theta = spio.loadmat('Theta.mat')
        Theta1 = np.mat(theta['Theta1'])
        Theta2 = np.mat(theta['Theta2'])
        tmp = np.zeros((1, 400))
        for p in range(400):
            tmp[0, p] = self.usrPixmap[p].color/255.0
        sr.ShowResult(pd.predict(Theta1, Theta2, tmp))
        
        
    def switchmode0(self):
        print "Mode: ", 0


    def switchmode1(self):
        print "Mode: ", 1
        
    def about(self):
        showinfo("Monkey Read Numbers", "I'm a Monkey that can read numbers.\nYou can also train me to do so.")
        
    def sample(self, event):
        from random import random
        import ShwoNum as sn
        import scipy.io as spio
        import numpy.core.fromnumeric as npfunc
        raw = spio.loadmat('reset.mat')
        X = raw['X']
        row = int(random()*npfunc.shape(X)[0])
#        print npfunc.shape(X)[0]
        sn.createImg(X[row]).show()
