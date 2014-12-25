# -*- coding: utf-8 -*- 

from Tkinter import *

class usrPixel:
    def __init__(self, col, obj):
        self.color = col
        self.object = obj

class MainWindow:
    def __init__(self):
        self.root = Tk()
        self.root.title('Monkey Read Numbers')
        self.root.geometry('500x500')
        self.cv = Canvas(self.root, width=400, height=400, bg="black")
        self.buttonFin = Button(self.root, text="Finish")
        self.buttonClear = Button(self.root, text="Clear")
        
        self.cv.bind('<B1-Motion>', self.usrDraw)
        self.buttonFin.bind('<Button-1>', self.usrFinish)
        self.buttonClear.bind('<Button-1>', self.usrClear)
        
        self.usrPixmap = list()
        for i in range(20):
            for j in range(20):
                self.usrPixmap.append(usrPixel(0, self.cv.create_rectangle(i*20, j*20, i*20+20, j*20+20, fill="black")))
        self.cv.pack()
        self.buttonFin.pack()
        self.buttonClear.pack()
        
        


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
        print "#####"
