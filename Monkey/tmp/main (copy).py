input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

from scipy import io as spio
raw = spio.loadmat('ex3data1.mat')

import numpy as np
X = np.mat(raw['X'])
y = np.mat(raw['y'])

import numpy.core.fromnumeric as npfunc
#print npfunc.shape(X)  // (5000, 400)
#print npfunc.shape(y)   // (5000, 1)

import Image
import shownum as sn
#for i in range(10):
#    im = sn.createImg(X[i*500])
#    im.show()

#theta = spio.loadmat('ex3weights.mat')
theta = spio.loadmat('finally.mat')
Theta1 = np.mat(theta['Theta1'])
Theta2 = np.mat(theta['Theta2'])


from Tkinter import *
root = Tk()
root.title('Monkey Read Numbers')
root.geometry('500x500')
cv = Canvas(root, width=400, height=400, bg="black")
buttonFin = Button(root, text="Finish")
buttonClear = Button(root, text="Clear")
buttonSave = Button(root, text="Save")

class usrPixel:
    def __init__(self, col, obj):
        self.color = col
        self.object = obj
usrPixmap = list()

def usrClear(event):
    for i in range(400):
        usrPixmap[i].color = 0
        cv.itemconfig(usrPixmap[i].object, fill="black")


def usrDraw(event):
    x = event.x / 20
    y = event.y / 20
    if (x >= 0 and x < 20 and y >= 0 and y < 20):
        #print x, y
        x1 = x 
        y1 = y
        if (usrPixmap[y1*20+x1].color < 255):
            usrPixmap[y1*20+x1].color = 255;
            cv.itemconfig(usrPixmap[x1*20+y1].object, fill='white')
    
savemat = {}
savepic3 = []
def usrSave(event):
    print len(savepic3)
    savemat['X'] = savepic3
    spio.savemat('test2.mat', savemat)
    
def usrFinish(event):
    print "Finish"
    import predict as pd
    tmp = np.zeros((1, 400))
    tmppic = []
    for p in range(400):
        tmp[0, p] = usrPixmap[p].color/255.0
        tmppic.append(usrPixmap[p].color/255)
    savepic3.append(tmppic)
    print "Now: ", len(savepic3)
#    print savepic3
#    raw = spio.loadmat('ex3data1.mat')
#    X = np.mat(raw['X'])
#    pd.predict(Theta1, Theta2, X[600])
    pd.predict(Theta1, Theta2, tmp)

cv.bind('<B1-Motion>', usrDraw)
buttonFin.bind('<Button-1>', usrFinish)
buttonClear.bind('<Button-1>', usrClear)
buttonSave.bind('<Button-1>', usrSave)

for i in range(20):
    for j in range(20):
        usrPixmap.append(usrPixel(0, cv.create_rectangle(i*20, j*20, i*20+20, j*20+20, fill="black")))


cv.pack()
buttonFin.pack()
buttonClear.pack()
buttonSave.pack()


root.mainloop()
