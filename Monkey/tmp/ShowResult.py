from Tkinter import *

class ShowResult:
    def __init__(self, vec):
        ans = 0
        maxp = 0
        Total = 0
        for i in range(10):
            if (maxp < vec[i]):
                maxp = vec[i]
                ans = i
            Total = Total + vec[i]
        print "The Number is: ", ans
        
        self.root = Tk()
        self.root.title('Result')
        self.root.geometry('400x400')
        self.cv = Canvas(self.root, width=400, height=400, bg="white")

        center = (50,10, 350, 310)
        pies = list()
        nowstart = 0
        labelx, labely = 50, 360
        colors = ("#DC143C", "#6A5ACD", "#E6E6FA", "#708090", "#87CEFA", 
                  "#AFEEEE", "#228B22", "#FFFFF0", "#D2691E", "#B22222")
        for i in range(10):
            pie = self.cv.create_arc(center, start=nowstart, extent=360.0*vec[i]/Total, fill=colors[i])
            nowstart = nowstart + 360.0*vec[i]/Total;
            pies.append(pie)
            
            self.cv.create_rectangle(labelx, labely, labelx+10, labely+10, fill=colors[i])
            self.cv.create_text(labelx + 20, labely-1, text="%d"%(i), anchor=N)
            labelx += 30
        self.cv.create_text(200, 340, text="I think the number is %d!"%(ans), anchor=N)
        self.cv.pack()
    
    def show(self):
        self.root.mainloop()