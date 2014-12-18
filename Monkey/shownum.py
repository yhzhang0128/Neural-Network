import Image
import ImageDraw
import numpy as np
import numpy.core.fromnumeric as npfunc

def createImg(vec):
    pixl = 10
    im = Image.new('L', (20*pixl, 20*pixl))
    draw = ImageDraw.Draw(im)
    for i in range(20):
        for j in range(20):
            idx = i*20+j
            draw.rectangle((i*pixl, j*pixl, i*pixl+pixl, j*pixl+pixl), fill=255*vec[0,idx])
    return im
