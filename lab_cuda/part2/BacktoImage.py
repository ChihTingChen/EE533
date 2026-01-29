import numpy as np
from PIL import Image
M=2048
N=7
out=M-N+1
d=np.fromfile("2048tunyuan7_conv.raw",dtype=np.uint32)
d=d.reshape((out,out))
mn=d.min()
mx=d.max()
if mx==mn:
    img=np.zeros_like(d,dtype=np.uint8)
else:
    img=((d-mn)/(mx-mn)*255).astype(np.uint8)
Image.fromarray(img).save("2048tunyuan7_conv.png")
