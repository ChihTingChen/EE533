from PIL import Image
import numpy as np
img=Image.open("tunyuan.png").convert("L")
img=img.resize((2048,2048))
image_matrix=np.array(img,dtype=np.uint32)
image_matrix.tofile("2048tunyuan.raw")
