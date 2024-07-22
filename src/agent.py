
from src.networking import Connection

import time

import io

from PIL import Image

class Agent:
    
    def run(self, ip, port):
            
        
        c = Connection(ip, int(port))
        
        while True:
            data = c.recieveScreenShot()
            
            image = Image.open(io.BytesIO(data))
            image.save("image.png")

            c.sendData("FRAME DONE!")