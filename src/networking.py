import socket

from threading import Thread

class Connection:
    
    def __init__(self, ip, port):
        self.socket = socket.socket()
        self.client = None
        
        self.thread = Thread(target=self.connect, args=[ip, port])
        self.thread.start()
    
    def connect(self, ip, port):
        self.socket.bind((ip, port))
        self.socket.listen()
        
        self.client, _ = self.socket.accept()
        print("Connnected to client!")

        connection_msg = self.recieveData(10)
        print("Recieved!", connection_msg)

        return True
    
    def disconnect(self):
        self.client.close()
    
    def recieveData(self, length=1024):
        msg = self.client.recv(length).decode()
        splitmesssage = msg.split(" ", 1)
        msg_length = splitmesssage[0]
        msg_data = splitmesssage[1]  

        while len(msg_data) < int(msg_length):
            msg_data += self.client.recv(int(msg_length) - len(msg_data)).decode()

        return msg_data
    
    def recieveScreenShot(self):
        buffer = self.client.recv(20000)
        
        msg = buffer.split(b' ', 1)
        
        return msg[1]

    
    def sendData(self, data):
        msg = str(len(data)) + " " + data
        self.client.send(msg.encode())