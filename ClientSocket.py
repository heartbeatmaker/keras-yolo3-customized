import socket

socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.connect(('192.168.0.150', 8999))

while 1:
        data2 = socket.recv(65535)
        print("received data from Server : ", data2.decode())