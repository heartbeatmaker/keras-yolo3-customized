'''
오렌지파이의 신호에 따라 스크린샷 찍는 코드 -- 전체화면 캡쳐 됨
'''


import d3dshot
import time
import cv2
import socket
import threading
import sys


# 결과물을 저장할 디렉토리
output_path  = './screenshot_output/'


# class Screenshot(threading.Thread):
#
#     def __init__(self):
#         threading.Thread.__init__(self)
#         self.signal = True
#
#     def run(self):
#         print("스크린샷 시작")
#
#         while self.signal:
#             d = d3dshot.create()
#             d.screenshot_to_disk_every(1, output_path)
#
#     def stop(self):
#         print("스크린샷 종료")
#         self.signal = False


'''오렌지파이(서버)와 연결'''
socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.connect(('192.168.0.77', 8999))
print("서버에 연결")

global capture

while 1:
    capture=True
    data2 = socket.recv(65535)
    message = data2.decode()
    print("received data from Server : ", message)
    # t = Screenshot()
    if capture:
        try:
            d = d3dshot.create()
        except:
            print("예외")

    if capture:
        if message == "on":  # 자동차가 주행을 시작한다는 메시지를 받으면 -> 1초에 한번씩 스크린샷을 찍는다
            print("On ON On On")
            # t.start()
            d.screenshot_to_disk_every(1, output_path)

    if message == "off":  # 자동차가 주행을 멈췄다는 메시지를 받으면 -> 스크린샷 정지
        # t.stop()
        print("Off Off Off Off")
        capture=False
        print("캡처 : "+ str(capture))
        time.sleep(1)
        d.stop()
        break

print("시스템 종료")
sys.exit()


