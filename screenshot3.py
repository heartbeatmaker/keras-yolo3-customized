'''
스크린샷 찍는 코드 -- 전체화면 캡쳐 됨
'''


import d3dshot
import time
import cv2

# 결과물을 저장할 디렉토리
output_path  = './screenshot_output/'


i=0

last_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
while True:

    i += 1
    filename = last_time + "_" + str(i)+".jpg"

    d= d3dshot.create()
    d.screenshot_to_disk_every(1, output_path)


    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break