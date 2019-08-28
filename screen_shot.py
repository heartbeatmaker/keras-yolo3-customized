'''
스크린샷 찍는 코드
'''

import numpy as np
import cv2
import time

'''
스크린에 떠있는 컨텐츠를 복사할 수 있는 모듈
스크린샷을 찍을 수 있다
'''
from PIL import ImageGrab


'''
Create a PNG/JPEG/GIF image object given raw data.
it will result in the image being displayed in the frontend.
'''

# 메인함수
def main():

    # 결과물을 저장할 디렉토리
    output_path  = './screenshot_output/'

    i=0

    last_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    while True:

        i += 1
        final_path = output_path + last_time + "_" + str(i)+".jpg"

        # 스크린 데이터를 얻는다
        # 계속해서 스크린샷을 찍는다
        screen =  np.array(ImageGrab.grab(bbox=None))

        # 이미지 저장
        cv2.imwrite(final_path, np.uint8(screen))

        time.sleep(2)

        if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

main()
