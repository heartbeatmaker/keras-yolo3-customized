#! /usr/bin/env python


'''
웹캠영상, 사진, 동영상에서 object를 찾아 분석하는 코드 -- 이미지 부분 수정함!!!!

<분석 결과를 도출하는 방식>
웹캠영상: 실시간
사진, 동영상: output 파일을 따로 저장
'''

import os
import argparse
import json
import shutil
import time

import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes, draw_boxes_for_dogs
from keras.models import load_model

# tqdm = 커맨드 창에 띄울 수 있는 프로그레스 바
from tqdm import tqdm
import numpy as np

def _main_(args):

    '''실행 명령어: python predict.py -c config.json -i 분석할 파일의 경로(웹캠 실행 시: webcam 입력)'''
    # 환경변수가 저장되어 있는 파일의 경로
    config_path  = args.conf

    # 분석할 파일의 경로
    input_path   = args.input

    # 결과물을 저장할 디렉토리
    output_path  = './predict_output/'


    # 환경설정 파일을 연다
    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)



    '''
    def makedirs(path):
    try:
        # 해당 path 의 디렉토리를 생성한다
        # 디렉토리가 이미 있으면 -> 에러 발생
        os.makedirs(path)
    except OSError:
        # 그러나 디렉토리가 이미 있는 경우는 에러를 발생시키지 않도록 처리한다
        if not os.path.isdir(path):
            # 그 외의 오류일 경우 -> 에러를 발생시킨다
            raise
    '''
    # 결과물을 저장할 폴더를 생성한다
    makedirs(output_path)


    ###############################
    #   Set some parameter
    ###############################       
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster

    # define the probability threshold for detected objects -- 60% 이상의 확실성을 가져야 해당 오브젝트로 인정
    # nms: non-maximal boxes -- 여러 박스가 중첩되었을 경우, 50% 이상의 확실성을 가져야 중첩 허용
    obj_thresh, nms_thresh = 0.7, 0.5

    ###############################
    #   Load the model -- 이미 학습된 모델을 로드한다. 환경설정에서 모델을 변경할 수 있음
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])



    ###############################
    #   Predict bounding boxes 
    ###############################

    # do detection on an image or a set of images
    '''사진을 분석하는 경우'''

    while True:

        image_paths = []

        # input_path 에 file 이름을 넣지 않고 디렉토리까지만 지정했을 경우
        if os.path.isdir(input_path):
            # os.listdir : 이 디렉토리에 있는 전체 파일의 이름을 리스트 형태로 반환한다
            for inp_file in os.listdir(input_path):
                image_paths += [input_path + inp_file]

        # 그중에서 jpg, png, jpeg 확장자를 가진 파일만 남긴다
        image_paths = [inp_file for inp_file in image_paths
                       if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

        # the main loop
        for image_path in image_paths:
            image = cv2.imread(image_path)

            # predict the bounding boxes -- 검출된 object 박스를 list 형태로 반환한다
            boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

            print("image path: "+image_path+" /// number of boxes: "+str(len(boxes)))

            labels = config['model']['labels']

            isDog = False

            # 이 사진에서 검출한 박스를 하나씩 검사한다. 조건에 맞는지 확인
            for box in boxes:
                for i in range(len(labels)):
                    if box.classes[i] > obj_thresh and "dog" == labels[i]:  # 60% 이상의 확률로 개인지 확인
                        print("It's a dog!! " + str(round(box.get_score()*100, 2)) + '%')
                        isDog = True

            filename = image_path.split('/')[-1]

            # draw bounding boxes on the image using labels
            if isDog:  # 이 사물이 개라면

                # 사진 위의 개 주변에 박스를 그린다 -- 굳이 안 그려도 됨. 앱 사용자한테 보여줄 사진이기 때문에, 박스처리하지 않음
                # draw_boxes_for_dogs(image, boxes, config['model']['labels'], obj_thresh)

                # output 폴더에 가공된 사진을 저장한다 -- 가공하지 않았음
                # write the image with bounding boxes to file
                # cv2.imwrite(output_path + "processed_" + filename, np.uint8(image))

                # 원본 사진을 output 폴더로 옮긴다
                shutil.move(image_path, output_path+"detected_"+filename)

            else:
                print("It's not a dog")
                # 원본 사진을 originals 폴더로 옮긴다
                shutil.move(image_path, "./originals/"+filename)

        time.sleep(3)


if __name__ == '__main__':

    # The argparse module makes it easy to write user-friendly command-line interfaces.
    # It's what you use to get command line arguments into your program.
    # python predict.py -h 나 --help 를 입력하면 각 argument 의 description 을 볼 수 있다
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')    
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')   
    
    args = argparser.parse_args()
    _main_(args)
