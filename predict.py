#! /usr/bin/env python


'''
웹캠영상, 사진, 동영상에서 object를 찾아 분석하는 코드

<분석 결과를 도출하는 방식>
웹캠영상: 실시간
사진, 동영상: output 파일을 따로 저장
'''

import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
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
    output_path  = './output/'


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
    obj_thresh, nms_thresh = 0.5, 0.45

    ###############################
    #   Load the model -- 이미 학습된 모델을 로드한다. 환경설정에서 모델을 변경할 수 있음
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])



    ###############################
    #   Predict bounding boxes 
    ###############################

    '''웹캠 영상을 분석하는 경우'''
    if 'webcam' in input_path:

        # do detection on the first webcam
        video_reader = cv2.VideoCapture(0)

        # the main loop
        batch_size  = 1 # 1 프레임 단위로 분석한다 -- 이걸 10으로 늘이면 화면이 멈춘다. 왜?
        images      = []
        while True:
            ret_val, frame = video_reader.read()  # 카메라에서 프레임 이미지를 얻는다
            if ret_val == True: images += [frame]

            if (len(images)==batch_size) or (ret_val==False and len(images)>0):

                # 사진에서 오브젝트를 검출한다
                batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                for i in range(len(images)):

                    # 오브젝트 주변에 박스를 그린다
                    draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh)

                    # 사진을 화면에 띄운다
                    cv2.imshow('video with bboxes', images[i])

                # 변수 초기화
                images = []

            if cv2.waitKey(1) == 27: 
                break  # esc to quit

        # 사용자가 esc 키를 누르면 웹캠 화면을 없앤다
        cv2.destroyAllWindows()

    elif input_path[-4:] == '.mp4': # do detection on a video
        '''mp4 영상을 분석하는 경우'''

        # 결과물의 이름과 경로를 지정한다
        # [-1]: 리스트의 맨 마지막 요소
        video_out = output_path + input_path.split('/')[-1]
        video_reader = cv2.VideoCapture(input_path)

        # 원본 영상의 총 프레임 수, 프레임당 크기를 가져온다
        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        # videoWriter: 비디오를 저장하는 객체
        # fourcc: 비디오 코덱을 특정하는 4바이트의 코드이다
        video_writer = cv2.VideoWriter(video_out,  # 비디오 경로
                               cv2.VideoWriter_fourcc(*'MPEG'),  # 비디오 코덱 -- MPEG 말고 다른걸로 바꾸면 어떻게 됨?
                               50.0,  # 프레임 수 -- 원래 50이었는데, 20으로 줄이면 어떻게 됨?
                               (frame_w, frame_h))  # 프레임 크기
        # the main loop
        batch_size  = 1
        images      = []
        start_point = 0 #%
        show_window = False

        # tqdm = 커맨드 창에 띄울 수 있는 프로그레스 바. A Fast, Extensible Progress Bar for Python and CLI
        # 영상의 총 프레임 수만큼 루프를 돌면서, 각각의 프레임에서 오브젝트를 검출한다. 검출된 오브젝트 주변에는 박스표시를 한다
        # 아래 절차는 웹캠 영상을 처리하는 것과 동일
        for i in tqdm(range(nb_frames)):
            _, frame = video_reader.read()  # 영상에서 프레임 이미지를 가져온다

            if (float(i+1)/nb_frames) > start_point/100.:
                images += [frame]

                if (i%batch_size == 0) or (i == (nb_frames-1) and len(images) > 0):
                    # predict the bounding boxes
                    batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                    for i in range(len(images)):
                        # draw bounding boxes on the image using labels
                        draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh)   

                        # show the video with detection bounding boxes          
                        if show_window: cv2.imshow('video with bboxes', images[i])  

                        # write result to the output video
                        video_writer.write(images[i]) 
                    images = []
                if show_window and cv2.waitKey(1) == 27: break  # esc to quit

        if show_window: cv2.destroyAllWindows()
        video_reader.release()
        video_writer.release()


    else: # do detection on an image or a set of images
        '''사진을 분석하는 경우'''
        image_paths = []

        # input_path 에 file 이름을 넣지 않고 디렉토리까지만 지정했을 경우
        if os.path.isdir(input_path):

            # os.listdir : 이 디렉토리에 있는 전체 파일의 이름을 리스트 형태로 반환한다
            for inp_file in os.listdir(input_path):
                image_paths += [input_path + inp_file]

        else:  # input_path 에 file 이름까지 지정했을 경우
            image_paths += [input_path]

        # 그중에서 jpg, png, jpeg 확장자를 가진 파일만 남긴다
        image_paths = [inp_file for inp_file in image_paths
                       if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

        # the main loop
        for image_path in image_paths:
            image = cv2.imread(image_path)
            print(image_path)

            # predict the bounding boxes
            boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

            # draw bounding boxes on the image using labels
            draw_boxes(image, boxes, config['model']['labels'], obj_thresh) 
     
            # write the image with bounding boxes to file
            cv2.imwrite(output_path + "processed_" + image_path.split('/')[-1], np.uint8(image))

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
