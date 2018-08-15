import subprocess
import json
import time
import platform
import sys
import logging
import requests
import zipfile
import os


process_ocr_url = "http://172.16.1.60:8888/ocr/process"



WORK_DIR = os.path.dirname(os.path.abspath(__file__))


class VideoOcr:

    def video_general(self, payload = None):
        r = requests.post(process_ocr_url, data = json.dumps(payload))
        result_dic = json.loads(r.content.decode("utf-8"))
        return result_dic


if __name__ == "__main__":

    payload = {'VGA_url':'/home/ocr/OCR/OCR_class/split_video/video.mp4',
               'save_url':'http://172.16.1.60:8889/test/write_data',
               'subject':'数学', 'grade':'九年级', 'curriculumId':'1234' }
    v_ocr = VideoOcr()
    
    result_dic = v_ocr.video_general(payload)
   
    print(result_dic) 
