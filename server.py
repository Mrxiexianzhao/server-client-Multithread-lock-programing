import tornado.ioloop
import tornado.web
import json
import logging
from OCR_test import OCR
import os
import requests
import time
import sys
import threading
from train_data.make_train_datas import *
from check_nvidia.check_nv import *

WORK_DIR = os.path.dirname(os.path.abspath(__file__))


# 线程锁安全的维护gpu资源
# 为每个请求分配gpu资源
class GPURes():

    def __init__(self, gpus):
        self.gpus_ = [[i, True] for i in gpus]
        self.lock_ = threading.RLock()

    def get_available(self):
        with self.lock_:
            available = []
            for g in self.gpus_:
                if g[1]:
                    available.append(g[0])
        return available
     
    # 获取可用的gpu id
    def acquire_gpu(self):
        idx = -1
        with self.lock_:
            for g in self.gpus_:
                if g[1]:
                    idx = g[0]
                    g[1] = False
                    break
        return idx

    # 释放gpu
    def release_gpu(self, gpu_id):
        with self.lock_:
            for g in self.gpus_:
                if g[0] == gpu_id:
                    g[1] = True
                    break



# 创建log文件
def creat_log_file():

    targetDirect = WORK_DIR + "/runtime/bin"
    log_file_name = targetDirect + "/zkocr.log"
    
    # 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # 创建一个handler，写入日志文件  
    file_handler = logging.FileHandler(log_file_name, mode='a+')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 创建handler, 在控制台打印日志
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.formatter = formatter

    # 添加handler到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


total_logger = creat_log_file()

# 获取系统有几个显卡
supported_gpus = get_supported_gpus()

# 初始化线程安全维护类
gpus = [i for i in range(len(supported_gpus))]
c_GPURes = GPURes(gpus)


class MainHandler(tornado.web.RequestHandler):

    def initialize(self, ioloop):
        self.ioloop = ioloop


    # 处理错误请求并返回错误信息
    def process_error(self, error_code, error_msg):
        error_dic = {}
        error_dic['error_code'] = error_code
        error_dic['error_msg'] = error_msg

        total_logger.error(error_dic)

        self.ioloop.add_callback(self.finish, json.dumps(error_dic))

     
    def ocr_process(self, gpu_idx):        
        try:
            total_logger.info(self.request.body)
            info_dic = json.loads(self.request.body.decode('utf-8'))

        except Exception as ex:
            self.process_error(1, 'request body error')
            c_GPURes.release_gpu(gpu_idx)
            return 

        total_logger.info(info_dic)

        video_url = info_dic.get("VGA_url", None)
        save_url = info_dic.get("save_url", None)
        subject = info_dic.get("subject", None)
        grade = info_dic.get("grade", None)
        curriculumId = info_dic.get("curriculumId", None)

        try:
            v_ocr = OCR(video_url, subject, grade, curriculumId, gpu_idx)
            result_dict = v_ocr.main()
 
            self.ioloop.add_callback(self.finish, json.dumps(result_dict))
        
        except:
            total_logger.error("OCR_test.py error !!!", exc_info = 1)
            self.ioloop.add_callback(self.finish, json.dumps('OCR_test.py is error !!!'))

        c_GPURes.release_gpu(gpu_idx)


    # 每来一个请求分配一个线程，并为该线程分配gpu
    @tornado.web.asynchronous
    def post(self, *args, **kwargs):
        
        total_logger.info('*****************************') 
        total_logger.info('*****************************')

        gpu_idx = c_GPURes.acquire_gpu()
        if gpu_idx == -1:
            error_dic = self.process_error(7, 'there is no free gpu !!!')
        else:        
            # 每来一个请求启动一个线程 
            t = threading.Thread(target = self.ocr_process, args=(gpu_idx,))
            t.start()


# 每次任务请求之前调用一下该服务，查看是否有可用的gpu
class GPU_State(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        result_dic = {}

        if len(supported_gpus) == 0:
            result_dic['gpu'] = -1
        else:
            aviable_gpus = c_GPURes.get_available()
            if len(aviable_gpus) > 0:
                result_dic['gpu'] = 1
            else:
                result_dic['gpu'] = 0
        
        return self.finish(result_dic)
    

         
class Train(tornado.web.RequestHandler):
    def post(self, *args, **kwargs):
        data_json = self.request.body.decode('utf-8')
        #param = json.loads(data_json)
        #print (param)
        data = make_train_datas(data_json)



class NVIDIA_SMI(tornado.web.RequestHandler):
    def post(self, *args, **kwargs):
        supported_gpus = get_supported_gpus()
        if len(supported_gpus) == 0:
            GPU_info = {"error":"no nvidia"}
            return self.finish(GPU_info)
        GPU_info = {}
        GPU_info["dev_cnt"] = len(supported_gpus)
        g = []
        for card in supported_gpus:
            g.append([card.no,card.product,card.uuid,card.mem_size])
        GPU_info["gpu"] = g

        return self.finish(GPU_info)


def make_app():
    ioloop = tornado.ioloop.IOLoop()
    application = tornado.web.Application([
        (r'/ocr/process', MainHandler, dict(ioloop=ioloop)),
        (r"/ocr/get_train_data", Train),
        (r"/ocr/get_nvidia_info", GPU_State)
    ])
    application.listen(8888)
    ioloop.start()


if __name__ == "__main__":
    make_app()
