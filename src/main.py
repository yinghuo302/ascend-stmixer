from typing import List
import cv2 as cv
import config as cfg
from engine import ACLEngine
import numpy as np
from dataprocess import DataProcess
import acl
from time import sleep
from concurrent.futures import ThreadPoolExecutor
# from multiprocessing import Pool,Value

engine = ACLEngine(cfg.MODEL_FILE)
cap = cv.VideoCapture(cfg.IN)
fps = cap.get(cv.CAP_PROP_FPS)
ori_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
ori_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
processor = DataProcess(fps,ori_w,ori_h)
current=1
# current = Value('i',1)


def inference(originFrames:List[tuple[np.ndarray,int]],inputFrames:List[tuple[np.ndarray,int]],id):
    global current
    acl.rt.set_context(engine.context)
    frames = [inputFrame[0] for inputFrame in inputFrames]
    frames = processor.processFrameList(frames)
    print(f"start {id}")
    # with open("input.bin", "wb") as f:
    #     frames.tofile(f)
    scores,boxes = engine.inference(frames)
    # with open("output.bin", "wb") as f:
    #     np.hstack((scores.flatten(), boxes.flatten())).tofile(f)
    frames = processor.postprocess(originFrames,inputFrames,boxes,scores)
    del originFrames,inputFrames,boxes,scores
    while current != id:
        print(f'sleep {id}')
        sleep(0.2)
    print(f'write {id}')
    processor.writeFrames(frames)
    current = (current + 1) & 1073741823
    print(f'finish {id}')
    

def main():
    pool = ThreadPoolExecutor(max_workers=5)
    # pool = Pool(processes=3)
    idx,cnt,sample,c = 0,0,0,0
    originFrame,inputFrame = [],[]
    while (cap.isOpened()):
        success, frame = cap.read()
        if success == False or c==1:
            break
        idx = (idx + 1) & 1073741823
        sample += 1
        originFrame.append((frame,idx))
        if sample >= cfg.SAMPLE:
            sample -= cfg.SAMPLE
            inputFrame.append((frame,idx))
            cnt += 1
            if cnt == cfg.NFRAME:
                c = (c + 1) & 1073741823
                inference(originFrame,inputFrame,c)
                # pool.submit(inference,originFrame.copy(),inputFrame.copy(),c)
                # pool.apply_async(inference,(originFrame.copy(),inputFrame.copy(),c))
                originFrame,inputFrame = [],[]
                cnt = 0
    pool.shutdown(wait=True)
    # pool.close()
    # pool.join()


if __name__ == '__main__':
    main()
    
        
