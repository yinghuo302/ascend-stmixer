import config as cfg
import cv2 as cv
import numpy as np
from typing import List
from videowriter import VideoWriter

class DataProcess:
    def __init__(self,fps,width,height) -> None:
        self.width = width
        self.height = height
        self.outStream = VideoWriter(cfg.OUT_TYPE,cfg.OUT,fps,width,height)


    def processFrameList(self,frames):
        for i in range(cfg.NFRAME):
            frames[i] = cv.resize(frames[i],(cfg.WIDTH,cfg.HEIGHT))
            frames[i] = cv.cvtColor(frames[i], cv.COLOR_BGR2RGB).transpose((2,0,1))
            frames[i] = frames[i].astype(np.float32) / 255.0
        return np.stack(frames,axis=1).reshape([1,3,cfg.NFRAME,cfg.HEIGHT,cfg.WIDTH])

    def postprocess(self,originFrames:List[tuple[np.ndarray,int]],inputFrames,boxes:np.ndarray,scores:np.ndarray) -> List[np.ndarray]:
        boxes = boxes.reshape((1,cfg.NFRAME,cfg.NQUERY,4))
        scores = scores.reshape((1,cfg.NQUERY,cfg.NLABEL+1))
        i = 0
        resFrames = []
        for (frame, idx) in originFrames:
            if i < cfg.NFRAME and inputFrames[i][1] < idx:
                i += 1
            for j in range(cfg.NQUERY):
                idx = np.argmax(scores[0,j,0:cfg.NLABEL])
                # print(f"{scores[0][j][cfg.NLABEL]},{scores[0][j][idx]}")
                if scores[0][j][cfg.NLABEL] > cfg.BG_THRESHOLD or scores[0][j][idx] < cfg.THRESHOLD:
                    continue
                x1,x2 = boxes[0][i][j][0] * self.width / cfg.WIDTH ,boxes[0][i][j][2] * self.width / cfg.WIDTH
                y1,y2 = boxes[0][i][j][1] * self.height / cfg.HEIGHT ,boxes[0][i][j][3] * self.height / cfg.HEIGHT
                x1,x2,y1,y2 = int(x1),int(x2),int(y1),int(y2)
                cv.rectangle(frame ,(x1,y1),(x2,y2),(0,0,255))
                cv.putText(frame,'{} {:.2f}'.format(cfg.LABELS[idx],scores[0][j][idx]),
                        (x1,y1),cv.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 1,8)
            resFrames.append(frame)
        return resFrames
    

    def writeFrames(self,frames:List[np.ndarray]):
        for frame in frames:
            self.outStream.write(frame)
