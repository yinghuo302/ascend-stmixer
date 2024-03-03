class VideoWriter():
    def __init__(self, type:str,outUrl,fps,width,height) -> None:
        type = type.lower()
        if type == 'opencv':
            self.writer = OpenCVWriter(outUrl,fps,width,height)
        else:
            self.writer = FFMpegWriter(type,outUrl,fps,width,height)
    
    def write(self,frame):
        self.writer.write(frame)

    def close(self):
        self.writer.close()


class FFMpegWriter():
    
    def __init__(self,type,outUrl,fps,width,height) -> None:
        import subprocess as sp
        if type == 'rtsp':
            command = [
                'ffmpeg', '-y', '-f', 'rawvideo',
                '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
                '-s', '{}x{}'.format(width, height),
                '-r', str(fps), '-i', '-',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast', '-f', 'rtsp', outUrl]
        elif type == 'rtmp':
            command = [ 
                'ffmpeg', '-y', '-an', '-f', 'rawvideo',
                '-vcodec','rawvideo', '-pix_fmt', 'bgr24',
                '-s', '{}x{}'.format(width, height),
                '-r', str(fps),'-i', '-',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast', '-f', 'flv', outUrl
            ]
        elif type == 'fbdev':
            command = [
                'ffmpeg', '-y', '-f', 'rawvideo',
                '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
                '-s', '{}x{}'.format(width, height),
                '-r', str(fps), '-i', '-',
                '-pix_fmt', 'bgra', '-preset', 'ultrafast',
                '-f' ,'fbdev' ,'/dev/fb0'
            ]
        else:
            AssertionError("Only support opencv, rtsp, rtmp, fbdev video writer")
        self.stream = sp.Popen(command,stdin=sp.PIPE)

    def write(self,frame):
        self.stream.stdin.write(frame)

    def close(self):
        self.stream.stdin.close()
        self.stream.wait()
    
class OpenCVWriter():
    def __init__(self,outUrl,fps,width,height) -> None:
        import cv2 as cv
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        self.writer = cv.VideoWriter(outUrl, fourcc,fps, (width,height))


    def write(self,frame):
        self.writer.write(frame)

    def close(self):
        self.writer.release()
