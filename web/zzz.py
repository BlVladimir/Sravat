import threading
import time
import cv2
from flask import Flask, render_template, Response


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        if not ret:
            return None
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            return None
        return jpeg.tobytes()


app = Flask(__name__)
video_stream = VideoCamera()
recording = True
N = 20


def record_frames(camera):
    global recording
    while recording:
        frame_bytes = camera.get_frame()
        sleep_time = 1 / N
        time.sleep(sleep_time)


recording_thread = threading.Thread(target=record_frames, args=(video_stream,))
recording_thread.start()


@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    try:
        app.run(port=8080, host='127.0.0.1', debug=True)
    finally:
        recording = False