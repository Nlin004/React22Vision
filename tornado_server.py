# Writing the Tornado server to initiate the client request

import webbrowser
import tornado.ioloop
import tornado.web
import tornado.websocket
import cv2
# from img_to_string import to_b64
import base64
from ball_detect import main, source, source2
from line_detector import detect_line
from tornado.options import define, options

# Converts image to string using base64 encoding.
def to_b64(filename):
    with open(filename, "rb") as img_file:
        my_string = base64.b64encode(img_file.read())

    return my_string

# Websocket defined with port 8080.
define('port', default=1111, type=int)

# you should know this
cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_EXPOSURE, -7)
cap2 = cv2.VideoCapture(source2, cv2.CAP_DSHOW)
cap2.set(cv2.CAP_PROP_EXPOSURE, -7)
# LIGHTING: -1 for internal camera, -7 for FISHEYE, -4 for Microsoft HD-3000


# This handler handles a call to the base of the server \
# (127.0.0.1:8888/ -> 127.0.0.1:8888/index.html)
class IndexHandler(tornado.web.RequestHandler):
    # GET request to get the base webpage
    # from the Tornado server
    def get(self):
        self.render('websocket/www/index.html')

# This handler handles a websocket connection
class WebSocketHandler(tornado.websocket.WebSocketHandler):
    # function to open a new connection to the WebSocket
    def open(self, *args):
        print('new cargo connection!')
        # self.write_message('welcome!')

    # function to respond to a message on the WebSocket
    def on_message(self, message):
        _, frame = cap.read()
        
        # enter open cv code here
        output_image = main(frame, message)
        cv2.imwrite("./websocket/frame.jpg", output_image)

        self.write_message(to_b64("./websocket/frame.jpg"))

    # function to close a connection on the WebSocket
    def on_close(self):
        print('connection closed')

class ShadowHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("websocket/www/line.html")

class ShadowSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self, *args):
        print("line websocket connection")
    
    def on_message(self, message):
        _, frame = cap2.read()
        output_image = detect_line(frame)
        cv2.imwrite("./websocket/line.jpg", output_image)
        self.write_message(to_b64("./websocket/line.jpg"))


    def on_close(self):
        print('connection closed')

class RoboRIOHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('./www/info.html')

class RoboRIOSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self, *args):
        print("info open")
    
    def on_close(self):
        print("info close")

    def on_message(self, message):
        _, frame = cap.read()
        
        angles = detect_line(frame, True)

        data = {
            "ball_angle" : 39,
            "ball_distance" : 10,
            "line_angle" : angles
        }   
        self.write_message(data)

app = tornado.web.Application([
    (r'/', IndexHandler),
    (r'/ws/', WebSocketHandler),
    (r'/line/', ShadowHandler),
    (r'/line/ws/', ShadowSocketHandler),
    (r'/info/', RoboRIOHandler),
    (r'/info/ws/', RoboRIOSocketHandler)
])

if __name__ == '__main__':
    app.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()