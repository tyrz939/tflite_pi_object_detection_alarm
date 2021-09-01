import time
import queue
import threading
import datetime
import re
from pygame import mixer

from gpiozero import LED

import numpy as np
import cv2
from object_detection.utils import visualization_utils as vis_util
from tflite_runtime.interpreter import Interpreter

class LiveView(threading.Thread):

    def __init__(self, queue_request_image, queue_live_out, stream):
        self.crashed = False
        threading.Thread.__init__(self)

        self.queue_live_out = queue_live_out
        self.queue_request_image = queue_request_image
        self.stream = stream
        self.connect_to_camera(stream)
        self.retry_counter = 0
    
    def connect_to_camera(self, stream):
        try: self.cap = cv2.VideoCapture(stream)
        except: print('Can not open camera stream...')

    def run(self):
        while True:
            try:
                status = 'frame'
                if not self.cap.isOpened():
                    raise IOError('Cannot open camera')
                else:
                    work, frame_cap = self.cap.read()
                    try:
                        if work:
                            self.retry_counter = 0 # Resets connection retries if everything is a-ok
                            self.queue_request_image.get(0)
                            ret = status, frame_cap
                            self.queue_live_out.put(ret)
                        else:
                            raise IOError('Image unable to be captured')
                    except queue.Empty:
                        pass
                    except Exception as e:
                        print('catch 2: ', e)
                        self.retry_counter = self.retry_counter + 1
                        self.cap.release
                        time.sleep(15)
                        self.connect_to_camera(self.stream)
            except Exception as e:
                print('catch 1: ', e)
                self.retry_counter = self.retry_counter + 1
                self.cap.release
                time.sleep(15)
                self.connect_to_camera(self.stream)
            if self.retry_counter >= 1000000:
                print('Retry limit reached, giving up!')
                self.crashed = True
                break
        self.cap.release
        return
    
class Inference(threading.Thread):
    def __init__(self, queue_request_image, queue_live_out, queue_out):
        threading.Thread.__init__(self)
        self.queue_request_image = queue_request_image
        self.queue_live_out = queue_live_out
        self.queue_out = queue_out

    def load_labels(self, path):
        """Loads the labels file. Supports files with or without index numbers."""
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            labels = {}
            for row_number, content in enumerate(lines):
                pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
                if len(pair) == 2 and pair[0].strip().isdigit():
                    labels[int(pair[0])] = pair[1].strip()
                else:
                    labels[row_number] = pair[0].strip()
            return labels
        
    def set_input_tensor(self, interpreter, image):
        """Sets the input tensor."""
        tensor_index = interpreter.get_input_details()[0]['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image


    def get_output_tensor(self, interpreter, index):
        """Returns the output tensor at the given index."""
        output_details = interpreter.get_output_details()[index]
        tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
        return tensor
    
    def detect_objects(self, interpreter, image, threshold):
        """Returns a list of detection results, each a dictionary of object info."""
        self.set_input_tensor(interpreter, image)
        interpreter.invoke()

        # Get all output details
        boxes = self.get_output_tensor(interpreter, 0)
        classes = self.get_output_tensor(interpreter, 1)
        scores = self.get_output_tensor(interpreter, 2)
        count = int(self.get_output_tensor(interpreter, 3))

        results = []
        for i in range(count):
            if scores[i] >= threshold:
                result = {
                    'bounding_box': boxes[i],
                    'class_id': classes[i],
                    'score': scores[i]
                }
                results.append(result)
        return results

    def run(self):
        # Points to model and label files
        labels = 'model_litev1/labelmap.txt'
        model = 'model_litev1/detect.tflite'
        # Anything below this threshold won't show
        threshold = 0.4
        labels = self.load_labels(labels)
        label_map = {}
        for item in labels.items():
            label = {'id': item[0], 'name':item[1]}
            label_map.update({item[0]: label})
        interpreter = Interpreter(model)
        interpreter.allocate_tensors()
        _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
        self.queue_request_image.put(True)
        timer = time.time()
        while True:
            try:
                status, image_np = self.queue_live_out.get(0)
                # Change this to false to stop it drawing the box
                # This allows you to run console only output and doesn't need desktop gui
                draw_boxes = True
                if status == 'frame':
                    # Crop Image, optional if you want to cut out parts of your image
                    #image_np = image_np[80:480, 0:350]
                    # Shrink image to 300x300 for the model we're using
                    image_np_small = cv2.resize(image_np, (300, 300))
                    # Runs the inference on the image and returns the results
                    results = self.detect_objects(interpreter, image_np_small, threshold)
                    # init arrays to zeros, convert results to be more easily usable
                    boxes = np.array([[0.0,0.0,0.0,0.0]])
                    classes = np.array([0])
                    scores = np.array([0.0])
                    for result in results:
                        a = np.array(result['bounding_box'])
                        boxes = np.vstack((boxes, a))
                        a = np.array([int(result['class_id'])])
                        classes = np.concatenate((classes, a))
                        a = np.array([result['score']])
                        scores = np.concatenate((scores, a))
                        
                    results = 0
                    for score, d_class in zip(scores, classes):
                        # <3 on the labelmap.txt means the first 4 (0, 1, 2, 3)
                        # being person, bicycle, car and motocycle are counted
                        if d_class <= 3:
                            if score > 0.5:
                                results += 1
                    if draw_boxes:
                        # Set window size (useful for high res cameras)
                        #image_np = cv2.resize(image_np, (800, 480))
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image_np,
                            boxes,
                            classes,
                            scores,
                            label_map,
                            skip_labels=False,
                            skip_scores=False,
                            use_normalized_coordinates=True,
                            line_thickness=2,
                            max_boxes_to_draw=None)
                        cv2.imshow('Frame', image_np)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                    inference_time = (time.time() - timer)
                    ret = True, image_np, results, inference_time
                    self.queue_out.put(ret)
                    if self.queue_live_out.qsize() == 0:
                        self.queue_request_image.put(True)

                    #print('Inference FPS: ', round(1 / inference_time, 2))
                    timer = time.time()
            except queue.Empty:
                #pass
                time.sleep(0.1)
                
class MainApplication:

    def __init__(self):
        # Relay for alarm, change number to set GPIO pin
        self.relay = LED(23)
        self.relay.off()
        
        # Live View + Inference Setup
        self.live_view = False
        self.queue_live_out = queue.Queue(maxsize=0)
        self.queue_live_inference_out = queue.Queue(maxsize=0)
        self.queue_request_image = queue.Queue(maxsize=0)
        
        # Setup and start live camera capture thread
        # note below, this is for hikvision cameras I am using
        # you may need to hunt down RTSP address for your own cameras
        # on mine 101 is main stream and 102 is substream (low resolution)
        self.live_thread = LiveView(
            self.queue_request_image,
            self.queue_live_out,
            'rtsp://admin:helpme03@192.168.50.206:554/Streaming/Channels/102/')
        
        self.live_thread.setDaemon(True)
        self.live_thread.start()
        
        # Setup and start inference thread
        self.live_inference = Inference(self.queue_request_image, self.queue_live_out, self.queue_live_inference_out)
        self.live_inference.setDaemon(True)
        self.live_inference.start()
        
    def cal_average(self, num):
        sum_num = 0
        for t in num:
            sum_num = sum_num + t
        avg = sum_num / len(num)
        return avg
    
    def trigger_alarm(self):
        self.relay.on()
        mixer.init()
        mixer.music.load("alarm.wav")
        mixer.music.play()
        # How long the alarm goes off for. 3 seconds here.
        time.sleep(3)
        self.relay.off()
        #mixer.music.stop()

    def run(self):
        # Final number is how many previous results to store to average.
        # 10 means last 8 results are stored and 4/8 images will have to contain an object to trigger the alarm
        # this helps prevent false alarms and can be tuned to your liking
        prev_results = [10] * 8
        prev_average = 10
        prev_single_result = 10
        timer = time.time()
        while True:
            if self.live_thread.crashed:
                break
            try:
                status, image_np, results, inference_time = self.queue_live_inference_out.get(0)
                averaged_results = round(self.cal_average(prev_results))
                elapsed_time = (time.time() - timer)
                if elapsed_time > 30:
                    prev_single_result = results
                    if averaged_results > prev_average:
                        print('ALARM!!!', elapsed_time, ' TIME: ', datetime.datetime.now())
                        cv2.imshow('ALARM IMAGE', image_np)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                        self.trigger_alarm()
                        timer = time.time()
                        # Show alarm image when object sets off alarm, disable for console only
                prev_results.pop()
                prev_results.insert(0, results)
                prev_average = averaged_results
            except queue.Empty:
                # print('nothing to show')
                pass
            except Exception as e:
                print('catch main: ', e)
                break
            time.sleep(0.1)

c = MainApplication()  # keeps the application open
c.run()