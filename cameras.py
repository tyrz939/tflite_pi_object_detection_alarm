import time
import queue
import threading
import datetime
import re

from gpiozero import Button
from gpiozero import LED

import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter
from object_detection.utils import visualization_utils as vis_util


class LiveView(threading.Thread):

    def __init__(self, queue_request_image, queue_live_out, stream):
        self.crashed = False
        self._stopper = threading.Event()
        threading.Thread.__init__(self)

        self.timer = time.time()
        self.queue_live_out = queue_live_out
        self.queue_request_image = queue_request_image
        self.stream = stream
        self.connect_to_camera(stream)
        
        self.retry_counter = 0
        global c
    
    def connect_to_camera(self, stream):
        try: self.cap = cv2.VideoCapture(stream)
        except: print('Can not open camera stream...')

    # function using _stop function
    def stop(self):
        self._stopper.set()

    def stopped(self):
        return self._stopper.isSet()

    def run(self):
        while True:
            self.timer = time.time()

            if self.stopped():
                return
            try:
                status = 'frame'
                if not self.cap.isOpened():
                    raise IOError('Cannot open camera')
                    ret = 'Cannot open camera', False, False
                    self.queue_live_out.put(ret)
                    break
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
                            ret = 'Image unable to be captured', False, False
                            self.queue_out.put(ret)
                            break
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
        self._stopper = threading.Event()
        threading.Thread.__init__(self)
        self.queue_request_image = queue_request_image
        self.queue_live_out = queue_live_out
        self.queue_out = queue_out

    # function using _stop function
    def stop(self):
        self._stopper.set()

    def stopped(self):
        return self._stopper.isSet()

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
        labels = 'model_litev1/labelmap.txt'
        model = 'model_litev1/detect.tflite'
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
            if self.stopped():
                break
            try:
                status, image_np = self.queue_live_out.get(0)
                draw_boxes = True
                if status == 'frame':
                    # Crop Image
                    #image_np = image_np[80:480, 0:350]
                    image_np_small = cv2.resize(image_np, (300, 300))
                    results = self.detect_objects(interpreter, image_np_small, threshold)

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
                        if d_class <= 4:
                            if score > 0.5:
                                results += 1
                    if draw_boxes:
                        #output_resolution = (800, 480)
                        #image_np = cv2.resize(image_np, output_resolution)
                            
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
                        inference_time = (time.time() - timer)
                        ret = True, image_np, results, inference_time
                        cv2.imshow('Frame', image_np)
                        # Press Q on keyboard to  exit
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                    else:
                        inference_time = (time.time() - timer)
                        ret = True, image_np, results, inference_time
                    self.queue_out.put(ret)
                    if self.queue_live_out.qsize() == 0:
                        self.queue_request_image.put(True)

                    print('Inference FPS: ', round(1 / inference_time, 2))
                    timer = time.time()
            except queue.Empty:
                #pass
                time.sleep(0.1)


class MainApplication:

    def __init__(self):
        # Relay for alarm
        self.relay = LED(23)
        self.relay.off()
        
        # Live View + Inference Setup
        self.live_view = False
        self.queue_live_out = queue.Queue(maxsize=0)
        self.queue_live_inference_out = queue.Queue(maxsize=0)
        self.queue_request_image = queue.Queue(maxsize=0)
        
        self.live_thread = LiveView(
            self.queue_request_image,
            self.queue_live_out,
            'rtsp://user:passwd@address:554/Streaming/Channels/102/')
        
        self.live_thread.setDaemon(True)
        self.live_thread.start()
        
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
        time.sleep(3)
        self.relay.off()

    def run(self):
        prev_results = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
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
                        self.trigger_alarm()
                        timer = time.time()
                        
                        # Show Alarm Image
                        #cv2.imshow('ALARM IMAGE', image_np)
                        #if cv2.waitKey(25) & 0xFF == ord('q'):
                        #    break
                prev_results.pop()
                prev_results.insert(0, results)
                prev_average = averaged_results
            except queue.Empty:
                # print('nothing to show')
                pass
            except Exception as e:
                print('catch main: ', e)
                break
            time.sleep(0.01)

c = MainApplication()  # keeps the application open
c.run()

print('Goodbye')


