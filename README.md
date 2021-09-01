# tflite_pi_object_detection_alarm
 Raspberry Pi TensorFlow Lite Object Detection for security cameras + alarm
 
For my [blog post here](https://johnkeen.tech/raspberry-pi-for-security-camera-object-detection/)

Pin 23 on the pi is used for either lighting up an LED or switching a Relay to activate an alarm.

## Installing Requirements

This script requires TensorFlow Lite runtime for inference as well as OpenCV for image capturing.

### Installing TensorFlow Lite:

<code>echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install python3-tflite-runtime</code>

### Using pip3 we can install required python libraries

<code>pip3 install opencv-python
pip3 install numpy==1.21.2
pip3 install tensorflow-object-detection-api</code>

### Now some extra library requirements

<code>sudo apt-get install -y libatlas-base-dev libhdf5-dev libhdf5-dev libatlas-base-dev libjasper-dev libqtgui4 libqt4-test</code>
