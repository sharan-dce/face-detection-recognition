# FaceRecognition

# Samples
![Alt text](https://github.com/sharan-dce/face-detection-recognition/blob/master/samples/test_image_1.jpg)
![Alt text](https://github.com/sharan-dce/face-detection-recognition/blob/master/samples/test_image_2.jpg)
![Alt text](https://github.com/sharan-dce/face-detection-recognition/blob/master/samples/test_image_3.jpg)
![Alt text](https://github.com/sharan-dce/face-detection-recognition/blob/master/samples/test_image_4.jpg)
![Alt text](https://github.com/sharan-dce/face-detection-recognition/blob/master/samples/test_image_5.jpg)
![Alt text](https://github.com/sharan-dce/face-detection-recognition/blob/master/samples/test_image_6.jpg)

# Sample Video
https://drive.google.com/file/d/1VZamGZdyFyJd1sNG2m3QScRvKZjyFeiV/view?usp=sharing

# Instructions
Clone the repo:
git clone https://github.com/sharan-dce/face-detection-recognition.git

## Training over your images
Store tightly cropped faces, to be recognized in a directory and negative ones in another.
Run the core.py file with arguments as the path to the positive directory and the negative one.
python3 core.py --train true --positives_dir ./path_to_the_positive_directory --negatives_dir ./path_to_the_negative_directory

## Running over images
python3 core.py --input_image_path ./input_image_path --output_image_path ./output_image_path

## Running over videos
python3 core.py --input_video_path ./input_video_path --output_video_path ./output_video_path

# Credits:
David Sandberg for FaceNet's implementation (https://github.com/davidsandberg/facenet)
