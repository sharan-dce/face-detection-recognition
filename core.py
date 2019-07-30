import cv2 as cv
from mtcnn.mtcnn import MTCNN
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
import os
from facenet import facecnn
from facenet.src import facenet as fc
from sklearn.svm import SVC
import pickle
# def process_video(input_video_path, output_dir):
# 	video_capture = cv.VideoCapture(input_video_path)
# 	success, frame = video_capture.read()
# 	count = 0
# 	detector = MTCNN()
# 	net = facecnn.FACECNN()
# 	classifier_filename_exp = './svm_weights/params'
# 	with open(classifier_filename_exp, 'rb') as infile:
# 		model = pickle.load(infile)
# 	print('Loaded classifier model from file "%s"' % classifier_filename_exp)
# 	while success:
# 		frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
# 		result = detector.detect_faces(frame)
# 		# print(result)
# 		frame_faces = []
# 		detected_faces = []
# 		for i in range(len(result)):
# 			bounding_box = result[i]['box']
# 			# now crop out the face
# 			# print(np.shape(frame))
# 			face = frame[bounding_box[1]: bounding_box[1] + bounding_box[3], bounding_box[0]: bounding_box[0] + bounding_box[2], :]
# 			if(face.size > 0):
# 				# cv.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (255, 0, 255), 2)
# 				# face = cv.cvtColor(face, cv.COLOR_RGB2BGR)
# 				face = cv.resize(face, (160, 160))
# 				frame_faces.append(face)
# 				detected_faces.append(result[i]['box'])
# 				# face_embedding = net.get_embeddings()
# 				# cv.imwrite(os.path.join(output_dir, 'frame' + str(count) + '_face' + str(i) + '.bmp'), face)

# 		# print(np.shape(frame_faces)[0])
# 		embeddings = net.get_embeddings(frame_faces)
# 		predictions = model.predict_proba(embeddings)
# 		best_class_indices = np.argmax(predictions, axis = 1)
# 		# print(np.shape(detected_faces), np.shape(best_class_indices))
# 		for box, prob in zip(detected_faces, best_class_indices):
# 			if(prob):
# 				cv.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
# 			else:
# 				cv.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0), 2)
# 		frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
# 		#do something with the result
# 		cv.imwrite(os.path.join(output_dir, 'frame' + str(count) + '.bmp'), frame)
# 		print('\rFrame {}'.format(count), end = '')
# 		success, frame = video_capture.read()
# 		count += 1
# 	print('\rDone')

def process_video(input_video_path, output_video_path):
	video_capture = cv.VideoCapture(input_video_path)
	video_writer = cv.VideoWriter(output_video_path, cv.VideoWriter_fourcc('F','M','P','4'), video_capture.get(cv.CAP_PROP_FPS), (int(video_capture.get(3)),int(video_capture.get(4))))
	success, frame = video_capture.read()
	count = 0
	detector = MTCNN()
	net = facecnn.FACECNN()
	classifier_filename_exp = './svm_weights/params'
	with open(classifier_filename_exp, 'rb') as infile:
		model = pickle.load(infile)
	print('Loaded classifier model from file "%s"' % classifier_filename_exp)
	while success:
		frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
		result = detector.detect_faces(frame)
		# print(result)
		frame_faces = []
		detected_faces = []
		for i in range(len(result)):
			bounding_box = result[i]['box']
			# now crop out the face
			# print(np.shape(frame))
			face = frame[bounding_box[1]: bounding_box[1] + bounding_box[3], bounding_box[0]: bounding_box[0] + bounding_box[2], :]
			if(face.size > 0):
				# cv.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (255, 0, 255), 2)
				# face = cv.cvtColor(face, cv.COLOR_RGB2BGR)
				face = cv.resize(face, (160, 160))
				frame_faces.append(face)
				detected_faces.append(result[i]['box'])
				# face_embedding = net.get_embeddings()
				# cv.imwrite(os.path.join(output_dir, 'frame' + str(count) + '_face' + str(i) + '.bmp'), face)

		# print(np.shape(frame_faces)[0])
		embeddings = net.get_embeddings(frame_faces)
		predictions = model.predict_proba(embeddings)
		best_class_indices = np.argmax(predictions, axis = 1)
		# print(np.shape(detected_faces), np.shape(best_class_indices))
		for box, prob in zip(detected_faces, best_class_indices):
			if(prob):
				cv.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
			else:
				cv.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0), 2)
		frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
		#do something with the result
		cv.imshow('Processed Frame', frame)
		if cv.waitKey(25) & 0xFF == ord('q'):
			break
		video_writer.write(frame)
		print('\rFrame {}/{}'.format(count, int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))), end = '')
		success, frame = video_capture.read()
		count += 1
	video_capture.release()
	video_writer.release()
	print('\nDone')

def train_decision_model(positives, negatives):
	positives_list = os.listdir(positives)
	negatives_list = os.listdir(negatives)
	for i in range(len(positives_list)):
		positives_list[i] = os.path.join(positives, positives_list[i])
	for i in range(len(negatives_list)):
		negatives_list[i] = os.path.join(negatives, negatives_list[i])
	for file_name in positives_list + negatives_list:
		loaded_image = cv.resize(cv.imread(file_name), (160, 160))
		cv.imwrite(file_name, loaded_image)
	net = facecnn.FACECNN()
	positive_embeddings = net.get_embeddings(positives_list)
	negative_embeddings = net.get_embeddings(negatives_list)
	# print(np.shape(positive_embeddings))
	# print(np.shape(negative_embeddings))
	labels = np.concatenate((np.full(shape = [len(positive_embeddings)], fill_value = True), np.full(shape = [len(negative_embeddings)], fill_value = False)))
	embeddings = np.concatenate((positive_embeddings, negative_embeddings))
	model = SVC(kernel = 'linear', probability = True)
	classifier_filename_exp = './svm_weights/params'
	model.fit(embeddings, labels)
	with open(classifier_filename_exp, 'wb') as outfile:
		pickle.dump(model, outfile)
	print('Saved classifier model to file "%s"' % classifier_filename_exp)

def process_image(image_path, output_path):
	detector = MTCNN()
	net = facecnn.FACECNN()
	classifier_filename_exp = './svm_weights/params'
	with open(classifier_filename_exp, 'rb') as infile:
		model = pickle.load(infile)
	print('Loaded classifier model from file "%s"' % classifier_filename_exp)

	frame = cv.imread(image_path)
	frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
	result = detector.detect_faces(frame)
	# print(result)
	frame_faces = []
	detected_faces = []
	for i in range(len(result)):
		bounding_box = result[i]['box']
		# now crop out the face
		# print(np.shape(frame))
		face = frame[bounding_box[1]: bounding_box[1] + bounding_box[3], bounding_box[0]: bounding_box[0] + bounding_box[2], :]
		if(face.size > 0):
			# cv.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (255, 0, 255), 2)
			# face = cv.cvtColor(face, cv.COLOR_RGB2BGR)
			face = cv.resize(face, (160, 160))
			frame_faces.append(face)
			detected_faces.append(result[i]['box'])
			# face_embedding = net.get_embeddings()
			# cv.imwrite(os.path.join(output_dir, 'frame' + str(count) + '_face' + str(i) + '.bmp'), face)

	# print(np.shape(frame_faces)[0])
	embeddings = net.get_embeddings(frame_faces)
	predictions = model.predict_proba(embeddings)
	best_class_indices = np.argmax(predictions, axis = 1)
	# print(np.shape(detected_faces), np.shape(best_class_indices))
	for box, prob in zip(detected_faces, best_class_indices):
		if(prob):
			cv.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
		else:
			cv.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0), 2)
	frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
	#do something with the result
	cv.imwrite(output_path, frame)

def main():
	argparse = ArgumentParser()
	argparse.add_argument('--input_video_path', type = str, help = 'Path to the input video file', default = './input_video')
	argparse.add_argument('--input_image_path', type = str, help = 'Path to input image', default = '-')
	argparse.add_argument('--output_image_path', type = str, help = 'Path to output image', default = '-')
	argparse.add_argument('--output_video_path', type = str, help = 'Path to the output video', default = './')
	argparse.add_argument('--train', type = str, help = 'true/false - to train or not', default = 'false')
	argparse.add_argument('--positives_dir', type = str, help = 'Path to positive images', default = './positives')
	argparse.add_argument('--negatives_dir', type = str, help = 'Path to negative images', default = './negatives')
	args = argparse.parse_args()
	if(args.train == 'true'):
		train_decision_model(args.positives_dir, args.negatives_dir)
	elif(args.input_image_path == '-'):
		process_video(args.input_video_path, args.output_video_path)
	else:
		process_image(args.input_image_path, args.output_image_path)

if __name__  == '__main__':
	main()