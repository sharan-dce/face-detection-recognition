import tensorflow as tf
import sys
sys.path.append('./facenet')
from tensorflow.python.platform import gfile
import numpy as np
from src import facenet

class FACECNN:
	def __init__(self, model = './facenet/models/20180402-114759/20180402-114759.pb'):
		print('Loading FaceNet from ProtoBuf {}'.format(model))
		self.model = model
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.sess = tf.Session()
			with gfile.FastGFile(self.model,'rb') as f:
				graph_def = tf.GraphDef()
				graph_def.ParseFromString(f.read())
				tf.import_graph_def(graph_def, name='')
			self.images_placeholder = self.graph.get_tensor_by_name("input:0")
			self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
			self.embedding_size = self.embeddings.get_shape()[1]

		print('FaceNet Loaded')

	def get_embeddings(self, images):
		if(len(np.shape(images)) == 1):
			iimages = facenet.load_data(images, do_random_crop = False, do_random_flip = False, image_size = 160)
		else:
			iimages = facenet.process_data(images, False, False, 160)
		feed_dict = {self.images_placeholder: iimages, self.phase_train_placeholder: False}
		embeddings = self.sess.run(self.embeddings, feed_dict = feed_dict)
		return embeddings



# net = FACECNN()
# net.test()
# images = net.get_embeddings(['../random.jpeg'])
# print(np.shape(images))