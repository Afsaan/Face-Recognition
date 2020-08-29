from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import asarray
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1

path = 'data/friends/'
 
# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	print(face_array.shape)
	return face_array
 
# specify folder to plot
# folder = 'data/train_images/afsan/'
# i = 1
# # enumerate files
# for filename in listdir(folder):
# 	# path
# 	path = folder + filename
# 	# get face
# 	face = extract_face(path)
# 	print(i, face.shape)
# 	# plot
# 	pyplot.subplot(2, 7, i)
# 	pyplot.axis('off')
# 	pyplot.imshow(face)
# 	i += 1
# pyplot.show()


# load images and extract faces for all images in a directory
def load_faces(directory):
	faces = list()
	# enumerate files
	for filename in listdir(directory):
		# path
		path = directory + filename
		# get face
		face = extract_face(path)
		# store
		faces.append(face)
	return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)

train_x , train_y = load_dataset(f'{path}/train_images/')
print(train_x)
print('-----------------------------------------------------------------------------------------------------------')
print(train_y)

print('------------------------------------------------------test section-----------------------------------------')

test_x , test_y = load_dataset(f'{path}/test_images/')
print(test_x)
print('-----------------------------------------------------------------------------------------------------------')
print(test_y)

savez_compressed('3-friends-faces-dataset.npz', train_x, train_y, test_x, test_y)






