import os
from PIL import Image
from glob import glob
import tensorflow as tf

#Data constants
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 256
MAX_VALUE_FOR_NORMALIZATION = 10

#Model constants
NUM_NEURONS_DENSE_LAYER_1 = 4000
NUM_NEURONS_DENSE_LAYER_2 = 2000
NUM_NEURONS_DENSE_LAYER_3 = NUM_NEURONS_DENSE_LAYER_1
NUM_NEURON_DT_NET = 200

#Training constants
BATCH_SIZE = 128
MIN_AFTER_DEQUEUE = 1000
NUM_THREADS= 32
CAPACITY = MIN_AFTER_DEQUEUE + (NUM_THREADS + 1)*BATCH_SIZE
NUM_EPOCHS = 5

#Optimizer constants
INITIAL_LEARNING_RATE = 0.0002
BETA1 = 0.5
EPSILON = 0.00000001
lamda_l2 = 1
lamda_adv = 10000

#Dir for model saving
log_dir_save = "DCGAN_dtnet_save/DCGAN_dtnet_save_03"

#path to DB file
train_DB = "train_cylinderSquareSemi_smaller_dtDB.txt"
test_DB = "test_cylinderSquareSemi_smaller_dtDB.txt"

def read_data(train_DB, test_DB):

	""" Read the pairs image,label in DB_fileNames and extract the data from the csv files 
			Input : DB_fileNames : file containing the path of the pairs image,label
			Output : Two tensors of shape(N,nx*ny) containing the N image/label vorticity grid 
	"""

	#Open the files containing the pairs (image_path, label_path) for training
	train_DB_file = open(train_DB, "r")
	train_DB_list=[]
	#read file to fill the fileNames_list
	for line in train_DB_file:
		(img_path,label_path, kdt) = line.strip().split(",")
		train_DB_list.append((img_path,label_path,kdt))

	train_DB_file.close()

	#Open the files containing the pairs (image_path, label_path) for testing 
	test_DB_file = open(test_DB, "r")
	test_DB_list=[]
	#read file to fill the fileNames_list
	for line in test_DB_file:
		(img_path,label_path, kdt) = line.strip().split(",")
		test_DB_list.append((img_path,label_path,kdt))

	test_DB_file.close()


	# Shuffle database
	random.shuffle(test_DB_list)
	random.shuffle(train_DB_list)

	# partition testing data for train and validation
	total_test_set_size = len(test_DB_list)
	print("Test dataset size : {}".format(total_test_set_size))
	test_set_size = int(0.7 * total_test_set_size) #take 70% of the dataset for testing and the rest for validation	
	validation_DB_list = test_DB_list[test_set_size:]
	test_DB_list = test_DB_list[:test_set_size]

	#for each image_file,label_file the vorticity grid is extracted and put into a list
	validation_images = []
	validation_labels = []
	validation_kdts = []
	for (img_path,label_path, kdt) in validation_DB_list:
		validation_images.append(img_path)
		validation_labels.append(label_path)
		validation_kdts.append(float(kdt))

	#for each image_file,label_file the vorticity grid is extracted and put into a list
	train_images = []
	train_labels = []
	train_kdts = []
	for (img_path,label_path,kdt) in train_DB_list:
		train_images.append(img_path)
		train_labels.append(label_path)
		train_kdts.append(float(kdt))


	#for each image_file,label_file the vorticity grid is extracted and put into a list
	test_images = []
	test_labels = []
	test_kdts = []
	for (img_path,label_path, kdt) in test_DB_list:
		test_images.append(img_path)
		test_labels.append(label_path)
		test_kdts.append(float(kdt))

	print("train image dataset size = {}".format(len(train_images)))
	print("train label dataset size = {}".format(len(train_labels)))
	print("validation image dataset size = {}".format(len(validation_images)))
	print("validation label dataset size = {}".format(len(validation_labels)))
	print("test image dataset size = {}".format(len(test_images)))
	print("test label dataset size = {}".format(len(test_labels)))


	return len(train_DB_list), len(test_DB_list), validation_images, train_images, test_images, validation_labels, train_labels, test_labels, validation_kdts, train_kdts, test_kdts

def get_files_and_time():
    train_set_size, test_set_size, validation_images, train_images, test_images, validation_labels, train_labels, test_labels, validation_kdts, train_kdts, test_kdts = read_data(train_DB, test_DB)
    validation_kdts = tf.stack(validation_kdts)
    train_kdts = tf.stack(train_kdts)
    test_kdts = tf.stack(test_kdts)

    validation_kdts = tf.cast(validation_kdts, tf.float32)
    train_kdts = tf.cast(train_kdts, tf.float32)
    test_kdts = tf.cast(test_kdts, tf.float32)

    # Make a queue of with the data.
    print("Create queues containing the pairs img, label")
    validation_image = tf.train.string_input_producer(validation_images, shuffle = False)
    train_image = tf.train.string_input_producer(train_images, shuffle = False)
    test_image = tf.train.string_input_producer(test_images, shuffle = False)
    validation_label = tf.train.string_input_producer(validation_labels, shuffle = False)
    train_label = tf.train.string_input_producer(train_labels, shuffle = False)
    test_label = tf.train.string_input_producer(test_labels, shuffle = False)
    validation_kdt = tf.train.input_producer(validation_kdts, element_shape = (), shuffle= False)
    train_kdt = tf.train.input_producer(train_kdts, element_shape = (), shuffle= False)
    test_kdt = tf.train.input_producer(test_kdts, element_shape = (), shuffle= False)


    # Read files
    reader =tf.WholeFileReader()
    _,validation_image = reader.read(validation_image)
    _,validation_label = reader.read(validation_label)
    _,train_image = reader.read(train_image)
    _,train_label = reader.read(train_label)
    _,test_image = reader.read(test_image)
    _,test_label = reader.read(test_label)

    #Decode raws
    validation_image = tf.decode_raw(validation_image, tf.float32)
    validation_label = tf.decode_raw(validation_label, tf.float32)
    train_image = tf.decode_raw(train_image, tf.float32)
    train_label = tf.decode_raw(train_label, tf.float32)
    test_image = tf.decode_raw(test_image, tf.float32)
    test_label = tf.decode_raw(test_label, tf.float32)

    validation_image.set_shape(IMAGE_HEIGHT*IMAGE_WIDTH)
    validation_label.set_shape(IMAGE_HEIGHT*IMAGE_WIDTH)
    train_image.set_shape(IMAGE_HEIGHT*IMAGE_WIDTH)
    train_label.set_shape(IMAGE_HEIGHT*IMAGE_WIDTH)
    test_image.set_shape(IMAGE_HEIGHT*IMAGE_WIDTH)
    test_label.set_shape(IMAGE_HEIGHT*IMAGE_WIDTH)

    #Normalize data
    NORM_VALUE = tf.constant(1.0/MAX_VALUE_FOR_NORMALIZATION, dtype=tf.float32)
    train_image = tf.scalar_mul(NORM_VALUE, train_image)
    train_label = tf.scalar_mul(NORM_VALUE, train_label)
    validation_image = tf.scalar_mul(NORM_VALUE, validation_image)
    validation_label = tf.scalar_mul(NORM_VALUE, validation_label)
    test_image = tf.scalar_mul(NORM_VALUE, test_image)
    test_label = tf.scalar_mul(NORM_VALUE, test_label)


    #Create batches of data
    print("Create batches of data")
    validation_image_batch, validation_label_batch, validation_kdt_batch = tf.train.shuffle_batch([validation_image, validation_label, validation_kdt.dequeue()], batch_size=BATCH_SIZE, capacity=CAPACITY,min_after_dequeue=MIN_AFTER_DEQUEUE, num_threads=NUM_THREADS)
    train_image_batch, train_label_batch, train_kdt_batch = tf.train.shuffle_batch([train_image, train_label, train_kdt.dequeue()], batch_size=BATCH_SIZE, capacity=CAPACITY,min_after_dequeue=MIN_AFTER_DEQUEUE, num_threads=NUM_THREADS)
    test_image_batch, test_label_batch, test_kdt_batch = tf.train.shuffle_batch([test_image, test_label, test_kdt.dequeue()], batch_size=BATCH_SIZE, capacity=CAPACITY,min_after_dequeue=MIN_AFTER_DEQUEUE, num_threads=NUM_THREADS)

    return train_image_batch, train_label_batch, train_kdt_batch

def get_loader(root, batch_size, scale_size, data_format, split=None, is_grayscale=False, seed=None):
    dataset_name = os.path.basename(root)
    if dataset_name in ['CelebA'] and split:
        root = os.path.join(root, 'splits', split)

    for ext in ["jpg", "png"]:
        paths = glob("{}/*.{}".format(root, ext))

        if ext == "jpg":
            tf_decode = tf.image.decode_jpeg
        elif ext == "png":
            tf_decode = tf.image.decode_png

        if len(paths) != 0:
            break

    with Image.open(paths[0]) as img:
        w, h = img.size
        shape = [h, w, 3]

    filename_queue = tf.train.string_input_producer(list(paths), shuffle=False, seed=seed)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    image = tf_decode(data, channels=3)

    if is_grayscale:
        image = tf.image.rgb_to_grayscale(image)
    image.set_shape(shape)

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size

    queue = tf.train.shuffle_batch(
        [image], batch_size=batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

    if dataset_name in ['CelebA']:
        queue = tf.image.crop_to_bounding_box(queue, 50, 25, 128, 128)
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])
    else:
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])

    if data_format == 'NCHW':
        queue = tf.transpose(queue, [0, 3, 1, 2])
    elif data_format == 'NHWC':
        pass
    else:
        raise Exception("[!] Unkown data_format: {}".format(data_format))

    return tf.to_float(queue)
