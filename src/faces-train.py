# # from cProfile import label
# # import os
# # from PIL import Image
# # import numpy as np
# # import cv2
# # import pickle

# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # image_dir = os.path.join(BASE_DIR, "images")

# # face_cascade = cv2.CascadeClassifier('src\cascades\data\haarcascade_frontalface_alt2.xml')
# # recognizer = cv2.face.LBHF

# # current_id = 0
# # label_ids ={}
# # x_train = []
# # y_labels = []

# # for root, dir, files in os.walk(image_dir):
# #     for file in files:
# #         if file.endswith("png") or file.endswith("jpg"):
# #             path = os.path.join(root, file)
# #             label = os.path.basename(root).replace(" ", "-").lower()
# #             print(label, path)
# #             if not label in label_ids:
# #                 label_ids[label] = current_id
# #                 current_id += 1
# #             id_ = label_ids[label]
# #             print(label_ids)
            
# #             # y_labels.append(label)
# #             # x_train.append(path)
# #             pil_image = Image.open(path).convert("L")
# #             image_array = np.array(pil_image, "uint8")
# #             #print(image_array)
# #             faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            
# #             for(x,y,w,h) in faces:
# #                 roi = image_array[y:y+h, x:x+w]
# #                 x_train.append(roi)
# #                 y_labels.append(id_)
# # #print(y_labels)
# # #print(x_train)

# # with open("label.pickle", 'wb') as f:
# #     pickle.dump(label_ids, f)
    
# import cv2
# import os
# import numpy as np
# from PIL import Image
# import pickle

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# image_dir = os.path.join(BASE_DIR, "images")

# face_cascade = cv2.CascadeClassifier('src\cascades\data\haarcascade_frontalface_alt.xml')
# recognizer = cv2.face.LBPHFaceRecognizer_create()

# current_id = 0
# label_ids = {}
# y_labels = []
# x_train = []

# for root, dirs, files in os.walk(image_dir):
# 	for file in files:
# 		if file.endswith("png") or file.endswith("jpg"):
# 			path = os.path.join(root, file)
# 			label = os.path.basename(root).replace(" ", "-").lower()
# 			print(label, path)
# 			if not label in label_ids:
# 				label_ids[label] = current_id
# 				current_id += 1
# 			id_ = label_ids[label]
# 			#print(label_ids)
# 			#y_labels.append(label) # some number
# 			#x_train.append(path) # verify this image, turn into a NUMPY arrray, GRAY
# 			pil_image = Image.open(path).convert("L") # grayscale
# 			size = (550, 550)
# 			final_image = pil_image.resize(size)
# 			image_array = np.array(final_image, "uint8")
# 			#print(image_array)
# 			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

# 			for (x,y,w,h) in faces:
# 				roi = image_array[y:y+h, x:x+w]
# 				x_train.append(roi)
# 				y_labels.append(id_)


# #print(y_labels)
# #print(x_train)

# with open(r"src\pickles\face-labels.pickle", 'wb') as f:
# 	pickle.dump(label_ids, f)

# recognizer.train(x_train, np.array(y_labels))
# print(np.array(y_labels))
# recognizer.save(r"src\recognizers\face-trainner.yml")
# from cProfile import label
# import os
# from PIL import Image
# import numpy as np
# import cv2
# import pickle

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# image_dir = os.path.join(BASE_DIR, "images")

# face_cascade = cv2.CascadeClassifier('src\cascades\data\haarcascade_frontalface_alt2.xml')
# recognizer = cv2.face.LBHF

# current_id = 0
# label_ids ={}
# x_train = []
# y_labels = []

# for root, dir, files in os.walk(image_dir):
#     for file in files:
#         if file.endswith("png") or file.endswith("jpg"):
#             path = os.path.join(root, file)
#             label = os.path.basename(root).replace(" ", "-").lower()
#             print(label, path)
#             if not label in label_ids:
#                 label_ids[label] = current_id
#                 current_id += 1
#             id_ = label_ids[label]
#             print(label_ids)
            
#             # y_labels.append(label)
#             # x_train.append(path)
#             pil_image = Image.open(path).convert("L")
#             image_array = np.array(pil_image, "uint8")
#             #print(image_array)
#             faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            
#             for(x,y,w,h) in faces:
#                 roi = image_array[y:y+h, x:x+w]
#                 x_train.append(roi)
#                 y_labels.append(id_)
# #print(y_labels)
# #print(x_train)

# with open("label.pickle", 'wb') as f:
#     pickle.dump(label_ids, f)
    
import cv2
import os
import numpy as np
from PIL import Image
import pickle

index = 0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('src\cascades\data\haarcascade_frontalface_alt.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(" ", "-").lower()
			print(label, path)
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]
			#print(label_ids)
			#y_labels.append(label) # some number
			#x_train.append(path) # verify this image, turn into a NUMPY arrray, GRAY
			pil_image = Image.open(path).convert("L") # grayscale
			size = (550, 550)
			final_image = pil_image.resize(size)
			image_array = np.array(final_image, "uint8")
			
			#print(image_array)
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100),flags=cv2.CASCADE_SCALE_IMAGE)
			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				cv2.imwrite(str(index)+".png", roi)
				index += 1
				y_labels.append(id_)


#print(y_labels)
#print(x_train)

with open(r"src\pickles\face-labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
print(np.array(y_labels))
recognizer.write(r"src\recognizers\face-trainner.yml")
