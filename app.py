import json
from config import env_loader
from middlewares import startup
from flask import Flask, jsonify, make_response, request, send_file
from bson import json_util
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from werkzeug.middleware.proxy_fix import ProxyFix
import numpy as np
import cv2
import time
print(cv2.__version__)
if(cv2.cuda.getCudaEnabledDeviceCount()):
    print("CUDA IS ACTIVE")
else:
    print("CUDA IS INACTIVE")

app = Flask(__name__)
mongo = startup.StartUp.db(app)

# #recommended by yolo authors, scale factor is 0.003922=1/255, width,height of blob is 320,320
# #accepted sizes are 320×320,416×416,608×608. More size means more accuracy but less speed

# # only single label
class_labels = ["benur"]

# #Declare only a single color
class_colors = ["100,25,0"]
class_colors = [
    np.array(every_color.split(",")).astype("int")
    for every_color in class_colors
]
class_colors = np.array(class_colors)
class_colors = np.tile(class_colors, (1, 1))

yolo_model = cv2.dnn.readNetFromDarknet(
    './dnn/benur_mobile_yolov4.cfg',
    './dnn/benur_yolov4_mobile.weights'
)

layer_names = yolo_model.getLayerNames()
yolo_output_layer = [layer_names[i - 1]
                        for i in yolo_model.getUnconnectedOutLayers()]

@app.get("/")
def index():
    json_docs = []
    for doc in mongo.db.devices.find():
        json_doc = json.dumps(doc, default=json_util.default)
        json_docs.append(json_doc)
    return jsonify(json_docs)


@app.post("/")
def upload_file():
    if request.method == 'POST':
        start_time = time.time()
        img_to_detect = cv2.imdecode(np.fromstring(
            request.files['file'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        img_height = img_to_detect.shape[0]
        img_width = img_to_detect.shape[1]
        
        # img_to_detect = cv2.resize(img_to_detect, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        detection = 0

        # convert to blob to pass into model
        img_blob = cv2.dnn.blobFromImage(img_to_detect,
                                         1/255, (1024, 1024),
                                         swapRB=True,
                                         crop=False)
        
        yolo_model.setInput(img_blob)
        obj_detection_layers = yolo_model.forward(yolo_output_layer)

        class_ids_list = []
        boxes_list = []
        confidences_list = []

        for object_detection_layer in obj_detection_layers:
            for object_detection in object_detection_layer:
                all_scores = object_detection[5:]
                predicted_class_id = np.argmax(all_scores)
                prediction_confidence = all_scores[predicted_class_id]

                if prediction_confidence > 0.01:
                    # get the predicted label
                    predicted_class_label = class_labels[predicted_class_id]
                    # obtain the bounding box co-oridnates for actual image from resized image size
                    bounding_box = object_detection[0:4] * np.array(
                        [img_width, img_height, img_width, img_height])
                    (box_center_x_pt, box_center_y_pt, box_width,
                     box_height) = bounding_box.astype("int")
                    start_x_pt = int(box_center_x_pt - (box_width / 2))
                    start_y_pt = int(box_center_y_pt - (box_height / 2))

                    ############## NMS Change 2 ###############
                    # save class id, start x, y, width & height, confidences in a list for nms processing
                    # make sure to pass confidence as float and width and height as integers
                    class_ids_list.append(predicted_class_id)
                    confidences_list.append(float(prediction_confidence))
                    boxes_list.append(
                        [start_x_pt, start_y_pt,
                         int(box_width),
                         int(box_height)])
                    ############## NMS Change 2 END ###########

        ############## NMS Change 3 ###############
        # Applying the NMS will return only the selected max value ids while suppressing the non maximum (weak) overlapping bounding boxes
        # Non-Maxima Suppression confidence set as 0.5 & max_suppression threhold for NMS as 0.4 (adjust and try for better perfomance)
        max_value_ids = cv2.dnn.NMSBoxes(
            boxes_list, confidences_list, 0.1, 0.1)

        # loop through the final set of detections remaining after NMS and draw bounding box and write text
        for max_valueid in max_value_ids:
            # print(max_valueid)
            max_class_id = max_valueid
            box = boxes_list[max_class_id]
            start_x_pt = box[0]
            start_y_pt = box[1]
            box_width = box[2]
            box_height = box[3]


            # #get the predicted class id and label
            predicted_class_id = class_ids_list[max_class_id]
            predicted_class_label = class_labels[predicted_class_id]
            prediction_confidence = confidences_list[max_class_id]
            # ############## NMS Change 3 END ###########

            end_x_pt = start_x_pt + box_width
            end_y_pt = start_y_pt + box_height

            # #get a random mask color from the numpy array of colors
            box_color = class_colors[predicted_class_id]

            # #convert the color numpy array as a list and apply to text and box
            box_color = [int(c) for c in box_color]

            # # print the prediction in console
            predicted_class_label = "{}: {:.2f}%".format(predicted_class_label,
                                                         prediction_confidence * 100)

            # # draw rectangle and text in the image
            cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt),
                          (end_x_pt, end_y_pt), box_color, 1)
            # cv2.putText(img_to_detect, predicted_class_label,
            #             (start_x_pt, start_y_pt - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             box_color, 1)
            detection = detection + 1

        cv2.putText(img_to_detect, "Perhitungan: " + str(len(max_value_ids)), (20, 40 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

        retval, buffer = cv2.imencode('.jpg', img_to_detect)
        response = make_response(buffer.tobytes())
        response.headers['Content-Type'] = 'image/png'
        cv2.imwrite("save.jpg", img_to_detect)
        print((time.time() - start_time))

        return response
        # return jsonify({
		# 	"count": detection
		# })

@app.post("/only-number")
def upload_file_get_number():
    if request.method == 'POST':
        img_to_detect = cv2.imdecode(np.fromstring(
            request.files['file'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img_height = img_to_detect.shape[0]
        img_width = img_to_detect.shape[1]
        detection = 0

        # convert to blob to pass into model
        img_blob = cv2.dnn.blobFromImage(img_to_detect,
                                         1/255, (1024, 1024),
                                         swapRB=True,
                                         crop=False)
        
        yolo_model.setInput(img_blob)
        obj_detection_layers = yolo_model.forward(yolo_output_layer)

        class_ids_list = []
        boxes_list = []
        confidences_list = []

        for object_detection_layer in obj_detection_layers:
            for object_detection in object_detection_layer:
                all_scores = object_detection[5:]
                predicted_class_id = np.argmax(all_scores)
                prediction_confidence = all_scores[predicted_class_id]

                if prediction_confidence > 0.01:
                    # get the predicted label
                    predicted_class_label = class_labels[predicted_class_id]
                    # obtain the bounding box co-oridnates for actual image from resized image size
                    bounding_box = object_detection[0:4] * np.array(
                        [img_width, img_height, img_width, img_height])
                    (box_center_x_pt, box_center_y_pt, box_width,
                     box_height) = bounding_box.astype("int")
                    start_x_pt = int(box_center_x_pt - (box_width / 2))
                    start_y_pt = int(box_center_y_pt - (box_height / 2))

                    ############## NMS Change 2 ###############
                    # save class id, start x, y, width & height, confidences in a list for nms processing
                    # make sure to pass confidence as float and width and height as integers
                    class_ids_list.append(predicted_class_id)
                    confidences_list.append(float(prediction_confidence))
                    boxes_list.append(
                        [start_x_pt, start_y_pt,
                         int(box_width),
                         int(box_height)])
                    ############## NMS Change 2 END ###########

        ############## NMS Change 3 ###############
        # Applying the NMS will return only the selected max value ids while suppressing the non maximum (weak) overlapping bounding boxes
        # Non-Maxima Suppression confidence set as 0.5 & max_suppression threhold for NMS as 0.4 (adjust and try for better perfomance)
        max_value_ids = cv2.dnn.NMSBoxes(
            boxes_list, confidences_list, 0.1, 0.1)

        # loop through the final set of detections remaining after NMS and draw bounding box and write text
        for max_valueid in max_value_ids:
            # print(max_valueid)
            max_class_id = max_valueid
            box = boxes_list[max_class_id]
            start_x_pt = box[0]
            start_y_pt = box[1]
            box_width = box[2]
            box_height = box[3]

            # #get the predicted class id and label
            predicted_class_id = class_ids_list[max_class_id]
            predicted_class_label = class_labels[predicted_class_id]
            prediction_confidence = confidences_list[max_class_id]
            # ############## NMS Change 3 END ###########

            end_x_pt = start_x_pt + box_width
            end_y_pt = start_y_pt + box_height

            # #get a random mask color from the numpy array of colors
            box_color = class_colors[predicted_class_id]

            # #convert the color numpy array as a list and apply to text and box
            box_color = [int(c) for c in box_color]

            # # print the prediction in console
            predicted_class_label = "{}: {:.2f}%".format(predicted_class_label,
                                                         prediction_confidence * 100)

            # # draw rectangle and text in the image
            cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt),
                          (end_x_pt, end_y_pt), box_color, 1)
            detection = detection + 1

        cv2.putText(img_to_detect, "Perhitungan: " + str(len(max_value_ids)), (20, 40 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

        retval, buffer = cv2.imencode('.jpg', img_to_detect)
        response = make_response(buffer.tobytes())
        response.headers['Content-Type'] = 'image/png'
        cv2.imwrite(str(detection) + ".jpg", img_to_detect)

        # return response
        return jsonify({
			"count": 10
		})

# @app.get("/check")
# def check():
#     if request.method == 'GET':
#         return jsonify({
# 			"count": "OKE"
# 		})