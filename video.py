import numpy as np
import cv2
print(cv2.__version__)
if(cv2.cuda.getCudaEnabledDeviceCount()):
    print("CUDA IS ACTIVE")
else:
    print("CUDA IS INACTIVE")

class_labels = ["benur"]

# #Declare only a single color
class_colors = ["100,25,0"]
class_colors = [
    np.array(every_color.split(",")).astype("int")
    for every_color in class_colors
]
class_colors = np.array(class_colors)
class_colors = np.tile(class_colors, (1, 1))

yolo_model = cv2.dnn.readNetFromDarknet('./dnn/benur_mobile_yolov4.cfg',
                                        './dnn/benur_yolov4_mobile.weights')

layer_names = yolo_model.getLayerNames()
yolo_output_layer = [layer_names[i - 1]
                        for i in yolo_model.getUnconnectedOutLayers()]


cam = cv2.VideoCapture(0)

while True:
  ret, img_to_detect = cam.read()
  img_height = img_to_detect.shape[0]
  img_width = img_to_detect.shape[1]
  img_to_detect = cv2.resize(img_to_detect, (416, 416), interpolation=cv2.INTER_LINEAR)

  # convert to blob to pass into model
  img_blob = cv2.dnn.blobFromImage(img_to_detect,
                                  0.0024038, (416, 416),
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

        if prediction_confidence > 0.05:
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
    cv2.putText(img_to_detect, predicted_class_label,
                (start_x_pt, start_y_pt - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                box_color, 1)

  # retval, buffer = cv2.imencode('.jpg', img_to_detect)
  # response = make_response(buffer.tobytes())
  # response.headers['Content-Type'] = 'image/png'
  cv2.imshow("frame", img_to_detect)
      
  # the 'q' button is set as the
  # quitting button you may use any
  # desired button of your choice
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  
# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()