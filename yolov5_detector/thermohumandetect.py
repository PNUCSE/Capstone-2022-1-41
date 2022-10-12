import pandas
import torch
import cv2
import os, datetime, sys


class ThermoCamDetection:
    def __init__(self, model_path='trained_models/exp2_best.pt', res_model='trained_models/ESPCN_x4.pb'):
        self.model = self.get_model(model_path)
        self.sr_model = self.load_superres(res_model)
        self.model.conf = 0.45
        self.model.iou = 0.50
        self.detect_table = None

        self.save_path = './record'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

    def get_model(self, path):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)
        return model

    def load_superres(self, res_model_path):
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(res_model_path)
        sr.setModel("espcn", 4)
        return sr

    def interfere(self, frame):
        interfered_frame = self.model(frame)
        # interfered_frame.print()
        interfered_frame.render()  # updates results.imgs with boxes and labels, returns nothing
        return (interfered_frame.imgs[0], interfered_frame.pandas().xyxy[0])

    def camStart(self, previewName, camID, camFourCC):
        cv2.namedWindow(previewName)
        cam = cv2.VideoCapture(camID)
        frame_number = 0
        if cam.isOpened():  # try to get the first frame
            if(camFourCC == cv2.VideoWriter.fourcc('Y','1','6',' ')):
                cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','1','6',' '))
                cam.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            else:
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
                cam.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off

            print('width: {}, height : {}, frame? : {}, ? : {}'.format(cam.get(3), cam.get(4), cam.get(5), cam.get(8)))

            rval, frame = cam.read()
        else:
            rval = False



        while rval:
            rval, frame = cam.read()
            if(camFourCC == cv2.VideoWriter.fourcc('Y','1','6',' ')):
                # In order to display image, should be scaled and normalize.
                cv2.normalize(frame, frame, 20000, 65535, cv2.NORM_MINMAX)  # Best Normalized
            frame = cv2.convertScaleAbs(frame, alpha=(255.0/65535.0))  # uint16 to uint8
            # For Big Frame
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            frame = self.sr_model.upsample(frame)
            # frame = cv2.resize(frame, (0, 0), fx=4.0, fy=4.0)
            rendered_frame, result_table = self.interfere(frame)
            if not result_table.empty:
                self.detect_table = result_table
                print(result_table)

            cv2.imshow(previewName, rendered_frame)

            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
            elif key == ord('s'):  # s to save image
                now = datetime.datetime.now().strftime("%y%m%d%H%M%S")
                next_path = os.path.join(self.save_path, f'{now}-{frame_number}.png')
                cv2.imwrite(next_path, rendered_frame)
                print(f'success to save : {next_path}')
                frame_number += 1

        if cam.isOpened():
            cam.release()
        cv2.destroyWindow(previewName)
