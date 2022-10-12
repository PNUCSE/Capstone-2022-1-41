from yolov5_detector.thermohumandetect import ThermoCamDetection
import sys
import cv2
import datetime, os


class ThermoDetector(ThermoCamDetection):
    def __init__(self, model_path='trained_models/exp2_best.pt'):
        super().__init__(model_path)
        self.bed_region = None
        self.bed_area = None
        self.fall_area_threshold = 0.4
        self.lying_box_ratio_threshold = 1.1

    def set_bed_region(self, xmin, ymin, xmax, ymax):
        self.bed_region = [xmin, ymin, xmax, ymax]
        self.bed_area = (xmax - xmin) * (ymax - ymin)

    def show_bed_region(self, rendered_frame):
        # x1 : left-top x2 : right-bottom
        p1 = (self.bed_region[0], self.bed_region[3])
        p2 = (self.bed_region[2], self.bed_region[1])
        rendered_frame = cv2.rectangle(rendered_frame, p1, p2, (0, 255, 0), 3)
        return rendered_frame

    def intersection_area(self):
        if not self.bed_region:
            raise ValueError('need init bed region')
        # xmin, ymin, xmax, ymax
        print(self.bed_region)
        print(self.detect_table)

        if self.bed_region[0] >= self.detect_table['xmax'].values[0]:
            # 왼쪽으로 벗어난 경우
            return 0
        if self.bed_region[2] <= self.detect_table['xmin'].values[0]:
            # 오른쪽으로 벗어난 경우
            return 0
        if self.bed_region[1] >= self.detect_table['ymax'].values[0]:
            # 아래쪽으로 벗어난 경우
            return 0
        if self.bed_region[3] <= self.detect_table['ymin'].values[0]:
            # 위쪽으로 벗어난 경우
            return 0

        in_xmin = max(self.bed_region[0], self.detect_table['xmin'].values[0])
        in_xmax = min(self.bed_region[2], self.detect_table['xmax'].values[0])
        in_ymin = max(self.bed_region[1], self.detect_table['ymin'].values[0])
        in_ymax = min(self.bed_region[3], self.detect_table['ymax'].values[0])
        return (in_xmax - in_xmin) * (in_ymax - in_ymin)

    def is_lying(self):
        width = self.detect_table['xmax'].values[0] - self.detect_table['xmin'].values[0]
        height = self.detect_table['ymax'].values[0] - self.detect_table['ymin'].values[0]
        ratio = max(width, height) / min(width, height)

        if ratio >= self.lying_box_ratio_threshold:
            return True, ratio
        else:
            return False, ratio

    def is_fall(self):
        inter_area = self.intersection_area()
        if inter_area != 0 and inter_area / self.bed_area <= self.fall_area_threshold:
            return True, inter_area
        else:
            return False, inter_area

    def detect(self, rendered_frame):
        font = cv2.FONT_HERSHEY_DUPLEX
        org = (0, 25)
        color = (0, 0, 0)
        lying, ratio = self.is_lying()
        if lying:
            falling, area = self.is_fall()
            if area != 0:
                self.area_ratio = area / self.bed_area
            else:
                self.area_ratio = 0
            if falling:
                cv2.putText(rendered_frame, f'fall!! intersect : {int(area)}, ratio : {self.area_ratio}', org, font,
                            1, color)
            else:
                cv2.putText(rendered_frame, f'not fall intersect : {int(area)}, ratio : {self.area_ratio}', org,
                            font, 1, color)
        else:
            cv2.putText(rendered_frame, f'not lying!! {ratio}', org, font, 1, color)
        return rendered_frame


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
                rendered_frame = self.detect(rendered_frame)

            rendered_frame = self.show_bed_region(rendered_frame)
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


if __name__ == '__main__':
    # --path "trained_models/exp4_best.pt"
    td = None
    if len(sys.argv) >= 2 and sys.argv[1] == '--path':
        td = ThermoDetector(sys.argv[2])
    else:
        td = ThermoDetector()
    td.set_bed_region(210, 50, 450, 450)
    td.camStart("thermal detection", "/dev/video0", cv2.VideoWriter.fourcc('Y', '1', '6', ' '))
