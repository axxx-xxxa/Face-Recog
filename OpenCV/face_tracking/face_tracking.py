import dlib
import cv2


def main():
    capture = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    tractor = dlib.correlation_tracker()
    tracking_state = False
    while True:
        ret,frame = capture.read()
        if tracking_state is False:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            dets = detector(gray,1)
            print(dets)
            if len(dets)>0:
                tractor.start_track(frame,dets[0])
                tracking_state = True
        if tracking_state is True:
            tractor.update(frame)
            position = tractor.get_position()
            print(position)
            cv2.rectangle(frame,(int(position.left()),int(position.top())),(int(position.right()),int(position.bottom())),(255,0,0),1)

        key = cv2.waitKey(1) & 0xFF
        if key ==ord('q'):
            break
        cv2.imshow("tttqwe", frame)
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()