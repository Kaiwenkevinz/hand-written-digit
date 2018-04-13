# revised version of manual_labelling.py
# written in python2
# Kaiwen Zhang

#!/usr/bin/env python
import cv2
import argparse
import cPickle as pickle
import sys, os, numpy as np

# disabled "cell" mode
# disabled draw digits on frame
# disabled frame changing

def parse_user_input():
    """
    parse the command line input argument
    """
    description = 'Manual labeling script.'
    parser = argparse.ArgumentParser(description=description,
                                     epilog='')

    parser.add_argument('-f','--file',
                        dest='labeling_file_path',
                        help='User input argument for the labeling file path.',
                        required=True)

    args = parser.parse_args(sys.argv[1:])

    return args

def try_open_video_file(labeling_file_path):
    """
    Try to open a video file using OpenCV
    """
    video_source = cv2.VideoCapture(labeling_file_path)
    if video_source.isOpened() == True :
        return video_source
    else:
        return None

class RuneLabel:
    def __init__(self, name, frame_num, frame):
        # the info for label identify
        self.video_name = name
        self.frame_num = frame_num
        self.raw_frame = frame
        # the info for label
        self.is_rune_in_frame = False
        self.grid_boding_box = []
        self.handwritten_number_boding_boxes = []
        self.handwritten_number_positions = [] # the center point of boding box
        self.handwritten_number_labels = []

    def __str__(self):
        # the info for label identify
        print self.video_name
        print self.frame_num

        # the info for label
        print self.is_rune_in_frame
        print self.grid_boding_box
        print self.handwritten_number_boding_boxes
        print self.handwritten_number_positions # the center point of boding box
        print self.handwritten_number_labels
        return ""

    def add_label(self, value, step=None):
        self.is_rune_in_frame = True
        if type(value)==type(list()) and value[0][0]>value[1][0]:
            value[0], value[1] = value[1], value[0]
        if step==None:
            step = self.get_step()
        if step==0:
            self.grid_boding_box = value
        elif step%2==0:
            self.handwritten_number_boding_boxes.append(value)
        else:
            self.handwritten_number_labels.append(value)

    def delet_label(self, step=None):
        if step==None:
            step = self.get_step()
        if step==0:
            print "[Error] Invalid command. Empty label noting to delete"
            pass # Empty label
        elif step==2:
            self.grid_boding_box = []
        elif step%2==1:
            self.handwritten_number_boding_boxes.pop()
        else:
            self.handwritten_number_labels.pop()

        if self.get_step() == 0:
            self.is_rune_in_frame = False

    def get_step(self):
        step = len(self.grid_boding_box)
        step += len(self.handwritten_number_boding_boxes)
        step += len(self.handwritten_number_labels)
        return step

    def draw_label(self, frame):
        drawed_frame = frame.copy()
        if self.is_rune_in_frame == False:
            return drawed_frame

        # drawing bounding box
        if len(self.grid_boding_box)!=0:
            points = self.grid_boding_box
            cv2.rectangle(drawed_frame,
                          (points[0][0],points[0][1]),
                          (points[1][0],points[1][1]),
                          self.get_draw_color("grid"),1)

        # drawing cells and handwritten number labels
        for cell in range(len(self.handwritten_number_boding_boxes)):
            points = self.handwritten_number_boding_boxes[cell]
            cv2.rectangle(drawed_frame,
                          (points[0][0],points[0][1]),
                          (points[1][0],points[1][1]),
                          self.get_draw_color("cell"),1)

        for cell in range(len(self.handwritten_number_labels)):
            points = self.handwritten_number_boding_boxes[cell]
            cv2.putText(drawed_frame,
                        self.handwritten_number_labels[cell],
                        (points[0][0],points[0][1]),
                        cv2.FONT_HERSHEY_PLAIN, 2,
                        self.get_draw_color("cell"),
                        2)

        return drawed_frame

    def get_draw_color(self, label=None):
        if label=="grid":
            color = (0,255,0)
        else:
            color = (255,0,255)
        return color


class FrameLabeler:
    def __init__(self, v_name, v_source):
        self.__video_name__ = v_name
        self.__video_source__ = v_source
        self.is_EOF = False
        self.output_file_name = v_name+".rune_label"
        self.label = []
        self.frame_num = 0

        self.drawing = False
        self.ix, self.iy = None, None
        self.ex, self.ey = None, None
        self.current_frame = None
        self.last_frame = None
        self.drawed_frame = None
        self.mode = "grid"

    def print_instruction(self):
        print "User manual:"
        print "r - reset label"
        print "q - Save label and exit"

    def output_coordinates(self):
        coordinates = [self.ix, self.iy, self.ex, self.ey]

        filename = "coordinates.txt"
        if os.path.exists(filename):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not
        file_content = open(filename,append_write)
        file_content.write(self.__video_name__ +': '+ str(coordinates) + '\n')
        file_content.close()
        print "Coordinates are appended to " + filename
        # print coordinates

    def reset_current_frame_variable(self):
        self.drawing = False
        self.ix, self.iy = None, None
        self.ex, self.ey = None, None
        self.drawed_frame = self.current_frame.copy()
        self.change_mode(mode="grid")

    def reset_label(self, index):
        self.reset_current_frame_variable()
        self.label[index] = RuneLabel(self.__video_name__,
                                      self.frame_num,
                                      self.current_frame.copy())
        cv2.imshow(self.__video_name__, self.current_frame)

    def read_next_frame(self):
        # print self.__video_source__.get(1)
        if self.frame_num==len(self.label):
            ret, frame = self.__video_source__.read()
        else:
            self.frame_num += 1
            # self.reset_current_frame_variable()
            self.load_labeled_frame(self.frame_num-1)
            # print "labeling %s frame #%d" \
                # %(self.__video_name__,self.frame_num)
                # %(self.__video_name__,len(self.label))
            cv2.imshow(self.__video_name__, self.drawed_frame)
            return
        # print "Frame NO. %d" %self.__video_source__.get(1)
        if ret==False:
            self.is_EOF = True
        else:
            self.frame_num += 1
            self.last_frame = self.current_frame
            self.current_frame = frame
            self.label.append(
                RuneLabel(self.__video_name__,
                          self.frame_num,
                          self.current_frame.copy()))
            self.reset_current_frame_variable()
            print "labeling %s frame #%d" \
                %(self.__video_name__,self.frame_num)
            cv2.imshow(self.__video_name__, self.current_frame)
        return frame

    def read_last_frame(self):
        self.frame_num -= 1
        if self.frame_num<1:
            self.frame_num = max(self.frame_num, 1)
            print "[Error] Invalid command. Reach the first frame in file."
        else:
            self.reset_current_frame_variable()
            self.load_labeled_frame(self.frame_num-1)
            print "labeling %s frame #%d" \
                %(self.__video_name__,self.frame_num)
            cv2.imshow(self.__video_name__, self.drawed_frame)
        return

    def load_labeled_frame(self, index):
        self.current_frame = self.label[index].raw_frame

        # load label from memory
        self.drawed_frame = self.label[index].draw_label(self.current_frame)

        # auto reset mode
        if len(self.label[index].grid_boding_box)<2:
            self.change_mode(mode="grid")
        else:
            self.change_mode(mode="cell")

    def start_labeling(self):
        self.load_labeled_frames_from_file()

        # adjust the next frame be read from source
        for i in range(len(self.label)):
            self.__video_source__.grab()

        self.read_next_frame()
        cv2.setMouseCallback(self.__video_name__,
                                self.mouse_events_callback)

        while (True):
            if self.is_EOF:
                break
            self.keyboard_callback(cv2.waitKey(0) & 0xFF)()

        # save to file when labeling done
        self.save_labeled_frames_to_file()

    def mouse_events_callback(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            if ((self.label[self.frame_num-1].get_step() % 2==0) or\
            self.mode=="grid"):
                self.ix, self.iy = x,y
                self.ex, self.ey = None,None
            if self.mode=="grid":
                self.label[self.frame_num-1].delet_label(step=2)
                # print self.label[self.frame_num-1]
                self.load_labeled_frame(self.frame_num-1)
                cv2.imshow(self.__video_name__, self.drawed_frame)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True and\
                    ((self.label[self.frame_num-1].get_step() % 2==0) or\
                    self.mode=="grid"):
                drawed_frame = self.drawed_frame.copy()
                cv2.rectangle(drawed_frame,
                              (self.ix,self.iy),
                              (x,y),
                              self.label[self.frame_num-1].get_draw_color(self.mode),
                              1)
                cv2.imshow(self.__video_name__,drawed_frame)
                self.ex, self.ey = x,y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.ex==None: # single click without moving
                # print "Single Click Preformed."
                return
            if self.label[self.frame_num-1].get_step() % 2==0:
                cv2.rectangle(self.drawed_frame,
                            (self.ix,self.iy),
                            (self.ex,self.ey),
                            self.label[self.frame_num-1].get_draw_color(self.mode),
                            1)

                if self.mode=="grid":
                    step = 0
                    self.change_mode()
                else:
                    step = None

                self.label[self.frame_num-1].add_label(
                                [(self.ix,self.iy),(self.ex,self.ey)],
                                step)
                # print self.label[self.frame_num-1]


    def keyboard_callback(self, key):
        def ukn_key_handle():
            pass
        command = lambda: ukn_key_handle()
        if key == 27:
            print "[Exit Interupt]",len(self.label),"frames labeled."
            self.save_labeled_frames_to_file()
            command = lambda: exit()
        elif key == 127:
            self.label[self.frame_num-1].delet_label()
            self.load_labeled_frame(self.frame_num-1)
            command = lambda: cv2.imshow(self.__video_name__, self.drawed_frame)
        elif key == ord("r"):
            command = lambda: self.reset_label(self.frame_num-1)
        elif key == ord("q"):
            print "Exited."
            self.output_coordinates()
            sys.exit()
        else:
            if key == ord("c"):
                print "mode changing is removed from this version"
            elif (key >= ord("0") and key <= ord("9")):
                print "digit-on-screen draing is removed from this version"
            elif key == ord("n") or key == ord("l"):
                print "frame changing is removed from this version"
            else:
                print "Invalid key input, ord(key) ==", key
        return command

    def change_mode(self, mode=None):
        if mode==None:
            if self.mode=="grid":
                self.mode = "cell"
            else:
                self.mode = "grid"
        else:
            self.mode = mode

    def save_labeled_frames_to_file(self):
        with open(self.output_file_name,"wb") as output:
            print "[Prompt] Saving labels ..."
            pickle.dump(self.label, output, pickle.HIGHEST_PROTOCOL)
            print "[Prompt] Saved labels to file "+self.output_file_name
            output.close()

    def load_labeled_frames_from_file(self):
        try:
            with open(self.output_file_name,"rb") as output:
                print "[Prompt] Loading labels ..."
                self.label = pickle.load(output)
                print "[Prompt] Loaded labels from file "+self.output_file_name
                output.close()
        except IOError, err:
            if err.errno==2:
                pass # Not labeled file found

if __name__ == "__main__":
    """
    Main function for testing
    """
    # parse user input
    args = parse_user_input()

    # read the file that will be labeled
    labeling_file_dir = os.path.dirname(os.path.abspath(__file__))
    labeling_file_path = os.path.join(labeling_file_dir, args.labeling_file_path)

    video_source = try_open_video_file(labeling_file_path)
    assert(video_source!=None)

    labeler = FrameLabeler(args.labeling_file_path, video_source)

    # comment this line to disable showing user manual every time
    labeler.print_instruction()

    labeler.start_labeling()
    cv2.destroyAllWindows()
