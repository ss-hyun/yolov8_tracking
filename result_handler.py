import time
import _thread


def sec_per_frame(sec):
    return sec * ResultHandler.fps


class State():
    def __init__(self, name="", inter_min = 0, inter_max = 10,  time_thres = 2):
        self.name = name
        self.frames = 0
        self.inter_thres_min = inter_min # interect ratio threshold
        self.inter_thres_max = inter_max 
        self.frame_thres = sec_per_frame(time_thres) # time threshold
        
        
    def update(self, inter_ROI, next_states):
        for next in next_states:
            if inter_ROI >= next.inter_thres_min and inter_ROI <= next.inter_thres_max:
                next.frames += 1
                if next.frames >= next.frame_thres:
                    self.frames = 0
                    return True, next
            else:
                next.frames = 0
        self.frames += 1
        return False, self


class Track():
    def __init__(self, trk_id, trk_tlbr, trk_cls, trk_conf):
        self.id = trk_id
        #  [x1, y1, x2, y2]
        self.tlbr = trk_tlbr
        self.cls = trk_cls
        self.conf = trk_conf
        
    def update(self, trk_tlbr, trk_cls, trk_conf):
        self.tlbr = trk_tlbr
        self.cls = trk_cls
        self.conf = trk_conf
        
    def ratio_intersect_in_ROI(self, ROI):
        xA = max(self.tlbr[0], ROI[0])
        yA = max(self.tlbr[1], ROI[1])
        xB = min(self.tlbr[2], ROI[2])
        yB = min(self.tlbr[3], ROI[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        bboxArea = max(0, self.tlbr[2] - self.tlbr[0] + 1) * max(0, self.tlbr[3] - self.tlbr[1] + 1)
        return (interArea * 100) / bboxArea
        

class Car(Track):
    def __init__(self, trk_id, trk_tlbr, trk_cls, trk_conf):
        super().__init__(trk_id, trk_tlbr, trk_cls, trk_conf)
        self.cur_state = State(name="", inter_min=0, inter_max=20,  time_thres=2)
        self.next_states = {
            "" : [
                State(name="entering", inter_min=20, inter_max=100, time_thres=2)
            ],
            "entering": [
                State(name="parking_in_progress", inter_min=40, inter_max=100,  time_thres=4),
                State(name="left", inter_min=0, inter_max=20,  time_thres=2)
            ],
            "parking_in_progress": [
                State(name="parking", inter_min=80, inter_max=100,  time_thres=5),
                State(name="parking", inter_min=40, inter_max=100,  time_thres=15),
                State(name="left", inter_min=0, inter_max=20,  time_thres=2)
            ],
            "parking": [
                State(name="leaving", inter_min=0, inter_max=30,  time_thres=2),
                State(name="leaving", inter_min=0, inter_max=50,  time_thres=4)
            ],
            "leaving": [
                State(name="left", inter_min=0, inter_max=20,  time_thres=2),
                State(name="parking", inter_min=50, inter_max=100,  time_thres=2)
            ],
            "left": [
                State(name="", inter_min=0, inter_max=20,  time_thres=2)
            ]
        }
        
    def update_state(self, ROI):
        is_change, self.cur_state = self.cur_state.update(self.ratio_intersect_in_ROI(ROI), self.next_states[self.cur_state.name])
        return is_change
    
    def print_info(self):
        print(f"ID: {self.id}\tState: {self.cur_state.name}\tTLBR: {self.tlbr}\tClass: {self.cls}\tConfidence: {self.conf}")
        
    
class ResultHandler():
    fps = 30
    ROI = [0,0,0,0]

    def __init__(self, size, fps):
        # size - input image size (height, width)
        self.img_height = size[0]
        self.img_width = size[1]
        
        ResultHandler.fps = fps
        
        #  [x1, y1, x2, y2]
        ResultHandler.ROI = [ int(0.2 * self.img_width)
                    , int(0 * self.img_height)
                    , int(0.45 * self.img_width)
                    , int(1 * self.img_height)]
        
        self.tracks = {} # {track_vid: Car()}
        
        self.ko_event_label = {
            "" : "",
            "entering": "진입",
            "parking_in_progress": "주차 진행중",
            "parking": "주차",
            "leaving": "출차 진행중",
            "left": "출차"
        }
        
        #_thread.start_new_thread(self.print_state,())


    # bbox가 ROI에 겹치는 영역의 비율
    def ratio_intersect_in_ROI(self, bbox):
        ROI = self.ROI
        xA = max(bbox[0], ROI[0])
        yA = max(bbox[1], ROI[1])
        xB = min(bbox[2], ROI[2])
        yB = min(bbox[3], ROI[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        bboxArea = max(0, bbox[2] - bbox[0] + 1) * max(0, bbox[3] - bbox[1] + 1)
        return (interArea * 100) / bboxArea


    def print_state(self):
        time.sleep(5)
        #_thread.start_new_thread(self.print_state,())

    # 주차 event 발생 확인
    def on_event(self, track_vid, track_tlbr, trk_cls, trk_conf):
        if track_vid not in self.tracks:
            self.tracks[track_vid] = Car(track_vid, track_tlbr, trk_cls, trk_conf)
        else:
            self.tracks[track_vid].update(track_tlbr, trk_cls, trk_conf)
            
        return self.tracks[track_vid].update_state(ResultHandler.ROI), self.ko_event_label[self.tracks[track_vid].cur_state.name]
