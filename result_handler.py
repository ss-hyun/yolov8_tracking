import time
import _thread

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
        

class Car(Track):
    def __init__(self, trk_id, trk_tlbr, trk_cls, trk_conf, state):
        super().__init__(trk_id, trk_tlbr, trk_cls, trk_conf)
        self.state = state
        self.frames = 0
        
    def update_state(self, state):
        if self.state == state:
            self.frames += 1
            return False
        self.state = state
        self.frames = 0
        return True
    
    def print_info(self):
        print(f"ID: {self.id}\tState: {self.state}\tTLBR: {self.tlbr}\tClass: {self.cls}\tConfidence: {self.conf}")
        
    
class ResultHandler():

    def __init__(self, size, fps):
        # size - input image size (height, width)
        self.img_height = size[0]
        self.img_width = size[1]
        
        self.fps = fps
        
        #  [x1, y1, x2, y2]
        self.ROI = [ int(0.2 * self.img_width)
                    , int(0 * self.img_height)
                    , int(0.45 * self.img_width)
                    , int(1 * self.img_height)]
        
        # { interect ratio threshold, time threshold(sec) }
        self.enter_threshold = { "inter" : 60, "time" : self.sec_per_frame(1) }
        self.parking_threshold = { "inter" : 95, "time" : self.sec_per_frame(15) }
        self.leaving_threshold = { "inter" : 80, "time" : self.sec_per_frame(1) }
        self.left_threshold = { "inter" : 10, "time" : self.sec_per_frame(1) }
        
        self.tracks = {} # {track_vid: Car()}
        self.check_event = {
            # return next state
            "" : self.entering,
            "entering": self.parking,
            "parking_in_progress": self.parking,
            "parking": self.leaving,
            "leaving": self.left,
            "left": self.entering
        }
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
        print(f"Parking: {len(self.parking)}\tEntering: {len(self.entering)}\tCandidate: {len(self.candidate)}")
        time.sleep(5)
        #_thread.start_new_thread(self.print_state,())


    def sec_per_frame(self, sec):
        return sec * self.fps


    def update(self, track):
        return track.update_state(self.check_event[track.state](track))


    def entering(self, track):
        if self.ratio_intersect_in_ROI(track.tlbr) > self.enter_threshold["inter"] \
        and track.frames > self.enter_threshold["time"]:
            return "entering"
        return ""
    
    
    def parking(self, track):
        if self.ratio_intersect_in_ROI(track.tlbr) > self.parking_threshold["inter"] \
        and track.frames > self.parking_threshold["time"]:
            return "parking"
        elif self.ratio_intersect_in_ROI(track.tlbr) < self.left_threshold["inter"] \
        and track.frames > self.left_threshold["time"]:
            return "left"
        else:
            return "parking_in_progress"

    
    def leaving(self, track):
        if self.ratio_intersect_in_ROI(track.tlbr) < self.leaving_threshold["inter"] \
        and track.frames > self.leaving_threshold["time"]:
                return "leaving"
        else:
            return "parking"

    def left(self,track):
        if self.ratio_intersect_in_ROI(track.tlbr) < self.left_threshold["inter"] \
        and track.frames > self.left_threshold["time"]:
            return "left"
        else:
            return "leaving"

    # 주차 event 발생 확인
    def on_event(self, track_vid, track_tlbr, trk_cls, trk_conf):
        if track_vid not in self.tracks:
            self.tracks[track_vid] = Car(track_vid, track_tlbr, trk_cls, trk_conf, "")
        else:
            self.tracks[track_vid].update(track_tlbr, trk_cls, trk_conf)
        
        return self.update(self.tracks[track_vid]), self.ko_event_label[self.tracks[track_vid].state]
