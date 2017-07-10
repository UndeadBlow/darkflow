import numpy as np

class BoundBox:
    def __init__(self, classes):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.class_num = classes
        self.probs = np.zeros((classes,))

    def containsPoint(self, x, y, raw_yolo_coords = False):
        if (raw_yolo_coords):
            me_x1 = self.x - self.w / 2
            me_x2 = self.x + self.w / 2

            me_y1 = self.y - self.h / 2
            me_y2 = self.y + self.h / 2

        else:
            me_x1 = self.x - self.w / 2
            me_x2 = self.x + self.w / 2

            me_y1 = self.y - self.h / 2
            me_y2 = self.y + self.h / 2

        return (x > me_x1) and (x < me_x2) and (y > me_y1) and (y < me_y2)

    def isMeInsideThat(self, that, raw_yolo_coords = False):
        if (raw_yolo_coords):
            me_x1 = self.x - self.w / 2
            me_x2 = self.x + self.w / 2

            me_y1 = self.y - self.h / 2
            me_y2 = self.y + self.h / 2

            he_x1 = that.x - that.w / 2
            he_x2 = that.x + that.w / 2

            he_y1 = that.y - that.h / 2
            he_y2 = that.y + that.h / 2

        else:
            me_x1 = self.x
            me_x2 = self.x + self.w

            me_y1 = self.y
            me_y2 = self.y + self.h

            he_x1 = that.x
            he_x2 = that.x + that.w

            he_y1 = that.y
            he_y2 = that.y + that.h

        return (me_x1 > he_x1) and (me_x2 < he_x2) and (me_y1 > he_y1) and (me_y2 < he_y2)

def checkRectsIntersect(this_rect, that_rect, raw_yolo_coords = False):
    if raw_yolo_coords:
        x1 = that_rect.x - that_rect.w / 2
        x2 = that_rect.x + that_rect.w / 2

        y1 = that_rect.y - that_rect.h / 2
        y2 = that_rect.y + that_rect.h / 2
    else:
        x1 = that_rect.x
        x2 = that_rect.x + that_rect.w

        y1 = that_rect.y
        y2 = that_rect.y + that_rect.h

    return (this_rect.containsPoint(x1, y1, raw_yolo_coords) or this_rect.containsPoint(x2, y1, raw_yolo_coords)\
    or this_rect.containsPoint(x1, y2, raw_yolo_coords) or this_rect.containsPoint(x2, y2, raw_yolo_coords))

def isRectsIntersect(this_rect, that_rect, raw_yolo_coords):
    return checkRectsIntersect(this_rect, that_rect, raw_yolo_coords) or checkRectsIntersect(that_rect, this_rect, raw_yolo_coords)

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w * a.h + b.w * b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b);

def prob_compare(box):
    return box.probs[box.class_num]

def prob_compare2(boxa, boxb):
    if (boxa.pi < boxb.pi):
        return 1
    elif(boxa.pi == boxb.pi):
        return 0
    else:
        return -1
