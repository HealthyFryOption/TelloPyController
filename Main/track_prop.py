'''
    track_prop contains TrackP class whose object is responsible to constantly remember the previous error values of required PID Control variables.
'''

class TrackP():
    
    def __init__(self):
        self.h_pError = 0
        self.v_pError = 0
        self.b_pError = 0