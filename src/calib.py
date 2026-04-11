import numpy as np

# From Github https://github.com/BerensRWU/DenseMap#
class Calibration:
    def __init__(self, calib_filepath):
        calibs = self.read_calib_file(calib_filepath)

        self.P0 = calibs['P0']
        self.P0 = np.reshape(self.P0, [3,4])

        self.P1 = calibs['P1']
        self.P1 = np.reshape(self.P1, [3,4])

        self.P2 = calibs['P2']
        self.P2 = np.reshape(self.P2, [3,4])

        self.Tr_velo_to_cam = calibs['Tr_velo_to_cam']
        self.Tr_velo_to_cam = np.reshape(self.Tr_velo_to_cam, [3,4])

        self.R0_rect = calibs['R0_rect']
        self.R0_rect = np.reshape(self.R0_rect,[3,3])

    @staticmethod
    def read_calib_file(filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data


