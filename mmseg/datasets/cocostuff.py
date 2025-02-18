from .builder import DATASETS
from .custom import CustomDataset
#from IPython import embed

@DATASETS.register_module()
class CocoStuff(CustomDataset):
    """Coco Stuff dataset.
    """
    nclass = 182
    CLASSES = [str(i) for i in range(nclass)]

    # random generated color
    PALETTE = [
         [167, 200, 7],
         [127, 228, 215],
         [26, 135, 248],
         [238, 73, 166],
         [91, 210, 215],
         [122, 20, 236],
         [234, 173, 35],
         [34, 98, 46],
         [115, 11, 206],
         [52, 251, 238],
         [209, 156, 236],
         [239, 10, 0],
         [26, 122, 36],
         [162, 181, 66],
         [26, 64, 22],
         [46, 226, 200],
         [89, 176, 6],
         [103, 36, 32],
         [74, 89, 159],
         [250, 215, 25],
         [57, 246, 82],
         [51, 156, 111],
         [139, 114, 219],
         [65, 208, 253],
         [33, 184, 119],
         [230, 239, 58],
         [176, 141, 158],
         [21, 29, 31],
         [135, 133, 163],
         [152, 241, 248],
         [253, 54, 7],
         [231, 86, 229],
         [179, 220, 46],
         [155, 217, 185],
         [58, 251, 190],
         [40, 201, 63],
         [236, 52, 220],
         [71, 203, 170],
         [96, 56, 41],
         [252, 231, 125],
         [255, 60, 100],
         [11, 172, 184],
         [127, 46, 248],
         [1, 105, 163],
         [191, 218, 95],
         [87, 160, 119],
         [149, 223, 79],
         [216, 180, 245],
         [58, 226, 163],
         [11, 43, 118],
         [20, 23, 100],
         [71, 222, 109],
         [124, 197, 150],
         [38, 106, 43],
         [115, 73, 156],
         [113, 110, 50],
         [94, 2, 184],
         [163, 168, 155],
         [83, 39, 145],
         [150, 169, 81],
         [134, 25, 2],
         [145, 49, 138],
         [46, 27, 209],
         [145, 187, 117],
         [197, 9, 211],
         [179, 12, 118],
         [107, 241, 133],
         [255, 176, 224],
         [49, 56, 217],
         [10, 227, 177],
         [152, 117, 25],
         [139, 76, 23],
         [53, 191, 10],
         [14, 244, 90],
         [247, 94, 189],
         [202, 160, 149],
         [24, 31, 150],
         [164, 236, 24],
         [47, 10, 204],
         [84, 187, 44],
         [17, 153, 55],
         [9, 191, 39],
         [216, 53, 216],
         [54, 13, 26],
         [241, 13, 196],
         [157, 90, 225],
         [99, 195, 27],
         [20, 186, 253],
         [175, 192, 0],
         [81, 11, 238],
         [137, 83, 196],
         [53, 186, 24],
         [231, 20, 101],
         [246, 223, 173],
         [75, 202, 249],
         [9, 188, 201],
         [216, 83, 7],
         [152, 92, 54],
         [137, 192, 79],
         [242, 169, 49],
         [99, 65, 207],
         [178, 112, 1],
         [120, 135, 40],
         [71, 220, 82],
         [180, 83, 172],
         [68, 137, 75],
         [46, 58, 15],
         [0, 80, 68],
         [175, 86, 173],
         [19, 208, 152],
         [215, 235, 142],
         [95, 30, 166],
         [246, 193, 8],
         [222, 19, 72],
         [177, 29, 183],
         [238, 61, 178],
         [246, 136, 87],
         [199, 207, 174],
         [218, 149, 231],
         [98, 179, 168],
         [23, 10, 10],
         [223, 9, 253],
         [206, 114, 95],
         [177, 242, 152],
         [115, 189, 142],
         [254, 105, 107],
         [59, 175, 153],
         [42, 114, 178],
         [50, 121, 91],
         [78, 238, 175],
         [232, 201, 123],
         [61, 39, 248],
         [76, 43, 218],
         [121, 191, 38],
         [13, 164, 242],
         [83, 70, 160],
         [109, 2, 64],
         [252, 81, 105],
         [151, 107, 83],
         [31, 95, 170],
         [7, 238, 218],
         [227, 49, 19],
         [56, 102, 49],
         [152, 241, 48],
         [110, 35, 108],
         [59, 198, 242],
         [186, 189, 39],
         [26, 157, 41],
         [183, 16, 169],
         [114, 26, 104],
         [131, 142, 127],
         [118, 85, 219],
         [203, 84, 210],
         [245, 16, 127],
         [57, 238, 110],
         [223, 225, 154],
         [143, 21, 231],
         [12, 215, 113],
         [117, 58, 3],
         [170, 201, 252],
         [60, 190, 197],
         [38, 22, 24],
         [37, 155, 237],
         [175, 41, 211],
         [188, 151, 129],
         [231, 92, 102],
         [229, 112, 245],
         [157, 182, 40],
         [1, 60, 204],
         [57, 58, 19],
         [156, 199, 180],
         [211, 47, 8],
         [153, 115, 233],
         [172, 117, 198],
         [33, 63, 208],
         [107, 80, 154],
         [217, 164, 13],
         [136, 83, 59],
         [53, 206, 6],
         [95, 127, 75],
         [110, 22, 240],
         [244, 212, 2]
    ]

    assert len(CLASSES) == len(PALETTE)

    def __init__(self, **kwargs):
        super(CocoStuff, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)