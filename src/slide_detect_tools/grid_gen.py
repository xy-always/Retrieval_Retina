# This script is to generate the grid

# Different from old grid crop method
# We do circle detect before crop the slide
# First to get the thumbnil of the WSI
# If the slide is ibl format, which has clean background, only to generate the non-background-color grid patch
# else
# we use the houghtransform the frind the circles in the image, we know there is a big circle in the slide which is the target cells part, but the texture of the 
# cell slide may confuse the agrorithm, we find many circel instead, to get the envelope of the circles and find the minium outter circle.



class Agrorihm:
    
    def __init__(self) -> None:
        pass
        
        
    def get_regions():
        pass


class GridGen():
    
    def __init__(self, agr: Agrorihm) -> None:
        pass
        
        


