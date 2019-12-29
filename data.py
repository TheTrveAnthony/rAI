import torch
import torchvision
from PIL import Image as im
import subprocess




def split_mask(mask):

	""" this function will split the masks into a H*W*2 array, each mask shall
		have its own channel.
		In gray scale, the mask values are 50 and 94 """
	
	split = torch.zeros((mask.shape[0], mask.shape[1], 2))

	for i, line in enumerate(mask):
		for j, pix in enumerate(line):

			##### There are also values close to 50 or 94 like 47 or 92 so we gonna catch em as well

			val_50 = pix/50
			val_94 = pix/94

			if val_50 > 45 and val_50 <= 50:

				split[i, j, 0] = 1

			if val_94 > 89 and val_50 <= 94:

				split[i, j, 1] = 1

	return split


class FormDataset(Dataset):    
	""" custom class to make our dataset."""

    def __init__(self, d, transforms):
        
        self.dir = d 			#### The directory we use (train or validate)  
        self.t = transforms	#### This will be a dictionnary containing one transform for frames and one for masks : t['frames'] and t['masks']
        
    
    def __getitem__(self, idx):
        
    	### get file name for the frame and mask
    	### it will be self.dir/frames/-0i.png or self.dir/masks/-0i.png
    	
    	i = str(idx)
    	fname = "-0"

    	if len(i) == 1:		# if i = 5 (for ex)
    		fname += "000" 

    	elif len(i) == 2:	# if i = 41 (for ex)
    		fname += "00" 

    	elif len(i) == 3:	# if i = 455 (for ex)
    		fname += "0" 

    	fname += i + ".png"	# when i > 999 we don't need to add zeros

    	fr_name = self.dir + "/frames/" + fname
    	m_name = self.dir + "/masks/" + fname 

    	#### Now we can load the images and apply the transforms to them

    	frame = im.open(fr_name).convert("L")	### read in grayscale mode
    	frame = self.t['frames'](frame)			### apply transform

    	mask = im.open(m_name).convert("L")
    	mask = self.t['masks'](mask)
    	s_mask = split_mask(mask)		### split different masks into their own channel


        return frame, mask
    
    def __len__(self):

    	ux = "ls -l" + self.d + "/frames/*.png | wc -l"	### assuming that we use png files
    	x = subprocess.check_output(ux, shell = True)

        return int(x)