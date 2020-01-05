import cv2 as cv
import numpy as np
from PIL import Image as im
import torch
import torchvision.transforms as trs



""" functions used for inference, which is executed on google colab """


def segment_frame(frame, model):
  """ performs segmentation on a frame """
  
  #### define the transformation
  trans = trs.Compose([ 
		trs.Resize(size=(720, 1280)),
		trs.ToTensor(),
    trs.Normalize([0.5], [0.5]) 
	])

  ### Convert the frame from numpy to PIL and transform it
  
  frame = im.fromarray(frame)
  frame = trans(frame)
  frame = frame.view(1, 1, 720, 1280).cuda()

  ### throw it into the network

  with torch.no_grad():

    model.eval()
    out = model(frame)
  
  return out.cpu().numpy()


def mask_frame(frame, mask):
  """ applies mask to the current frame """

  #### first we gonna create a rgb copy of the frame to apply colored masks

  rgb_frame = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
  
  #### now apply our mask to the frame

  for i, col in enumerate(mask[0, 0]):
    for j, pix in enumerate(col):
      
      if pix == 1:

        rgb_frame[i, j] = [255, 0, 0]

      elif pix == 2:

        rgb_frame[i, j] = [0, 255, 0]
  
  return rgb_frame


def make_video(video, n_frames, fps, model, name):

  """ creates a video with masks
      n_frames : the number of frames we will use
      fps : the frame rate
      the video shall then last n_frames/fps seconds
      model : our segmentation model, a pt file
      name : our video name, without the extension """

  ### create our video writer
  name += ".avi"
  out = cv.VideoWriter(name, cv.VideoWriter_fourcc('M','J','P','G'), fps, (1280, 720))

  ### read the input video

  vid = cv.VideoCapture(video)
  frame_count = 0

  while(vid.isOpened() and frame_count < n_frames):

    ### read the current frame and convert it into grayscales
    r, f = vid.read()
    gray = cv.cvtColor(f, cv.COLOR_RGB2GRAY)

    ### create and apply mask
    msk = segment_frame(gray, model)
    mf = mask_frame(gray, msk)
    plt.imshow(msk[0, 0])
    plt.show()
    ##### write our masked frame

    out.write(mf)
    frame_count += 1
      
  ### release everything and done

  vid.release()
  out.release()
  print("video ready.")