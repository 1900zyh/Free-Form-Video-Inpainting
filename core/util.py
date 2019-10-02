import os
import sys
import io
import cv2
import time
import argparse
import shutil
import random 
import zipfile 
from glob import glob
import numpy as np
from PIL import Image, ImageOps, ImageFilter

import torch
import torchvision
import torch.distributed as dist
from PIL import Image, ImageOps, ImageDraw,ImageFilter



def get_video_masks_by_moving_random_stroke(
    video_len, imageWidth=424, imageHeight=240, nStroke=3,
    nVertexBound=[5, 20], maxHeadSpeed=15, maxHeadAcceleration=(15, 3.14),
    brushWidthBound=(30, 50), boarderGap=50, nMovePointRatio=0.5, maxPiontMove=10,
    maxLineAcceleration=(5,0.5), maxInitSpeed=10
):
    '''
    Get video masks by random strokes which move randomly between each
    frame, including the whole stroke and its control points
    Parameters
    ----------
        imageWidth: Image width
        imageHeight: Image height
        nStroke: Number of drawed lines
        nVertexBound: Lower/upper bound of number of control points for each line
        maxHeadSpeed: Max head speed when creating control points
        maxHeadAcceleration: Max acceleration applying on the current head point (
            a head point and its velosity decides the next point)
        brushWidthBound (min, max): Bound of width for each stroke
        boarderGap: The minimum gap between image boarder and drawed lines
        nMovePointRatio: The ratio of control points to move for next frames
        maxPiontMove: The magnitude of movement for control points for next frames
        maxLineAcceleration: The magnitude of acceleration for the whole line
    Examples
    ----------
        object_like_setting = {
            "nVertexBound": [5, 20],
            "maxHeadSpeed": 15,
            "maxHeadAcceleration": (15, 3.14),
            "brushWidthBound": (30, 50),
            "nMovePointRatio": 0.5,
            "maxPiontMove": 10,
            "maxLineAcceleration": (5, 0.5),
            "boarderGap": 20,
            "maxInitSpeed": 10,
        }
        rand_curve_setting = {
            "nVertexBound": [10, 30],
            "maxHeadSpeed": 20,
            "maxHeadAcceleration": (15, 0.5),
            "brushWidthBound": (3, 10),
            "nMovePointRatio": 0.5,
            "maxPiontMove": 3,
            "maxLineAcceleration": (5, 0.5),
            "boarderGap": 20,
            "maxInitSpeed": 6
        }
        get_video_masks_by_moving_random_stroke(video_len=5, nStroke=3, **object_like_setting)
    '''
    assert(video_len >= 1)

    # Initilize a set of control points to draw the first mask
    mask = Image.new(mode='1', size=(imageWidth, imageHeight), color=0)
    control_points_set = []
    for _ in range(nStroke):
      brushWidth = np.random.randint(brushWidthBound[0], brushWidthBound[1])
      Xs, Ys, velocity = get_random_stroke_control_points(
        imageWidth=imageWidth, imageHeight=imageHeight,
        nVertexBound=nVertexBound, maxHeadSpeed=maxHeadSpeed,
        maxHeadAcceleration=maxHeadAcceleration, boarderGap=boarderGap,
        maxInitSpeed=maxInitSpeed)
      control_points_set.append((Xs, Ys, velocity, brushWidth))
      draw_mask_by_control_points(mask, Xs, Ys, brushWidth, fill=255)

    # Generate the following masks by randomly move strokes and their control points
    masks = [mask]
    for _ in range(video_len - 1):
      mask = Image.new(mode='1', size=(imageWidth, imageHeight), color=0)
      for j in range(len(control_points_set)):
        Xs, Ys, velocity, brushWidth = control_points_set[j]
        new_Xs, new_Ys, velocity = random_move_control_points(
          Xs, Ys, imageWidth, imageHeight, velocity, nMovePointRatio, maxPiontMove,
          maxLineAcceleration, boarderGap, maxInitSpeed)
        control_points_set[j] = (new_Xs, new_Ys, velocity, brushWidth)
      for Xs, Ys, velocity, brushWidth in control_points_set:
        draw_mask_by_control_points(mask, Xs, Ys, brushWidth, fill=255)
      masks.append(mask)
    return masks


def random_accelerate(velocity, maxAcceleration, dist='uniform'):
    speed, angle = velocity
    d_speed, d_angle = maxAcceleration

    if dist == 'uniform':
        speed += np.random.uniform(-d_speed, d_speed)
        angle += np.random.uniform(-d_angle, d_angle)
    elif dist == 'guassian':
        speed += np.random.normal(0, d_speed / 2)
        angle += np.random.normal(0, d_angle / 2)
    else:
        raise NotImplementedError(f'Distribution type {dist} is not supported.')

    return (speed, angle)


def random_move_control_points(Xs, Ys, imageWidth, imageHeight, lineVelocity, nMovePointRatio, maxPiontMove, maxLineAcceleration, boarderGap=15, maxInitSpeed=10):
    new_Xs = Xs.copy()
    new_Ys = Ys.copy()

    # move the whole line and accelerate
    speed, angle = lineVelocity
    new_velocity = False
    new_Xs += int(speed * np.cos(angle))
    new_Ys += int(speed * np.sin(angle))
    lineVelocity = random_accelerate(lineVelocity, maxLineAcceleration, dist='guassian')

    # choose points to move
    chosen = np.arange(len(Xs))
    np.random.shuffle(chosen)
    chosen = chosen[:int(len(Xs) * nMovePointRatio)]
    for i in chosen:
        new_Xs[i] += np.random.randint(-maxPiontMove, maxPiontMove)
        new_Ys[i] += np.random.randint(-maxPiontMove, maxPiontMove)
        if not new_velocity and ((new_Xs[i] > imageWidth) or (new_Xs[i] < 0) or (new_Ys[i]>imageHeight) or (new_Ys[i]<0)):
          new_velocity = True
        new_Xs[i] = np.clip(new_Xs[i], boarderGap, imageWidth - boarderGap)
        new_Ys[i] = np.clip(new_Ys[i], boarderGap, imageHeight - boarderGap)
    if new_velocity:
      lineVelocity = get_random_velocity(maxInitSpeed, dist='guassian')
    return new_Xs, new_Ys, lineVelocity


def get_random_stroke_control_points(
    imageWidth, imageHeight,
    nVertexBound=(10, 30), maxHeadSpeed=10, maxHeadAcceleration=(5, 0.5), boarderGap=20,
    maxInitSpeed=10
):
    '''
    Implementation the free-form training masks generating algorithm
    proposed by JIAHUI YU et al. in "Free-Form Image Inpainting with Gated Convolution"
    '''
    startX = np.random.randint(imageWidth)
    startY = np.random.randint(imageHeight)
    Xs = [startX]
    Ys = [startY]

    numVertex = np.random.randint(nVertexBound[0], nVertexBound[1])

    angle = np.random.uniform(0, 2 * np.pi)
    speed = np.random.uniform(0, maxHeadSpeed)

    for i in range(numVertex):
        speed, angle = random_accelerate((speed, angle), maxHeadAcceleration)
        speed = np.clip(speed, 0, maxHeadSpeed)

        nextX = startX + speed * np.sin(angle)
        nextY = startY + speed * np.cos(angle)

        if boarderGap is not None:
            nextX = np.clip(nextX, boarderGap, imageWidth - boarderGap)
            nextY = np.clip(nextY, boarderGap, imageHeight - boarderGap)

        startX, startY = nextX, nextY
        Xs.append(nextX)
        Ys.append(nextY)

    velocity = get_random_velocity(maxInitSpeed, dist='guassian')

    return np.array(Xs), np.array(Ys), velocity


def get_random_velocity(max_speed, dist='uniform'):
    if dist == 'uniform':
        speed = np.random.uniform(max_speed)
    elif dist == 'guassian':
        speed = np.abs(np.random.normal(0, max_speed / 2))
    else:
        raise NotImplementedError(f'Distribution type {dist} is not supported.')

    angle = np.random.uniform(0, 2 * np.pi)
    return (speed, angle)


def draw_mask_by_control_points(mask, Xs, Ys, brushWidth, fill=255):
    radius = brushWidth // 2 - 1
    for i in range(1, len(Xs)):
        draw = ImageDraw.Draw(mask)
        startX, startY = Xs[i - 1], Ys[i - 1]
        nextX, nextY = Xs[i], Ys[i]
        draw.line((startX, startY) + (nextX, nextY), fill=fill, width=brushWidth)
    for x, y in zip(Xs, Ys):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
    return mask


# modified from https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/generate_data.py
def get_random_walk_mask(imageWidth=320, imageHeight=180, length=None):
    action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    canvas = np.zeros((imageHeight, imageWidth)).astype("i")
    if length is None:
        length = imageWidth * imageHeight
    x = random.randint(0, imageHeight - 1)
    y = random.randint(0, imageWidth - 1)
    x_list = []
    y_list = []
    for i in range(length):
        r = random.randint(0, len(action_list) - 1)
        x = np.clip(x + action_list[r][0], a_min=0, a_max=imageHeight - 1)
        y = np.clip(y + action_list[r][1], a_min=0, a_max=imageWidth - 1)
        x_list.append(x)
        y_list.append(y)
    canvas[np.array(x_list), np.array(y_list)] = 1
    return Image.fromarray(canvas * 255).convert('1')


def get_masked_ratio(mask):
    """
    Calculate the masked ratio.
    mask: Expected a binary PIL image, where 0 and 1 represent
          masked(invalid) and valid pixel values.
    """
    hist = mask.histogram()
    return hist[0] / np.prod(mask.size)


# set random seed 
def set_seed(seed, base=0, is_set=True):
  seed += base
  assert seed >=0, '{} >= {}'.format(seed, 0)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

class ZipReader(object):
  file_dict = dict()
  def __init__(self):
    super(ZipReader, self).__init__()

  @staticmethod
  def build_file_dict(path):
    file_dict = ZipReader.file_dict
    if path in file_dict:
      return file_dict[path]
    else:
      file_handle = zipfile.ZipFile(path, 'r')
      file_dict[path] = file_handle
      return file_dict[path]

  @staticmethod
  def imread(path, image_name):
    zfile = ZipReader.build_file_dict(path)
    data = zfile.read(image_name)
    im = Image.open(io.BytesIO(data))
    return im


def stitch_images(inputs, *outputs, img_per_row=2):
  gap = 5
  columns = len(outputs) + 1
  height, width = inputs[0][:, :, 0].shape

  if len(inputs) <= 1:
    img = Image.new('RGB', (width, height * columns))
    images = [inputs, *outputs]
    for cat in range(len(images)):
      im = np.array((images[cat][0]).cpu()).astype(np.uint8).squeeze()
      im = Image.fromarray(im)
      img.paste(im, (0, cat*height))
  else:
    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]
    for ix in range(len(inputs)):
      xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
      yoffset = int(ix / img_per_row) * height
      for cat in range(len(images)):
        im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
        im = Image.fromarray(im)
        img.paste(im, (xoffset + cat * width, yoffset))
  return img


def imsave(img, path):
  im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
  im.save(path)


# set parameter to gpu or cpu
def set_device(args):
  if torch.cuda.is_available():
    if isinstance(args, list):
      return (item.cuda() for item in args)
    else:
      return args.cuda()
  return args

def postprocess(img):
  img = img * 255.0
  img = img.permute(0, 2, 3, 1)
  img = img.int().cpu().numpy().astype(np.uint8)
  return img


def read_frame_from_videos(vname):
  frames = []
  vidcap = cv2.VideoCapture(vname)
  success, image = vidcap.read()
  count = 0
  while success:
    frames.append(image)
    success,image = vidcap.read()
    count += 1
  return frames

class Progbar(object):
  """Displays a progress bar.

  Arguments:
    target: Total number of steps expected, None if unknown.
    width: Progress bar width on screen.
    verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
    stateful_metrics: Iterable of string names of metrics that
      should *not* be averaged over time. Metrics in this list
      will be displayed as-is. All others will be averaged
      by the progbar before display.
    interval: Minimum visual progress update interval (in seconds).
  """

  def __init__(self, target, width=25, verbose=1, interval=0.05, stateful_metrics=None):
    self.target = target
    self.width = width
    self.verbose = verbose
    self.interval = interval
    if stateful_metrics:
      self.stateful_metrics = set(stateful_metrics)
    else:
      self.stateful_metrics = set()

    self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
      sys.stdout.isatty()) or 'ipykernel' in sys.modules or 'posix' in sys.modules)
    self._total_width = 0
    self._seen_so_far = 0
    # We use a dict + list to avoid garbage collection
    # issues found in OrderedDict
    self._values = {}
    self._values_order = []
    self._start = time.time()
    self._last_update = 0

  def update(self, current, values=None):
    """Updates the progress bar.
    Arguments:
      current: Index of current step.
      values: List of tuples:
        `(name, value_for_last_step)`.
        If `name` is in `stateful_metrics`,
        `value_for_last_step` will be displayed as-is.
        Else, an average of the metric over time will be displayed.
    """
    values = values or []
    for k, v in values:
      if k not in self._values_order:
        self._values_order.append(k)
      if k not in self.stateful_metrics:
        if k not in self._values:
          self._values[k] = [v * (current - self._seen_so_far), current - self._seen_so_far]
        else:
          self._values[k][0] += v * (current - self._seen_so_far)
          self._values[k][1] += (current - self._seen_so_far)
      else:
        self._values[k] = v
    self._seen_so_far = current

    now = time.time()
    info = ' - %.0fs' % (now - self._start)
    if self.verbose == 1:
      if (now - self._last_update < self.interval and 
        self.target is not None and current < self.target):
          return

      prev_total_width = self._total_width
      if self._dynamic_display:
        sys.stdout.write('\b' * prev_total_width)
        sys.stdout.write('\r')
      else:
        sys.stdout.write('\n')

      if self.target is not None:
        numdigits = int(np.floor(np.log10(self.target))) + 1
        barstr = '%%%dd/%d [' % (numdigits, self.target)
        bar = barstr % current
        prog = float(current) / self.target
        prog_width = int(self.width * prog)
        if prog_width > 0:
          bar += ('=' * (prog_width - 1))
          if current < self.target:
            bar += '>'
          else:
            bar += '='
        bar += ('.' * (self.width - prog_width))
        bar += ']'
      else:
        bar = '%7d/Unknown' % current
      self._total_width = len(bar)
      sys.stdout.write(bar)
      if current:
        time_per_unit = (now - self._start) / current
      else:
        time_per_unit = 0
      if self.target is not None and current < self.target:
        eta = time_per_unit * (self.target - current)
        if eta > 3600:
          eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
        elif eta > 60:
          eta_format = '%d:%02d' % (eta // 60, eta % 60)
        else:
          eta_format = '%ds' % eta
        info = ' - ETA: %s' % eta_format
      else:
        if time_per_unit >= 1:
          info += ' %.0fs/step' % time_per_unit
        elif time_per_unit >= 1e-3:
          info += ' %.0fms/step' % (time_per_unit * 1e3)
        else:
          info += ' %.0fus/step' % (time_per_unit * 1e6)

      for k in self._values_order:
        info += ' - %s:' % k
        if isinstance(self._values[k], list):
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if abs(avg) > 1e-3:
            info += ' %.4f' % avg
          else:
            info += ' %.4e' % avg
        else:
          info += ' %s' % self._values[k]

      self._total_width += len(info)
      if prev_total_width > self._total_width:
        info += (' ' * (prev_total_width - self._total_width))
      if self.target is not None and current >= self.target:
        info += '\n'
      sys.stdout.write(info)
      sys.stdout.flush()
    elif self.verbose == 2:
      if self.target is None or current >= self.target:
        for k in self._values_order:
          info += ' - %s:' % k
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if avg > 1e-3:
            info += ' %.4f' % avg
          else:
            info += ' %.4e' % avg
        info += '\n'
        sys.stdout.write(info)
        sys.stdout.flush()
    self._last_update = now

  def add(self, n, values=None):
      self.update(self._seen_so_far + n, values)


