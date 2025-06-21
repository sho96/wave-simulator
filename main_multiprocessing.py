import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os

WAVE_SPEED = 100
OUTPUT_DIR = "./"

class PointWaveSource:
  def __init__(self, x, y, amplitude, frequency, init_phase):
    self.pos = np.array([x, y])
    self.amplitude = amplitude
    self.frequency = frequency
    self.init_phase = init_phase
  def get_value_at_point(self, point, t):
    distance = np.linalg.norm(point - self.pos)
    x = 2 * np.pi * self.frequency * (t - distance / WAVE_SPEED) + self.init_phase
    return self.amplitude * np.sin(x if x > 0 else 0)
  def render_old(self, canvas, t):
    height, width = canvas.shape
    xs = np.linspace(0, width - 1, width)
    ys = np.linspace(0, height - 1, height)
    meshgrid = np.meshgrid(xs, ys)
    distance_field = np.concatenate([np.expand_dims(meshgrid[0], axis=-1), np.expand_dims(meshgrid[1], axis=-1)], axis=-1) - self.pos
    distance = np.linalg.norm(distance_field, axis=-1)
    x = 2 * np.pi * self.frequency * (t - distance / WAVE_SPEED) + self.init_phase
    x[x < 0] = 0
    return self.amplitude * np.sin(x)
  def render(self, canvas, t):
    height, width = canvas.shape
    ys, xs = np.ogrid[0:height, 0:width]
    # Broadcasting subtraction for distance field
    dx = xs - self.pos[0]
    dy = ys - self.pos[1]
    distance = np.hypot(dx, dy)
    x = 2 * np.pi * self.frequency * (t - distance / WAVE_SPEED) + self.init_phase
    x = np.maximum(x, 0)
    return self.amplitude * np.sin(x)

def main():
  """ wave_sources = [
    PointWaveSource(100, 100, 1, 2, 0),
    PointWaveSource(100, 1024-100, 1, 2, 0),
    PointWaveSource(1024 - 100, 100, 1, 1, 0),
    PointWaveSource(1024 - 100, 1024 - 100, 1, 1, 0)
  ] """
  
  wave_source_xs = np.linspace(100, 1024 - 100, 5)
  wave_sources = [PointWaveSource(x, 100, 1, 1, x / 1024 * 2 * np.pi) for x in wave_source_xs]
  
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  for filename in os.listdir(OUTPUT_DIR):
    if filename.endswith(".png"):
      os.remove(os.path.join(OUTPUT_DIR, filename))
  with Pool(processes=12) as pool:
    pool.map(save_frame, [(i, (1024, 1024), wave_sources) for i in range(200)])
  plt.close()

def render(canvas_shape, wave_sources, t):
  canvas = np.zeros(canvas_shape, dtype=np.float32)
  for wave_source in wave_sources:
    canvas += wave_source.render(canvas, t) * 255
  canvas //= len(wave_sources)
  return canvas

def save_frame(args):
  i, canvas_shape, wave_sources = args
  t = i / 20
  print(f"Rendering frame {i} at t={t:.2f} {canvas_shape}")
  canvas = render(canvas_shape, wave_sources, t)
  plt.imsave(os.path.join(OUTPUT_DIR, f"{i:04d}.png"), canvas, cmap='viridis')

if __name__ == "__main__":
  main()
  print("Done")
