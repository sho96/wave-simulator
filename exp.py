import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os, sys
from color_gradient import Color, Gradient

gradient1 = Gradient({0: Color(255, 0, 0), 0.5: Color(0, 128, 0), 1: Color(0, 0, 255)})
gradient2 = Gradient({0: Color(0, 0, 0), 0.2: Color(0, 128, 128), 0.8: Color(50, 255, 0), 2: Color(128, 255, 0)})
gradient3 = Gradient({
    0: Color(204, 86, 65),
    0.5: Color(64//5, 207//5, 93//5),
    1: Color(64, 126, 207)
})

def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def grid_to_img(grid, minimum, maximum):
    raw_result = (grid - minimum) / (maximum - minimum) * 255
    raw_result[raw_result < 0] = 0
    raw_result[raw_result > 255] = 255
    return raw_result.astype(np.uint8)  
    
def grid_to_img_auto_exposure(grid, minimum=0, maximum=255):
    min_val = np.min(grid)
    max_val = np.max(grid)
    return np.array((grid - min_val) / (max_val - min_val) * (maximum - minimum) + minimum, dtype=np.uint8)

#def add_circle(grid, origin, radius, value):
#    mask = np.zeros_like(grid, dtype=np.float32)
#    mask = cv2.circle(mask, origin, int(radius), (1, 1, 1), 2)
#    return grid + (mask * value)
def add_circle(grid, origin, radius, value): #more accurate version (uses more computational power)
    mask = np.zeros_like(grid, dtype=np.float32)
    mask = cv2.circle(mask, (int(origin[0]), int(origin[1])), int(radius), (1, 1, 1), 1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return grid + (mask * value)

def plot_points(image, points, color=(0, 0, 255), thickness=1):
    cpy = image.copy()
    for point in points:
        cv2.circle(cpy, (int(point[0]), int(point[1])), thickness, color, -1)
    return cpy

class Wave:
    travel_speed = 1
    amplitude = 50
    decay = 100
    def __init__(self, origin, amplitude=amplitude):
        self.origin = origin
        self.current_radius = 0
        self.current_amplitude = amplitude
        self.initial_amplitude = amplitude
    def update(self, dt=1):
        self.current_radius += self.travel_speed * dt
        #self.current_amplitude = self.initial_amplitude / np.exp(self.current_radius / self.decay)
    def render(self, grid):
        return add_circle(grid, self.origin, self.current_radius, self.current_amplitude)
    def is_not_rendered(self, grid):
        distances_to_corners = (
            dist(self.origin, (0, 0)),
            dist(self.origin, (0, grid.shape[1])),
            dist(self.origin, (grid.shape[0], 0)),
            dist(self.origin, (grid.shape[0], grid.shape[1])),
        )
        return max(distances_to_corners) < self.current_radius

class Source:
    def __init__(self, origin, function, amplitude, shift_t=0):
        self.origin = origin
        self.waves = []
        self.func = function
        self.amplitude = amplitude
        self.shift_t = shift_t
        self.t = shift_t
        #print(f"time_shift: {self.t}")
    def update(self, dt=1):
        #print(f"time_shift: {self.t}")
        self.t += dt
        self.waves.append(Wave(self.origin, self.amplitude * self.func(self.t) * dt))
        for wave in self.waves:
            wave.update(dt)
    
    def update_with_amplitude_specified(self, amplitude, dt=1):
        self.t += dt
        self.waves.append(Wave(self.origin, amplitude * dt))
        for wave in self.waves:
            wave.update(dt)
    
    def render(self, grid):
        modified = grid.copy()
        for wave in self.waves:
            if wave.is_not_rendered(grid):
                self.waves.remove(wave)
                del wave
                continue
            modified = wave.render(modified)
        return modified
    def delete_waves(self):
        for _ in range(len(self.waves)):
            popped = self.waves.pop()
            del popped
    def draw_origin_point(self, img, radius=3, color=(0, 0, 255), scale=1, render_value=False, max_value=1):
        cpy = img.copy()
        if not render_value:
            return cv2.circle(cpy, self.origin * scale, radius, color, -1)
        
        current_value = self.func(self.t) / max_value
        if current_value < 0:
            #invert color
            color = (255 - color[0], 255 - color[1], 255 - color[2])
        r = int(abs(radius * current_value * scale))
        if r == 0:
            return cpy
        #print(self.origin, scale, self.origin * scale)
        return cv2.circle(cpy, (int(self.origin[0] * scale), int(self.origin[1] * scale)), r, color, -1)
    def draw_origin_point_with_amplitude_specified(self, img, amplitude, radius=3, color=(0, 0, 255), scale=1, render_value=False, max_value=1):
        cpy = img.copy()
        if not render_value:
            return cv2.circle(cpy, self.origin * scale, radius, color, -1)
        
        current_value = amplitude / max_value
        if current_value < 0:
            #invert color
            color = (255 - color[0], 255 - color[1], 255 - color[2])
        r = int(abs(radius * current_value * scale))
        if r == 0:
            return cpy
        #print(self.origin, scale, self.origin * scale)
        return cv2.circle(cpy, (int(self.origin[0] * scale), int(self.origin[1] * scale)), r, color, -1)

class LineOfSources:
    def __init__(self, n_sources, origin, spacing, amplitude, function, time_shift, total_intensity=1, shift_t=5):
        self.sources = [
            Source(
                (origin[0] + i * spacing[0], origin[1] + i * spacing[1]), 
                function, 
                amplitude / n_sources * total_intensity, 
                shift_t + i * time_shift
            ) for i in range(n_sources)
        ]
    def update(self, dt=1):
        for source in self.sources:
            source.update(dt)
    def render(self, grid):
        modified = grid.copy()
        for source in self.sources:
            modified = source.render(modified)
        return modified
    
    def delete_waves(self):
        for source in self.sources:
            source.delete_waves()

    def draw_origin_points(self, img, radius=3, color=(0, 0, 255), scale=1, render_value=False, max_value=1):
        cpy = img.copy()
        for source in self.sources:
            cpy = source.draw_origin_point(cpy, radius, color, scale, render_value, max_value)
        return cpy

class PhaseChangingLineOfSources(LineOfSources):
    def __init__(self, n_sources, origin, spacing, amplitude, base_function, time_shift_function, time_shift_multiplier, total_intensity=1, shift_t=5):
        super().__init__(n_sources, origin, spacing, amplitude, base_function, 0, total_intensity, shift_t=0)
        
        #self.sources = [
        #    Source(
        #        (origin[0] + i * spacing[0], origin[1] + i * spacing[1]), 
        #        base_function, 
        #        amplitude, 
        #        0
        #    ) for i in range(n_sources)
        #]
        self.timer = shift_t
        self.time_shift_function = time_shift_function
        self.time_shift_multiplier = time_shift_multiplier
        self.current_time_shift = self.time_shift_function(self.timer)
    def update(self, dt=1):
        self.timer += dt
        self.current_time_shift = self.time_shift_function(self.timer) * self.time_shift_multiplier
        for i, source in enumerate(self.sources):
            source.update_with_amplitude_specified(
                source.func(self.timer + self.current_time_shift * i) * source.amplitude, 
                dt
            )
    def draw_origin_points(self, img, radius=3, color=(0, 0, 255), scale=1, render_value=False, max_value=1):
        cpy = img.copy()
        for i, source in enumerate(self.sources):
            amplitude = source.func(self.timer + self.current_time_shift * i)
            cpy = source.draw_origin_point_with_amplitude_specified(cpy, amplitude, radius, color, scale, render_value, max_value)
        return cpy
class Probe:
    def __init__(self, sampling_pos, name, init_t, max_num_measurements, record_min_max=False):
        self.measurenments = np.zeros((max_num_measurements, 2), dtype=np.float32)
        self.timer = init_t
        self.sampling_pos = sampling_pos
        self.record_min_max = record_min_max
        self.min_value_so_far = None
        self.max_value_so_far = None
        self.name = name
    
    def update(self, grid, dt=1):
        self.timer += dt
        value = grid[int(self.sampling_pos[1]), int(self.sampling_pos[0])]
        self.measurenments = np.roll(self.measurenments, 1, axis=0)
        self.measurenments[0] = (self.timer, value)
        
        if self.record_min_max:
            if self.min_value_so_far is None or value < self.min_value_so_far:
                self.min_value_so_far = value
            if self.max_value_so_far is None or value > self.max_value_so_far:
                self.max_value_so_far = value
    
    def draw_origin_point(self, img, radius=1, color=(255, 255, 0), scale=1, render_value=False, max_value=1):
        current_value = self.measurenments[0, 1] / max_value
        #print(self.measurenments[0, 1], current_value)
        if current_value < 0:
            #invert color
            color = (255 - color[0], 255 - color[1], 255 - color[2])
        r = int(abs(current_value * scale))
        cpy = img.copy()
        cpy = cv2.circle(cpy, (int(self.sampling_pos[0] * scale), int(self.sampling_pos[1] * scale)), int(radius * scale) + r, color, int(radius * scale / 2))
        cpy = cv2.circle(cpy, (int(self.sampling_pos[0] * scale), int(self.sampling_pos[1] * scale)), int(radius * scale), (0, 255, 0), -1)
        return cpy
    
    def plot_measurenments(self, ax_plt: plt.Axes = None, y_range=1000):
        ax_plt.set_ylim(-y_range, y_range)
        ax_plt.plot(self.measurenments[:-1, 0], self.measurenments[:-1, 1], color="blue", label=f"{round(self.measurenments[0, 1])}")
        
        if self.record_min_max:
            oldest_measurenment_time = self.measurenments[-1, 0]
            newest_measurenment_time = self.measurenments[0, 0]
            
            ax_plt.plot(
                [oldest_measurenment_time, newest_measurenment_time], 
                [self.min_value_so_far, self.min_value_so_far],
                color="red",
                label=f"{round(self.min_value_so_far)}"
            )
            ax_plt.plot(
                [oldest_measurenment_time, newest_measurenment_time], 
                [self.max_value_so_far, self.max_value_so_far],
                color="green",
                label=f"{round(self.max_value_so_far)}"
            )
        
        ax_plt.set_ylabel(self.name)
        ax_plt.legend(loc="upper left")
    
    def reset(self):
        self.measurenments = np.zeros_like(self.measurenments)
        self.timer = 0
        self.min_value_so_far = None
        self.max_value_so_far = None

class Probes:
    def __init__(self, points, max_num_measurements=100, record_min_max=False):
        self.probes = [Probe(point, f"({point[0]}, {point[1]})", 0, max_num_measurements, record_min_max) for point in points]
        self.fig, self.axes = None, None

    
    def update(self, grid, dt=1):
        if self.fig is None:
            self.fig, self.axes = plt.subplots(len(self.probes), sharex=True)
        for probe in self.probes:
            probe.update(grid, dt)

    def draw_origin_points(self, img, radius=1, color=(255, 255, 0), scale=1, render_value=False, max_value=1):
        cpy = img.copy()
        for probe in self.probes:
            cpy = probe.draw_origin_point(cpy, radius, color, scale, render_value, max_value)
        return cpy
    
    def render_measurenments(self, y_range=1000):
        for probe, ax in zip(self.probes, self.axes):
            probe.plot_measurenments(ax, y_range)
    
    def clear_graphs(self):
        for ax in self.axes:
            ax.cla()
    
    def save_graph(self, path, y_range=1000):
        self.render_measurenments(y_range)
        plt.savefig(path)
        self.clear_graphs()
    
    def show_measurenments(self, y_range=1000):
        self.render_measurenments(y_range)
        plt.pause(0.01)
        self.clear_graphs()

    def reset(self):
        for probe in self.probes:
            probe.reset()
            
            
class Scene:
    def __init__(self, sources=[], line_sources=[], probes:Probes=None):
        self.sources = sources
        self.line_sources = line_sources
        self.probes = probes

    def update(self, dt=1):
        for source in self.sources:
            source.update(dt)
        for source in self.line_sources:
            source.update(dt)

    def render(self, grid):
        modified = grid.copy()
        for source in self.sources:
            modified = source.render(modified)
        for source in self.line_sources:
            modified = source.render(modified)
        return modified

    def draw_origin_points(self, img, radius=3, probe_radius=.5, color=(255, 255, 0), scale=1, render_value=False, max_value=1, probe_max_value=2000):
        cpy = img.copy()
        for source in self.sources:
            cpy = source.draw_origin_point(cpy, radius, color, scale, render_value, max_value)
        for source in self.line_sources:
            cpy = source.draw_origin_points(cpy, radius, color, scale, render_value, max_value)
        
        cpy = self.probes.draw_origin_points(cpy, probe_radius, color, scale, render_value, probe_max_value)
        return cpy
    
    def delete_waves(self):
        for source in self.sources:
            source.delete_waves()
        for source in self.line_sources:
            source.delete_waves()

def sine_fast(t):
    return np.sin(t/8 * 2 * np.pi)

def sine_slow(t):
    return np.sin(t/100 * 2 * np.pi)

height = 9 * 20 // 2
width = 8 * 20 // 2

img = np.zeros((height, width), dtype=np.float32)

single_point = Scene(
    [
        Source((width//2, height//2), sine_fast, 200, 0)
    ],
    [],
    Probes(
        [
            (width//4, height//2),
            #(width//2 - 20//2, height//2),
            #(width//2, height//2),
            #(width//2 + 20//2, height//2),
            (width//4 * 3, height//2),
        ],
        100,
        True
    )
)



spacing = 5
n_sources = 7
emit_to_the_right = Scene(
    [],
    [
        LineOfSources(
            n_sources, (width//2 - spacing * n_sources // 2 + spacing/2, height), (spacing, 0), 200, sine_fast, -2, 1, 0
        )
    ],
    Probes(
        [
            (width//4, height//2),
            #(width//2 - 20//2, height//2),
            (width//2, height//2),
            #(width//2 + 20//2, height//2),
            (width//4 * 3, height//2),
        ],
        100,
        True
    )
)
spacing = 5
n_sources = 7
emit_to_the_left = Scene(
    [],
    [
        LineOfSources(
            n_sources, (width//2 - spacing * n_sources // 2 + spacing/2, height), (spacing, 0), 200, sine_fast, 2, 1, 0
        )
    ],
    Probes(
        [
            (width//4, height//2),
            #(width//2 - 20//2, height//2),
            (width//2, height//2),
            #(width//2 + 20//2, height//2),
            (width//4 * 3, height//2),
        ],
        100,
        True
    )
)



spacing = 7
n_sources = 6
normal = Scene(
    [],
    [
        LineOfSources(
            n_sources, (width/2 - spacing * n_sources / 2 + spacing/2, height-10), (spacing, 0), 200, sine_fast, -0, 1, 0
        )
    ],
    Probes(
        [
            (width//5, height//2),
            #(width//2 - 20//2, height//2),
            (width//2, height//2),
            #(width//2 + 20//2, height//2),
            (width//5 * 4, height//2),
        ],
        100,
        True
    )
)



spacing = 4
n_sources = 5
stationary = Scene(
    [],
    [
        LineOfSources(n_sources, (width//2 - spacing * n_sources // 2 + spacing/2, 0), (spacing, 0), 200, sine_fast, 0, 1, 0),
        LineOfSources(n_sources, (width//2 - spacing * n_sources // 2 + spacing/2, height), (spacing, 0), 200, sine_fast, 0, 1, 0)
    ],
    Probes(
        [
            (width//2, height//3.5),
            #(width//2 - 20//2, height//2),
            (width//2, height//2),
            #(width//2 + 20//2, height//2),
            (width//2, height - height//3.5),
        ],
        100,
        True
    )
)



spacing = 4
n_sources = 7
direction_changing = Scene(
    [],
    [
        PhaseChangingLineOfSources(
            n_sources, (width//2 - spacing * n_sources // 2, height), (spacing, 0), 200, sine_fast, sine_slow, 1.5, 1, 0
        )
    ],
    Probes(
        [
            (width//5, height//2),
            #(width//2 - 20//2, height//2),
            (width//2, height//2),
            #(width//2 + 20//2, height//2),
            (width//5 * 4, height//2),
        ],
        100,
        True
    )
)




scene = normal

probes = scene.probes


scale = 10
dt = 1.25 ** -2
prev_dt = dt
frame_count = 0

what_to_render = input('name to record: ')
output_dir = f"{what_to_render}"
if os.path.exists(output_dir):
    if input("Directory already exists. enter 'y' to overwrite\n") != 'y':
        sys.exit()
else:
    os.mkdir(output_dir)
if not os.path.exists(f"{output_dir}/frames"):
    os.mkdir(f"{output_dir}/frames")
if not os.path.exists(f"{output_dir}/figs"):
    os.mkdir(f"{output_dir}/figs")
    
min_value = -200
max_value = 200

probe_max_value = 200
    
while True:
    frame_count += 1
    
    scene.update(max(prev_dt, 1/1.25**5))
    rendered = scene.render(img)
    
    probes.update(rendered, max(prev_dt, 1/1.25**5))
    
    result = grid_to_img(rendered, minimum=min_value, maximum=max_value)
    result = cv2.resize(result, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    result = gradient3.apply_to_grayscale_image(result)
    result = scene.draw_origin_points(result, render_value=True, scale=scale, radius=2, probe_max_value=probe_max_value)
    
    cv2.imwrite(f"{output_dir}/frames/{frame_count}.png", result)
    #probes.render_measurenments(probe_max_value)
    probes.save_graph(f"{output_dir}/figs/{frame_count}.png", probe_max_value)
    probes.show_measurenments(probe_max_value)
    
    cv2.imshow(what_to_render, result)
    waitkey_result = cv2.waitKeyEx(min(int(np.ceil(dt)), 1))
    if waitkey_result == ord('q'):
        break
    if dt != 0:
        if waitkey_result == 2621440:
            #up arrow
            dt /= 1.25
            if dt < 1/(1.25**5):
                dt = 1 / (1.25**5)
            prev_dt = dt
            print(dt)
        if waitkey_result == 2490368:
            #down arrow
            dt *= 1.25
            prev_dt = dt
            print(dt)
    if waitkey_result == 8:
        #clear waves
        scene.delete_waves()
    
    if waitkey_result == ord("r"):
        probes.reset()
        
    if waitkey_result == ord(" "):
        if dt != 0:
            prev_dt = dt
            dt = 0
        else:
            dt = prev_dt
        
cv2.destroyAllWindows()

#save the image to output.png
cv2.imwrite("output.png", result)
