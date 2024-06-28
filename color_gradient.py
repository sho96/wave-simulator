import numpy as np
class Color:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def rgb(self):
        return (self.r, self.g, self.b)
    
    def bgr(self):
        return (self.b, self.g, self.r)
    
    def hex(self):
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    def __add__(self, other):
        return Color(self.r + other.r, self.g + other.g, self.b + other.b)

    def __mul__(self, other):
        return Color(self.r * other, self.g * other, self.b * other)

    def __truediv__(self, other):
        return Color(self.r / other, self.g / other, self.b / other)

    def __str__(self):
        return f"({self.r}, {self.g}, {self.b})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.r == other.r and self.g == other.g and self.b == other.b

    def __hash__(self):
        return hash((self.r, self.g, self.b))
    
class Gradient:
    def __init__(self, grad: "dict[float, Color]"):
        self.points = list(grad.keys())
        self.colors = list(grad.values())
        self.points, self.colors = zip(*sorted(zip(self.points, self.colors)))

    def __call__(self, t):
        if t < self.points[0]:
            return self.colors[0]
        if t > self.points[-1]:
            return self.colors[-1]
        for i in range(len(self.points) - 1):
            if t >= self.points[i] and t <= self.points[i + 1]:
                return self.colors[i] + (self.colors[i + 1] - self.colors[i]) * (t - self.points[i]) / (self.points[i + 1] - self.points[i])
    
    def apply_to_grayscale_image(self, img):
        zero_to_one = img.astype(np.float32).copy() / 255
        result = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        #print(np.min(zero_to_one), np.max(zero_to_one))
        result[zero_to_one <= self.points[0]] = self.colors[0].bgr()
        result[zero_to_one >= self.points[-1]] = self.colors[-1].bgr()
        for i in range(len(self.points) - 1):
            minimum = self.points[i]
            maximum = self.points[i + 1]
            target_indices = np.logical_and(zero_to_one > minimum, zero_to_one <= maximum)
            
            rmin, gmin, bmin = self.colors[i].rgb()
            rmax, gmax, bmax = self.colors[i + 1].rgb()
            
            result[target_indices, 2] = rmin + (rmax - rmin) * (zero_to_one[target_indices] - minimum) / (maximum - minimum)
            result[target_indices, 1] = gmin + (gmax - gmin) * (zero_to_one[target_indices] - minimum) / (maximum - minimum)
            result[target_indices, 0] = bmin + (bmax - bmin) * (zero_to_one[target_indices] - minimum) / (maximum - minimum)
            
        return result
        