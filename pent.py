import math

def draw_pentagon(size=20, fill=False):
    """Print a regular pentagon made of '*' characters."""
    h, w = size * 2, size * 2
    cx, cy = w // 2, h // 2
    radius = int(0.8 * min(cx, cy))

    # five vertices (clockwise)
    vertices = []
    for k in range(5):
        angle = math.radians(90 + k * 72)          # point up first
        x = cx + radius * math.cos(angle)
        y = cy - radius * math.sin(angle)
        vertices.append((int(round(x)), int(round(y))))

    # blank canvas
    canvas = [[' ' for _ in range(w)] for _ in range(h)]

    # simple line drawer
    def line(x0, y0, x1, y1):
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            if 0 <= x0 < w and 0 <= y0 < h:
                canvas[y0][x0] = '*'
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    # draw the 5 edges
    for i in range(5):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % 5]
        line(x0, y0, x1, y1)

    # optional fill (horizontal scan-line flood fill)
    if fill:
        for y in range(h):
            inside = False
            for x in range(w):
                if canvas[y][x] == '*':
                    inside = not inside
                elif inside:
                    canvas[y][x] = '*'

    # print result
    print('\n'.join(''.join(row).rstrip() for row in canvas))

if __name__ == '__main__':
    draw_pentagon(size=25, fill=False)

