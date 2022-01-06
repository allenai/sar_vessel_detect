class GridIndex(object):
    '''
    Implements a grid index for spatial data.
    Supports inserting points or rectangles, and efficiently searching by bounding box.
    '''

    def __init__(self, size):
        self.size = size
        self.grid = {}

    # Insert a point with data.
    def insert(self, p, data):
        self.insert_rect([p[0], p[1], p[0], p[1]], data)

    # Insert a data with rectangle bounds.
    def insert_rect(self, rect, data):
        def f(cell):
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append(data)
        self.each_cell(rect, f)

    def each_cell(self, rect, f):
        for i in range(rect[0] // self.size, rect[2] // self.size + 1):
            for j in range(rect[1] // self.size, rect[3] // self.size + 1):
                f((i, j))

    def search(self, rect):
        matches = set()
        def f(cell):
            if cell not in self.grid:
                return
            for data in self.grid[cell]:
                matches.add(data)
        self.each_cell(rect, f)
        return matches
