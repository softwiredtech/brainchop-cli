from tinygrad import dtypes

class DynamicMeshnet:
    def __init__(self):
        self.convs = []
        self.acts = []

    def __call__(self, x):
        for i in range(len(self.convs)):
            if i < len(self.acts):
                x = self.acts[i](self.convs[i](x))
            else: 
                x = self.convs[i](x)
        return x.argmax(1).cast(dtypes.float32)
