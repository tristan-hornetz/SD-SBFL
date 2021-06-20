from dataclasses import dataclass
import inspect

@dataclass
class T:
    a: int = 1
    b: int = 2
    def sum(self):
        return self.a + self.b

t = T()
print(t)
print(inspect.getfile(t.sum.__self__.__class__))
