import math
class Value:
    def __init__(self, data, op='', prev = None, grad = 0.0, name = ''):
        self.data = data 
        self._op = op
        self._prev = [] if prev is None else prev
        self._grad = 0.0
        self.name = name
    
    def __add__(self, other):
        if( isinstance(other, Value)):  
            return Value(self.data + other.data, op='+', prev=[self, other], name=f"({self.name} + {other.name})")
        else:
            return Value(self.data + other, op='+', prev=[self, other], name=f"({self.name} + {other})")
    
    def __radd__(self, other):
        if( isinstance(other, Value)):  
            return Value(self.data + other.data, op='+', prev=[self, other], name=f"({self.name} + {other.name})")
        else:
            return Value(self.data + other, op='+', prev=[self, other], name=f"({self.name} + {other})")
    
    def __mul__(self, other):
        if( isinstance(other, Value)):  
            return Value(self.data * other.data, op='*', prev=[self, other], name=f"({self.name} * {other.name})")
        else:
            return Value(self.data * other, op='*', prev=[self, other], name=f"({self.name} * {other})")
    
    def __rmul__(self, other):
        if( isinstance(other, Value)):      
            return Value(self.data * other.data, op='*!', prev=[self, other], name=f"({self.name} * {other.name})")
        else:
            return Value(self.data * other, op='*!', prev=[self, other], name=f"({self.name} * {other})")
    
    def __pow__(self, other):
        if( isinstance(other, Value)):      
            return Value(self.data ** other.data, op='**', prev=[self, other], name=f"({self.name} ** {other.name})")
        else:
            return Value(self.data ** other, op='**', prev=[self, other], name=f"({self.name} ** {other})")
    def __rpow__(self, other):
        if( isinstance(other, Value)):      
            return Value(other.data ** self.data, op='**', prev=[other, self], name=f"({other.name} ** {self.name})")
        else:
            return Value(other ** self.data, op='**', prev=[other, self], name=f"({other} ** {self.name})")
    
    def __truediv__(self, other):
        if( isinstance(other, Value)):      
            return Value(self.data / other.data, op='/', prev=[self, other], name=f"({self.name} / {other.name})")
        else:
            return Value(self.data / other, op='/', prev=[self, other], name=f"({self.name} / {other})")
    
    def __rtruediv__(self, other):
        if( isinstance(other, Value)):      
            return Value(other.data / self.data, op='/', prev=[self, other], name=f"({other.name} / {self.name})")
        else:
            return Value(other / self.data, op='/', prev=[self, other], name=f"({other} / {self.name})")
    
    def __neg__(self):
        return Value(-self.data, op='-', prev=[self], name=f"(-{self.name})")
    
    def __sub__(self, other):
        if( isinstance(other, Value)):      
            return Value(self.data - other.data, op='-', prev=[self, other], name=f"({self.name} - {other.name})")
        else:
            return Value(self.data - other, op='-', prev=[self, other], name=f"({self.name} - {other})")
    
    def __rsub__(self, other):
        if( isinstance(other, Value)):      
            return Value(other.data - self.data, op='-', prev=[other, self], name=f"({other.name} - {self.name})")
        else:
            return Value(other - self.data, op='-', prev=[other, self], name=f"({other} - {self.name})")
    
    def __repr__(self):
        return f"Value(data={self.data}, op={self._op},  grad={self._grad}, name={self.name})"
    

    def tanh(self):
        a = math.exp(2 * self.data) - 1
        b = math.exp(2 * self.data) + 1
        c= Value(a/b, op='tanh', prev=[self], name=f"tanh({self.name})")
        return c
    
    def zero_grad(self):
        self._grad = 0.0
        for prev_node in self._prev:
            if isinstance(prev_node, Value):
                prev_node.zero_grad()
    
    def backward(self, child_node = None):
        if child_node is None:
            self._grad = 1.0
            for prev_node in self._prev:
                if isinstance(prev_node, Value):
                    prev_node.backward(self)
                
            return 
        
        child_op = child_node._op 
        child_grad = child_node._grad 

        if child_op == '+':
            self._grad += child_grad 
        elif child_op == '*' or child_op == '*!':
            child_prev_nodes = child_node._prev
            if isinstance(child_prev_nodes[0], Value):
                if self == child_prev_nodes[0]:
                    if isinstance(child_prev_nodes[1], Value):  
                        self._grad += child_grad * child_prev_nodes[1].data 
                    else:
                        self._grad += child_grad * child_prev_nodes[1]
            if isinstance(child_prev_nodes[1], Value):
                if self == child_prev_nodes[1]:
                    if isinstance(child_prev_nodes[0], Value):
                        self._grad += child_grad * child_prev_nodes[0].data 
                    else:
                        self._grad += child_grad * child_prev_nodes[0]
        elif child_op == '-':
            child_prev_nodes = child_node._prev
            if isinstance(child_prev_nodes[0], Value):
                if self == child_prev_nodes[0]:
                    self._grad += child_grad 
            if isinstance(child_prev_nodes[1], Value):
                if self == child_prev_nodes[1]:
                    self._grad -= child_grad 
        elif child_op == '**':
            child_prev_nodes = child_node._prev
            if self == child_prev_nodes[0]:
                if isinstance(child_prev_nodes[1], Value):
                    self._grad += child_grad * child_prev_nodes[1].data * (self.data ** (child_prev_nodes[1].data - 1))
                else:
                    self._grad += child_grad * child_prev_nodes[1] * (self.data ** (child_prev_nodes[1] - 1))
            if self == child_prev_nodes[1]:
                if isinstance(child_prev_nodes[0], Value):
                    self._grad += child_grad * child_node.data * math.log(child_prev_nodes[0].data)
                else:
                    self._grad += child_grad * child_node.data * math.log(child_prev_nodes[0])
        
        elif child_op == 'tanh':
            self._grad += child_grad * (1 - child_node.data ** 2)
            

            
        for prev_node in self._prev:
            if isinstance(prev_node, Value):
                prev_node.backward(self)
