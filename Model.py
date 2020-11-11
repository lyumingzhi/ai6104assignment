import cnn
import fc
import softmax
import crossentropy
import relu
import pooling

class Model:
	def __init__(self,layers):
		self.layers=layers
	def forward(self,X):
		output=X
		for module in self.layers:
			print('layer: ',type(module).__name__)
			output=module.forward(output)
		return output
	def backward(self,dz):
		for module_index in range(len(self.layers))[::-1]:
			dz=self.layers[module_index].backward(dz)
	def step(self,lr):
		for module in self.layers:
			if hasattr(module,'step'):
				module.step(lr)
	def zero_gradient(self):
		for module in self.layers:
			if hasattr(module,'zero_gradient'):
				module.zero_gradient()

