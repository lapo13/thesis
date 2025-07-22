from torch import nn

class RegressionNet(nn.Module):
     def __init__(self, input_size, output_size):
          super(RegressionNet, self).__init__()
          self.linear_relu = nn.Sequential(
               nn.Linear(input_size, 32),
               nn.ReLU(),
               nn.Linear(32, 64),
               nn.ReLU(),
               nn.Linear(64, output_size)
          )
          
          self._init_weights()

     
     def _init_weights(self):
          for m in self.modules():
               if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias,0)



     def forward(self, x):
          return self.linear_relu(x)