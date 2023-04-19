import torch

class VGG16(torch.nn.Module):
  def __init__(self) -> None:
    super(VGG16,self).__init__()
    self._model = torch.nn.Sequential(
      self.conv2Block(3,64),
      self.conv2Block(64,128),
      self.conv3Block(128,256),
      self.conv3Block(256,512),
      self.conv3Block(512,512),
      torch.nn.Flatten(),
      self.fcBlock(512,2)
    )
    
    self.fc_layer = torch.nn.Sequential(
      torch.nn.AdaptiveAvgPool1d(512*7*7),
      torch.nn.Linear(512*7*7,4096),
      torch.nn.ReLU(),
      torch.nn.Dropout(),
      torch.nn.Linear(4096,1000),
      torch.nn.ReLU(),
      torch.nn.Dropout(),
      torch.nn.Linear(1000,2)
    )
      
  def conv2Block(self,input_dim,output_dim):
    block = torch.nn.Sequential(
      torch.nn.Conv2d(input_dim,output_dim,kernel_size=3,padding=1),
      torch.nn.BatchNorm2d(output_dim),
      torch.nn.ReLU(),
      torch.nn.Conv2d(output_dim,output_dim,kernel_size=3,padding=1),
      torch.nn.BatchNorm2d(output_dim),
      torch.nn.ReLU(),
      torch.nn.MaxPool2d(2,2)
    )
    return block
  
  def conv3Block(self,input_dim,output_dim):
    block = torch.nn.Sequential(
      torch.nn.Conv2d(input_dim,output_dim,kernel_size=3,padding=1),
      torch.nn.BatchNorm2d(output_dim),
      torch.nn.ReLU(),
      torch.nn.Conv2d(output_dim,output_dim,kernel_size=3,padding=1),
      torch.nn.BatchNorm2d(output_dim),
      torch.nn.ReLU(),
      torch.nn.Conv2d(output_dim,output_dim,kernel_size=3,padding=1),
      torch.nn.BatchNorm2d(output_dim),
      torch.nn.ReLU(),
      torch.nn.MaxPool2d(2,2)
    )
    return block
    
  def fcBlock(self,input_dim,output_dim):
    block = torch.nn.Sequential(
      torch.nn.AdaptiveAvgPool1d(input_dim*7*7),
      torch.nn.Linear(input_dim*7*7,4096),
      torch.nn.ReLU(),
      torch.nn.Dropout(),
      torch.nn.Linear(4096,1000),
      torch.nn.ReLU(),
      torch.nn.Dropout(),
      torch.nn.Linear(1000,output_dim)
    )
    return block
  
  @property
  def info(self):
    return self._model
    
  def forward(self,x):
    x = self._model(x)
    return x
    