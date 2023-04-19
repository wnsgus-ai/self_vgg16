import torch
import os
import yaml
import argparse
import logging
import time
from data_loder import DataLoder
from model import VGG16
import datetime
from tensorboardX import SummaryWriter

class Train:
  def __init__(self,loggig_level=logging.info) -> None:
    self.__setLogger(loggig_level)
    self.writer = SummaryWriter()
    
  def __setLogger(self,log_level):
    self._logger = logging.getLogger("Train")
    logging.basicConfig(
    format="%(asctime)s[%(name)s][%(levelname)s]-> %(message)s",
    level=log_level,
    datefmt="(%I:%M:%S)",)
  
  @property
  def device(self):
    return self._device
      
  @device.setter
  def device(self,device):
    self._device = device
    self._logger.info("Device: {}".format(self._device))
    if str(self._device) == "cuda":
      self._logger.info("Current cuda device: {}".format(torch.cuda.current_device()))
    
  @property
  def trainData(self):
    return self._train_data
  
  @trainData.setter
  def trainData(self,data):
    self._train_data = data
  
  @property
  def testData(self):
    return self._test_data
  
  @testData.setter
  def testData(self,data):
    self._test_data = data
    
  @property
  def config(self):
    return self._config
  
  @config.setter
  def config(self,config):
    self._config = config
    
  @property
  def lossFunction(self):
    return self._loss_func
  
  @lossFunction.setter
  def lossFunction(self,func):
    self._loss_func = func
    
  @property
  def optimizer(self):
    return self._optimizer
  
  @optimizer.setter
  def optimizer(self,optimizer):
    self._optimizer = optimizer

  @property
  def model(self):
    return self._model
  
  @model.setter
  def model(self,model):
    self._model = model
    
  def save(self,path):
    model = path+"/{}".format(datetime.datetime.now())
    torch.save(self._model.state_dict(),model)
    
  @property
  def train(self):
    self._logger.info(self._model)
    current_loss = 0
    for epoch in range(self._config["param"]["epoch"]):
      avg_loss = 0
      start_time = time.time()
      for index, (images, labels) in enumerate(self._train_data):
          self._logger.info("{}/{} loss: {}".format(index,len(self._train_data),current_loss))
          
          x = images.to(self._device)
          y_ = labels.to(self.device)
          
          self.optimizer.zero_grad()
          output = self._model.forward(x)
          loss = self.lossFunction(output,y_)
          current_loss = loss
          avg_loss += loss
          self.writer.add_scalar("current_loss",loss,index)
          loss.backward()
          self.optimizer.step()
          
      self.writer.add_scalar("total_loss",avg_loss/len(self._train_data),epoch)
      end_time = time.time()
      self._logger.info("epech{} : {}(s)".format(epoch+1,round(end_time-start_time,2)))
    self.writer.close()
    
  @property
  def test(self):
    correct = 0
    total = 0
    self._model.eval()
    with torch.no_grad():
      for image,label in self._test_data:        
        x = image.to(device)
        _y = label.to(device)

        output = self._model.forward(x)
        _,output_index = torch.max(output,1)

        total += label.size(0)
        correct += (output_index == _y).sum().float()
      acc = 100*correct/total
      self._logger.info("Accuracy of Test Data: {}%".format(acc))
    return acc


if __name__ == "__main__":
  logging_level = logging.DEBUG
  logger = logging.getLogger("main")
  logging.basicConfig(
    format="%(asctime)s[%(name)s][%(levelname)s]-> %(message)s",
    level=logging_level,
    datefmt="(%I:%M:%S)",)
  
  parser = argparse.ArgumentParser()
  parser.add_argument("-c","--config",dest="config",action="store")
  args = parser.parse_args()
  logger.debug("args: {}".format(args))
  
  with open("config/"+args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    logger.debug("config: {}".format(config))
    
  loader = DataLoder(logging_level)
  loader.config = config
  loader.load
  train_set = loader.getTrainData
  test_set = loader.getTestData
  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = VGG16().to(device)
  
  train = Train(logging_level)
  train.config = config
  train.device = device
  train.model = model
  train.trainData = train_set
  train.testData = test_set
  train.lossFunction = torch.nn.CrossEntropyLoss()
  train.optimizer = torch.optim.Adam(model.parameters(),lr=config["param"]["learning_rate"])
  train.train
  train.test
  train.save(config["save"]["path"])
  
    
  
    