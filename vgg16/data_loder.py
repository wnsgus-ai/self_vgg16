import torch
import glob
import torchvision
from torchvision import transforms
import logging
import numpy as np
from PIL import Image

class DataLoder:
  def __init__(self,log_level=logging.info) -> None:
    self.__setLogger(log_level)
  
  def __setLogger(self,log_level):
    self._logger = logging.getLogger("DataLoder")
    logging.basicConfig(
    format="%(asctime)s[%(name)s][%(levelname)s]-> %(message)s",
    level=log_level,
    datefmt="(%I:%M:%S)",)
    
  @property
  def config(self):
    return self._config
  
  @config.setter
  def config(self,config):
    self._config = config
    
  @property
  def load(self):
    # Set data dir
    dir = self._config["data"]["root"] +"/"+ self._config["data"]["dataset"]
    self._train_dir = dir + "/" + self._config["data"]["train"]    
    self._test_dir = dir + "/" + self._config["data"]["test"]

    
    image_size = self._config["data"]["size"].split("x")
    self._logger.info("image resize {}x{}".format(image_size[0],image_size[1]))
    
    self._trans = transforms.Compose([transforms.Resize((int(image_size[0]),int(image_size[1]))),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
    self._logger.debug("train data root: {}".format(self._train_dir))
    self._logger.debug("test data root: {}".format(self._test_dir))
    
    self._train = torchvision.datasets.ImageFolder(root=self._train_dir,transform=self._trans)
    self._test = torchvision.datasets.ImageFolder(root=self._test_dir,transform=self._trans)
    
    self._logger.debug("train data info: {}".format(self._train))
    self._logger.debug("test data info: {}".format(self._test))
    
    self._logger.info("number of train data: {}".format(len(self._train)))
    self._logger.info("number of test data: {}".format(len(self._test)))
    self._logger.info("number of class: {}, {}".format(len(self._train.classes),self._train.classes))

    self._train = torch.utils.data.DataLoader(self._train, 
                                              batch_size=self._config["param"]["batch_size"], 
                                              shuffle=True)
    self._test = torch.utils.data.DataLoader(self._test, 
                                             batch_size=self._config["param"]["batch_size"], 
                                             shuffle=True)      

  @property 
  def getTrainData(self):
    return self._train
  
  @property 
  def getTestData(self):
    return self._test
