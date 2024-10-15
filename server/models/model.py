# Description: Parent class for all models
import os
import joblib

class Model:
  ml_path = os.path.join(os.path.dirname(__file__), '../ml')

  def __init__(self, dataset_path):
    self.dataset_path = dataset_path
    self.model = None
    self.model_path =  f'{self.ml_path}/{self.__class__.__name__.lower()}.pkl'
    self.load_model()
  
  def train(self):
    pass

  def load_model(self):
    print(f'Loading model from {self.model_path}')
    if os.path.exists(self.model_path):
      self.model = joblib.load(self.model_path)
    else:
      print('Model not found')
      self.train()
      self.load_model()

  def save_model(self):
    if self.model:
      joblib.dump(self.model, self.model_path)
      print(f'Model saved at {self.model_path}')
    else:
      print('Model not found')

  def predict(self):
    pass
