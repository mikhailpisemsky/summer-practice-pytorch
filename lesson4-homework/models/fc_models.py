import torch
import json
import torch.nn as nn

class FullyConnectedModel(nn.Module):
    def __init__(self, config_path: str=None, input_size: int=None, num_classes: int=None, **kwargs):
        super().__init__()
        
        if config_path:
            self.config = self.load_config(config_path)
        else:
            self.config = kwargs
        
        self.input_size = input_size or self.config.get('input_size', 784)
        self.num_classes = num_classes or self.config.get('num_classes', 10)
        
        self.layers = self._build_layers()
    
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _build_layers(self):
        layers = []
        prev_size = self.input_size
        
        layer_config = self.config.get('layers', [])
        
        for layer_spec in layer_config:
            layer_type = layer_spec['type']
            
            if layer_type == 'linear':
                out_size = layer_spec['size']
                layers.append(nn.Linear(prev_size, out_size))
                prev_size = out_size
                
            elif layer_type == 'relu':
                layers.append(nn.ReLU())
                
            elif layer_type == 'sigmoid':
                layers.append(nn.Sigmoid())
                
            elif layer_type == 'tanh':
                layers.append(nn.Tanh())
                
            elif layer_type == 'dropout':
                rate = layer_spec.get('rate', 0.5)
                layers.append(nn.Dropout(rate))
                
            elif layer_type == 'batch_norm':
                layers.append(nn.BatchNorm1d(prev_size))
                
            elif layer_type == 'layer_norm':
                layers.append(nn.LayerNorm(prev_size))
        
        layers.append(nn.Linear(prev_size, self.num_classes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)
        
