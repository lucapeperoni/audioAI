import torch, torch.nn as nn
from torchvision.models import resnet18
import torch.onnx as onnx

model = resnet18(weights=None)
model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
model.fc    = nn.Linear(512, 10)
model.load_state_dict(torch.load('best_resnet18.pt', map_location='cpu'))
model.eval()

dummy = torch.randn(1,1,128,128)
onnx.export(model, dummy, 'genre_resnet18.onnx',
            input_names=['mel'], output_names=['logits'],
            opset_version=17)
print("ONNX salvato → genre_resnet18.onnx")
