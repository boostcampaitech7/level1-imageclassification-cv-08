import timm

class ModelSelector:
    def __init__(self, model_name='resnet18', num_classes=500, pretrained=True, drop_rate=0.5):
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.drop_rate = drop_rate

    def get_model(self):
        return timm.create_model(self.model_name, 
                                 pretrained=self.pretrained, 
                                 num_classes=self.num_classes,
                                 drop_rate = self.drop_rate
                                 )