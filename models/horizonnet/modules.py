from external.HorizonNet.model import HorizonNet as BaseHorizon


class HorizonNetModule(BaseHorizon):
    def __init__(self):
        super().__init__('resnet50', True)
