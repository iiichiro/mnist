"""
modelsパッケージ

"""
from .tf_model import TensorflowTutorialModel
from .siwei_models import LaNet5, SimpleCNN, ComplexCNN

DEFINED_MODELS = {
    TensorflowTutorialModel.MODEL_NAME: TensorflowTutorialModel,  # TensorFlowチュートリアルモデル
    LaNet5.MODEL_NAME: LaNet5,
    SimpleCNN.MODEL_NAME: SimpleCNN,
    ComplexCNN.MODEL_NAME: ComplexCNN,
}
