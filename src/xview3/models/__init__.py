models = {}

from xview3.models.frcnn import FasterRCNNModel
models['frcnn'] = FasterRCNNModel

from xview3.models.frcnn_and_regress import FasterRCNNRegress
models['frcnn_and_regress'] = FasterRCNNRegress

from xview3.models.frcnn_soft_labels import FasterRCNNSoftLabels
models['frcnn_soft_labels'] = FasterRCNNSoftLabels

from xview3.models.frcnn_softer_labels import FasterRCNNSofterLabels
models['frcnn_softer_labels'] = FasterRCNNSofterLabels

from xview3.models.frcnn_l1 import FasterRCNNModelL1
models['frcnn_l1'] = FasterRCNNModelL1

from xview3.models.frcnn_multihead import FasterRCNNMultihead
models['frcnn_multihead'] = FasterRCNNMultihead

from xview3.models.frcnn_multihead_pseudo import FasterRCNNMultiheadPseudo
models['frcnn_multihead_pseudo'] = FasterRCNNMultiheadPseudo

from xview3.models.frcnn_multihead_pseudo_softer import FasterRCNNmps
models['frcnn_multihead_pseudo_softer'] = FasterRCNNmps

from xview3.models.frcnn_pseudo import FasterRCNNPseudo
models['frcnn_pseudo'] = FasterRCNNPseudo

from xview3.models.frcnn_pseudo_softer import FasterRCNNps
models['frcnn_pseudo_softer'] = FasterRCNNps

from xview3.models.frcnn_i2 import FasterRCNNi2
models['frcnn_i2'] = FasterRCNNi2

from xview3.models.yolov5 import create_model
models['yolov5'] = create_model

from xview3.models.retinanet import RetinaNetModel
models['retinanet'] = RetinaNetModel
