from .single_stage_parsing import SingleStageInsParsingDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class PARSING(SingleStageInsParsingDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 mask_feat_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PARSING, self).__init__(backbone, neck, bbox_head, mask_feat_head, train_cfg,
                                   test_cfg, pretrained)
