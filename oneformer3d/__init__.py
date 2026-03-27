from .oneformer3d import (
    ScanNetOneFormer3D,
    ScanNet200OneFormer3D,
    S3DISOneFormer3D,
    InstanceOnlyOneFormer3D,
    FloorplanOneFormer3D,
)
from .spconv_unet import SpConvUNet

try:
    from .mink_unet import Res16UNet34C
except ImportError:
    # MinkowskiEngine-backed models remain optional in a fresh environment.
    pass
from .query_decoder import ScanNetQueryDecoder, QueryDecoder
from .unified_criterion import ScanNetUnifiedCriterion, S3DISUnifiedCriterion
from .semantic_criterion import ScanNetSemanticCriterion, S3DISSemanticCriterion
from .instance_criterion import (
    InstanceCriterion,
    QueryClassificationCost,
    MaskBCECost,
    MaskDiceCost,
    HungarianMatcher,
    SparseMatcher,
    OneDataCriterion,
)
from .loading import LoadAnnotations3D_, NormalizePointsColor_, NormalizePointsCoord
from .formatting import Pack3DDetInputs_
from .transforms_3d import (
    ElasticTransfrom,
    AddSuperPointAnnotations,
    FloorscanAddSuperPointAnnotations,
    SwapChairAndFloor,
    PointSample_,
    FloorplanPointSample_,
)
from .transforms_pt import (
    PointceptDataAugmentation,
    PointceptPreprocess,
    # LoadPointceptPointsFromFile,
)
from .data_preprocessor import Det3DDataPreprocessor_
from .unified_metric import UnifiedSegMetric
from .scannet_dataset import ScanNetSegDataset_, ScanNet200SegDataset_
from .floorplan_dataset import (
    S3DFloorplanSegDataset_,
    FloorscanSegDataset,
)
from .s3dis_dataset import S3DISSegDataset_
from .structured3d_dataset import Structured3DSegDataset, ConcatDataset_
from .structures import InstanceData_
from .pooling import (
    SegmentMaxPooling,
    SegmentMeanPooling,
    PointColorPooling,
    MixedPooling,
)

try:
    from .point_transformer_seg import (
        PointTransformerSeg50,
    )
except ImportError:
    # `pointops` is only needed by PointTransformerSeg-style variants.
    # Keep the main PTv3 path importable when that optional extension is absent.
    pass
from .point_transformer_v3 import PointTransformerV3
