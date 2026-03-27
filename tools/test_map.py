import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main(ann_file, res_file):
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(res_file)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COCO bbox evaluation on prediction json.")
    parser.add_argument("--ann-file", default="aaai_test/annotations_test.json", help="Ground-truth annotation json.")
    parser.add_argument("--res-file", default="aaai_test/pred_test_coco_mixedpooling.json", help="Prediction json.")
    args = parser.parse_args()
    main(args.ann_file, args.res_file)
