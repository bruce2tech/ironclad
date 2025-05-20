import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from modules.utils.metrics import calculate_iou
from modules.utils.metrics import evaluate_detections
from modules.utils.metrics import calculate_precision_recall_curve


###############################################################################
# Tests for IOU
###############################################################################
class TestCalculateIOU(unittest.TestCase):
    def test_no_overlap(self):
        self.assertAlmostEqual(calculate_iou((0, 0, 1, 1), (2, 2, 1, 1)), 0)

    def test_partial_overlap(self):
        self.assertAlmostEqual(calculate_iou((0, 0, 2, 2), (1, 1, 2, 2)), 1 / 7)

    def test_full_overlap(self):
        self.assertAlmostEqual(calculate_iou((0, 0, 2, 2), (0, 0, 2, 2)), 1)

    def test_one_inside_another(self):
        self.assertAlmostEqual(calculate_iou((0, 0, 3, 3), (1, 1, 1, 1)), 1 / 9)

    def test_touching_boxes(self):
        self.assertAlmostEqual(calculate_iou((0, 0, 2, 2), (2, 2, 2, 2)), 0)

    def test_zero_sized_box(self):
        self.assertAlmostEqual(calculate_iou((0, 0, 0, 0), (1, 1, 1, 1)), 0)
        self.assertAlmostEqual(calculate_iou((0, 0, 1, 1), (1, 1, 0, 0)), 0)



###############################################################################
# Tests for Evaluate Detections
###############################################################################
class TestEvaluateDetections(unittest.TestCase):
    
    def test_objectness(self):
        boxes = [[(50, 50, 100, 100), (200, 200, 80, 80)]]
        classes = [[0, 1]]
        scores = [[0.9, 0.85]]
        cls_scores = [[0.95, 0.88]]
        gt_boxes = [[(48, 48, 100, 100), (205, 205, 75, 75)]]
        gt_classes = [[0, 1]]
        map_iou_threshold = 0.5

        y_true, pred_scores = evaluate_detections(boxes, classes, scores, cls_scores, gt_boxes, gt_classes, map_iou_threshold, eval_type="objectness")
        
        self.assertListEqual(y_true, [0, 1])
        self.assertListEqual(pred_scores, [0.9, 0.85])

    def test_class_scores(self):
        boxes = [[(50, 50, 100, 100), (200, 200, 80, 80)]]
        classes = [[0, 1]]
        scores = [[0.9, 0.85]]
        cls_scores = [[0.95, 0.88]]
        gt_boxes = [[(48, 48, 100, 100), (205, 205, 75, 75)]]
        gt_classes = [[0, 1]]
        map_iou_threshold = 0.5

        y_true, pred_scores = evaluate_detections(boxes, classes, scores, cls_scores, gt_boxes, gt_classes, map_iou_threshold, eval_type="class_scores")
        
        self.assertListEqual(y_true, [0, 1])
        self.assertListEqual(pred_scores, [0.95, 0.88])

    def test_combined(self):
        boxes = [[(50, 50, 100, 100), (200, 200, 80, 80)]]
        classes = [[0, 1]]
        scores = [[0.9, 0.85]]
        cls_scores = [[0.95, 0.88]]
        gt_boxes = [[(48, 48, 100, 100), (205, 205, 75, 75)]]
        gt_classes = [[0, 1]]
        map_iou_threshold = 0.5

        y_true, pred_scores = evaluate_detections(boxes, classes, scores, cls_scores, gt_boxes, gt_classes, map_iou_threshold, eval_type="combined")
        
        self.assertListEqual(y_true, [0, 1])
        self.assertListEqual(pred_scores, [0.855, 0.748])

    def test_no_overlap(self):
        boxes = [[(0, 0, 10, 10)]]
        classes = [[0]]
        scores = [[0.9]]
        cls_scores = [[0.95]]
        gt_boxes = [[(20, 20, 10, 10)]]
        gt_classes = [[0]]
        map_iou_threshold = 0.5

        y_true, pred_scores = evaluate_detections(boxes, classes, scores, cls_scores, gt_boxes, gt_classes, map_iou_threshold, eval_type="class_scores")
        
        self.assertTrue((len(y_true) == 1 and y_true[0] == 0) or (len(y_true) == 2 and y_true[0] == -1 and y_true[1] == 0))
        self.assertTrue((len(pred_scores) == 1 and pred_scores[0] == 0) or (len(pred_scores) == 2 and pred_scores[0] == 0 and pred_scores[1] == 0))

    def test_partial_overlap(self):
        boxes = [[(0, 0, 10, 10)]]
        classes = [[0]]
        scores = [[0.9]]
        cls_scores = [[0.95]]
        gt_boxes = [[(5, 5, 10, 10)]]
        gt_classes = [[0]]
        map_iou_threshold = 0.1

        y_true, pred_scores = evaluate_detections(boxes, classes, scores, cls_scores, gt_boxes, gt_classes, map_iou_threshold, eval_type="class_scores")
        
        self.assertListEqual(y_true, [0])
        self.assertListEqual(pred_scores, [0.95])

###############################################################################
# Tests for Calculate Precision Recall Curve
###############################################################################
class TestCalculatePrecisionRecallCurve(unittest.TestCase):
    
    def test_single_class(self):
        y_true = [0, 1, 0, 1, 0]
        pred_scores = [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.4, 0.6], [0.7, 0.3]]
        precision, recall, thresholds = calculate_precision_recall_curve(y_true, pred_scores, num_classes=2)

        expected_precision_class_0 = [1.0, 0.5, 0.6666666666666666, 0.5, 0.4]
        expected_recall_class_0 = [0.3333333333333333, 0.3333333333333333, 0.6666666666666666, 0.6666666666666666, 1.0]
        np.testing.assert_allclose(precision[0], expected_precision_class_0)
        np.testing.assert_allclose(recall[0], expected_recall_class_0)
    
    def test_multi_class(self):
        y_true = [0, 1, 2, 1, 0]
        pred_scores = [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.1, 0.85], [0.2, 0.7, 0.1], [0.8, 0.1, 0.1]]
        precision, recall, thresholds = calculate_precision_recall_curve(y_true, pred_scores, num_classes=3)

        expected_precision_class_0 = [1.0, 0.5, 0.6666666666666666, 0.5, 0.4]
        expected_recall_class_0 = [0.5, 0.5, 1.0, 1.0, 1.0]
        np.testing.assert_allclose(precision[0], expected_precision_class_0)
        np.testing.assert_allclose(recall[0], expected_recall_class_0)
    
    def test_no_predictions(self):
        y_true = []
        pred_scores = []
        try: 
            precision, recall, thresholds = calculate_precision_recall_curve(y_true, pred_scores, num_classes=3)
        except ValueError:
            self.assertTrue(True)
            return

        for i in range(3):
            self.assertEqual(precision[i], [])
            self.assertEqual(recall[i], [])
            self.assertEqual(thresholds[i], [])

if __name__ == '__main__':
    unittest.main()
