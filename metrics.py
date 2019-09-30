from collections import defaultdict
import numpy as np
import torch
from catalyst.dl import Callback, RunnerState, MetricCallback, CallbackOrder


class JaccardMetricPerImage(Callback):
    """
    Jaccard metric callback which computes IoU metric per image and is aware that image is tiled.
    """

    def __init__(self, input_key: str = "targets", output_key: str = "logits", image_id_key: str = "image_id", prefix: str = "jaccard"):
        """
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        """
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.image_id_key = image_id_key
        self.scores_per_image = defaultdict(lambda: {'intersection': 0.0, 'union': 0.0})

    def on_loader_start(self, state):
        self.scores_per_image = defaultdict(lambda: {'intersection': 0.0, 'union': 0.0})

    def on_batch_end(self, state: RunnerState):
        image_id = state.input[self.image_id_key]
        outputs = state.output[self.output_key].detach()
        targets = state.input[self.input_key].detach()

        # Flatten images for easy computing IoU
        outputs = outputs.view(outputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # Binarize outputs as we don't want to compute soft-jaccard
        outputs = (outputs > 0).float()

        intersection = torch.sum(targets * outputs, dim=1)
        union = torch.sum(targets, dim=1) + torch.sum(outputs, dim=1)
        for img_id, img_intersection, img_union in zip(image_id, intersection, union):
            self.scores_per_image[img_id]['intersection'] += float(img_intersection)
            self.scores_per_image[img_id]['union'] += float(img_union)

    def on_loader_end(self, state):
        metric_name = self.prefix
        eps = 1e-7

        ious_per_image = []

        for image_id, values in self.scores_per_image.items():
            intersection = values['intersection']
            union = values['union']
            metric = intersection / (union - intersection + eps)
            ious_per_image.append(metric)


        metric = float(np.mean(ious_per_image))
        state.metrics.epoch_values[state.loader_name][metric_name] = metric

        # logger = _get_tensorboard_logger(state)
        # logger.add_scalar(f"{self.prefix}/all", metric, global_step=state.epoch)



class OptimalThreshold(Callback):
    """
    Callback that computes an optimal threshold for binarizing logits and theoretical IoU score at given threshold.
    """

    def __init__(self, input_key: str = "targets", output_key: str = "logits", image_id_key: str = "image_id", prefix: str = "optimal_threshold"):
        """
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        """
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.image_id_key = image_id_key
        self.thresholds = np.arange(-2, 2, 0.01)
        self.scores_per_image = defaultdict(lambda: {'intersection': np.zeros_like(self.thresholds), 'union': np.zeros_like(self.thresholds)})

    def on_loader_start(self, state):
        self.scores_per_image = defaultdict(lambda: {'intersection': np.zeros_like(self.thresholds), 'union': np.zeros_like(self.thresholds)})

    def on_batch_end(self, state: RunnerState):
        image_id = state.input[self.image_id_key]
        outputs = state.output[self.output_key].detach()
        targets = state.input[self.input_key].detach()

        # Flatten images for easy computing IoU
        outputs = outputs.view(outputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        for i, threshold in enumerate(self.thresholds):

            # Binarize outputs
            outputs_i = (outputs > threshold).float()

            intersection = torch.sum(targets * outputs_i, dim=1)
            union = torch.sum(targets, dim=1) + torch.sum(outputs_i, dim=1)

            for img_id, img_intersection, img_union in zip(image_id, intersection, union):
                self.scores_per_image[img_id]['intersection'][i] += float(img_intersection)
                self.scores_per_image[img_id]['union'][i] += float(img_union)

    def on_loader_end(self, state: RunnerState):
        eps = 1e-7

        ious_per_image = []

        for image_id, values in self.scores_per_image.items():
            intersection = values['intersection']
            union = values['union']
            metric = intersection / (union - intersection + eps)
            ious_per_image.append(metric)

        iou = np.mean(ious_per_image, axis=0)
        assert len(iou) == len(self.thresholds)

        threshold_index = np.argmax(iou)
        iou_at_threshold = iou[threshold_index]
        threshold_value = self.thresholds[threshold_index]

        state.metrics.epoch_values[state.loader_name][self.prefix] = float(threshold_value)
        state.metrics.epoch_values[state.loader_name][self.prefix + '/' + 'iou'] = float(iou_at_threshold)
