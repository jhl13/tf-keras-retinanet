import tensorflow as tf
from utils.eval import evaluate

class MyCustomCallback(tf.keras.callbacks.Callback):
    def __init__(
		self,
		file_writer,
        predict_model,
        generator,
        iou_threshold=0.5,
		score_threshold=0.05,
		max_detections=100,
	):
        super(MyCustomCallback, self).__init__()
        self.file_writer = file_writer
        self.predict_model = predict_model
        self.generator = generator
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.save_path = None
        self.global_step = 0

    def on_batch_end(self, batch, logs=None):
        self.global_step += 1
        with self.file_writer.as_default():
            tf.summary.scalar("classification_loss", logs["classification_loss"], step=self.global_step)
            tf.summary.scalar("bbox_regression_loss", logs["bbox_regression_loss"], step=self.global_step)
            l2_loss = logs["loss"] - logs["classification_loss"] - logs["bbox_regression_loss"]
            tf.summary.scalar("l2_loss", l2_loss, step=self.global_step)
            tf.summary.scalar("loss", logs["loss"], step=self.global_step)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Run evaluation.
        average_precisions, inference_time = evaluate(
            self.generator,
            self.predict_model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            save_path=self.save_path
        )

        # Compute per class average precision.
        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations ) in average_precisions.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                        self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)
        if self.weighted_average:
            self.mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
        else:
            self.mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)

        with self.file_writer.as_default():
            tf.summary.scalar("mAP", self.mean_ap, step=epoch)

        logs['mAP'] = self.mean_ap

        if self.verbose == 1:
            print('mAP: {:.4f}'.format(self.mean_ap))