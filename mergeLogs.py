import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def merge_tensorboard_logs(logdir1, logdir2, logdir_combined):
    """
    Merge the TensorBoard logs from two directories into a single directory.
    Useful for combining logs from multiple runs of the same training job.
    Args:
        logdir1 (str): The directory containing the first set of logs.
        logdir2 (str): The directory containing the second set of logs.
        logdir_combined (str): The directory to which the combined logs should be written.
    """
    # Load the events from the TensorBoard logs
    event1 = EventAccumulator(logdir1)
    event2 = EventAccumulator(logdir2)
    event1.Reload()
    event2.Reload()

    # Create a writer for the combined logs
    writer = tf.summary.create_file_writer(logdir_combined)

    with writer.as_default():
        # Merge the scalar data
        for tag in event1.Tags()["scalars"]:
            for event in event1.Scalars(tag):
                tf.summary.scalar(tag, event.value, step=event.step)
        for tag in event2.Tags()["scalars"]:
            for event in event2.Scalars(tag):
                tf.summary.scalar(tag, event.value, step=event.step)

        # Merge the image data
        for tag in event1.Tags()["images"]:
            for event in event1.Images(tag):
                img = tf.image.decode_image(event.encoded_image_string)
                tf.summary.image(tag, [img], step=event.step)
        for tag in event2.Tags()["images"]:
            for event in event2.Images(tag):
                img = tf.image.decode_image(event.encoded_image_string)
                tf.summary.image(tag, [img], step=event.step)

    writer.close()


merge_tensorboard_logs(
    "./results/architecture/resnet101/scratch/res1/logsVal",
    "./results/architecture/resnet101/scratch/res2/logsVal",
    "./results/architecture/resnet101/scratch/logsVal",
)
