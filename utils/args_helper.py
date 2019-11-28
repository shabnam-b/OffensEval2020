"""

    Author: Sajad
    Desc: This file is a helper file for receiving arguments
    (either training/eval) from the commandline.

"""

def parse_training_parameters(flags):
    FLAGS = flags.FLAGS

    flags.DEFINE_integer(
        "embedding_size", None,
        "GloVe word embedding size --if used'"
    )

    flags.DEFINE_integer(
        "num_layers", 2,
        "Number of lstm layer'"
    )

    flags.DEFINE_integer(
        "num_hidden", 200,
        "Number of hidden units"
    )

    flags.DEFINE_integer(
        "num_epochs", 10,
        "Number of training epochs"
    )

    flags.DEFINE_integer(
        "max_doc_length", 115,
        "Maximum document length"
    )

    flags.DEFINE_integer(
        "batch_size", 10,
        "Batch size"
    )

    flags.DEFINE_integer(
        "dropout", 0.5,
        "Probability for dropout"
    )

    flags.DEFINE_integer(
        "learning_rate", 0.001,
        "Learning rate."
    )

    return FLAGS