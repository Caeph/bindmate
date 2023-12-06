import numpy as np

bases = list("ACGT")


##### background creation
def create_background(k, background_type, bg_size, **kwargs):
    if background_type == "random":
        bgkmers = np.array(["".join(np.random.choice(bases, size=k)) for _ in range(bg_size)])
    elif background_type == "sampled":
        # TODO, input file is in **kwargs
        raise NotImplementedError("No other background type is implemented.")
    else:
        raise NotImplementedError("No other background type is implemented.")

    return bgkmers
