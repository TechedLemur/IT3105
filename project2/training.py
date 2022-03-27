from rl import ReinforcementLearningAgent
from config import Config
from datetime import datetime
import numpy as np
import os
import sys



if __name__ == "__main__":

    suffix = f"{Config.k}x{Config.k}"
    timestamp = datetime.now().isoformat()[:19]
    # Save data for possible later training
    timestamp = timestamp.replace(":", "-")

    if len(sys.argv) > 1:
        path = f"{sys.argv[1]}/data/{timestamp}_{suffix}"
    else:
        path = f"data/{timestamp}_{suffix}"

    os.mkdir(path)
    os.mkdir(f"{path}/models")
    os.mkdir(f"{path}/dataset")

    rlAgent = ReinforcementLearningAgent(path=path)
    rlAgent.train(file_suffix=suffix, n_parallel=1)

    print(f"Saving {len(rlAgent.x_train)} cases")

    with open(f"{path}/dataset/{timestamp}_{suffix}.npy", "wb") as f:
        np.save(f, rlAgent.states)
        np.save(f, rlAgent.y_train)
        np.save(f, rlAgent.y_train_value)
