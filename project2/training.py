from os import times
from rl import ReinforcementLearningAgent
from config import Config
from datetime import datetime
import numpy as np

suffix = f"{Config.k}x{Config.k}"


if __name__ == '__main__':
    rlAgent = ReinforcementLearningAgent()

    rlAgent.train(file_suffix=suffix, n_parallel=10)

    print(f"Saving {len(rlAgent.x_train)} cases")

    # Because Windows cannot have : in filename ._.
    timestamp = datetime.now().isoformat()[:19]
    # Save data for possible later training
    timestamp = timestamp.replace(":", "-")
    with open(f'project2/data/{timestamp}_{suffix}.npy', 'wb') as f:
        np.save(f, rlAgent.x_train)
        np.save(f, rlAgent.y_train)
        np.save(f, rlAgent.y_train_value)
