from gdrive import upload_data
from rl import ReinforcementLearningAgent
from config import cfg, cfg_file
from datetime import datetime
import numpy as np
import os
import sys
import shutil


if __name__ == "__main__":

    start_model = None

    suffix = f"{cfg.k}x{cfg.k}"
    timestamp = datetime.now().isoformat()[:19]
    # Save data for possible later training

    timestamp = timestamp.replace(":", "-")

    if len(sys.argv) > 2:
        path = f"{sys.argv[2]}/data/{timestamp}_{suffix}"
    else:
        path = f"data/{timestamp}_{suffix}"

    data_folder_name = f"{timestamp}_{suffix}"

    os.makedirs(path)
    os.mkdir(f"{path}/models")
    os.mkdir(f"{path}/dataset")

    rlAgent = ReinforcementLearningAgent(
        path=path, starting_model_path=start_model)
    print("Starting training")
    rlAgent.train(file_suffix=suffix, n_parallel=1,
                  train_net=True, train_interval=1)

    print(f"Saving {len(rlAgent.x_train)} cases")

    with open(f"{path}/{timestamp}_{suffix}.npy", "wb") as f:
        np.save(f, rlAgent.states)
        np.save(f, rlAgent.y_train)
        np.save(f, rlAgent.y_train_value)

    shutil.copyfile(f"./configs/{cfg_file}", f"{path}/{cfg_file}")

    upload_data(data_folder_name)
