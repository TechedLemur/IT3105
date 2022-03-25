from rl import ReinforcementLearningAgent
from config import Config


suffix = f"{Config.k}x{Config.k}"


if __name__ == '__main__':
    rlAgent = ReinforcementLearningAgent()

    rlAgent.train(file_suffix=suffix)

    print(f"Length of train set: {len(rlAgent.x_train)}")
