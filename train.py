import art
import weave
from agent import run_agent_and_score
from load_data import load_scenarios
from local_db import generate_database
from art.local import LocalBackend
from art.utils import iterate_dataset
import dotenv


ROLLOUTS_PER_GROUP = 4
NUM_EPOCHS = 3
GROUPS_PER_STEP = 12
VALIDATION_NUM_SCENARIOS = 100
TRAINING_NUM_SCENARIOS = 1000

dotenv.load_dotenv()
weave.init("side-project/rl-agent")


async def train(model): 
    # Generate database
    generate_database(languages=["python", "go"], overwrite=True)

    # Get training scenarios
    training_data = load_scenarios(
        "JamesSED/synthetic_QA_code_search_net",
        "train",
        limit=TRAINING_NUM_SCENARIOS,
        shuffle=True,
    )

    # Define trainable model
    model = art.TrainableModel(
        base_model=model,
        project="rl-agent",
        name="model_1",
    )

    await model.register(LocalBackend())

    # Register model with vllm backend instance with LoRA
    training_iterator = iterate_dataset(
        training_data,
        groups_per_step=GROUPS_PER_STEP,
        num_epochs=NUM_EPOCHS,
        initial_step=await model.get_step(),
    )

    for batch, epoch, global_step, epoch_step in training_iterator:
        groups = []
        for scenario in batch:
            groups.append(
                art.TrajectoryGroup(
                    run_agent_and_score(model, scenario)
                    for _ in range(ROLLOUTS_PER_GROUP)
                )
            )

        finished_groups = await art.gather_trajectory_groups(groups)
        await model.train(finished_groups)


if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    model = args.model
    print(f"Args: {args}")
    asyncio.run(train(model)) 

