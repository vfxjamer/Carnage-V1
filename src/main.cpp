#include <GigaLearnCPP/Learner.h>

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

#include <cstdlib>

#include "NextoObs.h"
#include "CarnageRewards.h"

using namespace GGL; // GigaLearn
using namespace RLGC; // RLGymCPP

// Carnage v1 - Phase 1 (0 - 5B steps)
//   Obs:         Nexto/Necto-style flat obs, exactly 94 dims (1v1)
//   Network:     Policy [1024,1024,512,512], Critic [1024,1024,512,512], no shared head
//   PPO:         3 epochs, ts/itr 50k, batch 50k, minibatch 25k, LR 2e-4, entropy 0.05
//   Rewards:     Touch(50), SpeedTowardBall(5), FaceBall(1), Air(0.15) - no goal reward
//   Metrics:     optional wandb via --wandb <project> (logs the full Report every iteration:
//                Player/* step metrics, Rewards/* per-reward curves, Game/* events)
//   Terminal:    No touch for 10s, or a goal is scored
//   State set:   RandomState (random ball/car speed, cars can spawn in the air)
//   Device:      GPU (CUDA) on Colab T4

void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report) {
	// To prevent expensive metrics from eating at performance, we will only run them on 1/4th of steps
	bool doExpensiveMetrics = (rand() % 4) == 0;

	for (auto& state : states) {
		if (state.goalScored) {
			report.AddAvg("Game/Goal Scored", 1.f);
			report.AddAvg("Game/Goal Speed", state.ball.vel.Length());
		}

		if (doExpensiveMetrics) {
			for (auto& player : state.players) {
				report.AddAvg("Player/In Air Ratio", !player.isOnGround);
				report.AddAvg("Player/Ball Touch Ratio", player.ballTouchedStep);
				report.AddAvg("Player/Demoed Ratio", player.isDemoed);

				report.AddAvg("Player/Speed", player.vel.Length());
				Vec dirToBall = (state.ball.pos - player.pos).Normalized();
				report.AddAvg("Player/Speed Towards Ball", RS_MAX(0, player.vel.Dot(dirToBall)));

				report.AddAvg("Player/Boost", player.boost);

				if (player.ballTouchedStep)
					report.AddAvg("Player/Touch Height", state.ball.pos.z);
			}
		}
	}
}

int main(int argc, char* argv[]) {
	// CLI args:
	//   positional: collision meshes folder (default: "collision_meshes")
	//   --device cpu|cuda|auto   (default: cuda)
	//   --games N                parallel games (default: 256)
	//   --save-dir <path>        checkpoint folder (default: "checkpoints")
	//   --wandb <project>        enable wandb metrics (project name). Run/group/run-id
	//                            come from env: CARNAGE_WANDB_GROUP, CARNAGE_WANDB_RUN,
	//                            CARNAGE_WANDB_RUN_ID. WANDB_API_KEY must be set in env.
	std::string meshesPath = "collision_meshes";
	std::string deviceStr = "cuda";
	int numGames = 256;
	std::string saveDir = "checkpoints";
	std::string wandbProject = "";

	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		if (arg == "--device" && i + 1 < argc) {
			deviceStr = argv[++i];
		} else if (arg == "--games" && i + 1 < argc) {
			numGames = atoi(argv[++i]);
		} else if (arg == "--save-dir" && i + 1 < argc) {
			saveDir = argv[++i];
		} else if (arg == "--wandb" && i + 1 < argc) {
			wandbProject = argv[++i];
		} else {
			meshesPath = arg;
		}
	}

	// Initialize RocketSim with collision meshes
	RocketSim::Init(meshesPath);

	// Create the RLGymCPP environment for each of our games
	auto EnvCreateFunc = [](int index) -> EnvCreateResult {
		std::vector<WeightedReward> rewards = {
			WeightedReward(new TouchBallReward(), 50),
			WeightedReward(new SpeedTowardBallReward(), 5),
			WeightedReward(new FaceBallReward(), 1),
			WeightedReward(new AirReward(), 0.15f)
		};

		std::vector<TerminalCondition*> terminalConditions = {
			new NoTouchCondition(10), // 10s without touching the ball ends the episode
			new GoalScoreCondition()  // A goal ends the episode (but gives no reward)
		};

		// 1v1 arena
		auto arena = Arena::Create(GameMode::SOCCAR);
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);

		EnvCreateResult result = {};
		result.actionParser = new DefaultAction();
		result.obsBuilder = new NextoObs();
		result.stateSetter = new RandomState(true, true, false);
		result.terminalConditions = terminalConditions;
		result.rewards = rewards;

		result.arena = arena;

		return result;
	};

	// Make configuration for the learner
	LearnerConfig cfg = {};

	if (deviceStr == "cpu") {
		cfg.deviceType = LearnerDeviceType::CPU;
	} else if (deviceStr == "cuda") {
		cfg.deviceType = LearnerDeviceType::GPU_CUDA;
	} else {
		cfg.deviceType = LearnerDeviceType::AUTO;
	}

	cfg.tickSkip = 8;
	cfg.actionDelay = cfg.tickSkip - 1; // Normal value in other RLGym frameworks

	cfg.numGames = numGames;
	cfg.checkpointFolder = saveDir;

	// Leave this empty to use a random seed each run
	// The random seed can have a strong effect on the outcome of a run
	cfg.randomSeed = 123;

	int tsPerItr = 50'000;
	cfg.ppo.tsPerItr = tsPerItr;
	cfg.ppo.batchSize = tsPerItr;
	cfg.ppo.miniBatchSize = 25'000;

	cfg.ppo.epochs = 3;

	// Literal value chosen for Phase 1
	cfg.ppo.entropyScale = 0.05f;

	// Rate of reward decay
	cfg.ppo.gaeGamma = 0.99;
	cfg.ppo.gaeLambda = 0.95;

	cfg.ppo.policyLR = 2e-4;
	cfg.ppo.criticLR = 2e-4;

	cfg.ppo.sharedHead = {}; // No shared head: policy and critic are fully separate

	cfg.ppo.policy.layerSizes = { 1024, 1024, 512, 512 };
	cfg.ppo.critic.layerSizes = { 1024, 1024, 512, 512 };

	auto optim = ModelOptimType::ADAM;
	cfg.ppo.policy.optimType = optim;
	cfg.ppo.critic.optimType = optim;

	auto activation = ModelActivationType::RELU;
	cfg.ppo.policy.activationType = activation;
	cfg.ppo.critic.activationType = activation;

	cfg.ppo.policy.addLayerNorm = true;
	cfg.ppo.critic.addLayerNorm = true;

	// Save a checkpoint every 5M timesteps (5_000_000) into --save-dir.
	// Old checkpoints are pruned by checkpointsToKeep (default 8).
	cfg.tsPerSave = 5'000'000;

	// wandb metrics (opt-in via --wandb). All Report metrics are sent every
	// iteration: Player/* step metrics, Rewards/* per-reward curves, Game/* events.
	// Resuming a run happens automatically: the reframe restores the previous
	// run ID from stats.json and wandb's resume="allow" continues that run.
	if (!wandbProject.empty()) {
		const char* group = getenv("CARNAGE_WANDB_GROUP");
		const char* run = getenv("CARNAGE_WANDB_RUN");

		cfg.sendMetrics = true;
		cfg.metricsProjectName = wandbProject;
		cfg.metricsGroupName = group ? group : "Phase 1";
		cfg.metricsRunName = run ? run : "carnage-v1";
	}
	else {
		cfg.sendMetrics = false; // No metric receiver running (console metrics still print)
	}

	// Make the learner with the environment creation function and the config we just made
	Learner* learner = new Learner(EnvCreateFunc, cfg, StepCallback);

	// Start learning!
	learner->Start();

	return EXIT_SUCCESS;
}