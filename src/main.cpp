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
//   Obs:         Nexto/Necto-style 94-dim (NextoObs)
//   Network:     Policy [1024,1024,512,512], Critic [1024,1024,512,512], no shared head
//   PPO:         3 epochs, ts/itr 50k, batch 50k, minibatch 25k, LR 2e-4, entropy 0.05
//   Rewards:     Touch(50), SpeedTowardBall(5), FaceBall(1), Air(0.15) - no goal reward
//   Terminal:    No touch for 10s, or a goal is scored
//   State set:   RandomState (random ball/car speed, cars can spawn in the air)
//   Device:      GPU (CUDA) - target machine is a Colab T4

// Create the RLGymCPP environment for each of our games
EnvCreateResult EnvCreateFunc(int index) {
	std::vector<WeightedReward> rewards = {
		{ new TouchBallReward(), 50 },
		{ new SpeedTowardBallReward(), 5 },
		{ new FaceBallReward(), 1 },
		{ new AirReward(), 0.15f }
	};

	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(10), // 10s without touching the ball ends the episode
		new GoalScoreCondition()  // A goal ends the episode (but gives no reward)
	};

	// Make the arena
	int playersPerTeam = 1;
	auto arena = Arena::Create(GameMode::SOCCAR);
	for (int i = 0; i < playersPerTeam; i++) {
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
	}

	EnvCreateResult result = {};
	result.actionParser = new DefaultAction();
	result.obsBuilder = new NextoObs();
	result.stateSetter = new RandomState(true, true, false);
	result.terminalConditions = terminalConditions;
	result.rewards = rewards;

	result.arena = arena;

	return result;
}

void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report) {
	// To prevent expensive metrics from eating at performance, we will only run them on 1/4th of steps
	bool doExpensiveMetrics = (rand() % 4) == 0;

	for (auto& state : states) {
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

		if (state.goalScored)
			report.AddAvg("Game/Goal Speed", state.ball.vel.Length());
	}
}

int main(int argc, char* argv[]) {
	// CLI args:
	//   positional: collision meshes folder (default: "collision_meshes")
	//   --device cpu|cuda|auto   (default: cuda)
	//   --games N                (default: 256)
	//   --save-dir <path>        checkpoint folder (default: "checkpoints")
	std::string meshesPath = "collision_meshes";
	std::string deviceStr = "cuda";
	int numGames = 256;
	std::string saveDir = "checkpoints";

	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		if (arg == "--device" && i + 1 < argc) {
			deviceStr = argv[++i];
		} else if (arg == "--games" && i + 1 < argc) {
			numGames = atoi(argv[++i]);
		} else if (arg == "--save-dir" && i + 1 < argc) {
			saveDir = argv[++i];
		} else {
			meshesPath = arg;
		}
	}

	// Initialize RocketSim with collision meshes
	RocketSim::Init(meshesPath);

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

	// This reads more like a big network but trains fine
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

	// Save a checkpoint each iteration (every 50k steps)
	cfg.tsPerSave = 0;

	cfg.sendMetrics = false; // No metric receiver running (console metrics still print)

	// Make the learner with the environment creation function and the config we just made
	Learner* learner = new Learner(EnvCreateFunc, cfg, StepCallback);

	// Start learning!
	learner->Start();

	return EXIT_SUCCESS;
}