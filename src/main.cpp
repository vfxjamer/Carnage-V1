#include <GigaLearnCPP/Learner.h>

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

#include <cstdlib>
#include <iostream>

#include "NextoObs.h"
#include "CarnageRewards.h"
#include "ReplayStateSetter.h"

using namespace GGL;
using namespace RLGC;

// Carnage v1 - Phase 2 (Learning to Score)
// Gamma: 0.993

void StepCallback(
    Learner* learner,
    const std::vector<GameState>& states,
    Report& report
) {
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
                report.AddAvg(
                    "Player/Speed Towards Ball",
                    RS_MAX(0, player.vel.Dot(dirToBall))
                );
                report.AddAvg("Player/Boost", player.boost);

                if (player.ballTouchedStep)
                    report.AddAvg("Player/Touch Height", state.ball.pos.z);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    std::string meshesPath = "collision_meshes";
    std::string deviceStr = "cpu";
    int numGames = 1250;
    std::string saveDir = "checkpoints";
    std::string replayPath = "serialized_replays.bin";
    float replayProbability = 0.50f;
    std::string wandbProject = "";
    bool renderMode = false;
    float renderTimeScale = 1.0f;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc)
            deviceStr = argv[++i];
        else if (arg == "--games" && i + 1 < argc)
            numGames = atoi(argv[++i]);
        else if (arg == "--save-dir" && i + 1 < argc)
            saveDir = argv[++i];
        else if (arg == "--replay-path" && i + 1 < argc)
            replayPath = argv[++i];
        else if (arg == "--replay-prob" && i + 1 < argc)
            replayProbability = static_cast<float>(atof(argv[++i]));
        else if (arg == "--wandb" && i + 1 < argc)
            wandbProject = argv[++i];
        else if (arg == "--render")
            renderMode = true;
        else if (arg == "--render-timescale" && i + 1 < argc)
            renderTimeScale = static_cast<float>(atof(argv[++i]));
        else
            meshesPath = arg;
    }

    if (replayProbability < 0.f || replayProbability > 1.f) {
        std::cerr << "Invalid --replay-prob " << replayProbability
                  << "; expected a value in [0, 1].\n";
        return EXIT_FAILURE;
    }

    RocketSim::Init(meshesPath);

    auto EnvCreateFunc = [replayPath, replayProbability](int index) -> EnvCreateResult {
        std::vector<WeightedReward> rewards = {
            WeightedReward(new VelocityBallToGoalReward(), 7.0f),
            // GoalReward is zero-sum: +1 when our team scores, -1 when the opponent scores.
            WeightedReward(new GoalReward(-1.0f), 70.0f),
            WeightedReward(new TouchQualityReward(), 1.0f),
            WeightedReward(new SpeedTowardBallReward(), 5.0f),
            WeightedReward(new FaceBallReward(), 1.5f),
            WeightedReward(new AirReward(), 0.16f),
            WeightedReward(new PickupBoostReward(), 0.7f),
            WeightedReward(new TotalEnergyReward(), 1.0f)
        };

        std::vector<TerminalCondition*> terminalConditions = {
            new NoTouchCondition(10),
            new GoalScoreCondition()
        };

        auto arena = Arena::Create(GameMode::SOCCAR);
        arena->AddCar(Team::BLUE);
        arena->AddCar(Team::ORANGE);

        EnvCreateResult result = {};
        result.actionParser = new DefaultAction();
        result.obsBuilder = new NextoObs();
        result.stateSetter = new ReplayStateSetter(replayPath, replayProbability);
        result.terminalConditions = terminalConditions;
        result.rewards = rewards;
        result.arena = arena;
        return result;
    };

    LearnerConfig cfg = {};

    if (deviceStr == "cpu")
        cfg.deviceType = LearnerDeviceType::CPU;
    else if (deviceStr == "cuda")
        cfg.deviceType = LearnerDeviceType::GPU_CUDA;
    else
        cfg.deviceType = LearnerDeviceType::AUTO;

    cfg.tickSkip = 8;
    cfg.actionDelay = cfg.tickSkip - 1;
    cfg.numGames = numGames;
    cfg.checkpointFolder = saveDir;
    cfg.randomSeed = 123;

    int tsPerItr = 50'000;
    cfg.ppo.tsPerItr = tsPerItr;
    cfg.ppo.batchSize = tsPerItr;
    cfg.ppo.miniBatchSize = 25'000;
    cfg.ppo.epochs = 3;

    cfg.ppo.entropyScale = 0.05f;
    cfg.ppo.gaeGamma = 0.993;
    cfg.ppo.gaeLambda = 0.95;

    cfg.ppo.policyLR = 2e-4;
    cfg.ppo.criticLR = 2e-4;
    cfg.ppo.sharedHead = {};

    cfg.ppo.policy.layerSizes = {1024, 1024, 512, 512};
    cfg.ppo.critic.layerSizes = {1024, 1024, 512, 512};

    auto optim = ModelOptimType::ADAM;
    cfg.ppo.policy.optimType = optim;
    cfg.ppo.critic.optimType = optim;

    auto activation = ModelActivationType::RELU;
    cfg.ppo.policy.activationType = activation;
    cfg.ppo.critic.activationType = activation;

    cfg.ppo.policy.addLayerNorm = true;
    cfg.ppo.critic.addLayerNorm = true;

    cfg.tsPerSave = 5'000'000;

    if (!wandbProject.empty()) {
        const char* group = getenv("CARNAGE_WANDB_GROUP");
        const char* run = getenv("CARNAGE_WANDB_RUN");
        cfg.sendMetrics = true;
        cfg.metricsProjectName = wandbProject;
        cfg.metricsGroupName = group ? group : "Phase 2";
        cfg.metricsRunName = run ? run : "carnage-v1";
    } else {
        cfg.sendMetrics = false;
    }

    cfg.renderMode = renderMode;
    cfg.renderTimeScale = renderTimeScale;

    Learner* learner = new Learner(EnvCreateFunc, cfg, StepCallback);
    learner->Start();

    return EXIT_SUCCESS;
}
