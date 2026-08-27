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

using namespace GGL;   // GigaLearn
using namespace RLGC;  // RLGymCPP

// Carnage v1 - Phase 2 (Learning to Score)
//
// Obs:
//   Nexto/Necto-style flat obs, exactly 94 dims (1v1)
//
// Network:
//   Policy [1024,1024,512,512]
//   Critic [1024,1024,512,512]
//   No shared head
//
// PPO:
//   3 epochs
//   50k timesteps/iteration
//   batch 50k
//   minibatch 25k
//   LR 2e-4
//   entropy 0.05
//
// Rewards:
//   VelocityBallToGoal  = 7.0
//   GoalReward          = 25.0
//   TouchBall           = 0.5
//   TouchAcceleration   = 1.5
//   TouchHeight         = 1.0
//   SpeedTowardBall     = 5.0
//   FaceBall            = 1.5
//   Air                 = 0.16
//   PickupBoost         = 0.7
//   TotalEnergy         = 1.0
//
// State setter:
//   50% real replay frames
//   50% RandomState fallback
//
// Replay:
//   serialized_replays.bin
//   override with --replay-path
//
// Terminal:
//   No touch for 10s
//   OR a goal is scored


void StepCallback(
    Learner* learner,
    const std::vector<GameState>& states,
    Report& report
) {
    // Only run expensive metrics on 1/4 of steps.
    bool doExpensiveMetrics = (rand() % 4) == 0;

    for (auto& state : states) {

        if (state.goalScored) {
            report.AddAvg(
                "Game/Goal Scored",
                1.f
            );

            report.AddAvg(
                "Game/Goal Speed",
                state.ball.vel.Length()
            );
        }

        if (doExpensiveMetrics) {

            for (auto& player : state.players) {

                report.AddAvg(
                    "Player/In Air Ratio",
                    !player.isOnGround
                );

                report.AddAvg(
                    "Player/Ball Touch Ratio",
                    player.ballTouchedStep
                );

                report.AddAvg(
                    "Player/Demoed Ratio",
                    player.isDemoed
                );

                report.AddAvg(
                    "Player/Speed",
                    player.vel.Length()
                );

                Vec dirToBall =
                    (state.ball.pos - player.pos).Normalized();

                report.AddAvg(
                    "Player/Speed Towards Ball",
                    RS_MAX(
                        0,
                        player.vel.Dot(dirToBall)
                    )
                );

                report.AddAvg(
                    "Player/Boost",
                    player.boost
                );

                if (player.ballTouchedStep) {

                    report.AddAvg(
                        "Player/Touch Height",
                        state.ball.pos.z
                    );
                }
            }
        }
    }
}


int main(
    int argc,
    char* argv[]
) {

    // ========================================================
    // CLI arguments
    // ========================================================

    std::string meshesPath =
        "collision_meshes";

    std::string deviceStr =
        "cpu";

    int numGames =
        1024;

    std::string saveDir =
        "checkpoints";

    std::string replayPath =
        "serialized_replays.bin";

    // 50% replay / 50% random
    float replayProbability =
        0.50f;

    std::string wandbProject =
        "";

    bool renderMode =
        false;

    float renderTimeScale =
        1.0f;


    // ========================================================
    // Parse arguments
    // ========================================================

    for (int i = 1; i < argc; i++) {

        std::string arg =
            argv[i];

        if (
            arg == "--device"
            && i + 1 < argc
        ) {
            deviceStr =
                argv[++i];
        }

        else if (
            arg == "--games"
            && i + 1 < argc
        ) {
            numGames =
                atoi(argv[++i]);
        }

        else if (
            arg == "--save-dir"
            && i + 1 < argc
        ) {
            saveDir =
                argv[++i];
        }

        else if (
            arg == "--replay-path"
            && i + 1 < argc
        ) {
            replayPath =
                argv[++i];
        }

        else if (
            arg == "--replay-prob"
            && i + 1 < argc
        ) {
            replayProbability =
                static_cast<float>(
                    atof(argv[++i])
                );
        }

        else if (
            arg == "--wandb"
            && i + 1 < argc
        ) {
            wandbProject =
                argv[++i];
        }

        else if (
            arg == "--render"
        ) {
            renderMode =
                true;
        }

        else if (
            arg == "--render-timescale"
            && i + 1 < argc
        ) {
            renderTimeScale =
                static_cast<float>(
                    atof(argv[++i])
                );
        }

        else {
            meshesPath =
                arg;
        }
    }


    // ========================================================
    // Validate replay probability
    // ========================================================

    if (
        replayProbability < 0.f
        ||
        replayProbability > 1.f
    ) {

        std::cerr
            << "Invalid --replay-prob "
            << replayProbability
            << "; expected a value in [0, 1].\n";

        return EXIT_FAILURE;
    }


    // ========================================================
    // Initialize RocketSim
    // ========================================================

    RocketSim::Init(
        meshesPath
    );


    // ========================================================
    // Environment creation
    // ========================================================

    auto EnvCreateFunc =
        [
            replayPath,
            replayProbability
        ](int index)
        -> EnvCreateResult {

        // ====================================================
        // Carnage v1 reward stack
        // ====================================================

        std::vector<WeightedReward> rewards = {

            // ------------------------------------------------
            // Scoring
            // ------------------------------------------------

            WeightedReward(
                new VelocityBallToGoalReward(),
                7.0f
            ),

            WeightedReward(
                new GoalReward(-1.0f),
                25.0f
            ),


            // ------------------------------------------------
            // Ball interaction
            // ------------------------------------------------

            // Small flat touch reward.
            //
            // This is intentionally low because TouchBallReward
            // can be repeatedly obtained during ball control.
            WeightedReward(
                new TouchBallReward(),
                0.5f
            ),

            // Rewards useful changes in ball velocity.
            //
            // This is intended to make the bot value the
            // QUALITY of a touch rather than merely touching.
            WeightedReward(
                new TouchAccelerationReward(),
                1.5f
            ),

            // Small touch-height signal.
            //
            // Kept low so aerial control is encouraged without
            // making high/roof touches an easy farming strategy.
            WeightedReward(
                new TouchHeightReward(),
                1.0f
            ),


            // ------------------------------------------------
            // Ball approach / orientation
            // ------------------------------------------------

            WeightedReward(
                new SpeedTowardBallReward(),
                5.0f
            ),

            WeightedReward(
                new FaceBallReward(),
                1.5f
            ),

            WeightedReward(
                new AirReward(),
                0.16f
            ),


            // ------------------------------------------------
            // Boost / energy
            // ------------------------------------------------

            WeightedReward(
                new PickupBoostReward(),
                0.7f
            ),

            WeightedReward(
                new TotalEnergyReward(),
                1.0f
            )
        };


        // ====================================================
        // Terminal conditions
        // ====================================================

        std::vector<TerminalCondition*> terminalConditions = {

            new NoTouchCondition(10),

            new GoalScoreCondition()
        };


        // ====================================================
        // 1v1 arena
        // ====================================================

        auto arena =
            Arena::Create(
                GameMode::SOCCAR
            );

        arena->AddCar(
            Team::BLUE
        );

        arena->AddCar(
            Team::ORANGE
        );


        // ====================================================
        // Environment result
        // ====================================================

        EnvCreateResult result = {};

        result.actionParser =
            new DefaultAction();

        result.obsBuilder =
            new NextoObs();

        result.stateSetter =
            new ReplayStateSetter(
                replayPath,
                replayProbability
            );

        result.terminalConditions =
            terminalConditions;

        result.rewards =
            rewards;

        result.arena =
            arena;

        return result;
    };


    // ========================================================
    // Learner configuration
    // ========================================================

    LearnerConfig cfg = {};


    // ========================================================
    // Device
    // ========================================================

    if (
        deviceStr == "cpu"
    ) {

        cfg.deviceType =
            LearnerDeviceType::CPU;
    }

    else if (
        deviceStr == "cuda"
    ) {

        cfg.deviceType =
            LearnerDeviceType::GPU_CUDA;
    }

    else {

        cfg.deviceType =
            LearnerDeviceType::AUTO;
    }


    // ========================================================
    // Environment timing
    // ========================================================

    cfg.tickSkip =
        8;

    cfg.actionDelay =
        cfg.tickSkip - 1;

    cfg.numGames =
        numGames;

    cfg.checkpointFolder =
        saveDir;


    // ========================================================
    // Random seed
    // ========================================================

    cfg.randomSeed =
        123;


    // ========================================================
    // PPO
    // ========================================================

    int tsPerItr =
        50'000;

    cfg.ppo.tsPerItr =
        tsPerItr;

    cfg.ppo.batchSize =
        tsPerItr;

    cfg.ppo.miniBatchSize =
        25'000;

    cfg.ppo.epochs =
        3;


    // ========================================================
    // Entropy / GAE
    // ========================================================

    cfg.ppo.entropyScale =
        0.05f;

    cfg.ppo.gaeGamma =
        0.99;

    cfg.ppo.gaeLambda =
        0.95;


    // ========================================================
    // Learning rates
    // ========================================================

    cfg.ppo.policyLR =
        2e-4;

    cfg.ppo.criticLR =
        2e-4;


    // ========================================================
    // No shared head
    // ========================================================

    cfg.ppo.sharedHead =
        {};


    // ========================================================
    // Policy network
    // ========================================================

    cfg.ppo.policy.layerSizes = {
        1024,
        1024,
        512,
        512
    };


    // ========================================================
    // Critic network
    // ========================================================

    cfg.ppo.critic.layerSizes = {
        1024,
        1024,
        512,
        512
    };


    // ========================================================
    // Optimizer
    // ========================================================

    auto optim =
        ModelOptimType::ADAM;

    cfg.ppo.policy.optimType =
        optim;

    cfg.ppo.critic.optimType =
        optim;


    // ========================================================
    // Activation
    // ========================================================

    auto activation =
        ModelActivationType::RELU;

    cfg.ppo.policy.activationType =
        activation;

    cfg.ppo.critic.activationType =
        activation;


    // ========================================================
    // Layer normalization
    // ========================================================

    cfg.ppo.policy.addLayerNorm =
        true;

    cfg.ppo.critic.addLayerNorm =
        true;


    // ========================================================
    // Checkpointing
    // ========================================================

    cfg.tsPerSave =
        5'000'000;


    // ========================================================
    // W&B
    // ========================================================

    if (
        !wandbProject.empty()
    ) {

        const char* group =
            getenv(
                "CARNAGE_WANDB_GROUP"
            );

        const char* run =
            getenv(
                "CARNAGE_WANDB_RUN"
            );

        cfg.sendMetrics =
            true;

        cfg.metricsProjectName =
            wandbProject;

        cfg.metricsGroupName =
            group
            ? group
            : "Phase 2";

        cfg.metricsRunName =
            run
            ? run
            : "carnage-v1";
    }

    else {

        cfg.sendMetrics =
            false;
    }


    // ========================================================
    // Rendering
    // ========================================================

    cfg.renderMode =
        renderMode;

    cfg.renderTimeScale =
        renderTimeScale;


    // ========================================================
    // Create learner
    // ========================================================

    Learner* learner =
        new Learner(
            EnvCreateFunc,
            cfg,
            StepCallback
        );


    // ========================================================
    // Start training
    // ========================================================

    learner->Start();

    return EXIT_SUCCESS;
}
