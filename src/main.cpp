#include <GigaLearnCPP/Learner.h>

#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "CarnageRewards.h"
#include "CarnageTraining.h"
#include "NextoObs.h"

using namespace GGL;
using namespace RLGC;

namespace {

struct Options {
    std::string meshesPath = "collision_meshes";
    std::string device = "auto";
    std::string deviceLayout = "single";
    int policyCudaDevice = 0;
    int criticCudaDevice = 1;
    int actionDelay = Carnage::DEFAULT_ACTION_DELAY;
    int games = 300;
    std::filesystem::path checkpointRoot = "checkpoints";
    std::string resume = "auto";
    std::string wandbProject;
    bool approveP0 = false;
    bool scoringConfirmed = false;
    int scoringRollout = 200'000;
    bool render = false;
    bool smokeTest = false;
    float renderTimeScale = 1.0f;
};

std::string RequireValue(int& index, int argc, char** argv, const std::string& option) {
    if (index + 1 >= argc)
        throw std::invalid_argument("Missing value for " + option);
    return argv[++index];
}

Options ParseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--device")
            options.device = RequireValue(index, argc, argv, argument);
        else if (argument == "--device-layout")
            options.deviceLayout = RequireValue(index, argc, argv, argument);
        else if (argument == "--policy-cuda-device")
            options.policyCudaDevice = std::stoi(RequireValue(index, argc, argv, argument));
        else if (argument == "--critic-cuda-device")
            options.criticCudaDevice = std::stoi(RequireValue(index, argc, argv, argument));
        else if (argument == "--action-delay")
            options.actionDelay = std::stoi(RequireValue(index, argc, argv, argument));
        else if (argument == "--games")
            options.games = std::stoi(RequireValue(index, argc, argv, argument));
        else if (argument == "--checkpoint-root" || argument == "--save-dir")
            options.checkpointRoot = RequireValue(index, argc, argv, argument);
        else if (argument == "--resume")
            options.resume = RequireValue(index, argc, argv, argument);
        else if (argument == "--wandb")
            options.wandbProject = RequireValue(index, argc, argv, argument);
        else if (argument == "--approve-p0")
            options.approveP0 = true;
        else if (argument == "--scoring-confirmed")
            options.scoringConfirmed = true;
        else if (argument == "--scoring-rollout")
            options.scoringRollout = std::stoi(RequireValue(index, argc, argv, argument));
        else if (argument == "--render")
            options.render = true;
        else if (argument == "--smoke-test")
            options.smokeTest = true;
        else if (argument == "--render-timescale")
            options.renderTimeScale = std::stof(RequireValue(index, argc, argv, argument));
        else if (argument == "--help") {
            std::cout
                << "Carnage fresh Phase 0 -> Phase 1 trainer\n"
                << "  --checkpoint-root PATH --resume auto|none|PATH\n"
                << "  --approve-p0 --scoring-confirmed --scoring-rollout 200000|300000\n"
                << "  --action-delay 0..8 --games N --device cpu|cuda|auto\n"
                << "  --device-layout single|split --policy-cuda-device N --critic-cuda-device N\n"
                << "  --smoke-test (one small integration iteration)\n";
            std::exit(EXIT_SUCCESS);
        } else if (argument.starts_with("--")) {
            throw std::invalid_argument("Unknown option: " + argument);
        } else {
            options.meshesPath = argument;
        }
    }

    if (options.actionDelay < 0 || options.actionDelay > Carnage::TICK_SKIP)
        throw std::invalid_argument("--action-delay must be between 0 and tick skip 8");
    if (options.games <= 0)
        throw std::invalid_argument("--games must be positive");
    if (options.device != "cpu" && options.device != "cuda" && options.device != "auto")
        throw std::invalid_argument("--device must be cpu, cuda, or auto");
    if (options.deviceLayout != "single" && options.deviceLayout != "split")
        throw std::invalid_argument("--device-layout must be single or split");
    if (options.scoringRollout != 200'000 && options.scoringRollout != 300'000)
        throw std::invalid_argument("--scoring-rollout must be 200000 or 300000");
    if (options.deviceLayout == "split" && options.device == "cpu")
        throw std::invalid_argument("Split actor/critic layout requires CUDA");
    return options;
}

void ApplySettings(Learner* learner, const Carnage::RuntimeSettings& settings) {
    constexpr std::size_t REWARD_COUNT = 6;
    const float weights[REWARD_COUNT] = {
        static_cast<float>(settings.rewards.touch),
        static_cast<float>(settings.rewards.speedTowardBall),
        static_cast<float>(settings.rewards.faceBall),
        static_cast<float>(settings.rewards.air),
        static_cast<float>(settings.rewards.velocityBallToGoal),
        static_cast<float>(settings.rewards.goal)
    };
    for (auto& arenaRewards : learner->envSet->rewards) {
        if (arenaRewards.size() != REWARD_COUNT)
            throw std::runtime_error("Unexpected reward layout; refusing an unsafe curriculum update");
        for (std::size_t index = 0; index < REWARD_COUNT; ++index)
            arenaRewards[index].weight = weights[index];
    }

    learner->ApplyPPOSettings(
        settings.rolloutSize,
        settings.batchSize,
        settings.minibatchSize,
        settings.epochs,
        static_cast<float>(settings.policyLearningRate),
        static_cast<float>(settings.criticLearningRate)
    );
}

void AddCurriculumMetrics(
    const Carnage::CurriculumState& state,
    const Carnage::RuntimeSettings& settings,
    std::uint64_t totalTimesteps,
    Report& report
) {
    report["curriculum/phase"] = static_cast<int>(settings.phase);
    report["curriculum/transition_progress"] = settings.transitionProgress;
    report["curriculum/transition_start_step"] = state.transitionStartStep;
    report["curriculum/transition_end_step"] = state.p0Approved
        ? state.transitionStartStep + state.transitionLength : 0;
    report["curriculum/actual_total_timesteps"] = totalTimesteps;
    report["curriculum/awaiting_p0_approval"] = state.awaitingP0Approval;
    report["curriculum/p0_approved"] = state.p0Approved;
    report["curriculum/scoring_confirmed"] = state.scoringConfirmed;
    report["curriculum/next_rotating_checkpoint_step"] = state.nextRotatingCheckpointStep;
    report["rewards/touch_weight"] = settings.rewards.touch;
    report["rewards/speed_toward_ball_weight"] = settings.rewards.speedTowardBall;
    report["rewards/face_ball_weight"] = settings.rewards.faceBall;
    report["rewards/air_weight"] = settings.rewards.air;
    report["rewards/velocity_ball_to_goal_weight"] = settings.rewards.velocityBallToGoal;
    report["rewards/goal_weight"] = settings.rewards.goal;
    report["ppo/policy_lr"] = settings.policyLearningRate;
    report["ppo/critic_lr"] = settings.criticLearningRate;
    report["ppo/rollout_size"] = settings.rolloutSize;
    report["ppo/batch_size"] = settings.batchSize;
    report["ppo/minibatch_size"] = settings.minibatchSize;
    report["ppo/epochs"] = settings.epochs;
}

void StepMetrics(Learner*, const std::vector<GameState>& states, Report& report) {
    for (const GameState& state : states) {
        if (state.goalScored) {
            report.AddAvg("game/goals", 1.0);
            report.AddAvg("game/goal_speed", state.ball.vel.Length());
        }
        for (const Player& player : state.players) {
            report.AddAvg("player/touch_rate", player.ballTouchedStep);
            report.AddAvg("player/in_air_rate", !player.isOnGround);
            report.AddAvg("player/speed", player.vel.Length());
            report.AddAvg("player/boost", player.boost);
        }
    }
}

std::string CurrentWandbRunId(const Learner* learner) {
    if (learner->metricSender)
        return learner->metricSender->curRunID;
    return learner->runID;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        Carnage::InstallSafeStopHandlers();

        const auto compatibility = Carnage::CompatibilitySignature(options.actionDelay);
        Carnage::CheckpointStore checkpointStore(options.checkpointRoot);
        Carnage::CurriculumState curriculum;
        std::optional<std::filesystem::path> resumePath;

        if (options.resume == "auto") {
            std::vector<std::string> diagnostics;
            resumePath = checkpointStore.FindLatestValid(compatibility, &diagnostics);
            for (const std::string& diagnostic : diagnostics)
                std::cerr << "Skipping checkpoint: " << diagnostic << '\n';
        } else if (options.resume != "none") {
            const std::filesystem::path requested = options.resume;
            const auto validation = checkpointStore.Validate(requested, compatibility);
            if (!validation.valid)
                throw std::runtime_error("Requested checkpoint is invalid: " + validation.error);
            resumePath = requested;
        }
        if (resumePath) {
            curriculum = checkpointStore.ReadCurriculum(*resumePath);
            std::cout << "Resume candidate: " << resumePath->string() << '\n';
        }

        if (curriculum.awaitingP0Approval && !options.approveP0) {
            std::cerr
                << "Phase 0 capability review is required. Inspect 200M_phase0_final, then restart "
                << "with --approve-p0 to continue from the exact saved state.\n";
            return EXIT_FAILURE;
        }
        if (options.approveP0 && !curriculum.awaitingP0Approval)
            throw std::invalid_argument("--approve-p0 was supplied, but the resumed checkpoint is not awaiting approval");

        RocketSim::Init(options.meshesPath);

        auto environmentFactory = [](int) -> EnvCreateResult {
            auto arena = Arena::Create(GameMode::SOCCAR);
            arena->AddCar(Team::BLUE);
            arena->AddCar(Team::ORANGE);

            EnvCreateResult result = {};
            result.arena = arena;
            result.actionParser = new DefaultAction();
            result.obsBuilder = new NextoObs();
            result.stateSetter = new RandomState(true, true, false);
            result.terminalConditions = {
                new NoTouchCondition(10),
                new GoalScoreCondition()
            };
            // The order is an invariant used for safe boundary weight updates.
            result.rewards = {
                WeightedReward(new TouchBallReward(), 50.0f),
                WeightedReward(new SpeedTowardBallReward(), 5.0f),
                WeightedReward(new FaceBallReward(), 1.0f),
                WeightedReward(new AirReward(), 0.15f),
                WeightedReward(new VelocityBallToGoalReward(), 0.0f),
                WeightedReward(new GoalReward(-1.0f), 0.0f)
            };
            return result;
        };

        LearnerConfig config = {};
        config.deviceType = options.device == "cpu" ? LearnerDeviceType::CPU
            : options.device == "cuda" ? LearnerDeviceType::GPU_CUDA
            : LearnerDeviceType::AUTO;
        config.policyCudaDevice = options.policyCudaDevice;
        config.criticCudaDevice = options.criticCudaDevice;
        config.splitActorCriticDevices = options.deviceLayout == "split";
        config.tickSkip = Carnage::TICK_SKIP;
        config.actionDelay = options.actionDelay;
        config.numGames = options.games;
        config.checkpointFolder = options.checkpointRoot;
        config.checkpointLoadFolder = resumePath.value_or(std::filesystem::path{});
        config.autoLoadLatestCheckpoint = false;
        config.autoSaveCheckpoints = false;
        config.enableQuitKey = false;
        config.checkpointsToKeep = -1;
        config.randomSeed = 123;

        const auto startupSettings = Carnage::SettingsFor(curriculum, resumePath
            ? checkpointStore.ReadMetadata(*resumePath).at("total_timesteps").get<std::uint64_t>()
            : 0);
        config.ppo.tsPerItr = startupSettings.rolloutSize;
        config.ppo.batchSize = startupSettings.batchSize;
        config.ppo.miniBatchSize = 25'000;
        config.ppo.epochs = 3;
        config.ppo.gaeGamma = 0.99f;
        config.ppo.gaeLambda = 0.95f;
        config.ppo.maskEntropy = false;
        config.ppo.entropyScale = static_cast<float>(Carnage::RuntimeEntropyCoefficient(
            Carnage::GUIDE_ENTROPY_COEFFICIENT,
            Carnage::ACTION_COUNT,
            Carnage::EntropyNormalization::DivideByLogActionCount
        ));
        config.ppo.policyLR = static_cast<float>(startupSettings.policyLearningRate);
        config.ppo.criticLR = static_cast<float>(startupSettings.criticLearningRate);
        config.ppo.sharedHead = {};
        config.ppo.policy.layerSizes = {2048, 2048, 1024, 1024};
        config.ppo.critic.layerSizes = {2048, 2048, 1024, 1024};
        config.ppo.policy.optimType = ModelOptimType::ADAM;
        config.ppo.critic.optimType = ModelOptimType::ADAM;
        config.ppo.policy.activationType = ModelActivationType::RELU;
        config.ppo.critic.activationType = ModelActivationType::RELU;
        config.ppo.policy.addLayerNorm = true;
        config.ppo.critic.addLayerNorm = true;

        if (options.smokeTest) {
            config.ppo.tsPerItr = 256;
            config.ppo.batchSize = 256;
            config.ppo.miniBatchSize = 256;
        }
        config.tsPerSave = Carnage::ROTATING_INTERVAL;
        config.renderMode = options.render;
        config.renderTimeScale = options.renderTimeScale;

        if (!options.wandbProject.empty()) {
            const char* group = std::getenv("CARNAGE_WANDB_GROUP");
            const char* run = std::getenv("CARNAGE_WANDB_RUN");
            config.sendMetrics = true;
            config.metricsProjectName = options.wandbProject;
            config.metricsGroupName = group ? group : "Phase 0 to Phase 1";
            config.metricsRunName = run ? run : "carnage-v1-fresh";
        } else {
            config.sendMetrics = false;
        }

        std::unique_ptr<Learner> learner;
        auto saveCheckpoint = [&](const std::string& label, std::uint64_t nominalMilestone) {
            const auto saved = checkpointStore.SaveAtomic(
                label,
                learner->totalTimesteps,
                learner->totalIterations,
                nominalMilestone,
                curriculum,
                compatibility,
                CurrentWandbRunId(learner.get()),
                [&](const std::filesystem::path& path) {
                    learner->SaveStats(path / "RUNNING_STATS.json");
                    learner->SavePPOTo(path);
                }
            );
            checkpointStore.PruneRotating(8);
            std::cout << "Published checkpoint: " << saved.string()
                      << " (nominal=" << nominalMilestone
                      << ", actual=" << learner->totalTimesteps << ")\n";
        };

        auto iterationCallback = [&](Learner* currentLearner, Report& report) {
            const Carnage::BoundaryDecision boundary =
                Carnage::ProcessSafeBoundary(curriculum, currentLearner->totalTimesteps);
            const Carnage::RuntimeSettings settings =
                Carnage::SettingsFor(curriculum, currentLearner->totalTimesteps);
            ApplySettings(currentLearner, settings);
            AddCurriculumMetrics(curriculum, settings, currentLearner->totalTimesteps, report);
            report["runtime/action_delay_ticks"] = options.actionDelay;
            report["runtime/action_delay_ms"] = Carnage::ActionDelaySeconds(options.actionDelay) * 1000.0;
            report["runtime/device_layout_split"] = options.deviceLayout == "split";

            const bool requestedStop = Carnage::StopRequested();
            if (boundary.save || requestedStop || options.smokeTest) {
                saveCheckpoint(
                    boundary.save ? boundary.checkpointLabel
                        : options.smokeTest ? "smoke_test" : "safe_stop",
                    boundary.nominalMilestone
                );
                report["checkpoint/actual_saved_timestep"] = currentLearner->totalTimesteps;
                report["checkpoint/nominal_milestone"] = boundary.nominalMilestone;
                report["checkpoint/valid"] = 1;
            }
            return boundary.stop || requestedStop || options.smokeTest;
        };

        learner = std::make_unique<Learner>(
            environmentFactory,
            config,
            StepMetrics,
            iterationCallback
        );
        if (learner->obsSize != Carnage::OBSERVATION_SIZE || learner->numActions != Carnage::ACTION_COUNT)
            throw std::runtime_error("NextoObs/DefaultAction compatibility invariant failed");

        if (options.approveP0) {
            Carnage::ApprovePhase0(curriculum, learner->totalTimesteps);
            ApplySettings(learner.get(), Carnage::SettingsFor(curriculum, learner->totalTimesteps));
            saveCheckpoint("p0_approved_transition_start", Carnage::PHASE0_NOMINAL_END);
        }
        if (options.scoringConfirmed) {
            Carnage::ConfirmScoring(curriculum, options.scoringRollout);
            ApplySettings(learner.get(), Carnage::SettingsFor(curriculum, learner->totalTimesteps));
            saveCheckpoint("scoring_confirmed", 0);
        }

        const double runtimeEntropy = config.ppo.entropyScale;
        std::cout << std::setprecision(10)
                  << "Guide entropy coefficient: " << Carnage::GUIDE_ENTROPY_COEFFICIENT << '\n'
                  << "GigaLearn entropy formula: H_raw / ln(90), loss -= runtime_coef * H_normalized\n"
                  << "Configured GigaLearn runtime coefficient: " << runtimeEntropy << '\n'
                  << "Action delay: " << options.actionDelay << " physics ticks ("
                  << Carnage::ActionDelaySeconds(options.actionDelay) * 1000.0 << " ms at 120 Hz)\n"
                  << "Device layout: " << options.deviceLayout << '\n';

        if (!options.smokeTest)
            ApplySettings(learner.get(), Carnage::SettingsFor(curriculum, learner->totalTimesteps));
        learner->Start();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Carnage trainer error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
