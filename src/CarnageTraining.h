#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace Carnage {

constexpr std::uint64_t PHASE0_NOMINAL_END = 200'000'000;
constexpr std::uint64_t TRANSITION_LENGTH = 50'000'000;
constexpr std::uint64_t PHASE1_NOMINAL_START = 250'000'000;
constexpr std::uint64_t ROTATING_INTERVAL = 5'000'000;
constexpr std::uint64_t TRAINING_END = 2'000'000'000;

constexpr int OBSERVATION_SIZE = 94;
constexpr int ACTION_COUNT = 90;
constexpr int TICK_SKIP = 8;
constexpr int DEFAULT_ACTION_DELAY = 7;
constexpr double GUIDE_ENTROPY_COEFFICIENT = 0.01;

enum class EntropyNormalization {
    None,
    DivideByLogActionCount,
    DivideByLogValidActionCount
};

double RuntimeEntropyCoefficient(
    double guideEquivalentCoefficient,
    int actionCount,
    EntropyNormalization normalization
);
double GuideEquivalentEntropyCoefficient(
    double runtimeCoefficient,
    int actionCount,
    EntropyNormalization normalization
);

double ActionDelaySeconds(int actionDelay, double physicsTicksPerSecond = 120.0);
bool ShouldApplyDelayedAction(int elapsedTicks, int actionDelay);

enum class Phase {
    Phase0,
    Transition,
    Phase1
};

std::string ToString(Phase phase);

struct RewardWeights {
    double touch = 50.0;
    double speedTowardBall = 5.0;
    double faceBall = 1.0;
    double air = 0.15;
    double velocityBallToGoal = 0.0;
    double goal = 0.0;
};

struct RuntimeSettings {
    Phase phase = Phase::Phase0;
    RewardWeights rewards = {};
    double policyLearningRate = 2e-4;
    double criticLearningRate = 2e-4;
    std::int64_t rolloutSize = 50'000;
    std::int64_t batchSize = 50'000;
    std::int64_t minibatchSize = 25'000;
    int epochs = 3;
    double transitionProgress = 0.0;
};

struct CurriculumState {
    int schemaVersion = 1;
    bool awaitingP0Approval = false;
    bool p0Approved = false;
    bool phase1StartSaved = false;
    bool scoringConfirmed = false;
    int scoringRollout = 100'000;
    std::uint64_t transitionStartStep = 0;
    std::uint64_t transitionLength = TRANSITION_LENGTH;
    std::uint64_t nextRotatingCheckpointStep = ROTATING_INTERVAL;
    std::size_t nextPermanentMilestoneIndex = 0;
};

double TransitionProgress(const CurriculumState& state, std::uint64_t totalTimesteps);
RuntimeSettings SettingsFor(const CurriculumState& state, std::uint64_t totalTimesteps);
void ApprovePhase0(CurriculumState& state, std::uint64_t totalTimesteps);
void ConfirmScoring(CurriculumState& state, int rollout);

struct BoundaryDecision {
    bool save = false;
    bool permanent = false;
    bool stop = false;
    std::string checkpointLabel;
    std::uint64_t nominalMilestone = 0;
    std::vector<std::uint64_t> crossedMilestones;
};

BoundaryDecision ProcessSafeBoundary(CurriculumState& state, std::uint64_t totalTimesteps);

nlohmann::json CurriculumToJson(const CurriculumState& state);
CurriculumState CurriculumFromJson(const nlohmann::json& value);
nlohmann::json CompatibilitySignature(int actionDelay);

struct CheckpointValidation {
    bool valid = false;
    std::string error;
    nlohmann::json metadata;
};

class CheckpointStore {
public:
    using StateWriter = std::function<void(const std::filesystem::path&)>;

    explicit CheckpointStore(std::filesystem::path root);

    std::filesystem::path SaveAtomic(
        const std::string& label,
        std::uint64_t totalTimesteps,
        std::uint64_t totalIterations,
        std::uint64_t nominalMilestone,
        const CurriculumState& curriculum,
        const nlohmann::json& compatibility,
        const std::string& wandbRunId,
        const StateWriter& writer
    ) const;

    CheckpointValidation Validate(
        const std::filesystem::path& checkpoint,
        const nlohmann::json& expectedCompatibility
    ) const;

    std::optional<std::filesystem::path> FindLatestValid(
        const nlohmann::json& expectedCompatibility,
        std::vector<std::string>* diagnostics = nullptr
    ) const;

    CurriculumState ReadCurriculum(const std::filesystem::path& checkpoint) const;
    nlohmann::json ReadMetadata(const std::filesystem::path& checkpoint) const;
    void PruneRotating(std::size_t keep) const;

private:
    std::filesystem::path root_;
};

void InstallSafeStopHandlers();
bool StopRequested();
void RequestStopForTest();
void ResetStopRequestForTest();

} // namespace Carnage
