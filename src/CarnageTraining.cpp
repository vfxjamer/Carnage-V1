#include "CarnageTraining.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

namespace Carnage {
namespace {

constexpr std::array<std::uint64_t, 4> PERMANENT_MILESTONES = {
    500'000'000ULL,
    1'000'000'000ULL,
    1'500'000'000ULL,
    2'000'000'000ULL
};

constexpr std::array<const char*, 5> REQUIRED_FILES = {
    "POLICY.lt",
    "POLICY_OPTIM.lt",
    "CRITIC.lt",
    "CRITIC_OPTIM.lt",
    "RUNNING_STATS.json"
};

volatile std::sig_atomic_t gStopRequested = 0;

void StopSignalHandler(int) {
    gStopRequested = 1;
}

double Lerp(double from, double to, double progress) {
    return from + (to - from) * progress;
}

std::string FileHash(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Cannot hash " + path.string());

    std::uint64_t hash = 14695981039346656037ULL;
    std::array<char, 64 * 1024> buffer = {};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        for (std::streamsize i = 0; i < count; ++i) {
            hash ^= static_cast<unsigned char>(buffer[static_cast<std::size_t>(i)]);
            hash *= 1099511628211ULL;
        }
    }

    std::ostringstream result;
    result << std::hex << std::setfill('0') << std::setw(16) << hash;
    return result.str();
}

void WriteJson(const std::filesystem::path& path, const nlohmann::json& value) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("Cannot write " + path.string());
    output << value.dump(2) << '\n';
    output.flush();
    if (!output)
        throw std::runtime_error("Failed while writing " + path.string());
}

nlohmann::json ReadJson(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Cannot read " + path.string());
    nlohmann::json value;
    input >> value;
    return value;
}

void AtomicReplaceFile(const std::filesystem::path& source, const std::filesystem::path& target) {
#ifdef _WIN32
    if (!MoveFileExW(
            source.c_str(),
            target.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
        )) {
        throw std::runtime_error("Cannot atomically publish " + target.string());
    }
#else
    if (std::rename(source.c_str(), target.c_str()) != 0)
        throw std::runtime_error("Cannot atomically publish " + target.string());
#endif
}

void AtomicPublishDirectory(const std::filesystem::path& source, const std::filesystem::path& target) {
#ifdef _WIN32
    constexpr int ATTEMPTS = 100;
    for (int attempt = 0; attempt < ATTEMPTS; ++attempt) {
        if (MoveFileExW(source.c_str(), target.c_str(), MOVEFILE_WRITE_THROUGH))
            return;
        const DWORD error = GetLastError();
        if (error != ERROR_ACCESS_DENIED && error != ERROR_SHARING_VIOLATION)
            throw std::runtime_error("Cannot publish checkpoint directory, Windows error " + std::to_string(error));
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    throw std::runtime_error("Checkpoint directory remained locked for five seconds: " + source.string());
#else
    if (std::rename(source.c_str(), target.c_str()) != 0)
        throw std::runtime_error("Cannot publish checkpoint directory " + target.string());
#endif
}

std::string SafeLabel(std::string label) {
    for (char& value : label) {
        const bool allowed =
            (value >= 'a' && value <= 'z') ||
            (value >= 'A' && value <= 'Z') ||
            (value >= '0' && value <= '9') || value == '_' || value == '-';
        if (!allowed)
            value = '_';
    }
    return label;
}

std::string UniqueSuffix() {
    std::random_device random;
    std::ostringstream suffix;
    suffix << std::hex << random() << random();
    return suffix.str();
}

} // namespace

double RuntimeEntropyCoefficient(
    double guideEquivalentCoefficient,
    int actionCount,
    EntropyNormalization normalization
) {
    if (guideEquivalentCoefficient < 0.0)
        throw std::invalid_argument("Entropy coefficient cannot be negative");
    if (normalization == EntropyNormalization::None)
        return guideEquivalentCoefficient;
    if (normalization == EntropyNormalization::DivideByLogValidActionCount)
        throw std::invalid_argument(
            "A fixed coefficient cannot exactly undo per-sample valid-action normalization"
        );
    if (actionCount <= 1)
        throw std::invalid_argument("Entropy action count must be greater than one");
    return guideEquivalentCoefficient * std::log(static_cast<double>(actionCount));
}

double GuideEquivalentEntropyCoefficient(
    double runtimeCoefficient,
    int actionCount,
    EntropyNormalization normalization
) {
    if (normalization == EntropyNormalization::None)
        return runtimeCoefficient;
    if (normalization == EntropyNormalization::DivideByLogValidActionCount)
        throw std::invalid_argument(
            "A fixed coefficient cannot exactly undo per-sample valid-action normalization"
        );
    if (actionCount <= 1)
        throw std::invalid_argument("Entropy action count must be greater than one");
    return runtimeCoefficient / std::log(static_cast<double>(actionCount));
}

double ActionDelaySeconds(int actionDelay, double physicsTicksPerSecond) {
    if (actionDelay < 0 || physicsTicksPerSecond <= 0.0)
        throw std::invalid_argument("Invalid action-delay timing inputs");
    return static_cast<double>(actionDelay) / physicsTicksPerSecond;
}

bool ShouldApplyDelayedAction(int elapsedTicks, int actionDelay) {
    if (actionDelay < 0)
        throw std::invalid_argument("Action delay cannot be negative");
    return elapsedTicks == -1 || elapsedTicks >= actionDelay;
}

std::string ToString(Phase phase) {
    switch (phase) {
        case Phase::Phase0: return "phase0";
        case Phase::Transition: return "transition";
        case Phase::Phase1: return "phase1";
    }
    throw std::invalid_argument("Unknown curriculum phase");
}

double TransitionProgress(const CurriculumState& state, std::uint64_t totalTimesteps) {
    if (!state.p0Approved || state.transitionLength == 0)
        return 0.0;
    if (totalTimesteps <= state.transitionStartStep)
        return 0.0;
    const double progress = static_cast<double>(totalTimesteps - state.transitionStartStep) /
        static_cast<double>(state.transitionLength);
    return std::clamp(progress, 0.0, 1.0);
}

RuntimeSettings SettingsFor(const CurriculumState& state, std::uint64_t totalTimesteps) {
    RuntimeSettings settings;
    if (!state.p0Approved) {
        settings.phase = Phase::Phase0;
        return settings;
    }

    const double progress = TransitionProgress(state, totalTimesteps);
    if (progress < 1.0) {
        settings.phase = Phase::Transition;
        settings.transitionProgress = progress;
        settings.rewards.touch = Lerp(50.0, 5.0, progress);
        settings.rewards.speedTowardBall = Lerp(5.0, 1.0, progress);
        settings.rewards.faceBall = Lerp(1.0, 0.1, progress);
        settings.rewards.air = 0.15;
        settings.rewards.velocityBallToGoal = Lerp(0.0, 2.0, progress);
        settings.rewards.goal = Lerp(0.0, 20.0, progress);
        settings.policyLearningRate = Lerp(2e-4, 1e-4, progress);
        settings.criticLearningRate = Lerp(2e-4, 1e-4, progress);
        settings.rolloutSize = 100'000;
        settings.batchSize = 100'000;
        return settings;
    }

    settings.phase = Phase::Phase1;
    settings.transitionProgress = 1.0;
    settings.rewards = {5.0, 1.0, 0.1, 0.15, 2.0, 20.0};
    settings.policyLearningRate = 1e-4;
    settings.criticLearningRate = 1e-4;
    settings.rolloutSize = state.scoringConfirmed ? state.scoringRollout : 100'000;
    settings.batchSize = settings.rolloutSize;
    return settings;
}

void ApprovePhase0(CurriculumState& state, std::uint64_t totalTimesteps) {
    if (!state.awaitingP0Approval)
        throw std::logic_error("Phase 0 can only be approved at its review checkpoint");
    state.p0Approved = true;
    state.awaitingP0Approval = false;
    state.transitionStartStep = totalTimesteps;
    state.transitionLength = TRANSITION_LENGTH;
}

void ConfirmScoring(CurriculumState& state, int rollout) {
    if (!state.phase1StartSaved)
        throw std::logic_error("Scoring rollout can only be confirmed after Phase 1 starts");
    if (rollout != 200'000 && rollout != 300'000)
        throw std::invalid_argument("Scoring rollout must be 200000 or 300000");
    state.scoringConfirmed = true;
    state.scoringRollout = rollout;
}

BoundaryDecision ProcessSafeBoundary(CurriculumState& state, std::uint64_t totalTimesteps) {
    BoundaryDecision decision;

    if (!state.p0Approved && !state.awaitingP0Approval && totalTimesteps >= PHASE0_NOMINAL_END) {
        state.awaitingP0Approval = true;
        decision.save = true;
        decision.permanent = true;
        decision.stop = true;
        decision.checkpointLabel = "200M_phase0_final";
        decision.nominalMilestone = PHASE0_NOMINAL_END;
        decision.crossedMilestones.push_back(PHASE0_NOMINAL_END);
    }

    if (state.p0Approved && !state.phase1StartSaved &&
        TransitionProgress(state, totalTimesteps) >= 1.0) {
        state.phase1StartSaved = true;
        decision.save = true;
        decision.permanent = true;
        decision.checkpointLabel = "250M_phase1_start";
        decision.nominalMilestone = PHASE1_NOMINAL_START;
        decision.crossedMilestones.push_back(PHASE1_NOMINAL_START);
    }

    std::uint64_t highestPermanent = 0;
    while (state.nextPermanentMilestoneIndex < PERMANENT_MILESTONES.size() &&
           totalTimesteps >= PERMANENT_MILESTONES[state.nextPermanentMilestoneIndex]) {
        highestPermanent = PERMANENT_MILESTONES[state.nextPermanentMilestoneIndex];
        decision.crossedMilestones.push_back(highestPermanent);
        ++state.nextPermanentMilestoneIndex;
    }
    if (highestPermanent != 0) {
        decision.save = true;
        decision.permanent = true;
        decision.nominalMilestone = highestPermanent;
        if (highestPermanent == 500'000'000ULL) decision.checkpointLabel = "500M_milestone";
        if (highestPermanent == 1'000'000'000ULL) decision.checkpointLabel = "1B_milestone";
        if (highestPermanent == 1'500'000'000ULL) decision.checkpointLabel = "1_5B_milestone";
        if (highestPermanent == 2'000'000'000ULL) decision.checkpointLabel = "2B_final";
    }

    bool crossedRotating = false;
    while (totalTimesteps >= state.nextRotatingCheckpointStep) {
        crossedRotating = true;
        state.nextRotatingCheckpointStep += ROTATING_INTERVAL;
    }
    if (crossedRotating && !decision.save) {
        decision.save = true;
        decision.checkpointLabel = "rotating";
    }

    if (totalTimesteps >= TRAINING_END)
        decision.stop = true;
    return decision;
}

nlohmann::json CurriculumToJson(const CurriculumState& state) {
    return {
        {"schema_version", state.schemaVersion},
        {"awaiting_p0_approval", state.awaitingP0Approval},
        {"p0_approved", state.p0Approved},
        {"phase1_start_saved", state.phase1StartSaved},
        {"scoring_confirmed", state.scoringConfirmed},
        {"scoring_rollout", state.scoringRollout},
        {"transition_start_step", state.transitionStartStep},
        {"transition_length", state.transitionLength},
        {"transition_end_step", state.transitionStartStep + state.transitionLength},
        {"next_rotating_checkpoint_step", state.nextRotatingCheckpointStep},
        {"next_permanent_milestone_index", state.nextPermanentMilestoneIndex}
    };
}

CurriculumState CurriculumFromJson(const nlohmann::json& value) {
    CurriculumState state;
    state.schemaVersion = value.at("schema_version").get<int>();
    if (state.schemaVersion != 1)
        throw std::runtime_error("Unsupported curriculum metadata schema");
    state.awaitingP0Approval = value.at("awaiting_p0_approval").get<bool>();
    state.p0Approved = value.at("p0_approved").get<bool>();
    state.phase1StartSaved = value.at("phase1_start_saved").get<bool>();
    state.scoringConfirmed = value.at("scoring_confirmed").get<bool>();
    state.scoringRollout = value.at("scoring_rollout").get<int>();
    state.transitionStartStep = value.at("transition_start_step").get<std::uint64_t>();
    state.transitionLength = value.at("transition_length").get<std::uint64_t>();
    state.nextRotatingCheckpointStep = value.at("next_rotating_checkpoint_step").get<std::uint64_t>();
    state.nextPermanentMilestoneIndex = value.at("next_permanent_milestone_index").get<std::size_t>();
    if (state.transitionLength != TRANSITION_LENGTH)
        throw std::runtime_error("Checkpoint transition length is incompatible");
    if (state.scoringConfirmed && state.scoringRollout != 200'000 && state.scoringRollout != 300'000)
        throw std::runtime_error("Checkpoint scoring rollout is invalid");
    return state;
}

nlohmann::json CompatibilitySignature(int actionDelay) {
    return {
        {"signature_version", 1},
        {"observation_size", OBSERVATION_SIZE},
        {"action_count", ACTION_COUNT},
        {"tick_skip", TICK_SKIP},
        {"action_delay", actionDelay},
        {"team_size", 1},
        {"policy_layers", {2048, 2048, 1024, 1024}},
        {"critic_layers", {2048, 2048, 1024, 1024}},
        {"activation", "relu"},
        {"layer_norm", true},
        {"shared_head", false},
        {"entropy_normalization", "divide_by_log_action_count"},
        {"mask_entropy", false}
    };
}

CheckpointStore::CheckpointStore(std::filesystem::path root) : root_(std::move(root)) {}

std::filesystem::path CheckpointStore::SaveAtomic(
    const std::string& label,
    std::uint64_t totalTimesteps,
    std::uint64_t totalIterations,
    std::uint64_t nominalMilestone,
    const CurriculumState& curriculum,
    const nlohmann::json& compatibility,
    const std::string& wandbRunId,
    const StateWriter& writer
) const {
    std::filesystem::create_directories(root_);
    const std::string finalName = SafeLabel(label) + "_" + std::to_string(totalTimesteps) +
        "_i" + std::to_string(totalIterations);
    auto finalPath = root_ / finalName;
    if (std::filesystem::exists(finalPath))
        finalPath = root_ / (finalName + "-" + UniqueSuffix());
    const auto temporaryPath = root_ / ("." + finalName + ".tmp-" + UniqueSuffix());

    std::filesystem::create_directories(temporaryPath);
    try {
        writer(temporaryPath);

        nlohmann::json files = nlohmann::json::object();
        for (const char* required : REQUIRED_FILES) {
            const auto path = temporaryPath / required;
            if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) == 0)
                throw std::runtime_error(std::string("Missing or empty checkpoint file: ") + required);
            files[required] = {
                {"size", std::filesystem::file_size(path)},
                {"fnv1a64", FileHash(path)}
            };
        }

        const RuntimeSettings settings = SettingsFor(curriculum, totalTimesteps);
        nlohmann::json metadata = {
            {"metadata_schema_version", 1},
            {"checkpoint_label", label},
            {"nominal_milestone", nominalMilestone},
            {"total_timesteps", totalTimesteps},
            {"total_iterations", totalIterations},
            {"curriculum_phase", ToString(settings.phase)},
            {"transition_progress", settings.transitionProgress},
            {"wandb_run_id", wandbRunId},
            {"compatibility", compatibility},
            {"curriculum", CurriculumToJson(curriculum)},
            {"files", files}
        };
        WriteJson(temporaryPath / "CARNAGE_METADATA.json", metadata);

        const auto validation = Validate(temporaryPath, compatibility);
        if (!validation.valid)
            throw std::runtime_error("Temporary checkpoint validation failed: " + validation.error);
        AtomicPublishDirectory(temporaryPath, finalPath);

        const auto latestTemporary = root_ / (".LATEST.tmp-" + UniqueSuffix());
        WriteJson(latestTemporary, {
            {"checkpoint", finalPath.filename().string()},
            {"total_timesteps", totalTimesteps},
            {"total_iterations", totalIterations}
        });
        AtomicReplaceFile(latestTemporary, root_ / "LATEST.json");
        return finalPath;
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove_all(temporaryPath, ignored);
        throw;
    }
}

CheckpointValidation CheckpointStore::Validate(
    const std::filesystem::path& checkpoint,
    const nlohmann::json& expectedCompatibility
) const {
    CheckpointValidation result;
    try {
        if (!std::filesystem::is_directory(checkpoint))
            throw std::runtime_error("Not a checkpoint directory");
        const auto metadataPath = checkpoint / "CARNAGE_METADATA.json";
        if (!std::filesystem::is_regular_file(metadataPath))
            throw std::runtime_error("Missing CARNAGE_METADATA.json");
        result.metadata = ReadJson(metadataPath);
        if (result.metadata.at("metadata_schema_version").get<int>() != 1)
            throw std::runtime_error("Unsupported checkpoint metadata schema");
        if (result.metadata.at("compatibility") != expectedCompatibility)
            throw std::runtime_error("Compatibility signature mismatch");
        CurriculumFromJson(result.metadata.at("curriculum"));

        const auto stats = ReadJson(checkpoint / "RUNNING_STATS.json");
        if (stats.at("total_timesteps").get<std::uint64_t>() !=
            result.metadata.at("total_timesteps").get<std::uint64_t>()) {
            throw std::runtime_error("Stats and metadata timestep counters differ");
        }

        const auto& files = result.metadata.at("files");
        for (const char* required : REQUIRED_FILES) {
            const auto path = checkpoint / required;
            if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) == 0)
                throw std::runtime_error(std::string("Missing or empty checkpoint file: ") + required);
            const auto& entry = files.at(required);
            if (entry.at("size").get<std::uintmax_t>() != std::filesystem::file_size(path) ||
                entry.at("fnv1a64").get<std::string>() != FileHash(path)) {
                throw std::runtime_error(std::string("Checkpoint file integrity failure: ") + required);
            }
        }
        result.valid = true;
    } catch (const std::exception& error) {
        result.error = error.what();
    }
    return result;
}

std::optional<std::filesystem::path> CheckpointStore::FindLatestValid(
    const nlohmann::json& expectedCompatibility,
    std::vector<std::string>* diagnostics
) const {
    if (!std::filesystem::is_directory(root_))
        return std::nullopt;

    struct Candidate {
        std::filesystem::path path;
        std::uint64_t timesteps;
        std::uint64_t iterations;
    };
    std::vector<Candidate> candidates;
    for (const auto& entry : std::filesystem::directory_iterator(root_)) {
        if (!entry.is_directory() || entry.path().filename().string().starts_with("."))
            continue;
        try {
            const auto metadata = ReadJson(entry.path() / "CARNAGE_METADATA.json");
            candidates.push_back({
                entry.path(),
                metadata.at("total_timesteps").get<std::uint64_t>(),
                metadata.at("total_iterations").get<std::uint64_t>()
            });
        } catch (const std::exception& error) {
            if (diagnostics)
                diagnostics->push_back(entry.path().string() + ": " + error.what());
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& left, const Candidate& right) {
        return std::tie(left.timesteps, left.iterations) > std::tie(right.timesteps, right.iterations);
    });
    for (const Candidate& candidate : candidates) {
        const auto validation = Validate(candidate.path, expectedCompatibility);
        if (validation.valid)
            return candidate.path;
        if (diagnostics)
            diagnostics->push_back(candidate.path.string() + ": " + validation.error);
    }
    return std::nullopt;
}

CurriculumState CheckpointStore::ReadCurriculum(const std::filesystem::path& checkpoint) const {
    return CurriculumFromJson(ReadMetadata(checkpoint).at("curriculum"));
}

nlohmann::json CheckpointStore::ReadMetadata(const std::filesystem::path& checkpoint) const {
    return ReadJson(checkpoint / "CARNAGE_METADATA.json");
}

void CheckpointStore::PruneRotating(std::size_t keep) const {
    struct Rotating {
        std::filesystem::path path;
        std::uint64_t timesteps;
    };
    std::vector<Rotating> checkpoints;
    if (!std::filesystem::is_directory(root_))
        return;
    for (const auto& entry : std::filesystem::directory_iterator(root_)) {
        if (!entry.is_directory())
            continue;
        try {
            const auto metadata = ReadJson(entry.path() / "CARNAGE_METADATA.json");
            if (metadata.at("checkpoint_label").get<std::string>() == "rotating") {
                checkpoints.push_back({
                    entry.path(), metadata.at("total_timesteps").get<std::uint64_t>()
                });
            }
        } catch (...) {
        }
    }
    std::sort(checkpoints.begin(), checkpoints.end(), [](const Rotating& left, const Rotating& right) {
        return left.timesteps > right.timesteps;
    });
    for (std::size_t index = keep; index < checkpoints.size(); ++index)
        std::filesystem::remove_all(checkpoints[index].path);
}

void InstallSafeStopHandlers() {
    std::signal(SIGINT, StopSignalHandler);
    std::signal(SIGTERM, StopSignalHandler);
}

bool StopRequested() {
    return gStopRequested != 0;
}

void RequestStopForTest() {
    gStopRequested = 1;
}

void ResetStopRequestForTest() {
    gStopRequested = 0;
}

} // namespace Carnage
