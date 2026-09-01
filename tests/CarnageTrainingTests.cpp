#include "CarnageTraining.h"

#include <cmath>
#include <csignal>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace Carnage;

namespace {

int failures = 0;

void Check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void CheckNear(double actual, double expected, double tolerance, const char* message) {
    Check(std::abs(actual - expected) <= tolerance, message);
}

template <typename Fn>
void CheckThrows(Fn&& function, const char* message) {
    try {
        function();
        Check(false, message);
    } catch (...) {
    }
}

void WriteDummyState(const std::filesystem::path& path, std::uint64_t timesteps) {
    for (const char* name : {"POLICY.LT", "POLICY_OPTIM.LT", "CRITIC.LT", "CRITIC_OPTIM.LT"}) {
        std::ofstream output(path / name, std::ios::binary);
        output << name << "-dummy-state";
    }
    std::ofstream stats(path / "RUNNING_STATS.json");
    stats << "{\"total_timesteps\":" << timesteps << ",\"total_iterations\":1}";
}

void TestEntropy() {
    const double runtime = RuntimeEntropyCoefficient(
        0.01, 90, EntropyNormalization::DivideByLogActionCount
    );
    CheckNear(runtime, 0.044998096, 1e-9, "90-action entropy mapping");
    CheckNear(
        GuideEquivalentEntropyCoefficient(runtime, 90, EntropyNormalization::DivideByLogActionCount),
        0.01,
        1e-12,
        "entropy reverse mapping"
    );
    CheckNear(
        RuntimeEntropyCoefficient(0.01, 90, EntropyNormalization::None),
        0.01,
        1e-12,
        "unnormalized entropy maps directly"
    );
    CheckThrows([] {
        RuntimeEntropyCoefficient(0.01, 90, EntropyNormalization::DivideByLogValidActionCount);
    }, "valid-action normalization must not claim a fixed exact mapping");
}

void TestCurriculum() {
    CurriculumState state;
    auto decision = ProcessSafeBoundary(state, 199'999'999);
    Check(!decision.permanent && !decision.stop, "Phase 0 review must not trigger before threshold");
    decision = ProcessSafeBoundary(state, 200'041'984);
    Check(decision.save && decision.stop && decision.permanent, "Phase 0 crossing gate");
    Check(state.awaitingP0Approval, "awaiting approval persisted in state");
    Check(decision.nominalMilestone == 200'000'000, "Phase 0 nominal milestone");

    ApprovePhase0(state, 200'041'984);
    Check(state.p0Approved && !state.awaitingP0Approval, "approval state transition");
    Check(state.transitionStartStep == 200'041'984, "transition uses actual approved step");

    auto start = SettingsFor(state, state.transitionStartStep);
    auto middle = SettingsFor(state, state.transitionStartStep + 25'000'000);
    auto end = SettingsFor(state, state.transitionStartStep + 50'000'000);
    Check(start.phase == Phase::Transition, "transition starts at 0 percent");
    CheckNear(start.transitionProgress, 0.0, 1e-12, "transition 0 percent");
    CheckNear(middle.transitionProgress, 0.5, 1e-12, "transition 50 percent");
    CheckNear(middle.rewards.touch, 27.5, 1e-12, "touch midpoint interpolation");
    CheckNear(middle.rewards.velocityBallToGoal, 1.0, 1e-12, "ball-to-goal midpoint interpolation");
    CheckNear(middle.policyLearningRate, 1.5e-4, 1e-12, "LR midpoint interpolation");
    Check(end.phase == Phase::Phase1, "transition reaches Phase 1");
    CheckNear(end.transitionProgress, 1.0, 1e-12, "transition 100 percent");
    CheckNear(TransitionProgress(state, state.transitionStartStep + 80'000'000), 1.0, 1e-12, "progress clamps high");
    CheckNear(TransitionProgress(state, state.transitionStartStep - 1), 0.0, 1e-12, "progress clamps low");

    decision = ProcessSafeBoundary(state, state.transitionStartStep + 50'000'000);
    Check(decision.save && decision.permanent, "Phase 1 start checkpoint crossing");
    Check(decision.nominalMilestone == 250'000'000, "Phase 1 nominal milestone");
    Check(state.phase1StartSaved, "Phase 1 checkpoint persistence");

    ConfirmScoring(state, 200'000);
    Check(state.scoringConfirmed, "scoring confirmation persists");
    Check(SettingsFor(state, state.transitionStartStep + 50'000'001).rolloutSize == 200'000,
          "100k to 200k scoring rollout switch");
    ConfirmScoring(state, 300'000);
    Check(SettingsFor(state, state.transitionStartStep + 50'000'001).rolloutSize == 300'000,
          "100k to 300k scoring rollout switch");
    CheckThrows([&] { ConfirmScoring(state, 250'000); }, "invalid scoring rollout rejection");
}

void TestRotatingThresholds() {
    CurriculumState state;
    auto decision = ProcessSafeBoundary(state, 5'012'000);
    Check(decision.save && !decision.permanent, "5M rotating threshold crossing");
    Check(state.nextRotatingCheckpointStep == 10'000'000, "next rotating target advances once");

    state = CurriculumState{};
    decision = ProcessSafeBoundary(state, 17'000'000);
    Check(decision.save, "multi-interval crossing saves once");
    Check(state.nextRotatingCheckpointStep == 20'000'000, "multi-interval crossing catches target up");
}

void TestCheckpointValidation() {
    const auto root = std::filesystem::temp_directory_path() /
        ("carnage-tests-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    try {
        CheckpointStore store(root);
        CurriculumState state;
        const auto signature = CompatibilitySignature(7);
        const auto first = store.SaveAtomic(
            "rotating", 5'010'000, 1, 0, state, signature, "wandb-test",
            [](const auto& path) { WriteDummyState(path, 5'010'000); }
        );
        Check(store.Validate(first, signature).valid, "complete checkpoint validates");
        Check(std::filesystem::is_regular_file(root / "LATEST.json"), "latest pointer published");

        const auto incomplete = root / "incomplete";
        std::filesystem::create_directories(incomplete);
        std::ofstream(incomplete / "CARNAGE_METADATA.json") << "{}";
        Check(!store.Validate(incomplete, signature).valid, "incomplete checkpoint rejected");

        auto legacy = signature;
        legacy["policy_layers"] = {1024, 1024, 512, 512};
        Check(!store.Validate(first, legacy).valid, "legacy network signature rejected");

        const auto second = store.SaveAtomic(
            "rotating", 10'020'000, 2, 0, state, signature, "wandb-test",
            [](const auto& path) { WriteDummyState(path, 10'020'000); }
        );
        std::ofstream(second / "POLICY.LT", std::ios::app) << "corruption";
        std::vector<std::string> diagnostics;
        const auto fallback = store.FindLatestValid(signature, &diagnostics);
        Check(fallback && *fallback == first, "corrupt latest falls back to previous valid checkpoint");
        Check(!diagnostics.empty(), "corrupt checkpoint produces diagnostics");
    } catch (...) {
        std::filesystem::remove_all(root);
        throw;
    }
    std::filesystem::remove_all(root);
}

void TestSafeStopAndDelay() {
    ResetStopRequestForTest();
    Check(!StopRequested(), "stop flag begins clear");
    InstallSafeStopHandlers();
    std::raise(SIGTERM);
    Check(StopRequested(), "SIGTERM only raises the observable stop request");
    ResetStopRequestForTest();
    CheckNear(ActionDelaySeconds(7), 7.0 / 120.0, 1e-12, "action delay is seven physics ticks");
    Check(DEFAULT_ACTION_DELAY < TICK_SKIP, "new action receives one tick in its inference interval");
    Check(!ShouldApplyDelayedAction(6, 7), "RLBot keeps the old action through tick six");
    Check(ShouldApplyDelayedAction(7, 7), "RLBot applies the new action at tick seven");
}

} // namespace

int main() {
    TestEntropy();
    TestCurriculum();
    TestRotatingThresholds();
    TestCheckpointValidation();
    TestSafeStopAndDelay();
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All Carnage training tests passed\n";
    return 0;
}
