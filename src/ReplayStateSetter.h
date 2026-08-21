#pragma once

#include <RLGymCPP/StateSetters/StateSetter.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/Framework.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace RLGC;

// Binary replay format used by the existing serialized_replays.bin dataset.
// Keep this layout in sync with the replay serializer.
struct ReplayFrame {
    Vec ballPos;
    Vec ballVel;
    Vec ballAngVel;

    Vec carPos[2];
    Vec carRotEuler[2];
    Vec carVel[2];
    Vec carAngVel[2];

    float carBoost[2];
    bool carOnGround[2];

    int blueScore;
    int orangeScore;
};

// Starts episodes from real Rocket League replay states while retaining a
// configurable RandomState fallback. Replay states are sampled uniformly.
//
// The replay pool is loaded once and shared by every environment, so 1024
// parallel games do not each keep their own 1.8M-frame copy.
class ReplayStateSetter : public StateSetter {
public:
    ReplayStateSetter(const std::string& replayPath, float replayProbability = 0.30f);

    void ResetArena(Arena* arena) override;

    bool IsReplayLoaded() const;
    size_t ReplayFrameCount() const;

private:
    static std::shared_ptr<std::vector<ReplayFrame>> _sharedReplayFrames;
    static std::once_flag _replayLoadOnce;

    float _replayProbability;
    RandomState _randomState;

    static bool _LoadReplays(const std::string& path);
    static int64_t _SampleReplayFrame();
    static void _SetFromReplay(Arena* arena, const ReplayFrame& frame);
};
