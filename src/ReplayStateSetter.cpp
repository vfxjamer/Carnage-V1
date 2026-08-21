#include "ReplayStateSetter.h"

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Math.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

using RocketSim::Math::RandFloat;
using namespace RLGC;

std::shared_ptr<std::vector<ReplayFrame>> ReplayStateSetter::_sharedReplayFrames;
std::once_flag ReplayStateSetter::_replayLoadOnce;

namespace {
    std::mt19937& ReplayRng() {
        static thread_local std::mt19937 rng(std::random_device{}());
        return rng;
    }
}

bool ReplayStateSetter::_LoadReplays(const std::string& path) {
    auto frames = std::make_shared<std::vector<ReplayFrame>>();

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "ReplayStateSetter: replay file not found: " << path
                  << ". Falling back to RandomState.\n";
        _sharedReplayFrames = frames;
        return false;
    }

    const std::streamoff fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if (fileSize < static_cast<std::streamoff>(sizeof(int64_t))) {
        std::cerr << "ReplayStateSetter: replay file is too small: " << path << "\n";
        _sharedReplayFrames = frames;
        return false;
    }

    int64_t nFrames = 0;
    file.read(reinterpret_cast<char*>(&nFrames), sizeof(nFrames));

    if (!file.good() || nFrames <= 0) {
        std::cerr << "ReplayStateSetter: invalid frame count: " << nFrames << "\n";
        _sharedReplayFrames = frames;
        return false;
    }

    // Existing serialized_replays.bin format:
    // 11 Vecs + 2 floats + 2 one-byte bools + 2 ints per frame.
    constexpr size_t BYTES_PER_VEC = sizeof(float) * 3;
    constexpr size_t BYTES_PER_FRAME =
        11 * BYTES_PER_VEC +
        2 * sizeof(float) +
        2 +
        2 * sizeof(int);

    const uint64_t expectedSize =
        sizeof(int64_t) + static_cast<uint64_t>(nFrames) * BYTES_PER_FRAME;

    if (static_cast<uint64_t>(fileSize) != expectedSize) {
        std::cerr << "ReplayStateSetter: replay size mismatch. Expected "
                  << expectedSize << " bytes for " << nFrames
                  << " frames, got " << fileSize << ".\n"
                  << "Replay file will NOT be used; falling back to RandomState.\n";
        _sharedReplayFrames = frames;
        return false;
    }

    try {
        frames->resize(static_cast<size_t>(nFrames));
    }
    catch (const std::bad_alloc&) {
        std::cerr << "ReplayStateSetter: unable to allocate " << nFrames
                  << " replay frames. Falling back to RandomState.\n";
        _sharedReplayFrames = std::make_shared<std::vector<ReplayFrame>>();
        return false;
    }

    for (int64_t i = 0; i < nFrames; ++i) {
        ReplayFrame& f = (*frames)[static_cast<size_t>(i)];

        auto readVec = [&](Vec& v) {
            file.read(reinterpret_cast<char*>(&v), sizeof(float) * 3);
        };
        auto readFloat = [&](float& v) {
            file.read(reinterpret_cast<char*>(&v), sizeof(float));
        };
        auto readInt = [&](int& v) {
            file.read(reinterpret_cast<char*>(&v), sizeof(int));
        };
        auto readByte = [&](bool& v) {
            uint8_t b = 0;
            file.read(reinterpret_cast<char*>(&b), 1);
            v = b != 0;
        };

        readVec(f.ballPos);
        readVec(f.ballVel);
        readVec(f.ballAngVel);

        for (int c = 0; c < 2; ++c) {
            readVec(f.carPos[c]);
            readVec(f.carRotEuler[c]);
            readVec(f.carVel[c]);
            readVec(f.carAngVel[c]);
            readFloat(f.carBoost[c]);
            readByte(f.carOnGround[c]);
        }

        readInt(f.blueScore);
        readInt(f.orangeScore);

        if (!file.good()) {
            std::cerr << "ReplayStateSetter: truncated/corrupt replay at frame "
                      << i << "/" << nFrames
                      << ". Falling back to RandomState.\n";
            frames->clear();
            _sharedReplayFrames = frames;
            return false;
        }

        // Never inject NaN/Inf into RocketSim.
        const Vec* vecs[] = {
            &f.ballPos, &f.ballVel, &f.ballAngVel,
            &f.carPos[0], &f.carPos[1],
            &f.carRotEuler[0], &f.carRotEuler[1],
            &f.carVel[0], &f.carVel[1],
            &f.carAngVel[0], &f.carAngVel[1]
        };
        for (const Vec* v : vecs) {
            if (!std::isfinite(v->x) || !std::isfinite(v->y) || !std::isfinite(v->z)) {
                std::cerr << "ReplayStateSetter: non-finite value at frame " << i
                          << ". Falling back to RandomState.\n";
                frames->clear();
                _sharedReplayFrames = frames;
                return false;
            }
        }
    }

    _sharedReplayFrames = frames;
    std::cout << "ReplayStateSetter: Loaded " << frames->size()
              << " replay frames from " << path << "\n";
    return true;
}

int64_t ReplayStateSetter::_SampleReplayFrame() {
    if (!_sharedReplayFrames || _sharedReplayFrames->empty())
        return -1;

    std::uniform_int_distribution<int64_t> dist(
        0, static_cast<int64_t>(_sharedReplayFrames->size() - 1));
    return dist(ReplayRng());
}

void ReplayStateSetter::_SetFromReplay(Arena* arena, const ReplayFrame& frame) {
    BallState bs = {};
    bs.pos = frame.ballPos;
    bs.vel = frame.ballVel;
    bs.angVel = frame.ballAngVel;
    arena->ball->SetState(bs);

    for (Car* car : arena->_cars) {
        const int idx = (car->team == Team::ORANGE) ? 1 : 0;

        CarState cs = {};
        cs.pos = frame.carPos[idx];
        cs.vel = frame.carVel[idx];
        cs.angVel = frame.carAngVel[idx];
        cs.boost = std::clamp(frame.carBoost[idx], 0.f, 100.f);
        cs.isOnGround = frame.carOnGround[idx];
        cs.rotMat = Angle(
            frame.carRotEuler[idx].x,
            frame.carRotEuler[idx].y,
            frame.carRotEuler[idx].z
        ).ToRotMat();

        car->SetState(cs);
    }
}

ReplayStateSetter::ReplayStateSetter(
    const std::string& replayPath,
    float replayProbability
) : _replayProbability(std::clamp(replayProbability, 0.f, 1.f)),
    _randomState(true, true, false) {

    std::call_once(_replayLoadOnce, [&]() {
        _sharedReplayFrames = std::make_shared<std::vector<ReplayFrame>>();
        _LoadReplays(replayPath);
    });
}

void ReplayStateSetter::ResetArena(Arena* arena) {
    if (_sharedReplayFrames && !_sharedReplayFrames->empty() &&
        RandFloat(0.f, 1.f) < _replayProbability) {
        const int64_t index = _SampleReplayFrame();
        if (index >= 0) {
            _SetFromReplay(arena, (*_sharedReplayFrames)[static_cast<size_t>(index)]);
            return;
        }
    }

    _randomState.ResetArena(arena);
}

bool ReplayStateSetter::IsReplayLoaded() const {
    return _sharedReplayFrames && !_sharedReplayFrames->empty();
}

size_t ReplayStateSetter::ReplayFrameCount() const {
    return _sharedReplayFrames ? _sharedReplayFrames->size() : 0;
}
