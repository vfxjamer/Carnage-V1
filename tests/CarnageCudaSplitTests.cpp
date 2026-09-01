#include <private/GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <private/GigaLearnCPP/PPO/PPOLearner.h>

#include <torch/cuda.h>

#include <cmath>
#include <iostream>

using namespace GGL;

namespace {

void CopyModel(Model* source, Model* destination) {
    torch::NoGradGuard noGrad;
    const auto sourceParameters = source->parameters();
    const auto destinationParameters = destination->parameters();
    if (sourceParameters.size() != destinationParameters.size())
        throw std::runtime_error("Model parameter layout differs");
    for (std::size_t index = 0; index < sourceParameters.size(); ++index)
        destinationParameters[index].copy_(sourceParameters[index].to(destination->device));
}

ExperienceBuffer MakeExperience() {
    constexpr int batch = 32;
    constexpr int observations = 4;
    constexpr int actions = 3;
    torch::manual_seed(9876);
    ExperienceBuffer experience(123, torch::kCPU);
    experience.data.states = torch::randn({batch, observations});
    experience.data.actions = torch::randint(actions, {batch});
    experience.data.logProbs = torch::full({batch}, std::log(1.0f / actions));
    experience.data.targetValues = torch::randn({batch});
    experience.data.actionMasks = torch::ones({batch, actions}, torch::kUInt8);
    experience.data.advantages = torch::randn({batch});
    return experience;
}

} // namespace

int main() {
    if (!torch::cuda::is_available() || torch::cuda::device_count() < 2) {
        std::cout << "SKIP: two CUDA devices are not available\n";
        return 0;
    }

    PPOLearnerConfig config;
    config.policy.layerSizes = {8, 8};
    config.critic.layerSizes = {8, 8};
    config.sharedHead = {};
    config.batchSize = 32;
    config.miniBatchSize = 16;
    config.epochs = 1;
    config.policyLR = 1e-4f;
    config.criticLR = 1e-4f;
    config.entropyScale = 0.01f;

    torch::manual_seed(1234);
    PPOLearner single(4, 3, config, torch::Device(torch::kCUDA, 0), torch::Device(torch::kCUDA, 0));
    torch::manual_seed(1234);
    PPOLearner split(4, 3, config, torch::Device(torch::kCUDA, 0), torch::Device(torch::kCUDA, 1));
    CopyModel(single.models["policy"], split.models["policy"]);
    CopyModel(single.models["critic"], split.models["critic"]);

    auto singleExperience = MakeExperience();
    auto splitExperience = MakeExperience();
    Report singleReport;
    Report splitReport;
    single.Learn(singleExperience, singleReport, false);
    split.Learn(splitExperience, splitReport, false);

    for (const char* name : {"policy", "critic"}) {
        const auto difference =
            (single.models[name]->CopyParams() - split.models[name]->CopyParams()).abs().max().item<float>();
        if (difference > 2e-5f) {
            std::cerr << name << " update mismatch: " << difference << '\n';
            return 1;
        }
    }
    std::cout << "Single/split CUDA PPO updates match within tolerance\n";
    return 0;
}
