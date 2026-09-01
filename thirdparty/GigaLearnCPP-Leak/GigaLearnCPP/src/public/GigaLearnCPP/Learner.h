#pragma once

#include <RLGymCPP/EnvSet/EnvSet.h>
#include "Util/MetricSender.h"
#include "Util/RenderSender.h"
#include "LearnerConfig.h"
#include "PPO/TransferLearnConfig.h"

namespace GGL {

	typedef std::function<void(class Learner*, const std::vector<RLGC::GameState>& states, Report& report)> StepCallbackFn;
	typedef std::function<bool(class Learner*, Report& report)> IterationCallbackFn;

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py
	class RG_IMEXPORT Learner {
	public:
		LearnerConfig config;

		RLGC::EnvSet* envSet;

		class PPOLearner* ppo;
		class PolicyVersionManager* versionMgr;

		RLGC::EnvCreateFn envCreateFn;
		MetricSender* metricSender;
		RenderSender* renderSender;

		int obsSize;
		int numActions;

		struct WelfordStat* returnStat;
		struct BatchedWelfordStat* obsStat;

		std::string runID = {};

		uint64_t
			totalTimesteps = 0,
			totalIterations = 0;

		StepCallbackFn stepCallback = NULL;
		IterationCallbackFn iterationCallback = NULL;

		Learner(
			RLGC::EnvCreateFn envCreateFunc,
			LearnerConfig config,
			StepCallbackFn stepCallback = NULL,
			IterationCallbackFn iterationCallback = NULL
		);
		void Start();

		void StartTransferLearn(const TransferLearnConfig& transferLearnConfig);

		void StartQuitKeyThread(bool& quitPressed, std::thread& outThread);

		void Save();
		void Load();
		void LoadFrom(std::filesystem::path path);
		void SaveStats(std::filesystem::path path);
		void LoadStats(std::filesystem::path path);
		void SavePPOTo(std::filesystem::path path);
		void ApplyPPOSettings(
			int64_t timestepsPerIteration,
			int64_t batchSize,
			int64_t minibatchSize,
			int epochs,
			float policyLearningRate,
			float criticLearningRate
		);

		RG_NO_COPY(Learner);

		~Learner();
	};
}
