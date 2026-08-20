#pragma once
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>

// Nexto/Necto-style flat observation for 1v1 or 2v2 (works for any team size).
//   Ball:      pos/5000(3), rotation matrix(9), vel/2300(3), angVel/3(3)   = 18
//   PrevAct:   8 discrete action values                                    = 8
//   BoostPads: 34 pad availability values (1.0 active, else 1/(1+timer))   = 34
//   Self:      pos(3), fwd(3), up(3), vel(3), angVel(3), boost(1),
//              onGround(1), hasFlip(1), isDemoed(1)                        = 19
//   Teammates: 19 per teammate (same block as Self)
//   Opponents: pos(3), fwd(3), up(3), vel(3), angVel(3) per opponent       = 15
// Total for 1v1: 94. Total for 2v2: 128.
namespace RLGC {
	class NextoObs : public ObsBuilder {
	public:
		constexpr static int BALL_DIM = 18;
		constexpr static int PREV_ACTION_DIM = Action::ELEM_AMOUNT; // 8
		constexpr static int PADS_DIM = CommonValues::BOOST_LOCATIONS_AMOUNT; // 34
		constexpr static int SELF_DIM = 19;
		constexpr static int OPPONENT_DIM = 15;

		// 1v1 reference size (the classic Nexto-style 94-dim obs)
		constexpr static size_t OBS_DIM_1V1 = BALL_DIM + PREV_ACTION_DIM + PADS_DIM + SELF_DIM + OPPONENT_DIM; // 94

		static size_t ExpectedDim(int playersPerTeam) {
			return BALL_DIM + PREV_ACTION_DIM + PADS_DIM
				+ SELF_DIM * playersPerTeam
				+ OPPONENT_DIM * playersPerTeam;
		}

		virtual FList BuildObs(const Player& player, const GameState& state) override;
	};
}