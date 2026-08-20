#pragma once
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>

// Nexto/Necto-style flat 94-dim observation for a 1v1 arena
// (the built-in DefaultObs is 89-dim, AdvancedObs is 109-dim).
//
// Layout: Ball(18) + PrevAction(8) + BoostPads(34) + Self(19) + Opponent(15) = 94
//   Ball:      pos/5000(3), rotation matrix(9), vel/2300(3), angVel/3(3)
//   PrevAct:   8 discrete action values
//   BoostPads: 34 pad availability values (1.0 if active, else 1/(1+timer))
//   Self:      pos(3), fwd(3), up(3), vel(3), angVel(3), boost/100(1),
//              onGround(1), hasFlip(1), isDemoed(1)
//   Opponent:  pos(3), fwd(3), up(3), vel(3), angVel(3) only
namespace RLGC {
	class NextoObs : public ObsBuilder {
	public:
		constexpr static int BALL_DIM = 18;
		constexpr static int PREV_ACTION_DIM = Action::ELEM_AMOUNT; // 8
		constexpr static int PADS_DIM = CommonValues::BOOST_LOCATIONS_AMOUNT; // 34
		constexpr static int SELF_DIM = 19;
		constexpr static int OPPONENT_DIM = 15;

		// Only valid for a 1v1 arena (exactly one opponent)
		constexpr static size_t OBS_DIM = BALL_DIM + PREV_ACTION_DIM + PADS_DIM + SELF_DIM + OPPONENT_DIM; // 94

		virtual FList BuildObs(const Player& player, const GameState& state) override;
	};
}