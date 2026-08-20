#pragma once
#include <RLGymCPP/Rewards/CommonRewards.h>

// speed_toward_ball from the RLGym-PPO guide's early-stage reward stack:
// the speed at which the car is moving towards the ball, normalized by
// CAR_MAX_SPEED and clamped at 0 (no negative reward for moving away).
namespace RLGC {
	class SpeedTowardBallReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			Vec dirToBall = (state.ball.pos - player.pos).Normalized();
			return RS_MAX(0, dirToBall.Dot(player.vel)) / CommonValues::CAR_MAX_SPEED;
		}
	};
}