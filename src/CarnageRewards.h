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

	class TotalEnergyReward : public Reward {
	public:
		explicit TotalEnergyReward(float weight = 1.0f) : weight_(weight) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (player.isDemoed) return 0.0f;

			constexpr float MASS = CommonValues::CAR_MASS;
			constexpr float GRAVITY = CommonValues::GRAVITY_Z;
			constexpr float CEILING_Z = CommonValues::CEILING_Z;
			constexpr float CAR_MAX_SPEED = CommonValues::CAR_MAX_SPEED;
			constexpr float JUMP_IMPULSE = 292.0f;
			constexpr float BOOST_ENERGY_PER_100 = 7.87e5f;

			// Max theoretical energy (ceiling height + supersonic + full boost + jump + flip)
			const float max_energy = (MASS * GRAVITY * (CEILING_Z - 17.0f))
			                       + (0.5f * MASS * (CAR_MAX_SPEED * CAR_MAX_SPEED))
			                       + (BOOST_ENERGY_PER_100 * 100.0f)
			                       + (0.8f * 0.5f * MASS * (JUMP_IMPULSE * JUMP_IMPULSE))
			                       + (0.9f * 0.5f * MASS * (600.0f * 600.0f));

			float energy = 0.0f;

			// Potential energy (height) - 1.1 factor encourages aerial play
			energy += 1.1f * MASS * GRAVITY * player.pos.z;

			// Kinetic energy
			float velocity = player.vel.Length();
			energy += 0.5f * MASS * (velocity * velocity);

			// Boost energy (7.87e5 per 100 boost)
			energy += BOOST_ENERGY_PER_100 * player.boost * 100.0f;

			// Jump energy
			if (player.hasJump) {
				energy += 0.8f * 0.5f * MASS * (JUMP_IMPULSE * JUMP_IMPULSE);
			}

			// Flip/dodge energy
			if (player.hasFlip) {
				float dodge_impulse = (velocity <= 1700.0f)
					? 500.0f + (velocity / 17.0f)
					: 600.0f - (velocity - 1700.0f);
				dodge_impulse = RS_MAX(dodge_impulse - 25.0f, 0.0f);
				energy += 0.9f * 0.5f * MASS * (dodge_impulse * dodge_impulse);
			}

			float norm_energy = energy / max_energy;
			return norm_energy * weight_;
		}

	private:
		float weight_;
	};
}