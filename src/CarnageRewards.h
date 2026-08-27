#pragma once

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/CommonValues.h>

namespace RLGC {

	// ============================================================
	// Speed Toward Ball
	// ============================================================

	class SpeedTowardBallReward : public Reward {
	public:
		virtual float GetReward(
			const Player& player,
			const GameState& state,
			bool isFinal
		) override {
			Vec dirToBall = (state.ball.pos - player.pos).Normalized();

			return RS_MAX(0, dirToBall.Dot(player.vel))
				/ CommonValues::CAR_MAX_SPEED;
		}
	};


	// ============================================================
	// Touch Acceleration Reward
	//
	// Necto-inspired:
	// Rewards changing the ball's velocity when the player touches it.
	//
	// This is intentionally capped so a strange physics interaction
	// cannot produce a massive reward.
	// ============================================================

	class TouchAccelerationReward : public Reward {
	public:
		virtual float GetReward(
			const Player& player,
			const GameState& state,
			bool isFinal
		) override {
			if (!player.ballTouchedStep)
				return 0.0f;

			// Change in ball velocity since the previous state.
			Vec deltaVelocity = state.ball.vel - previousBallVelocity;

			float deltaSpeed = deltaVelocity.Length();

			// Normalize by maximum car speed.
			float reward =
				deltaSpeed / CommonValues::CAR_MAX_SPEED;

			// Keep the raw reward bounded.
			return RS_MIN(reward, 1.0f);
		}

		virtual void Reset(const GameState& state) override {
			previousBallVelocity = state.ball.vel;
		}

		virtual void PreStep(const GameState& state) override {
			previousBallVelocity = state.ball.vel;
		}

	private:
		Vec previousBallVelocity = Vec(0, 0, 0);
	};


	// ============================================================
	// Touch Height Reward
	//
	// Necto-inspired concept, but intentionally much smaller.
	//
	// We DO NOT reward ordinary roof dribbling heavily.
	//
	// Below normal dribble height:
	//     reward = 0
	//
	// Above normal dribble height:
	//     reward rises gradually.
	//
	// The squared curve prevents low/mid-height touches from
	// becoming a large source of reward.
	// ============================================================

	class TouchHeightReward : public Reward {
	public:
		virtual float GetReward(
			const Player& player,
			const GameState& state,
			bool isFinal
		) override {
			if (!player.ballTouchedStep)
				return 0.0f;

			constexpr float DRIBBLE_HEIGHT = 150.0f;
			constexpr float CEILING_Z = CommonValues::CEILING_Z;

			float ballHeight = state.ball.pos.z;

			// Normal ground/roof-dribble region gives no
			// additional height reward.
			if (ballHeight <= DRIBBLE_HEIGHT)
				return 0.0f;

			float normalized =
				(ballHeight - DRIBBLE_HEIGHT)
				/ (CEILING_Z - DRIBBLE_HEIGHT);

			normalized = RS_MAX(
				0.0f,
				RS_MIN(normalized, 1.0f)
			);

			// Quadratic scaling.
			//
			// Small increase above dribble height:
			//     tiny reward
			//
			// Very high aerial touch:
			//     larger reward
			float heightFactor =
				normalized * normalized;

			// Deliberately small.
			return 0.25f * heightFactor;
		}
	};


	// ============================================================
	// Total Energy Reward
	// ============================================================

	class TotalEnergyReward : public Reward {
	public:
		explicit TotalEnergyReward(float weight = 1.0f)
			: weight_(weight) {}

		virtual float GetReward(
			const Player& player,
			const GameState& state,
			bool isFinal
		) override {
			if (player.isDemoed)
				return 0.0f;

			constexpr float MASS = 180.0f;
			constexpr float GRAVITY = 650.0f;
			constexpr float CEILING_Z =
				CommonValues::CEILING_Z;
			constexpr float CAR_MAX_SPEED =
				CommonValues::CAR_MAX_SPEED;
			constexpr float JUMP_IMPULSE = 292.0f;
			constexpr float BOOST_ENERGY_PER_100 = 7.87e5f;

			const float max_energy =
				(MASS * GRAVITY * (CEILING_Z - 17.0f))
				+ (0.5f * MASS *
					(CAR_MAX_SPEED * CAR_MAX_SPEED))
				+ (BOOST_ENERGY_PER_100 * 100.0f)
				+ (0.8f * 0.5f * MASS *
					(JUMP_IMPULSE * JUMP_IMPULSE))
				+ (0.9f * 0.5f * MASS *
					(600.0f * 600.0f));

			float energy = 0.0f;

			// Potential energy.
			energy +=
				1.1f * MASS * GRAVITY * player.pos.z;

			// Kinetic energy.
			float velocity = player.vel.Length();

			energy +=
				0.5f * MASS * (velocity * velocity);

			// Boost energy.
			energy +=
				BOOST_ENERGY_PER_100
				* player.boost
				* 100.0f;

			// Jump energy.
			if (player.hasJumped) {
				energy +=
					0.8f * 0.5f * MASS
					* (JUMP_IMPULSE * JUMP_IMPULSE);
			}

			// Flip/dodge energy.
			if (player.hasFlipped) {
				float dodge_impulse =
					(velocity <= 1700.0f)
					? 500.0f + (velocity / 17.0f)
					: 600.0f - (velocity - 1700.0f);

				dodge_impulse =
					RS_MAX(
						dodge_impulse - 25.0f,
						0.0f
					);

				energy +=
					0.9f * 0.5f * MASS
					* (dodge_impulse * dodge_impulse);
			}

			float normalizedEnergy =
				energy / max_energy;

			return normalizedEnergy * weight_;
		}

	private:
		float weight_;
	};

}
