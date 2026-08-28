#pragma once

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/CommonValues.h>

namespace RLGC {

	// ============================================================
	// Speed Toward Ball
	// ============================================================
	// Guide-equivalent implementation:
	// positive velocity toward the ball / CAR_MAX_SPEED.
	// Moving away from the ball is not punished.
	// ============================================================

	class SpeedTowardBallReward : public Reward {
	public:
		virtual float GetReward(
			const Player& player,
			const GameState& state,
			bool isFinal
		) override {
			Vec posDiff = state.ball.pos - player.pos;
			float distToBall = posDiff.Length();

			// Avoid normalizing a zero-length vector.
			if (distToBall <= 0.001f)
				return 0.0f;

			Vec dirToBall = posDiff / distToBall;
			float speedTowardBall = dirToBall.Dot(player.vel);

			if (speedTowardBall > 0.0f)
				return speedTowardBall / CommonValues::CAR_MAX_SPEED;

			return 0.0f;
		}
	};


	// ============================================================
	// Touch Quality Reward
	// ============================================================
	//
	// Replaces the old binary TouchBallReward.
	//
	// A touch is no longer worth a flat 1.0. Instead, its reward is
	// proportional to how much the touch actually changes the
	// ball's velocity.
	//
	// 0.00 -> negligible touch / no meaningful ball impact
	// 0.25 -> light touch
	// 0.50 -> moderate touch
	// 0.75 -> strong touch
	// 1.00 -> maximum-quality impact
	//
	// Direction is intentionally NOT evaluated here. Goal/ZeroSum
	// rewards are responsible for determining whether the resulting
	// ball movement was actually beneficial.
	// ============================================================

	class TouchQualityReward : public Reward {
	public:
		virtual float GetReward(
			const Player& player,
			const GameState& state,
			bool isFinal
		) override {
			if (!player.ballTouchedStep)
				return 0.0f;

			Vec deltaVelocity =
				state.ball.vel - previousBallVelocity;

			float deltaVelocityMagnitude =
				deltaVelocity.Length();

			float quality =
				deltaVelocityMagnitude / CommonValues::CAR_MAX_SPEED;

			// Keep the quality signal bounded to [0, 1].
			return RS_MAX(
				0.0f,
				RS_MIN(quality, 1.0f)
			);
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
	// Touch Acceleration Reward
	// ============================================================
	//
	// Kept as the second signal in the original 5 : 1.5 : 1
	// touch-ratio stack. This remains available independently from
	// the main TouchQuality signal so the existing weighting is
	// preserved exactly.
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

			Vec deltaVelocity =
				state.ball.vel - previousBallVelocity;

			float deltaSpeed =
				deltaVelocity.Length();

			float reward =
				deltaSpeed / CommonValues::CAR_MAX_SPEED;

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

			if (ballHeight <= DRIBBLE_HEIGHT)
				return 0.0f;

			float normalized =
				(ballHeight - DRIBBLE_HEIGHT)
				/ (CEILING_Z - DRIBBLE_HEIGHT);

			normalized = RS_MAX(
				0.0f,
				RS_MIN(normalized, 1.0f)
			);

			float heightFactor = normalized * normalized;

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
			constexpr float CEILING_Z = CommonValues::CEILING_Z;
			constexpr float CAR_MAX_SPEED = CommonValues::CAR_MAX_SPEED;
			constexpr float JUMP_IMPULSE = 292.0f;
			constexpr float BOOST_ENERGY_PER_100 = 7.87e5f;

			const float max_energy =
				(MASS * GRAVITY * (CEILING_Z - 17.0f))
				+ (0.5f * MASS * (CAR_MAX_SPEED * CAR_MAX_SPEED))
				+ (BOOST_ENERGY_PER_100 * 100.0f)
				+ (0.8f * 0.5f * MASS * (JUMP_IMPULSE * JUMP_IMPULSE))
				+ (0.9f * 0.5f * MASS * (600.0f * 600.0f));

			float energy = 0.0f;

			energy +=
				1.1f * MASS * GRAVITY * player.pos.z;

			float velocity = player.vel.Length();

			energy +=
				0.5f * MASS * (velocity * velocity);

			energy +=
				BOOST_ENERGY_PER_100 * player.boost * 100.0f;

			if (player.hasJumped) {
				energy +=
					0.8f * 0.5f * MASS
					* (JUMP_IMPULSE * JUMP_IMPULSE);
			}

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

			float normalizedEnergy = energy / max_energy;

			return normalizedEnergy * weight_;
		}

	private:
		float weight_;
	};

}
