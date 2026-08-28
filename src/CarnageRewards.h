#pragma once

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/CommonValues.h>
#include <cmath>

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
	// Replaces the old binary TouchBallReward + separate touch
	// acceleration/height signals with ONE continuous touch-quality
	// signal.
	//
	// A touch is scored using three properties:
	//
	//   1. Impact       - how much the ball velocity vector changed
	//   2. Speed change - how much the ball's scalar speed changed
	//   3. Height       - aerial/high-ball touch quality
	//
	// These retain the user's original relative touch ratio:
	//
	//   Impact       : Speed change : Height
	//        5       :     1.5      :   1
	//
	// The three normalized signals are combined into a weighted
	// average and then scaled to a maximum reward of 5.0.
	//
	// Direction/goal value is intentionally NOT evaluated here.
	// Goal/ZeroSum rewards handle whether the resulting play was
	// actually beneficial.
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

			// --------------------------------------------------------
			// 1. Impact quality
			// --------------------------------------------------------
			Vec deltaVelocity =
				state.ball.vel - previousBallVelocity;

			float impact =
				deltaVelocity.Length()
				/ CommonValues::CAR_MAX_SPEED;

			impact = RS_MAX(
				0.0f,
				RS_MIN(impact, 1.0f)
			);


			// --------------------------------------------------------
			// 2. Ball-speed-change quality
			// --------------------------------------------------------
			float previousSpeed =
				previousBallVelocity.Length();

			float currentSpeed =
				state.ball.vel.Length();

			float speedChange =
				std::fabs(currentSpeed - previousSpeed)
				/ CommonValues::CAR_MAX_SPEED;

			speedChange = RS_MAX(
				0.0f,
				RS_MIN(speedChange, 1.0f)
			);


			// --------------------------------------------------------
			// 3. Height quality
			// --------------------------------------------------------
			constexpr float DRIBBLE_HEIGHT = 150.0f;
			constexpr float CEILING_Z = CommonValues::CEILING_Z;

			float heightQuality = 0.0f;
			float ballHeight = state.ball.pos.z;

			if (ballHeight > DRIBBLE_HEIGHT) {
				float normalizedHeight =
					(ballHeight - DRIBBLE_HEIGHT)
					/ (CEILING_Z - DRIBBLE_HEIGHT);

				normalizedHeight = RS_MAX(
					0.0f,
					RS_MIN(normalizedHeight, 1.0f)
				);

				heightQuality =
					normalizedHeight * normalizedHeight;
			}


			// --------------------------------------------------------
			// Preserve the original 5 : 1.5 : 1 ratio
			// --------------------------------------------------------
			constexpr float IMPACT_WEIGHT = 5.0f;
			constexpr float SPEED_CHANGE_WEIGHT = 1.5f;
			constexpr float HEIGHT_WEIGHT = 1.0f;
			constexpr float TOTAL_WEIGHT =
				IMPACT_WEIGHT
				+ SPEED_CHANGE_WEIGHT
				+ HEIGHT_WEIGHT;

			float quality =
				(
					impact * IMPACT_WEIGHT
					+ speedChange * SPEED_CHANGE_WEIGHT
					+ heightQuality * HEIGHT_WEIGHT
				)
				/ TOTAL_WEIGHT;

			// Scale the normalized quality back to the original
			// TouchBall maximum of +5.0.
			return quality * 5.0f;
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
